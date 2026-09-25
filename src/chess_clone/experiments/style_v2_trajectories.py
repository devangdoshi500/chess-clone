"""Frozen, development-only matched autonomous-play diagnostic for v1 residuals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter

import chess
from catboost import CatBoostRanker
import numpy as np
import pyarrow.parquet as pq

from chess_clone.experiments.generated_rollout import move_record, pawn_structure
from chess_clone.experiments.sampled_policy import sample_move
from chess_clone.modeling.boosted import predict_relevance_scores
from chess_clone.modeling.legal_policy import build_all_legal_inference_rows
from chess_clone.modeling.style_residual import DIMENSION, VERSION, predict_residual, shared_residual

ARMS = ("population", "shared", "wrong", "personal")
PIECES = ("pawn", "knight", "bishop", "rook", "queen", "king")
WINGS = ("queenside", "center", "kingside")
PHASES = ("opening", "middlegame", "endgame")
BEHAVIOR = {
    "capture": "is_capture", "check": "is_check", "castle": "is_castle",
    "queen_trade": "is_queen_trade", "pawn_push": "is_pawn_push",
    "queen_move": "is_queen_move", "opponent_territory": "is_opponent_territory",
    "king_zone": "is_king_zone",
}
WHITE_OPENINGS = {"e2e4": "white_e4", "d2d4": "white_d4", "c2c4": "white_c4", "g1f3": "white_nf3"}
BLACK_OPENINGS = {"e7e5": "black_e5", "c7c5": "black_c5", "e7e6": "black_e6",
                  "c7c6": "black_c6", "d7d5": "black_d5", "g8f6": "black_nf6"}
PAWNS = ("remaining_pawns", "doubled_pawns", "isolated_pawns", "pawn_islands")
FAMILIES = {
    "behavior": tuple(BEHAVIOR), "piece": PIECES, "wing": WINGS, "phase": PHASES,
    "sequence": ("repeat_piece", "repeat_wing", "capture_after_capture"),
    "pawn_structure": PAWNS,
    "opening_choice": (*WHITE_OPENINGS.values(), "white_other", *BLACK_OPENINGS.values(), "black_other"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_sealed(path: Path, value: object) -> None:
    _write_json(path, value)
    _write_json(path.with_suffix(path.suffix + ".seal"), {"sha256": digest(path)})


def load_sealed(path: Path) -> dict:
    seal = json.loads(path.with_suffix(path.suffix + ".seal").read_text())
    if digest(path) != seal["sha256"]:
        raise ValueError(f"Changed sealed artifact: {path}")
    return json.loads(path.read_text())


def uniform(seed: int, player: str, rollout: int, ply: int, role: str) -> float:
    value = hashlib.sha256(f"{seed}:{player}:{rollout}:{ply}:{role}".encode()).digest()
    return int.from_bytes(value[:8], "big") / 2**64


def first_color(seed: int, player: str) -> bool:
    value = hashlib.sha256(f"{seed}:{player}:color".encode()).digest()
    return not bool(value[0] & 1)


def color_for(seed: int, player: str, rollout: int) -> chess.Color:
    return chess.WHITE if first_color(seed, player) == (rollout % 2 == 0) else chess.BLACK


def _ordered_games(seed: int, player: str, games: list[str]) -> list[str]:
    return sorted(games, key=lambda game: (hashlib.sha256(
        f"{seed}:{player}:{game}".encode()).hexdigest(), game))


def _move_record(board: chess.Board, uci: str, *, game_key: str, ply: int) -> dict:
    """Match the leakage-safe legal-candidate behavioral definitions."""
    record = move_record(board, uci, game_key=game_key, ply=ply)
    move = chess.Move.from_uci(uci)
    piece = board.piece_at(move.from_square)
    opponent_king = board.king(not board.turn)
    post = board.copy(stack=False)
    post.push(move)
    king_zone = (post.attacks(opponent_king) | chess.BB_SQUARES[opponent_king]
                 if opponent_king is not None else chess.SquareSet())
    rank = chess.square_rank(move.to_square)
    record.update(
        is_pawn_push=piece.piece_type == chess.PAWN,
        is_queen_move=piece.piece_type == chess.QUEEN,
        is_opponent_territory=rank >= 4 if board.turn == chess.WHITE else rank <= 3,
        is_king_zone=bool(post.attacks(move.to_square) & king_zone),
    )
    return record


def _human_game(rows: list[dict], player: str, game_id: str) -> dict:
    moves = []
    last_board = None
    color = None
    for row in sorted(rows, key=lambda item: item["ply"]):
        if str(row["player_username"]).casefold() != player:
            raise ValueError("Wrong player in human positions")
        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["actual_move_uci"])
        if move not in board.legal_moves:
            raise ValueError("Illegal recorded human move")
        local_color = chess.WHITE if row["player_color"] == "white" else chess.BLACK
        if color is not None and color != local_color:
            raise ValueError("Inconsistent human game color")
        color = local_color
        moves.append(_move_record(board, move.uci(), game_key=game_id, ply=int(row["ply"])))
        board.push(move)
        last_board = board
    return {"game_key": game_id, "moves": moves,
            "color": None if color is None else ("white" if color else "black"),
            "last_target_pawns": pawn_structure(last_board, color) if last_board is not None else None}


def style_vector(games: list[dict]) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """All rates have explicit target-move, transition, color, or game support."""
    moves = [move for game in games for move in game["moves"]]
    transitions = [(left, right) for game in games
                   for left, right in zip(game["moves"], game["moves"][1:])]
    pawns = [game["last_target_pawns"] for game in games if game["last_target_pawns"] is not None]
    openings = {color: [game["moves"][0]["move_uci"] for game in games
                        if game["color"] == color and game["moves"]] for color in ("white", "black")}
    out = {family: {} for family in FAMILIES}
    for name, field in BEHAVIOR.items():
        out["behavior"][name] = mean(float(move[field]) for move in moves) if moves else 0.0
    for category in PIECES:
        out["piece"][category] = sum(move["piece"] == category for move in moves) / len(moves) if moves else 0.0
    for category in WINGS:
        out["wing"][category] = sum(move["wing"] == category for move in moves) / len(moves) if moves else 0.0
    for category in PHASES:
        out["phase"][category] = sum(move["phase"] == category for move in moves) / len(moves) if moves else 0.0
    out["sequence"] = {
        "repeat_piece": sum(a["piece"] == b["piece"] for a, b in transitions) / len(transitions) if transitions else 0.0,
        "repeat_wing": sum(a["wing"] == b["wing"] for a, b in transitions) / len(transitions) if transitions else 0.0,
        "capture_after_capture": sum(a["is_capture"] and b["is_capture"] for a, b in transitions) / len(transitions) if transitions else 0.0,
    }
    for field in PAWNS:
        out["pawn_structure"][field] = mean(float(item[field]) for item in pawns) if pawns else 0.0
    for color, categories, prefix in (("white", WHITE_OPENINGS, "white"),
                                      ("black", BLACK_OPENINGS, "black")):
        choices = openings[color]
        for move, name in categories.items():
            out["opening_choice"][name] = sum(choice == move for choice in choices) / len(choices) if choices else 0.0
        out["opening_choice"][f"{prefix}_other"] = sum(choice not in categories for choice in choices) / len(choices) if choices else 0.0
    denominators = {"assigned_games": len(games), "games_with_target_moves": sum(bool(game["moves"]) for game in games),
                    "target_moves": len(moves), "target_transitions": len(transitions),
                    "post_target_pawn_states": len(pawns),
                    "white_openings": len(openings["white"]), "black_openings": len(openings["black"])}
    return out, denominators


def train_reference(history: dict[str, list[dict]], seed: int) -> dict:
    blocks = {}
    for player, games in sorted(history.items()):
        ordered_ids = _ordered_games(seed, player, [game["game_key"] for game in games])
        by_id = {game["game_key"]: game for game in games}
        ordered = [by_id[game_id] for game_id in ordered_ids]
        if len(ordered) != 400:
            raise ValueError(f"Expected 400 assigned history games for {player}")
        blocks[player] = [style_vector(ordered[index:index + 25])[0] for index in range(0, 400, 25)]
    scale = {}
    for family, fields in FAMILIES.items():
        floor = 0.5 if family == "pawn_structure" else 0.05
        scale[family] = {field: max(floor, pstdev(block[family][field]
                                                 for local in blocks.values() for block in local))
                         for field in fields}
    centroids = {player: {family: {field: mean(block[family][field] for block in local)
                                   for field in fields} for family, fields in FAMILIES.items()}
                 for player, local in blocks.items()}
    return {"scale": scale, "centroids": centroids, "history_blocks_per_player": 16,
            "training_block_count": sum(map(len, blocks.values()))}


def identity_distances(vector: dict, reference: dict) -> dict[str, float]:
    scale = reference["scale"]
    result = {}
    for player, centroid in reference["centroids"].items():
        result[player] = mean(mean(((vector[family][field] - centroid[family][field]) /
                                    scale[family][field]) ** 2 for field in fields)
                              for family, fields in FAMILIES.items())
    return result


def identity_score(vector: dict, target: str, reference: dict) -> dict:
    distances = identity_distances(vector, reference)
    ranked = sorted(distances, key=lambda player: (distances[player], player))
    return {"predicted": ranked[0], "target_rank": ranked.index(target) + 1,
            "target_rank_error": ranked.index(target) / (len(ranked) - 1),
            "target_distance": distances[target], "nearest_distance": distances[ranked[0]]}


def behavior_distance(vector: dict, human: dict, scale: dict) -> dict:
    families = {family: min(4.0, mean(abs(vector[family][field] - human[family][field]) /
                                     scale[family][field] for field in fields)) / 4.0
                for family, fields in FAMILIES.items()}
    return {"mean": mean(families.values()), "families": families}


def _validate_config(config: dict) -> None:
    if (config.get("schema_version") != 1 or tuple(config.get("arms", ())) != ARMS
            or config.get("rollouts_per_player_arm") != 25 or config.get("max_plies") != 160
            or config.get("start_fen") != chess.STARTING_FEN
            or config.get("style_families") != {key: list(value) for key, value in FAMILIES.items()}):
        raise ValueError("Changed or unsupported frozen trajectory protocol")


def _inputs(config_path: Path, config: dict) -> dict[str, str]:
    cohort = Path(config["cohort"])
    data = Path(config["data"])
    models = Path(config["baseline_models"])
    population = Path(config["population_artifacts"])
    bundle = json.loads((data / "bundle.json").read_text())
    cohort_data = json.loads(cohort.read_text())
    players = sorted(cohort_data["development_players"])
    if (players != sorted(bundle["development_players"]) or len(players) != 14
            or set(players) & set(cohort_data["evaluation_players"])):
        raise ValueError("Development cohort changed")
    paths = [config_path, cohort, data / "bundle.json", data / "selection.json", Path(__file__)]
    paths += [population / name for name in ("safe_population.cbm", "feature_sets.json", "metrics.json")]
    for player in players:
        paths.extend((models / f"{player}-v1.json", models / f"{player}-v1.json.seal"))
        paths.extend(data / "positions" / split / f"{player}.parquet" for split in ("history", "validation"))
        paths.append(data / "candidate-cache/history" / f"{player}.parquet")
    hashes = {str(path): digest(path) for path in paths}
    if hashes[str(data / "selection.json")] != bundle["selection_sha256"]:
        raise ValueError("Selection hash differs from frozen bundle")
    for name, expected in bundle["population_artifacts"].items():
        if hashes[str(population / name)] != expected:
            raise ValueError(f"Population artifact differs from prototype input: {name}")
    return hashes


def _load_humans(config: dict, players: list[str]) -> tuple[dict, dict, dict]:
    data = Path(config["data"])
    selection = json.loads((data / "selection.json").read_text())
    reserved = {game for player, item in selection.items() if player not in players
                for split in ("validation", "evaluation") for game in item["game_ids"][split]}
    if set(selection["juliowi2"]["game_ids"]["validation"]) & reserved != {"gMUToUOs"}:
        raise ValueError("Reserved overlap/quarantine changed")
    history, validation, contexts = {}, {}, {}
    for player in players:
        player_sets = {}
        ratings, opponent_ratings = [], []
        for split in ("history", "validation"):
            positions = pq.read_table(data / "positions" / split / f"{player}.parquet").to_pylist()
            by_game = defaultdict(list)
            for row in positions:
                if split == "validation" and row["game_id"] == "gMUToUOs":
                    if player != "juliowi2":
                        raise ValueError("Unexpected reserved game")
                    continue
                by_game[str(row["game_id"])].append(row)
                if split == "history":
                    ratings.append(int(row["player_rating"]))
                    opponent_ratings.append(int(row["opponent_rating"]))
            assigned = [game for game in selection[player]["game_ids"][split]
                        if not (split == "validation" and player == "juliowi2" and game == "gMUToUOs")]
            if set(by_game) - set(assigned):
                raise ValueError("Positions outside frozen allocation")
            player_sets[split] = [_human_game(by_game[game], player, game) for game in assigned]
        history[player] = player_sets["history"]
        validation[player] = player_sets["validation"]
        contexts[player] = {"player_rating": round(median(ratings)),
                            "opponent_rating": round(median(opponent_ratings)),
                            "speed": "blitz", "time_control": "180+0"}
    return history, validation, contexts


class FrozenPolicy:
    def __init__(self, config: dict, players: list[str]):
        population = Path(config["population_artifacts"])
        self.model = CatBoostRanker().load_model(population / "safe_population.cbm")
        self.fields = tuple(json.loads((population / "feature_sets.json").read_text())["safe_population"])
        self.temperature = float(json.loads((population / "metrics.json").read_text())["safe_population"]["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid population temperature")
        source = Path(config["baseline_models"])
        personal = {}
        for player in players:
            wrapper = load_sealed(source / f"{player}-v1.json")
            model = wrapper["model"]
            if model["version"] != VERSION or len(model["coefficients"]) != DIMENSION:
                raise ValueError("Invalid matched-history v1 residual")
            history = Path(config["data"]) / "candidate-cache/history" / f"{player}.parquet"
            if wrapper["inputs"]["history_sha256"] != digest(history):
                raise ValueError("Residual history input changed")
            personal[player] = model
        self.personal = personal
        self.shared = shared_residual(list(personal.values()))
        self.wrong = {player: players[(index + 1) % len(players)]
                      for index, player in enumerate(players)}

    def distribution(self, board: chess.Board, context: dict, *, arm: str,
                     player: str, target_turn: bool) -> dict[str, float]:
        if target_turn:
            own, other = context["player_rating"], context["opponent_rating"]
            residual = {"population": None, "shared": self.shared,
                        "wrong": self.personal[self.wrong[player]], "personal": self.personal[player]}[arm]
        else:
            own, other, residual = context["opponent_rating"], context["player_rating"], None
        rows = build_all_legal_inference_rows(
            board.fen(en_passant="fen"), player_username=None,
            player_rating=own, opponent_rating=other, speed="blitz", time_control="180+0",
        )
        if len(rows) != board.legal_moves.count():
            raise ValueError("Legal candidate coverage changed")
        scores = predict_relevance_scores(self.model, rows, self.fields)
        key = " ".join(board.fen().split()[:4])
        for row, score in zip(rows, scores, strict=True):
            row["position_key"] = key
            row["base_logit"] = float(score) / self.temperature
        probabilities = predict_residual(rows, residual)
        return {row["candidate_move_uci"]: probability
                for row, probability in zip(rows, probabilities, strict=True)}


def _generate_unit(config: dict, policy: FrozenPolicy, player: str, context: dict, rollout: int) -> dict:
    games = {}
    color = color_for(config["seed"], player, rollout)
    for arm in ARMS:
        board = chess.Board(config["start_fen"])
        moves = []
        legal_decisions = legal_samples = 0
        last_target_pawns = None
        while not board.is_game_over(claim_draw=True) and board.ply() < config["max_plies"]:
            target_turn = board.turn == color
            distribution = policy.distribution(board, context, arm=arm, player=player,
                                               target_turn=target_turn)
            role = "target" if target_turn else "opponent"
            selected = sample_move(distribution, uniform(config["seed"], player, rollout,
                                                         board.ply() + 1, role))
            legal_decisions += 1
            move = chess.Move.from_uci(selected)
            if move in board.legal_moves:
                legal_samples += 1
            else:
                raise ValueError(f"Illegal generated move: {selected}")
            if target_turn:
                moves.append(_move_record(board, selected,
                                          game_key=f"{player}:{arm}:{rollout}", ply=board.ply() + 1))
            board.push(move)
            if target_turn:
                last_target_pawns = pawn_structure(board, color)
        outcome = board.outcome(claim_draw=True)
        games[arm] = {"game_key": f"{player}:{arm}:{rollout}", "color": "white" if color else "black",
                      "moves": moves, "last_target_pawns": last_target_pawns,
                      "plies": board.ply(), "terminal": outcome is not None,
                      "truncated": outcome is None,
                      "termination": outcome.termination.name.lower() if outcome else "max_plies",
                      "result": outcome.result() if outcome else "truncated",
                      "legal_decisions": legal_decisions, "legal_samples": legal_samples}
    return {"player": player, "rollout": rollout, "context": context, "games": games}


def paired_interval(deltas: list[float], seed: int) -> dict:
    if len(deltas) != 14:
        raise ValueError("Paired interval requires all 14 players")
    rng = np.random.default_rng(seed)
    values = np.asarray(deltas, dtype=float)
    sampled = rng.choice(values, size=(2000, 14), replace=True).mean(axis=1)
    return {"players": 14, "mean_delta": float(values.mean()),
            "paired_player_95_interval": np.quantile(sampled, [0.025, 0.975]).tolist()}


def _summarize(config: dict, players: list[str], history: dict, validation: dict,
               units: list[dict], reference: dict, input_hashes: dict) -> dict:
    human = {}
    confusion = {player: Counter() for player in players}
    heldout_blocks = 0
    heldout_correct = 0
    for player in players:
        ordered = sorted(validation[player], key=lambda game: hashlib.sha256(
            f"{config['seed']}:{player}:{game['game_key']}".encode()).hexdigest())
        if len(ordered) != (49 if player == "juliowi2" else 50):
            raise ValueError("Held-out human game denominator changed")
        full, denominators = style_vector(ordered)
        blocks = []
        for start in range(0, len(ordered), 25):
            vector, support = style_vector(ordered[start:start + 25])
            classification = identity_score(vector, player, reference)
            confusion[player][classification["predicted"]] += 1
            heldout_blocks += 1
            heldout_correct += classification["predicted"] == player
            blocks.append({"games": support["assigned_games"], "identity": classification})
        human[player] = {"validation_vector": full, "denominators": denominators, "blocks": blocks}
    by_player = defaultdict(lambda: defaultdict(list))
    for unit in units:
        for arm, game in unit["games"].items():
            by_player[unit["player"]][arm].append(game)
    per_player = {}
    arms = {}
    for player in players:
        per_player[player] = {}
        for arm in ARMS:
            games = sorted(by_player[player][arm], key=lambda game: int(game["game_key"].split(":")[-1]))
            if len(games) != 25:
                raise ValueError("Incomplete matched rollout set")
            vector, denominators = style_vector(games)
            identity = identity_score(vector, player, reference)
            behavior = behavior_distance(vector, human[player]["validation_vector"], reference["scale"])
            per_player[player][arm] = {
                "vector": vector, "denominators": denominators, "identity": identity,
                "behavior": behavior, "composite": (identity["target_rank_error"] + behavior["mean"]) / 2,
                "terminal_games": sum(game["terminal"] for game in games),
                "truncated_games": sum(game["truncated"] for game in games),
                "mean_plies": mean(game["plies"] for game in games),
                "termination_counts": dict(Counter(game["termination"] for game in games)),
                "legal_samples": sum(game["legal_samples"] for game in games),
                "legal_decisions": sum(game["legal_decisions"] for game in games),
            }
    for arm in ARMS:
        scored = [per_player[player][arm] for player in players]
        legal = sum(item["legal_samples"] for item in scored)
        decisions = sum(item["legal_decisions"] for item in scored)
        arms[arm] = {
            "games": 25 * len(players), "players": len(players),
            "target_moves": sum(item["denominators"]["target_moves"] for item in scored),
            "target_transitions": sum(item["denominators"]["target_transitions"] for item in scored),
            "legal_samples": legal, "legal_decisions": decisions,
            "legal_move_rate": legal / decisions,
            "terminal_games": sum(item["terminal_games"] for item in scored),
            "truncated_games": sum(item["truncated_games"] for item in scored),
            "mean_plies_player_macro": mean(item["mean_plies"] for item in scored),
            "termination_counts": dict(sum((Counter(item["termination_counts"]) for item in scored), Counter())),
            "identity_accuracy_players": sum(item["identity"]["predicted"] == player
                                             for player, item in zip(players, scored)) / len(players),
            "identity_target_rank_error_player_macro": mean(item["identity"]["target_rank_error"] for item in scored),
            "behavior_distance_player_macro": mean(item["behavior"]["mean"] for item in scored),
            "composite_player_macro": mean(item["composite"] for item in scored),
            "behavior_family_player_macro": {family: mean(item["behavior"]["families"][family]
                                                   for item in scored) for family in FAMILIES},
        }
    comparisons = {}
    for control in ("population", "shared", "wrong"):
        comparisons[f"personal_minus_{control}"] = {
            "composite": paired_interval([per_player[player]["personal"]["composite"] -
                                          per_player[player][control]["composite"]
                                          for player in players], config["seed"])
        }
        comparisons[f"personal_minus_{control}"]["behavior"] = paired_interval([
            per_player[player]["personal"]["behavior"]["mean"] -
            per_player[player][control]["behavior"]["mean"] for player in players], config["seed"] + 101)
        comparisons[f"personal_minus_{control}"]["identity_rank_error"] = paired_interval([
            per_player[player]["personal"]["identity"]["target_rank_error"] -
            per_player[player][control]["identity"]["target_rank_error"] for player in players], config["seed"] + 102)
        comparisons[f"personal_minus_{control}"]["behavior_families"] = {
            family: paired_interval([per_player[player]["personal"]["behavior"]["families"][family] -
                                     per_player[player][control]["behavior"]["families"][family]
                                     for player in players], config["seed"] + 200 + index)
            for index, family in enumerate(FAMILIES)}
    if any(arm["legal_move_rate"] != 1.0 for arm in arms.values()):
        raise ValueError("Legal move requirement failed")
    accuracy = heldout_correct / heldout_blocks
    return {
        "scope": config["evidence_scope"], "input_sha256": input_hashes,
        "players": len(players), "rollouts_per_player_arm": 25,
        "human_validation": {"assigned_games": 700, "scored_games": 699,
                             "quarantined_game": "juliowi2:gMUToUOs",
                             "heldout_blocks": heldout_blocks, "classifier_correct": heldout_correct,
                             "classifier_accuracy": accuracy, "chance_accuracy": 1 / 14,
                             "resolution_gate_passed": accuracy > 2 / 14,
                             "confusion": {player: dict(confusion[player]) for player in players}},
        "training_reference": reference, "human_players": human,
        "arms": arms, "comparisons": comparisons, "per_player": per_player,
        "interpretation_limit": "Generated identity scores are diagnostic only; no identity-fidelity claim when the human resolution gate fails. No independent confirmation or chess-quality equivalence claim.",
    }


def run(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text())
    _validate_config(config)
    started = perf_counter()
    input_hashes = _inputs(config_path, config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["input_sha256"] != input_hashes:
            raise ValueError("Trajectory inputs changed since checkpoint")
        if manifest["status"] == "complete":
            return load_sealed(output / "report.json")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Nonempty trajectory output without manifest")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"status": "running", "input_sha256": input_hashes}
        _write_json(manifest_path, manifest)
    cohort = json.loads(Path(config["cohort"]).read_text())
    players = sorted(cohort["development_players"])
    history, validation, contexts = _load_humans(config, players)
    reference = train_reference(history, config["seed"])
    policy = FrozenPolicy(config, players)
    units = []
    completed = 0
    for player in players:
        for rollout in range(25):
            path = output / "units" / f"{player}-{rollout:02d}.json"
            if path.exists():
                unit = load_sealed(path)
            else:
                unit = _generate_unit(config, policy, player, contexts[player], rollout)
                save_sealed(path, unit)
            if (unit["player"] != player or unit["rollout"] != rollout
                    or unit["context"] != contexts[player] or set(unit["games"]) != set(ARMS)):
                raise ValueError("Changed or malformed matched rollout checkpoint")
            units.append(unit)
            completed += 1
        print(f"Trajectory matched units: {completed}/350 ({player})", flush=True)
    report = _summarize(config, players, history, validation, units, reference, input_hashes)
    report["runtime_seconds_this_invocation"] = perf_counter() - started
    save_sealed(output / "report.json", report)
    manifest.update(status="complete", report_sha256=digest(output / "report.json"),
                    matched_units=completed, generated_games=completed * len(ARMS))
    _write_json(manifest_path, manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/style_v2_trajectories_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/style-v2-trajectories-v1"))
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps({"arms": report["arms"], "comparisons": report["comparisons"],
                      "human_validation": report["human_validation"]}, indent=2))


if __name__ == "__main__":
    main()
