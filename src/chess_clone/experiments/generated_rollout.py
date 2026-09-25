"""Matched generated-game trajectories for the frozen deep-style policy."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.deep_style import donor_for
from chess_clone.experiments.deep_style_cohort import load_cohort
from chess_clone.experiments.move_quality import _write_json
from chess_clone.experiments.profile_ablation import digest
from chess_clone.experiments.sampled_policy import sample_move, selected_game_ids
from chess_clone.experiments.style_adaptation import paired_player_interval
from chess_clone.features.board import classify_game_phase, extract_move_behavior_features
from chess_clone.modeling.inference import DeepStylePredictor, PredictionContext

BOOLEAN_TRAITS = (
    "is_capture", "is_check", "is_castle", "is_promotion", "is_queen_trade",
)
PIECES = ("pawn", "knight", "bishop", "rook", "queen", "king")
WINGS = ("queenside", "center", "kingside")
PHASES = ("opening", "middlegame", "endgame")


def rollout_uniform(seed: int, player: str, rollout: int, ply: int, role: str) -> float:
    value = hashlib.sha256(
        f"{seed}:{player.casefold()}:{rollout}:{ply}:{role}".encode()
    ).digest()[:8]
    return int.from_bytes(value, "big") / 2**64


def _wing(square: int) -> str:
    file_index = chess.square_file(square)
    if file_index <= 2:
        return "queenside"
    if file_index <= 4:
        return "center"
    return "kingside"


def move_record(board: chess.Board, uci: str, *, game_key: str, ply: int) -> dict:
    move = chess.Move.from_uci(uci)
    behavior = extract_move_behavior_features(board, uci)
    piece = board.piece_at(move.from_square)
    return {
        "game_key": game_key,
        "ply": ply,
        "phase": classify_game_phase(board),
        "move_uci": uci,
        "piece": chess.piece_name(piece.piece_type),
        "wing": _wing(move.to_square),
        "is_capture": behavior.is_capture,
        "is_check": behavior.is_check,
        "is_castle": behavior.is_castle,
        "is_promotion": behavior.is_promotion,
        "is_queen_trade": behavior.is_queen_trade,
    }


def pawn_structure(board: chess.Board, color: chess.Color) -> dict[str, int]:
    files = Counter(chess.square_file(square) for square in board.pieces(chess.PAWN, color))
    occupied = set(files)
    return {
        "remaining_pawns": sum(files.values()),
        "doubled_pawns": sum(max(0, count - 1) for count in files.values()),
        "isolated_pawns": sum(count for file_index, count in files.items()
                              if file_index - 1 not in occupied and file_index + 1 not in occupied),
        "pawn_islands": sum(1 for file_index in sorted(occupied) if file_index - 1 not in occupied),
    }


def _rates(records, field, categories):
    counts = Counter(record[field] for record in records)
    return {category: counts[category] / len(records) if records else None for category in categories}


def trajectory_summary(records: list[dict], games: list[dict]) -> dict:
    ordered = defaultdict(list)
    for record in records:
        ordered[record["game_key"]].append(record)
    repeated, transitions = 0, 0
    for local in ordered.values():
        local.sort(key=lambda row: row["ply"])
        for left, right in zip(local, local[1:]):
            transitions += 1
            repeated += left["piece"] == right["piece"]
    structures = [game["pawn_structure"] for game in games]
    return {
        "games": len(games),
        "target_moves": len(records),
        "terminal_rate": mean(game["terminal"] for game in games) if games else None,
        "truncation_rate": mean(game["truncated"] for game in games) if games else None,
        "mean_plies": mean(game["plies"] for game in games) if games else None,
        "results": dict(Counter(game["result"] for game in games)),
        "boolean_rates": {field: mean(record[field] for record in records) if records else None
                          for field in BOOLEAN_TRAITS},
        "piece_distribution": _rates(records, "piece", PIECES),
        "wing_distribution": _rates(records, "wing", WINGS),
        "phase_distribution": _rates(records, "phase", PHASES),
        "repeated_piece_transition_rate": repeated / transitions if transitions else None,
        "pawn_structure": {field: mean(item[field] for item in structures) if structures else None
                           for field in ("remaining_pawns", "doubled_pawns", "isolated_pawns", "pawn_islands")},
    }


def distribution_tv(left, right):
    return sum(abs(left[key] - right[key]) for key in left) / 2


def compare_to_human(generated: dict, human: dict) -> dict:
    boolean_errors = [abs(generated["boolean_rates"][field] - human["boolean_rates"][field])
                      for field in BOOLEAN_TRAITS]
    return {
        "mean_boolean_rate_error": mean(boolean_errors),
        "piece_tv": distribution_tv(generated["piece_distribution"], human["piece_distribution"]),
        "wing_tv": distribution_tv(generated["wing_distribution"], human["wing_distribution"]),
        "phase_tv": distribution_tv(generated["phase_distribution"], human["phase_distribution"]),
        "sequence_rate_error": abs(generated["repeated_piece_transition_rate"] - human["repeated_piece_transition_rate"]),
        "pawn_structure_mean_absolute_error": mean(
            abs(generated["pawn_structure"][field] - human["pawn_structure"][field])
            for field in generated["pawn_structure"]
        ),
    }


def _policy_distribution(predictor, board, base_context, *, arm, target, donor, target_turn):
    if not target_turn:
        context = PredictionContext(
            base_context.opponent_rating, base_context.player_rating,
            base_context.speed, base_context.time_control, None,
        )
        policy = "population"
    elif arm in {"population", "shared"}:
        context = base_context
        policy = arm
    else:
        identity = target if arm == "personal" else donor
        context = PredictionContext(
            base_context.player_rating, base_context.opponent_rating,
            base_context.speed, base_context.time_control, identity,
        )
        policy = "personal"
    result = predictor.predict(board.fen(en_passant="fen"), context, policy=policy, top_k=None)
    return {move.uci: move.probability for move in result.moves}


def run(config_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    required = {"schema_version", "seed", "source", "sampled_policy_protocol", "players", "arms",
                "rollouts_per_player_arm", "start_position", "opponent", "coupling",
                "max_plies", "termination", "metrics", "scope"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("Unsupported rollout declaration")
    if config["arms"] != ["population", "shared", "personal", "wrong"]:
        raise ValueError("v1 requires the four frozen arms")
    sampled = json.loads(Path(config["sampled_policy_protocol"]).read_text())
    if sampled["schema_version"] != 2 or sampled["seed"] != config["seed"]:
        raise ValueError("Rollout and sampled-policy declarations disagree")

    source = Path(config["source"])
    _, cohort = load_cohort(source)
    development = {p: r for p, r in cohort.items() if r["role"] == "development"}
    output.mkdir(parents=True)
    manifest = {"status": "running", "config_sha256": digest(config_path),
                "sampled_protocol_sha256": digest(Path(config["sampled_policy_protocol"])),
                "deep_style_declaration_sha256": digest(source / "declaration.json"),
                "deep_style_acquisition_sha256": digest(source / "acquisition.json")}
    _write_json(output / "manifest.json", manifest)
    started = perf_counter()
    try:
        predictor = DeepStylePredictor(source)
        target_inputs = {}
        unsupported = []
        for player, record in sorted(development.items()):
            donor = donor_for(player, cohort, predictor.personal_models, "validation")
            if donor is None:
                unsupported.append(player)
                continue
            rows = pq.read_table(record["paths"]["validation"]).to_pylist()
            game = selected_game_ids(rows, player=player, seed=config["seed"], count=1)[0]
            selected = sorted((row for row in rows if str(row["game_id"]) == game), key=lambda row: int(row["ply"]))
            target_inputs[player] = {"donor": donor, "game_id": game, "rows": selected}
        if len(target_inputs) != sampled["minimum_supported_players"]:
            raise ValueError("Rollout common-support cohort changed")

        move_rows, game_rows = [], []
        human_summaries = {}
        for player, item in target_inputs.items():
            human_moves = []
            final_board = None
            for row in item["rows"]:
                board = chess.Board(row["fen"])
                human_moves.append(move_record(board, row["actual_move_uci"], game_key=item["game_id"], ply=int(row["ply"])))
                board.push_uci(row["actual_move_uci"])
                final_board = board
            color = chess.WHITE if item["rows"][0]["player_color"] == "white" else chess.BLACK
            human_game = {"terminal": False, "truncated": False, "plies": max(r["ply"] for r in human_moves),
                          "result": "observed_partial", "pawn_structure": pawn_structure(final_board, color)}
            human_summaries[player] = trajectory_summary(human_moves, [human_game])

            first = item["rows"][0]
            context = PredictionContext(first.get("player_rating"), first.get("opponent_rating"),
                                        first.get("speed"), first.get("time_control"), player)
            for arm in config["arms"]:
                for rollout in range(config["rollouts_per_player_arm"]):
                    board = chess.Board()
                    game_key = f"{player}:{arm}:{rollout}"
                    local_moves = []
                    while not board.is_game_over(claim_draw=True) and board.ply() < config["max_plies"]:
                        is_target = board.turn == color
                        probabilities = _policy_distribution(
                            predictor, board, context, arm=arm, target=player,
                            donor=item["donor"], target_turn=is_target,
                        )
                        role = "target" if is_target else "opponent"
                        uniform = rollout_uniform(config["seed"], player, rollout, board.ply() + 1, role)
                        move = sample_move(probabilities, uniform)
                        if is_target:
                            local_moves.append(move_record(board, move, game_key=game_key, ply=board.ply() + 1))
                        board.push_uci(move)
                    outcome = board.outcome(claim_draw=True)
                    terminal = outcome is not None
                    game = {"game_key": game_key, "player_username": player, "arm": arm,
                            "rollout": rollout, "target_color": "white" if color else "black",
                            "plies": board.ply(), "terminal": terminal,
                            "truncated": not terminal, "result": outcome.result() if outcome else "truncated",
                            "termination": outcome.termination.name.lower() if outcome else None,
                            "pawn_structure": pawn_structure(board, color)}
                    game_rows.append(game)
                    move_rows.extend({**record, "player_username": player, "arm": arm, "rollout": rollout}
                                     for record in local_moves)
            print(f"Generated trajectories: {len(game_rows)}/{len(target_inputs) * len(config['arms']) * config['rollouts_per_player_arm']} games", flush=True)

        report = {"scope": config["scope"], "unsupported_players": unsupported,
                  "players": len(target_inputs), "games": len(game_rows), "arms": {},
                  "comparisons": {}, "runtime_seconds": perf_counter() - started}
        per_player = defaultdict(dict)
        for arm in config["arms"]:
            arm_moves = [row for row in move_rows if row["arm"] == arm]
            arm_games = [row for row in game_rows if row["arm"] == arm]
            generated = trajectory_summary(arm_moves, arm_games)
            comparisons = []
            for player in target_inputs:
                local = trajectory_summary(
                    [row for row in arm_moves if row["player_username"] == player],
                    [row for row in arm_games if row["player_username"] == player],
                )
                comparison = compare_to_human(local, human_summaries[player])
                per_player[player][arm] = comparison
                comparisons.append(comparison)
            generated["human_comparison_player_macro"] = {
                field: mean(item[field] for item in comparisons) for field in comparisons[0]
            }
            report["arms"][arm] = generated
        for control in ("population", "shared", "wrong"):
            report["comparisons"][f"personal_minus_{control}"] = {
                field: paired_player_interval([
                    per_player[player]["personal"][field] - per_player[player][control][field]
                    for player in target_inputs
                ])
                for field in per_player[next(iter(target_inputs))]["personal"]
            }
        pq.write_table(pa.Table.from_pylist(move_rows), output / "moves.parquet")
        pq.write_table(pa.Table.from_pylist(game_rows), output / "games.parquet")
        _write_json(output / "players.json", {p: {"human": human_summaries[p], **per_player[p]} for p in target_inputs})
        _write_json(output / "report.json", report)
        manifest.update(status="complete", runtime_seconds=report["runtime_seconds"])
        _write_json(output / "manifest.json", manifest)
        return report
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _write_json(output / "manifest.json", manifest)
        raise
