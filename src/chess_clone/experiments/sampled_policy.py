"""Predeclared sampled-policy strength diagnostics on development validation games."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.analysis import EngineSettings, FileAnalysisCache, StockfishAnalyzer
from chess_clone.benchmark.engine import CoverageEngine
from chess_clone.experiments.deep_style import donor_for
from chess_clone.experiments.deep_style_cohort import load_cohort
from chess_clone.experiments.move_quality import compare_scores, score_move, _write_json
from chess_clone.experiments.profile_ablation import digest
from chess_clone.experiments.winning_chance import (
    convert_row,
    paired_comparison,
    summary,
)
from chess_clone.features.board import classify_game_phase
from chess_clone.modeling.inference import DeepStylePredictor, PredictionContext


def selected_game_ids(rows: list[dict], *, player: str, seed: int, count: int) -> list[str]:
    games = {str(row["game_id"]) for row in rows}
    return sorted(
        games,
        key=lambda game: hashlib.sha256(
            f"{seed}:{player.casefold()}:{game}".encode()
        ).hexdigest(),
    )[:count]


def shared_uniform(seed: int, decision_id: str, sample: int) -> float:
    value = hashlib.sha256(f"{seed}:{decision_id}:{sample}".encode()).digest()[:8]
    return int.from_bytes(value, "big") / 2**64


def sample_move(probabilities: dict[str, float], uniform: float) -> str:
    if not probabilities or not 0 <= uniform < 1:
        raise ValueError("A non-empty distribution and uniform in [0, 1) are required")
    total = sum(probabilities.values())
    if abs(total - 1) > 1e-9:
        raise ValueError("Move probabilities must sum to one")
    cumulative = 0.0
    ordered = sorted(probabilities.items())
    for move, probability in ordered:
        if probability < 0:
            raise ValueError("Move probabilities must be non-negative")
        cumulative += probability
        if uniform < cumulative:
            return move
    return ordered[-1][0]


def _distribution(predictor, row, *, player):
    result = predictor.predict(
        row["fen"],
        PredictionContext(
            player_username=player,
            player_rating=row.get("player_rating"),
            opponent_rating=row.get("opponent_rating"),
            speed=row.get("speed"),
            time_control=row.get("time_control"),
        ),
        policy="personal" if player else "population",
        top_k=None,
    )
    return {move.uci: move.probability for move in result.moves}


def _severity(values):
    bins = Counter()
    for value in values:
        if value < 1:
            bins["lt_1pp"] += 1
        elif value < 3:
            bins["1_to_lt_3pp"] += 1
        elif value < 5:
            bins["3_to_lt_5pp"] += 1
        elif value < 10:
            bins["5_to_lt_10pp"] += 1
        else:
            bins["gte_10pp"] += 1
    return {name: {"count": bins[name], "rate": bins[name] / len(values) if values else None}
            for name in ("lt_1pp", "1_to_lt_3pp", "3_to_lt_5pp", "5_to_lt_10pp", "gte_10pp")}


def run(config_path: Path, output: Path, *, stockfish_path="stockfish",
        cache_dir=Path("data/cache/candidate-coverage"), analyzer_factory=StockfishAnalyzer):
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    required = {"schema_version", "seed", "source", "split", "games_per_player",
                "samples_per_decision", "minimum_supported_players", "arms", "sampling", "wrong_player_control",
                "engine", "thresholds", "promotion_scope"}
    if set(config) != required or config["schema_version"] != 2:
        raise ValueError("Unsupported sampled-policy declaration")
    if config["split"] != "development_validation":
        raise ValueError("v1 may only inspect development validation games")
    if config["arms"] != ["population", "shared", "personal", "wrong"]:
        raise ValueError("v1 requires all four frozen arms")
    engine_config = config["engine"]
    if engine_config != {"stockfish_version": 18, "nodes": 20000, "multipv": 1,
                         "threads": 1, "hash_mb": 16}:
        raise ValueError("v1 engine settings changed")
    source = Path(config["source"])
    thresholds_path = Path(config["thresholds"])
    declaration, cohort = load_cohort(source)
    development = {p: r for p, r in cohort.items() if r["role"] == "development"}
    if len(development) != 20:
        raise ValueError("v1 requires the sealed 20-player development cohort")

    output.mkdir(parents=True)
    manifest = {
        "status": "running",
        "config_sha256": digest(config_path),
        "deep_style_declaration_sha256": digest(source / "declaration.json"),
        "deep_style_acquisition_sha256": digest(source / "acquisition.json"),
        "thresholds_sha256": digest(thresholds_path),
    }
    _write_json(output / "manifest.json", manifest)
    started = perf_counter()
    try:
        predictor = DeepStylePredictor(source)
        selected, donor_map = {}, {}
        positions = []
        unsupported = []
        for player, record in sorted(development.items()):
            donor = donor_for(player, cohort, predictor.personal_models, "validation")
            if donor is None:
                unsupported.append(player)
                continue
            rows = pq.read_table(record["paths"]["validation"]).to_pylist()
            games = selected_game_ids(
                rows, player=player, seed=config["seed"], count=config["games_per_player"]
            )
            if len(games) != config["games_per_player"]:
                raise ValueError(f"Insufficient validation games for {player}")
            selected[player] = games
            positions.extend(row for row in rows if str(row["game_id"]) in set(games))
            donor_map[player] = donor
        if len(selected) < config["minimum_supported_players"]:
            raise ValueError("Insufficient players with valid wrong-player controls")
        positions.sort(key=lambda row: (str(row["player_username"]).casefold(), str(row["game_id"]), int(row["ply"])))
        sampling = {"players": selected, "donors": donor_map, "unsupported_players": unsupported,
                    "decisions": len(positions),
                    "games": len({r["game_id"] for r in positions})}
        _write_json(output / "sampling.json", sampling)

        settings = EngineSettings(nodes=20000, multipv=20, threads=1, hash_mb=16)
        all_rows = []
        with analyzer_factory(stockfish_path) as analyzer:
            engine = CoverageEngine(analyzer, FileAnalysisCache(cache_dir), settings)
            manifest["engine_identity"] = analyzer.engine_identity
            _write_json(output / "manifest.json", manifest)
            for index, row in enumerate(positions, 1):
                target = str(row["player_username"]).casefold()
                context = PredictionContext(row.get("player_rating"), row.get("opponent_rating"),
                                            row.get("speed"), row.get("time_control"), target)
                distributions = {}
                for arm, identity in (("population", None), ("shared", target),
                                      ("personal", target), ("wrong", donor_map[target])):
                    if arm == "population":
                        result = predictor.predict(row["fen"], context, policy="population", top_k=None)
                    elif arm == "shared":
                        result = predictor.predict(row["fen"], context, policy="shared", top_k=None)
                    else:
                        arm_context = PredictionContext(row.get("player_rating"), row.get("opponent_rating"),
                                                        row.get("speed"), row.get("time_control"), identity)
                        result = predictor.predict(row["fen"], arm_context, policy="personal", top_k=None)
                    distributions[arm] = {move.uci: move.probability for move in result.moves}

                board = chess.Board(row["fen"])
                reference_line, _, _ = engine.analyze_quality(row["fen"])
                reference_move = reference_line[0].best_move_uci
                actual_move = row["actual_move_uci"]
                sampled = {}
                for sample in range(config["samples_per_decision"]):
                    uniform = shared_uniform(config["seed"], f"{row['game_id']}:{row['ply']}", sample)
                    for arm in config["arms"]:
                        sampled[arm, sample] = sample_move(distributions[arm], uniform)
                moves = {actual_move, reference_move, *sampled.values()}
                scores = {move: score_move(board, move, engine) for move in sorted(moves)}
                for (arm, sample), move in sampled.items():
                    record = {
                        "method": arm,
                        "decision_id": f"{row['game_id']}:{row['ply']}:sample{sample}",
                        "source_decision_id": f"{row['game_id']}:{row['ply']}",
                        "sample": sample,
                        "game_id": row["game_id"],
                        "player_username": target,
                        "game_phase": classify_game_phase(board),
                        "player_color": row["player_color"],
                        "time_control": str(row.get("time_control") or "unknown"),
                        "actual_move_uci": actual_move,
                        "predicted_move_uci": move,
                        "reference_move_uci": reference_move,
                        "exact_correct": move == actual_move,
                        **compare_scores(scores[actual_move], scores[move], scores[reference_move]),
                    }
                    all_rows.append(convert_row(record))
                if index % 100 == 0 or index == len(positions):
                    print(f"Sampled policy: {index}/{len(positions)} decisions; {engine.stats.engine_calls} searches", flush=True)
            engine_stats = engine.snapshot()

        thresholds = json.loads(thresholds_path.read_text())
        by_arm = {arm: [r for r in all_rows if r["method"] == arm] for arm in config["arms"]}
        report = {"scope": config["promotion_scope"], "sampling": sampling,
                  "engine": engine_stats, "methods": {}, "comparisons": {},
                  "runtime_seconds": perf_counter() - started}
        for arm, rows in by_arm.items():
            arm_summary = summary(rows, thresholds)
            losses = [r["predicted_loss_pp"] for r in rows if r["predicted_loss_pp"] is not None]
            arm_summary["winning_chance_loss_severity"] = _severity(losses)
            player_values = {}
            for player in sorted({r["player_username"] for r in rows}):
                local = summary([r for r in rows if r["player_username"] == player], thresholds)
                player_values[player] = local
            arm_summary["player_macro"] = {
                "players": len(player_values),
                "similar_rate": mean(v["all"]["similar_rate"] for v in player_values.values()),
                "sound_rate": mean(v["all"]["sound_rate"] for v in player_values.values()),
                "mean_loss_pp": mean(v["predicted_loss_pp"]["mean"] for v in player_values.values()),
            }
            report["methods"][arm] = arm_summary
        for control in ("population", "shared", "wrong"):
            report["comparisons"][f"personal_minus_{control}"] = paired_comparison(
                by_arm[control], by_arm["personal"], thresholds, unit="player_username"
            )
        pq.write_table(pa.Table.from_pylist(all_rows), output / "decisions.parquet")
        _write_json(output / "report.json", report)
        manifest.update(status="complete", runtime_seconds=report["runtime_seconds"], engine=engine_stats)
        _write_json(output / "manifest.json", manifest)
        return report
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _write_json(output / "manifest.json", manifest)
        raise
