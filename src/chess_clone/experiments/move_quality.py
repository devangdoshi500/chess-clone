"""Post-hoc chess-quality diagnostics for frozen human-move predictions."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.analysis import EngineSettings, FileAnalysisCache, StockfishAnalyzer
from chess_clone.benchmark.engine import CoverageEngine
from chess_clone.experiments.population_policy import load_population_sources
from chess_clone.features.board import classify_game_phase


@dataclass(frozen=True)
class QualityScore:
    cp: int | None
    outcome: str
    mate_in: int | None = None


def score_move(board: chess.Board, uci: str, engine: CoverageEngine) -> QualityScore:
    """Evaluate every move via a separate SinglePV child search, in mover POV."""
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move: {uci}")
    child = board.copy(stack=False)
    child.push(move)
    if child.is_checkmate():
        return QualityScore(None, "winning_mate", 0)
    if child.is_stalemate() or child.is_insufficient_material() or child.is_seventyfive_moves():
        return QualityScore(0, "draw")
    lines, _, _ = engine.analyze_quality(child.fen(en_passant="fen"))
    line = lines[0]
    if line.mate_in is not None:
        mate = -line.mate_in
        return QualityScore(None, "winning_mate" if mate > 0 else "losing_mate", mate)
    if line.score_cp is None:
        return QualityScore(None, "missing")
    return QualityScore(-line.score_cp, "finite")


def compare_scores(actual: QualityScore, predicted: QualityScore, reference: QualityScore) -> dict:
    finite_pair = actual.cp is not None and predicted.cp is not None
    delta = predicted.cp - actual.cp if finite_pair else None
    gap = abs(delta) if delta is not None else None
    regret = (
        max(0, reference.cp - predicted.cp)
        if reference.cp is not None and predicted.cp is not None else None
    )
    human_regret = (
        max(0, reference.cp - actual.cp)
        if reference.cp is not None and actual.cp is not None else None
    )
    return {
        "actual_cp": actual.cp, "predicted_cp": predicted.cp, "reference_cp": reference.cp,
        "actual_outcome": actual.outcome, "predicted_outcome": predicted.outcome,
        "reference_outcome": reference.outcome,
        "actual_mate_in": actual.mate_in, "predicted_mate_in": predicted.mate_in,
        "reference_mate_in": reference.mate_in,
        "evaluation_gap_cp": gap, "predicted_minus_actual_cp": delta,
        "predicted_regret_cp": regret, "actual_regret_cp": human_regret,
        "predicted_exceeds_reference": (
            predicted.cp > reference.cp
            if predicted.cp is not None and reference.cp is not None else None
        ),
        **{f"within_{threshold}_cp": gap <= threshold if gap is not None else None
           for threshold in (25, 50, 100)},
    }


def summarize_quality(rows: list[dict]) -> dict:
    """Keep denominators explicit; never substitute mate sentinels into CP means."""
    result = {"decisions": len(rows), "games": len({r["game_id"] for r in rows})}
    finite = [r for r in rows if r["evaluation_gap_cp"] is not None]
    different = [r for r in finite if not r["exact_correct"]]
    result.update({"finite_pair_decisions": len(finite),
                   "excluded_pair_decisions": len(rows) - len(finite),
                   "different_move_finite_decisions": len(different),
                   "exact_accuracy": mean(r["exact_correct"] for r in rows) if rows else None})
    for field in ("evaluation_gap_cp", "predicted_minus_actual_cp", "predicted_regret_cp", "actual_regret_cp"):
        values = [r[field] for r in rows if r[field] is not None]
        result[field] = {"count": len(values), "mean": mean(values) if values else None,
                         "median": median(values) if values else None}
    for threshold in (25, 50, 100):
        key = f"within_{threshold}_cp"
        result[key] = mean(r[key] for r in finite) if finite else None
        result[f"different_move_{key}"] = mean(r[key] for r in different) if different else None
    mate_rows = [r for r in rows if "mate" in r["actual_outcome"] or "mate" in r["predicted_outcome"]]
    result["mate_pair_decisions"] = len(mate_rows)
    result["mate_outcome_agreement"] = (
        mean(r["actual_outcome"] == r["predicted_outcome"] for r in mate_rows) if mate_rows else None
    )
    result["missing_pair_decisions"] = sum(
        "missing" in (r["actual_outcome"], r["predicted_outcome"]) for r in rows
    )
    result["predicted_exceeds_reference_count"] = sum(r["predicted_exceeds_reference"] is True for r in rows)
    return result


def load_inputs(cohort: Path, artifact_dir: Path, split: str = "test") -> tuple[dict, dict, dict]:
    """Join prediction identities to positions; reject mismatches before engine work."""
    if split not in ("validation", "test"):
        raise ValueError("Quality split must be validation or test")
    paths = sorted(artifact_dir.glob(f"{split}_predictions_*.parquet"))
    if not paths:
        raise ValueError(f"No frozen {split}_predictions_*.parquet files found")
    hashes = {str(cohort): hashlib.sha256(cohort.read_bytes()).hexdigest()}
    positions = {}
    for source in load_population_sources(cohort):
        hashes[str(source.positions)] = hashlib.sha256(source.positions.read_bytes()).hexdigest()
        for row in pq.read_table(source.positions).to_pylist():
            if str(row["player_username"]).casefold() != source.username.casefold():
                continue
            key = f"{row['game_id']}:{int(row['ply'])}"
            if key in positions:
                raise ValueError(f"Duplicate source decision: {key}")
            positions[key] = row
    methods = {}
    for path in paths:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        name = path.stem.removeprefix(f"{split}_predictions_")
        indexed = {}
        for row in pq.read_table(path).to_pylist():
            key = str(row["decision_id"])
            if key in indexed:
                raise ValueError(f"Duplicate prediction: {key}")
            if key not in positions:
                raise ValueError(f"Missing source position: {key}")
            source = positions[key]
            if (str(row["game_id"]) != str(source["game_id"])
                    or str(row["player_username"]).casefold() != str(source["player_username"]).casefold()
                    or row["actual_move_uci"] != source["actual_move_uci"]):
                raise ValueError(f"Prediction/source mismatch: {key}")
            board = chess.Board(source["fen"])
            expected_color = "white" if board.turn else "black"
            if not board.is_valid() or source["player_color"] != expected_color:
                raise ValueError(f"Invalid board or mover: {key}")
            for field in ("actual_move_uci", "predicted_move_uci"):
                if chess.Move.from_uci(row[field]) not in board.legal_moves:
                    raise ValueError(f"Illegal {field}: {key}")
            indexed[key] = row
        if not indexed:
            raise ValueError(f"Empty predictions: {name}")
        if methods and set(indexed) != set(next(iter(methods.values()))):
            raise ValueError("Methods must score identical decisions")
        methods[name] = indexed
    return positions, methods, hashes


def run_move_quality(cohort: Path, artifact_dir: Path, output_dir: Path, *,
                     stockfish_path: str = "stockfish",
                     cache_dir: Path = Path("data/cache/candidate-coverage"),
                     split: str = "test",
                     analyzer_factory=StockfishAnalyzer) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    positions, methods, hashes = load_inputs(cohort, artifact_dir, split)
    output_dir.mkdir(parents=True)
    started = perf_counter()
    manifest = {"status": "running", "input_sha256": hashes, "schema_version": 1, "split": split,
                "protocol": "SinglePV 20000 nodes; separate child evaluations; mover POV"}
    _write_json(output_dir / "manifest.json", manifest)
    try:
        settings = EngineSettings(nodes=20_000, multipv=20, threads=1, hash_mb=16)
        all_rows = []
        with analyzer_factory(stockfish_path) as analyzer:
            engine = CoverageEngine(analyzer, FileAnalysisCache(cache_dir), settings)
            manifest["engine_identity"] = analyzer.engine_identity
            manifest["engine_settings"] = {**settings.cache_payload(), "multipv": 1}
            _write_json(output_dir / "manifest.json", manifest)
            keys = sorted(next(iter(methods.values())))
            for index, key in enumerate(keys, 1):
                source = positions[key]
                board = chess.Board(source["fen"])
                lines, _, _ = engine.analyze_quality(source["fen"])
                reference_move = lines[0].best_move_uci
                actual_move = source["actual_move_uci"]
                moves = {actual_move, reference_move} | {
                    predictions[key]["predicted_move_uci"] for predictions in methods.values()
                }
                scores = {move: score_move(board, move, engine) for move in sorted(moves)}
                for name, predictions in methods.items():
                    predicted = predictions[key]["predicted_move_uci"]
                    all_rows.append({
                        "method": name, "decision_id": key, "game_id": source["game_id"],
                        "player_username": source["player_username"], "fen": source["fen"],
                        "game_phase": classify_game_phase(board), "player_color": source["player_color"],
                        "rating_band": str(int(source["player_rating"]) // 200 * 200),
                        "time_control": str(source.get("time_control") or "unknown"),
                        "actual_move_uci": actual_move, "predicted_move_uci": predicted,
                        "reference_move_uci": reference_move, "exact_correct": actual_move == predicted,
                        **compare_scores(scores[actual_move], scores[predicted], scores[reference_move]),
                    })
                if index % 100 == 0 or index == len(keys):
                    print(f"Quality: {index}/{len(keys)} decisions; {engine.stats.engine_calls} searches", flush=True)
            stats = engine.snapshot()
        report = {"methods": {}, "engine": stats, "runtime_seconds": perf_counter() - started}
        for name in methods:
            rows = [r for r in all_rows if r["method"] == name]
            summary = summarize_quality(rows)
            summary["breakdowns"] = {}
            for field in ("player_username", "game_phase", "rating_band", "player_color", "time_control"):
                groups = defaultdict(list)
                for row in rows:
                    groups[row[field]].append(row)
                summary["breakdowns"][field] = {k: summarize_quality(v) for k, v in sorted(groups.items())}
            players = list(summary["breakdowns"]["player_username"].values())
            summary["player_macro"] = {}
            for field in ("exact_accuracy", "within_25_cp", "within_50_cp", "within_100_cp", "different_move_within_50_cp"):
                values = [p[field] for p in players if p[field] is not None]
                summary["player_macro"][field] = {"players": len(values), "mean": mean(values) if values else None}
            report["methods"][name] = summary
        pq.write_table(pa.Table.from_pylist(all_rows), output_dir / "decisions.parquet")
        _write_json(output_dir / "report.json", report)
        (output_dir / "REPORT.md").write_text(render_report(report))
        manifest.update(status="complete", runtime_seconds=report["runtime_seconds"], engine=stats)
        _write_json(output_dir / "manifest.json", manifest)
        return report
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _write_json(output_dir / "manifest.json", manifest)
        raise


def render_report(report: dict) -> str:
    lines = ["# Frozen-prediction chess-quality benchmark", "",
             "CP agreement measures evaluation similarity, not human preference or equivalent plans.",
             "Finite-score pairs only; mate/missing exclusions and differing-move denominators are explicit.", "",
             "| Method | Exact | CP pairs | Excluded | Within 25 | Within 50 | Within 100 | Different-move pairs | Different within 50 | Median gap CP | Mean regret CP |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def pct(value):
        return "n/a" if value is None else f"{value:.2%}"
    def number(value):
        return "n/a" if value is None else f"{value:.1f}"
    for name, m in report["methods"].items():
        lines.append(f"| {name} | {pct(m['exact_accuracy'])} | {m['finite_pair_decisions']} | {m['excluded_pair_decisions']} | "
                     f"{pct(m['within_25_cp'])} | {pct(m['within_50_cp'])} | {pct(m['within_100_cp'])} | "
                     f"{m['different_move_finite_decisions']} | {pct(m['different_move_within_50_cp'])} | "
                     f"{number(m['evaluation_gap_cp']['median'])} | {number(m['predicted_regret_cp']['mean'])} |")
    lines += ["", "Regret is clipped at zero relative to the engine-selected move, with all moves evaluated by identical child searches.",
              "Finite search can rank a different move above that reference; counts are in report.json.",
              "FEN analysis preserves the halfmove clock but cannot reconstruct repetition history or claimable draws.",
              "No model is fitted or selected here. This reused test window is exploratory; confirm on fresh games before tuning further.", ""]
    return "\n".join(lines)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
