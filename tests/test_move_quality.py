import json

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from chess_clone.analysis import EngineLine, EngineSettings, FileAnalysisCache
from chess_clone.benchmark.engine import CoverageEngine
from chess_clone.cli import app
from chess_clone.experiments.move_quality import (
    QualityScore, compare_scores, load_inputs, run_move_quality, score_move, summarize_quality,
)


class FakeFish:
    engine_identity = "Stockfish 18|sha256:quality-test"

    def __init__(self, *args):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def analyze(self, fen, settings):
        self.calls.append((fen, settings))
        move = sorted(chess.Board(fen).legal_moves, key=lambda m: m.uci())[0]
        return [EngineLine(1, 30, None, move.uci(), move.uci(), 8, 8, 20000, .01)]


def test_score_perspective_and_cache(tmp_path):
    fish = FakeFish()
    engine = CoverageEngine(fish, FileAnalysisCache(tmp_path), EngineSettings(nodes=20000, multipv=20))
    board = chess.Board()
    assert score_move(board, "e2e4", engine).cp == -30
    assert score_move(board, "e2e4", engine).cp == -30
    assert len(fish.calls) == 1
    board.push_uci("e2e4")
    assert score_move(board, "e7e5", engine).cp == -30
    assert all(settings.multipv == 1 for _, settings in fish.calls)
    with pytest.raises(ValueError, match="Illegal"):
        score_move(board, "e2e4", engine)


def test_terminal_moves_and_mates_do_not_become_cp(tmp_path):
    engine = CoverageEngine(FakeFish(), FileAnalysisCache(tmp_path), EngineSettings(nodes=20000, multipv=20))
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(move)
    assert score_move(board, "d8h4", engine) == QualityScore(None, "winning_mate", 0)
    board = chess.Board("7k/5K2/4Q3/8/8/8/8/8 w - - 0 1")
    assert score_move(board, "e6g6", engine) == QualityScore(0, "draw")
    assert engine.stats.engine_calls == 0
    result = compare_scores(QualityScore(None, "winning_mate", 3), QualityScore(400, "finite"), QualityScore(None, "winning_mate", 2))
    assert result["evaluation_gap_cp"] is None
    assert result["within_100_cp"] is None
    assert result["predicted_regret_cp"] is None


def test_thresholds_denominators_and_no_false_regret():
    def row(a, p, exact=False):
        return {"game_id": "game", "exact_correct": exact,
                **compare_scores(a, p, QualityScore(100, "finite"))}
    rows = [row(QualityScore(10, "finite"), QualityScore(10, "finite"), True),
            row(QualityScore(0, "finite"), QualityScore(50, "finite")),
            row(QualityScore(None, "losing_mate", -2), QualityScore(0, "finite"))]
    summary = summarize_quality(rows)
    assert summary["finite_pair_decisions"] == 2
    assert summary["excluded_pair_decisions"] == 1
    assert summary["within_25_cp"] == .5
    assert summary["within_50_cp"] == 1
    assert summary["different_move_within_25_cp"] == 0
    assert summary["different_move_within_50_cp"] == 1
    result = compare_scores(QualityScore(50, "finite"), QualityScore(120, "finite"), QualityScore(100, "finite"))
    assert result["predicted_regret_cp"] == 0
    assert result["predicted_exceeds_reference"] is True
    assert summarize_quality([])["within_50_cp"] is None


def fixture_inputs(tmp_path):
    artifact = tmp_path / "models"
    artifact.mkdir()
    players = []
    predictions = []
    for index, name in enumerate(("Alice", "Bob")):
        position = {"game_id": f"game{index}", "ply": 1, "fen": chess.STARTING_FEN,
                    "player_username": name, "player_color": "white", "player_rating": 1500,
                    "actual_move_uci": "e2e4", "time_control": "180+0"}
        pq.write_table(pa.Table.from_pylist([position]), tmp_path / f"{name}.parquet")
        pq.write_table(pa.Table.from_pylist([{"game_id": f"game{index}"}]), tmp_path / f"{name}_games.parquet")
        players.append({"username": name, "positions": f"{name}.parquet", "games": f"{name}_games.parquet"})
        predictions.append({"decision_id": f"game{index}:1", "game_id": f"game{index}",
                            "player_username": name, "actual_move_uci": "e2e4", "predicted_move_uci": "d2d4"})
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"schema_version": 1, "players": players}))
    pq.write_table(pa.Table.from_pylist(predictions), artifact / "test_predictions_population.parquet")
    return cohort, artifact, predictions


def test_runner_provenance_reports_cache_and_no_overwrite(tmp_path):
    cohort, artifact, _ = fixture_inputs(tmp_path)
    output = tmp_path / "out"
    report = run_move_quality(cohort, artifact, output, cache_dir=tmp_path / "cache", analyzer_factory=FakeFish)
    assert report["methods"]["population"]["within_50_cp"] == 1
    assert report["methods"]["population"]["player_macro"]["within_50_cp"]["players"] == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["engine_settings"]["multipv"] == 1
    assert len(manifest["input_sha256"]) == 4
    assert pq.read_table(output / "decisions.parquet").num_rows == 2
    cached = run_move_quality(cohort, artifact, tmp_path / "out2", cache_dir=tmp_path / "cache", analyzer_factory=FakeFish)
    assert cached["engine"]["engine_calls"] == 0
    with pytest.raises(FileExistsError):
        run_move_quality(cohort, artifact, output, analyzer_factory=FakeFish)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "illegal", "actual", "unequal"])
def test_reject_corrupt_or_unmatched_predictions(tmp_path, mutation):
    cohort, artifact, rows = fixture_inputs(tmp_path)
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "missing":
        rows[0]["decision_id"] = "absent:1"
    elif mutation == "illegal":
        rows[0]["predicted_move_uci"] = "e2e5"
    elif mutation == "actual":
        rows[0]["actual_move_uci"] = "g1f3"
    else:
        pq.write_table(pa.Table.from_pylist(rows[:1]), artifact / "test_predictions_other.parquet")
    pq.write_table(pa.Table.from_pylist(rows), artifact / "test_predictions_population.parquet")
    with pytest.raises(ValueError):
        load_inputs(cohort, artifact)


def test_cli_errors_without_engine_work(tmp_path):
    result = CliRunner().invoke(app, ["benchmark-move-quality", "--cohort", str(tmp_path / "missing"),
                                     "--artifact-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "No frozen" in result.output


def test_engine_failure_leaves_failed_manifest(tmp_path):
    cohort, artifact, _ = fixture_inputs(tmp_path)
    class BrokenFish(FakeFish):
        def analyze(self, fen, settings):
            raise RuntimeError("engine stopped")
    with pytest.raises(RuntimeError, match="engine stopped"):
        run_move_quality(cohort, artifact, tmp_path / "out", cache_dir=tmp_path / "cache", analyzer_factory=BrokenFish)
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("mate,outcome", [(3, "losing_mate"), (-3, "winning_mate")])
def test_engine_mate_perspective(tmp_path, mate, outcome):
    class MateFish(FakeFish):
        def analyze(self, fen, settings):
            move = next(iter(chess.Board(fen).legal_moves)).uci()
            return [EngineLine(1, None, mate, move, move, 8, 8, 20000, .01)]
    engine = CoverageEngine(MateFish(), FileAnalysisCache(tmp_path), EngineSettings(nodes=20000, multipv=20))
    assert score_move(chess.Board(), "e2e4", engine) == QualityScore(None, outcome, -mate)
