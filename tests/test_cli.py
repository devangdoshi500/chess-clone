from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import chess_clone.cli as cli
from chess_clone.analysis import EngineLine
from chess_clone.ingestion.pgn_parser import parse_pgn
from chess_clone.ingestion.storage import save_positions_parquet

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def test_cli_ingest_prints_summary(monkeypatch, tmp_path: Path) -> None:
    payload = (FIXTURES / "sample_games.pgn").read_bytes()
    monkeypatch.setattr(
        cli.LichessProvider,
        "download_games",
        lambda self, username, **kwargs: payload,
    )

    result = runner.invoke(
        cli.app,
        [
            "ingest",
            "TargetPlayer",
            "--max-games",
            "2",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--processed-dir",
            str(tmp_path / "processed"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Ingestion complete" in result.output
    assert "Games: 2" in result.output
    assert "Player positions: 5" in result.output


def test_cli_inspect_model_prints_metrics_and_repeated_examples(tmp_path: Path) -> None:
    parsed = parse_pgn(
        (FIXTURES / "sample_games.pgn").read_bytes(), "TargetPlayer"
    )
    positions_path = tmp_path / "positions_targetplayer_batch.parquet"
    save_positions_parquet(parsed.positions, positions_path)

    result = runner.invoke(
        cli.app,
        [
            "inspect-model",
            "targetplayer",
            "--positions",
            str(positions_path),
            "--examples",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Historical move model" in result.output
    assert "PositionRecords: 5" in result.output
    assert "Unique positions: 5" in result.output
    assert "Repeated positions: 0" in result.output


def test_cli_analyze_positions_runs_small_cached_batch(monkeypatch, tmp_path: Path) -> None:
    class FakeAnalyzer:
        engine_identity = "Fakefish CLI"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def analyze(self, fen, settings):
            return [
                EngineLine(
                    rank=1,
                    score_cp=10,
                    mate_in=None,
                    best_move_uci="e2e4",
                    pv_uci="e2e4",
                    depth=2,
                    seldepth=2,
                    nodes_searched=settings.nodes,
                    time_seconds=0.001,
                )
            ]

    parsed = parse_pgn(
        (FIXTURES / "sample_games.pgn").read_bytes(), "TargetPlayer"
    )
    positions_path = tmp_path / "positions_targetplayer_batch.parquet"
    output_path = tmp_path / "analysis.parquet"
    save_positions_parquet(parsed.positions, positions_path)
    monkeypatch.setattr(cli, "StockfishAnalyzer", lambda executable: FakeAnalyzer())

    result = runner.invoke(
        cli.app,
        [
            "analyze-positions",
            "targetplayer",
            "--positions",
            str(positions_path),
            "--stockfish-path",
            "ignored",
            "--nodes",
            "50",
            "--max-positions",
            "2",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Engine analysis complete" in result.output
    assert "PositionRecords: 2" in result.output
    assert "Engine calls: 4" in result.output
    assert output_path.is_file()


def test_cli_build_features_prints_join_accounting(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "features.parquet"
    captured = {}

    def fake_build(position_path, analysis_path, username, **kwargs):
        captured["thresholds"] = kwargs["thresholds"]
        return SimpleNamespace(
            username=username,
            available_position_records=12,
            analyzed_position_records=10,
            feature_rows=10,
            unanalyzed_position_records=2,
            output_path=kwargs["output_path"],
        )

    monkeypatch.setattr(cli, "build_behavior_features", fake_build)
    result = runner.invoke(
        cli.app,
        [
            "build-features",
            "TargetPlayer",
            "--positions",
            str(tmp_path / "positions.parquet"),
            "--analysis",
            str(tmp_path / "analysis.parquet"),
            "--output",
            str(output_path),
            "--low-time-fraction",
            "0.15",
            "--low-time-seconds",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Behavioral feature build complete" in result.output
    assert "Feature rows: 10" in result.output
    assert "Unanalyzed PositionRecords: 2" in result.output
    assert captured["thresholds"].fraction == 0.15
    assert captured["thresholds"].seconds == 20


def test_cli_train_model_prints_dataset_accounting(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"

    def fake_train(username, **kwargs):
        return SimpleNamespace(
            username=username,
            total_decisions=100,
            inside_top_5_decisions=88,
            outside_top_5_decisions=12,
            runtime_seconds=1.25,
            artifact_dir=kwargs["artifact_dir"],
            metrics={"models": {}},
        )

    monkeypatch.setattr(cli, "train_personalized_ranker", fake_train)
    result = runner.invoke(
        cli.app,
        [
            "train-model",
            "TargetPlayer",
            "--features",
            str(tmp_path / "features.parquet"),
            "--analysis",
            str(tmp_path / "analysis.parquet"),
            "--games",
            str(tmp_path / "games.parquet"),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Inside top 5: 88" in result.output
    assert f"Artifact directory: {artifact_dir}" in result.output


def test_cli_evaluate_model_prints_saved_metrics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli, "evaluate_saved_artifact", lambda path: {"test": {"accuracy": 0.5}}
    )
    result = runner.invoke(cli.app, ["evaluate-model", str(tmp_path / "model")])

    assert result.exit_code == 0, result.output
    assert '"accuracy": 0.5' in result.output


def test_cli_train_boosted_model_prints_artifact(monkeypatch, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "catboost"

    def fake_train(username, **kwargs):
        return SimpleNamespace(
            username=username,
            total_decisions=100,
            usable_decisions=88,
            outside_top_5_decisions=12,
            runtime_seconds=2.5,
            artifact_dir=kwargs["artifact_dir"],
        )

    monkeypatch.setattr(cli, "train_boosted_rankers", fake_train)
    result = runner.invoke(
        cli.app,
        [
            "train-boosted-model",
            "TargetPlayer",
            "--features",
            str(tmp_path / "features.parquet"),
            "--analysis",
            str(tmp_path / "analysis.parquet"),
            "--games",
            str(tmp_path / "games.parquet"),
            "--rf-artifact",
            str(tmp_path / "rf"),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Grouped CatBoost training complete" in result.output
    assert "Usable top-5 decisions: 88" in result.output
