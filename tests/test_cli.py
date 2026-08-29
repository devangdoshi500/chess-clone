from pathlib import Path

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
    assert "Engine calls: 2" in result.output
    assert output_path.is_file()
