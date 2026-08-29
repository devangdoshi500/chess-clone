from pathlib import Path

from typer.testing import CliRunner

import chess_clone.cli as cli

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

