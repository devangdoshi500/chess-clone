from pathlib import Path

import pyarrow.parquet as pq

from chess_clone.ingestion.pipeline import ingest_games
from chess_clone.providers.base import DateInput, GameProvider

FIXTURES = Path(__file__).parent / "fixtures"


class FixtureProvider(GameProvider):
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def download_games(
        self,
        username: str,
        *,
        max_games: int | None = None,
        since: DateInput = None,
        until: DateInput = None,
        perf_type: str | None = None,
    ) -> bytes:
        return self.payload


def test_pipeline_preserves_raw_bytes_and_writes_parquet(tmp_path: Path) -> None:
    payload = (FIXTURES / "sample_games.pgn").read_bytes()
    summary = ingest_games(
        FixtureProvider(payload),
        "TargetPlayer",
        max_games=2,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    assert summary.games == 2
    assert summary.positions == 5
    assert summary.raw_path.read_bytes() == payload
    games = pq.read_table(summary.games_path)
    positions = pq.read_table(summary.positions_path)
    assert games.num_rows == 2
    assert positions.num_rows == 5
    assert "played_at" in games.column_names
    assert "clock_seconds_after_move" in positions.column_names
