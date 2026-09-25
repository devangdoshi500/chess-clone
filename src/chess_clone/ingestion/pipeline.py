"""End-to-end ingestion orchestration."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re

from chess_clone.ingestion.pgn_parser import parse_pgn
from chess_clone.ingestion.storage import (
    save_games_parquet,
    save_positions_parquet,
    save_raw_pgn,
)
from chess_clone.providers.base import DateInput, GameProvider


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    username: str
    games: int
    positions: int
    skipped_games: int
    raw_path: Path
    games_path: Path
    positions_path: Path


def ingest_games(
    provider: GameProvider,
    username: str,
    *,
    max_games: int | None = None,
    since: DateInput = None,
    until: DateInput = None,
    perf_type: str | None = None,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
) -> IngestionSummary:
    """Download, preserve, normalize, and persist one ingestion batch."""

    pgn_bytes = provider.download_games(
        username,
        max_games=max_games,
        since=since,
        until=until,
        perf_type=perf_type,
    )
    safe_username = re.sub(r"[^A-Za-z0-9_-]", "_", username.strip()).lower()
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{safe_username}_{batch_id}"
    raw_path = raw_dir / f"{stem}.pgn"
    games_path = processed_dir / f"games_{stem}.parquet"
    positions_path = processed_dir / f"positions_{stem}.parquet"

    # Preserve the response before any decoding or parsing takes place.
    save_raw_pgn(pgn_bytes, raw_path)
    parsed = parse_pgn(pgn_bytes, username)
    save_games_parquet(parsed.games, games_path)
    save_positions_parquet(parsed.positions, positions_path)

    return IngestionSummary(
        username=username,
        games=len(parsed.games),
        positions=len(parsed.positions),
        skipped_games=parsed.skipped_games,
        raw_path=raw_path,
        games_path=games_path,
        positions_path=positions_path,
    )
