"""Filesystem and Parquet persistence for ingested records."""

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.models import GameRecord, PositionRecord

GAME_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("provider", pa.string()),
        ("game_url", pa.string()),
        ("played_at", pa.timestamp("ms", tz="UTC")),
        ("white_username", pa.string()),
        ("black_username", pa.string()),
        ("white_rating", pa.int64()),
        ("black_rating", pa.int64()),
        ("result", pa.string()),
        ("rated", pa.bool_()),
        ("variant", pa.string()),
        ("speed", pa.string()),
        ("time_control", pa.string()),
        ("eco", pa.string()),
        ("opening_name", pa.string()),
        ("opening_variation", pa.string()),
        ("termination", pa.string()),
        ("total_plies", pa.int64()),
    ]
)

POSITION_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("ply", pa.int64()),
        ("move_number", pa.int64()),
        ("player_username", pa.string()),
        ("player_color", pa.string()),
        ("fen", pa.string()),
        ("actual_move_uci", pa.string()),
        ("actual_move_san", pa.string()),
        ("player_rating", pa.int64()),
        ("opponent_rating", pa.int64()),
        ("white_rating", pa.int64()),
        ("black_rating", pa.int64()),
        ("speed", pa.string()),
        ("time_control", pa.string()),
        ("eco", pa.string()),
        ("opening_name", pa.string()),
        ("opening_variation", pa.string()),
        ("clock_seconds_after_move", pa.float64()),
    ]
)


def save_raw_pgn(pgn_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pgn_bytes)


def save_games_parquet(records: Sequence[GameRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([record.to_dict() for record in records], schema=GAME_SCHEMA)
    pq.write_table(table, path)


def save_positions_parquet(records: Sequence[PositionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [record.to_dict() for record in records], schema=POSITION_SCHEMA
    )
    pq.write_table(table, path)

