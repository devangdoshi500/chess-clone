"""Normalized records produced by the ingestion pipeline."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One normalized game and its game-level metadata."""

    game_id: str
    provider: str
    game_url: str | None
    played_at: datetime | None
    white_username: str
    black_username: str
    white_rating: int | None
    black_rating: int | None
    result: str
    rated: bool
    variant: str
    speed: str | None
    time_control: str | None
    eco: str | None
    opening_name: str | None
    opening_variation: str | None
    termination: str | None
    total_plies: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """A pre-move position where the requested player is to move."""

    game_id: str
    ply: int
    move_number: int
    player_username: str
    player_color: Literal["white", "black"]
    fen: str
    actual_move_uci: str
    actual_move_san: str
    player_rating: int | None
    opponent_rating: int | None
    white_rating: int | None
    black_rating: int | None
    speed: str | None
    time_control: str | None
    eco: str | None
    opening_name: str | None
    opening_variation: str | None
    clock_seconds_after_move: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

