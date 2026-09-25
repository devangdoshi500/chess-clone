"""Exact-position historical move-frequency model."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import chess
import pyarrow.parquet as pq

from chess_clone.models import PositionRecord

MoveDistributionRecord = dict[str, str | int | float]
RepeatedPositionRecord = dict[str, object]


def canonical_position_key(fen: str) -> str:
    """Return a normalized key containing the first four FEN fields.

    The key keeps piece placement, side to move, castling rights, and the raw
    en-passant target while discarding the halfmove and fullmove counters.
    Four-field position keys are accepted as input as well as full FENs.
    """

    fields = fen.split()
    if len(fields) == 4:
        fen = f"{fen} 0 1"
    elif len(fields) != 6:
        raise ValueError(f"Invalid FEN: expected 4 or 6 fields, got {len(fields)}")

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {fen}") from exc

    # python-chess normally omits an en-passant square when no legal capture
    # exists. The model key must preserve the FEN target itself.
    return " ".join(board.fen(en_passant="fen").split()[:4])


@dataclass(frozen=True, slots=True)
class HistoricalModelSummary:
    """Aggregate coverage metrics for a historical model."""

    total_observations: int
    unique_positions: int
    repeated_positions: int
    average_observations_per_position: float
    max_observations_for_one_position: int
    repeated_position_records_percentage: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class HistoricalMoveModel:
    """Empirical move distributions indexed by exact canonical positions."""

    def __init__(
        self,
        move_counts: Mapping[str, Mapping[str, int]],
        *,
        player_username: str | None = None,
        source_path: Path | None = None,
    ) -> None:
        self._move_counts = {
            key: Counter({move: int(count) for move, count in counts.items()})
            for key, counts in move_counts.items()
        }
        self.player_username = player_username
        self.source_path = source_path

    @classmethod
    def from_observations(
        cls,
        observations: Iterable[tuple[str, str]],
        *,
        player_username: str | None = None,
    ) -> "HistoricalMoveModel":
        """Build a model from ``(fen, move_uci)`` observations."""

        counts: dict[str, Counter[str]] = {}
        for fen, move_uci in observations:
            key = canonical_position_key(fen)
            counts.setdefault(key, Counter())[move_uci] += 1
        return cls(counts, player_username=player_username)

    @classmethod
    def from_position_records(
        cls,
        records: Iterable[PositionRecord],
        *,
        player_username: str | None = None,
    ) -> "HistoricalMoveModel":
        """Build a model from normalized PositionRecords."""

        selected = records
        if player_username is not None:
            requested = player_username.casefold()
            selected = (
                record
                for record in records
                if record.player_username.casefold() == requested
            )
        return cls.from_observations(
            ((record.fen, record.actual_move_uci) for record in selected),
            player_username=player_username,
        )

    @classmethod
    def from_parquet(
        cls, path: str | Path, player_username: str
    ) -> "HistoricalMoveModel":
        """Load one player's PositionRecords from a processed Parquet file."""

        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Position dataset not found: {source_path}")

        required = ["fen", "actual_move_uci", "player_username"]
        try:
            table = pq.read_table(source_path, columns=required)
        except Exception as exc:
            raise ValueError(
                f"Could not read PositionRecords from {source_path}: {exc}"
            ) from exc

        requested = player_username.casefold()
        observations = [
            (row["fen"], row["actual_move_uci"])
            for row in table.to_pylist()
            if row["player_username"].casefold() == requested
        ]
        if not observations:
            raise ValueError(
                f"No PositionRecords for player '{player_username}' in {source_path}"
            )

        model = cls.from_observations(observations, player_username=player_username)
        model.source_path = source_path
        return model

    def get_move_distribution(self, fen: str) -> list[MoveDistributionRecord]:
        """Return the empirical move distribution for ``fen``.

        Records are sorted by descending count and then UCI move. An unknown
        position returns an empty list.
        """

        counts = self._move_counts.get(canonical_position_key(fen))
        if not counts:
            return []
        total = sum(counts.values())
        return [
            {
                "move_uci": move,
                "count": count,
                "probability": count / total,
            }
            for move, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    def sample_move(self, fen: str, seed: int | None = None) -> str | None:
        """Sample a historical move for ``fen``, or return ``None`` if unknown."""

        distribution = self.get_move_distribution(fen)
        if not distribution:
            return None
        rng = random.Random(seed)
        return rng.choices(
            [str(record["move_uci"]) for record in distribution],
            weights=[int(record["count"]) for record in distribution],
            k=1,
        )[0]

    @property
    def summary(self) -> HistoricalModelSummary:
        observations_by_position = [
            sum(counts.values()) for counts in self._move_counts.values()
        ]
        total = sum(observations_by_position)
        unique = len(observations_by_position)
        repeated_counts = [count for count in observations_by_position if count > 1]
        repeated_records = sum(repeated_counts)
        return HistoricalModelSummary(
            total_observations=total,
            unique_positions=unique,
            repeated_positions=len(repeated_counts),
            average_observations_per_position=total / unique if unique else 0.0,
            max_observations_for_one_position=max(observations_by_position, default=0),
            repeated_position_records_percentage=(
                repeated_records / total * 100 if total else 0.0
            ),
        )

    def get_repeated_positions(
        self, limit: int | None = None
    ) -> list[RepeatedPositionRecord]:
        """Return repeated positions ordered by observation count."""

        repeated = [
            (key, sum(counts.values()))
            for key, counts in self._move_counts.items()
            if sum(counts.values()) > 1
        ]
        repeated.sort(key=lambda item: (-item[1], item[0]))
        if limit is not None:
            repeated = repeated[:limit]
        return [
            {
                "position_key": key,
                "total_observations": total,
                "moves": self.get_move_distribution(key),
            }
            for key, total in repeated
        ]
