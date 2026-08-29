"""Descriptive, Insight-style summaries of behavioral feature datasets."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class BehaviorSummary:
    move_count: int
    average_centipawn_loss: float | None
    median_centipawn_loss: float | None
    top_1_move_rate: float | None
    top_3_move_rate: float | None
    top_5_move_rate: float | None
    average_move_time_seconds: float | None
    time_pressure_percentage: float | None
    move_counts_by_game_phase: dict[str, int]
    move_counts_by_piece_moved: dict[str, int]
    capture_rate: float | None
    check_rate: float | None
    castling_rate: float | None
    queen_trade_rate: float | None
    average_material_balance_by_phase: dict[str, float]
    average_approximate_winning_chance_by_phase: dict[str, float]
    opening_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_behavior_features(path: str | Path) -> BehaviorSummary:
    rows = pq.read_table(path).to_pylist()
    if not rows:
        return BehaviorSummary(
            move_count=0,
            average_centipawn_loss=None,
            median_centipawn_loss=None,
            top_1_move_rate=None,
            top_3_move_rate=None,
            top_5_move_rate=None,
            average_move_time_seconds=None,
            time_pressure_percentage=None,
            move_counts_by_game_phase={},
            move_counts_by_piece_moved={},
            capture_rate=None,
            check_rate=None,
            castling_rate=None,
            queen_trade_rate=None,
            average_material_balance_by_phase={},
            average_approximate_winning_chance_by_phase={},
            opening_distribution={},
        )

    centipawn_losses = _present(rows, "centipawn_loss")
    move_times = _present(rows, "seconds_spent_on_move")
    phases = Counter(str(row["game_phase"]) for row in rows)
    pieces = Counter(str(row["piece_moved"]) for row in rows)
    material_by_phase: dict[str, list[float]] = defaultdict(list)
    chance_by_phase: dict[str, list[float]] = defaultdict(list)
    openings: Counter[str] = Counter()
    for row in rows:
        phase = str(row["game_phase"])
        if row.get("material_balance") is not None:
            material_by_phase[phase].append(float(row["material_balance"]))
        if row.get("approximate_winning_chance_before") is not None:
            chance_by_phase[phase].append(
                float(row["approximate_winning_chance_before"])
            )
        eco = row.get("opening_eco") or "?"
        opening = row.get("opening_name") or "Unknown"
        openings[f"{eco} | {opening}"] += 1

    return BehaviorSummary(
        move_count=len(rows),
        average_centipawn_loss=_mean_or_none(centipawn_losses),
        median_centipawn_loss=(median(centipawn_losses) if centipawn_losses else None),
        top_1_move_rate=_boolean_rate(rows, "actual_move_in_top_1"),
        top_3_move_rate=_boolean_rate(rows, "actual_move_in_top_3"),
        top_5_move_rate=_boolean_rate(rows, "actual_move_in_top_5"),
        average_move_time_seconds=_mean_or_none(move_times),
        time_pressure_percentage=_boolean_rate(rows, "low_time"),
        move_counts_by_game_phase=dict(sorted(phases.items())),
        move_counts_by_piece_moved=dict(sorted(pieces.items())),
        capture_rate=_boolean_rate(rows, "is_capture"),
        check_rate=_boolean_rate(rows, "is_check"),
        castling_rate=_boolean_rate(rows, "is_castle"),
        queen_trade_rate=_boolean_rate(rows, "is_queen_trade"),
        average_material_balance_by_phase={
            phase: mean(values) for phase, values in sorted(material_by_phase.items())
        },
        average_approximate_winning_chance_by_phase={
            phase: mean(values) for phase, values in sorted(chance_by_phase.items())
        },
        opening_distribution=dict(
            sorted(openings.items(), key=lambda item: (-item[1], item[0]))
        ),
    )


def _present(rows: list[dict[str, object]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) is not None]


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _boolean_rate(rows: list[dict[str, object]], field: str) -> float | None:
    values = [bool(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) * 100 if values else None
