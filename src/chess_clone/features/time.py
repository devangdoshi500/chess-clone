"""Clock parsing, increment-aware move time, and time-pressure features."""

from dataclasses import dataclass
import re

_CLOCK_RE = re.compile(r"^(\d+)\+(\d+)$")


@dataclass(frozen=True, slots=True)
class TimePressureThresholds:
    fraction: float = 0.10
    seconds: float = 30.0

    def __post_init__(self) -> None:
        if not 0 <= self.fraction <= 1:
            raise ValueError("time-pressure fraction must be between 0 and 1")
        if self.seconds < 0:
            raise ValueError("time-pressure seconds must not be negative")


@dataclass(frozen=True, slots=True)
class TimeFeatures:
    player_clock_seconds_after_move: float | None
    seconds_spent_on_move: float | None
    initial_time_seconds: int | None
    increment_seconds: int | None
    fraction_of_initial_time_remaining: float | None
    time_pressure: bool | None
    time_remaining_seconds: float | None
    time_remaining_fraction: float | None
    low_time: bool | None


def parse_time_control(time_control: str | None) -> tuple[int | None, int | None]:
    if not time_control:
        return None, None
    match = _CLOCK_RE.fullmatch(time_control)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def derive_time_features(
    *,
    time_control: str | None,
    clock_after_move: float | None,
    previous_player_clock_after_move: float | None,
    thresholds: TimePressureThresholds,
) -> TimeFeatures:
    """Derive clock features without guessing unavailable first-move time.

    For a non-first observed player move with clocks on both player moves:

        seconds_spent = previous_clock_after + increment - current_clock_after

    Lichess applies increment after a completed move, so subtracting increment
    would be incorrect. The first player move and inconsistent/negative values
    remain null because the reliable pre-move clock is unavailable.
    """

    initial, increment = parse_time_control(time_control)
    remaining = float(clock_after_move) if clock_after_move is not None else None
    fraction = (
        remaining / initial
        if remaining is not None and initial is not None and initial > 0
        else None
    )

    spent: float | None = None
    if (
        remaining is not None
        and previous_player_clock_after_move is not None
        and increment is not None
    ):
        candidate = float(previous_player_clock_after_move) + increment - remaining
        if candidate >= 0:
            spent = candidate

    low_time: bool | None = None
    if remaining is not None:
        low_time = remaining < thresholds.seconds or (
            fraction is not None and fraction < thresholds.fraction
        )

    return TimeFeatures(
        player_clock_seconds_after_move=remaining,
        seconds_spent_on_move=spent,
        initial_time_seconds=initial,
        increment_seconds=increment,
        fraction_of_initial_time_remaining=fraction,
        time_pressure=low_time,
        time_remaining_seconds=remaining,
        time_remaining_fraction=fraction,
        low_time=low_time,
    )
