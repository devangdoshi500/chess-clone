"""Ordered, smoothed player-tendency features fitted on training decisions only."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path

from chess_clone.modeling.ranker import validate_candidate_labels

PLAYER_HISTORY_FEATURE_FIELDS = (
    "history_piece_probability",
    "history_rank_probability",
    "history_capture_match_probability",
    "history_check_match_probability",
    "history_queen_trade_match_probability",
    "history_castle_match_probability",
    "history_phase_rank_probability",
    "history_time_pressure_rank_probability",
    "history_opening_rank_probability",
    "history_total_observations",
    "history_phase_observations",
    "history_time_pressure_observations",
    "history_opening_observations",
)

_PIECES = ("pawn", "knight", "bishop", "rook", "queen", "king")
_RANKS = (1, 2, 3, 4, 5)


@dataclass(slots=True)
class _HistoryState:
    total: int = 0
    piece_counts: Counter[str] = field(default_factory=Counter)
    rank_counts: Counter[int] = field(default_factory=Counter)
    boolean_true_counts: Counter[str] = field(default_factory=Counter)
    phase_rank_counts: dict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    phase_totals: Counter[str] = field(default_factory=Counter)
    pressure_rank_counts: dict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    pressure_totals: Counter[str] = field(default_factory=Counter)
    opening_rank_counts: dict[str, Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    opening_totals: Counter[str] = field(default_factory=Counter)


class PlayerHistoryEncoder:
    """Create target-player priors without reading validation/test outcomes."""

    def __init__(
        self,
        *,
        smoothing_strength: float = 10.0,
        opening_min_samples: int = 30,
    ) -> None:
        if smoothing_strength <= 0:
            raise ValueError("smoothing_strength must be positive")
        if opening_min_samples < 1:
            raise ValueError("opening_min_samples must be positive")
        self.smoothing_strength = float(smoothing_strength)
        self.opening_min_samples = int(opening_min_samples)
        self._state = _HistoryState()
        self._fitted = False
        self.fit_decision_ids: frozenset[str] = frozenset()

    def fit_transform_ordered(
        self, train_rows: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Encode each training decision from strictly earlier train decisions."""

        validate_candidate_labels(train_rows)
        state = _HistoryState()
        result: list[dict[str, object]] = []
        decision_ids: list[str] = []
        for group in _decision_groups(train_rows):
            result.extend(self._encode_group(group, state))
            self._update(group, state)
            decision_ids.append(str(group[0]["decision_id"]))
        self._state = state
        self._fitted = True
        self.fit_decision_ids = frozenset(decision_ids)
        return result

    def fit(self, train_rows: list[dict[str, object]]) -> "PlayerHistoryEncoder":
        validate_candidate_labels(train_rows)
        state = _HistoryState()
        decision_ids: list[str] = []
        for group in _decision_groups(train_rows):
            self._update(group, state)
            decision_ids.append(str(group[0]["decision_id"]))
        self._state = state
        self._fitted = True
        self.fit_decision_ids = frozenset(decision_ids)
        return self

    def transform(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """Apply frozen training history without updating from transformed labels."""

        if not self._fitted:
            raise RuntimeError("PlayerHistoryEncoder must be fitted before transform")
        result: list[dict[str, object]] = []
        for group in _decision_groups(rows):
            result.extend(self._encode_group(group, self._state))
        return result

    def to_dict(self) -> dict[str, object]:
        if not self._fitted:
            raise RuntimeError("Cannot serialize an unfitted history encoder")
        state = self._state
        return {
            "smoothing_strength": self.smoothing_strength,
            "opening_min_samples": self.opening_min_samples,
            "fit_decision_ids": sorted(self.fit_decision_ids),
            "state": {
                "total": state.total,
                "piece_counts": dict(state.piece_counts),
                "rank_counts": {str(k): v for k, v in state.rank_counts.items()},
                "boolean_true_counts": dict(state.boolean_true_counts),
                "phase_rank_counts": _nested_counts(state.phase_rank_counts),
                "phase_totals": dict(state.phase_totals),
                "pressure_rank_counts": _nested_counts(state.pressure_rank_counts),
                "pressure_totals": dict(state.pressure_totals),
                "opening_rank_counts": _nested_counts(state.opening_rank_counts),
                "opening_totals": dict(state.opening_totals),
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def _encode_group(
        self, group: list[dict[str, object]], state: _HistoryState
    ) -> list[dict[str, object]]:
        phase = str(group[0].get("game_phase"))
        pressure = str(group[0].get("pre_move_time_pressure"))
        opening = str(group[0].get("opening_eco"))
        result: list[dict[str, object]] = []
        for row in group:
            rank = int(row["engine_rank"])
            piece = str(row.get("candidate_piece_moved"))
            encoded = dict(row)
            encoded.update(
                {
                    "history_piece_probability": _categorical_probability(
                        state.piece_counts,
                        state.total,
                        piece,
                        len(_PIECES),
                        self.smoothing_strength,
                    ),
                    "history_rank_probability": self._rank_probability(
                        state.rank_counts, state.total, rank, state
                    ),
                    "history_capture_match_probability": self._boolean_match(
                        state, "capture", bool(row["candidate_is_capture"])
                    ),
                    "history_check_match_probability": self._boolean_match(
                        state, "check", bool(row["candidate_gives_check"])
                    ),
                    "history_queen_trade_match_probability": self._boolean_match(
                        state, "queen_trade", bool(row["candidate_trades_queens"])
                    ),
                    "history_castle_match_probability": self._boolean_match(
                        state, "castle", bool(row["candidate_is_castle"])
                    ),
                    "history_phase_rank_probability": self._conditional_rank(
                        state.phase_rank_counts[phase],
                        state.phase_totals[phase],
                        rank,
                        state,
                    ),
                    "history_time_pressure_rank_probability": (
                        self._conditional_rank(
                            state.pressure_rank_counts[pressure],
                            state.pressure_totals[pressure],
                            rank,
                            state,
                        )
                    ),
                    "history_opening_rank_probability": (
                        self._opening_rank_probability(opening, rank, state)
                    ),
                    "history_total_observations": state.total,
                    "history_phase_observations": state.phase_totals[phase],
                    "history_time_pressure_observations": (
                        state.pressure_totals[pressure]
                    ),
                    "history_opening_observations": state.opening_totals[opening],
                }
            )
            result.append(encoded)
        return result

    def _rank_probability(
        self,
        counts: Counter[int],
        total: int,
        rank: int,
        state: _HistoryState,
    ) -> float:
        del state
        return _categorical_probability(
            counts, total, rank, len(_RANKS), self.smoothing_strength
        )

    def _conditional_rank(
        self,
        counts: Counter[int],
        total: int,
        rank: int,
        state: _HistoryState,
    ) -> float:
        global_probability = self._rank_probability(
            state.rank_counts, state.total, rank, state
        )
        return (
            counts[rank] + self.smoothing_strength * global_probability
        ) / (total + self.smoothing_strength)

    def _opening_rank_probability(
        self, opening: str, rank: int, state: _HistoryState
    ) -> float:
        if state.opening_totals[opening] < self.opening_min_samples:
            return self._rank_probability(state.rank_counts, state.total, rank, state)
        return self._conditional_rank(
            state.opening_rank_counts[opening],
            state.opening_totals[opening],
            rank,
            state,
        )

    def _boolean_match(
        self, state: _HistoryState, name: str, candidate_value: bool
    ) -> float:
        probability_true = (
            state.boolean_true_counts[name] + self.smoothing_strength * 0.5
        ) / (state.total + self.smoothing_strength)
        return probability_true if candidate_value else 1.0 - probability_true

    @staticmethod
    def _update(group: list[dict[str, object]], state: _HistoryState) -> None:
        chosen = next(row for row in group if bool(row["chosen"]))
        rank = int(chosen["engine_rank"])
        piece = str(chosen.get("candidate_piece_moved"))
        phase = str(chosen.get("game_phase"))
        pressure = str(chosen.get("pre_move_time_pressure"))
        opening = str(chosen.get("opening_eco"))
        state.total += 1
        state.piece_counts[piece] += 1
        state.rank_counts[rank] += 1
        for name, field in (
            ("capture", "candidate_is_capture"),
            ("check", "candidate_gives_check"),
            ("queen_trade", "candidate_trades_queens"),
            ("castle", "candidate_is_castle"),
        ):
            state.boolean_true_counts[name] += int(bool(chosen[field]))
        state.phase_rank_counts[phase][rank] += 1
        state.phase_totals[phase] += 1
        state.pressure_rank_counts[pressure][rank] += 1
        state.pressure_totals[pressure] += 1
        state.opening_rank_counts[opening][rank] += 1
        state.opening_totals[opening] += 1


def _decision_groups(
    rows: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    seen: set[str] = set()
    current_id: str | None = None
    current: list[dict[str, object]] = []
    for row in rows:
        decision_id = str(row["decision_id"])
        if decision_id != current_id:
            if current:
                groups.append(current)
                seen.add(str(current_id))
            if decision_id in seen:
                raise ValueError(f"Candidate decision group is not contiguous: {decision_id}")
            current_id = decision_id
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return groups


def _categorical_probability(
    counts: Counter[object],
    total: int,
    value: object,
    category_count: int,
    smoothing_strength: float,
) -> float:
    prior = 1.0 / category_count
    return (counts[value] + smoothing_strength * prior) / (
        total + smoothing_strength
    )


def _nested_counts(values: dict[str, Counter[int]]) -> dict[str, dict[str, int]]:
    return {
        key: {str(rank): count for rank, count in counts.items()}
        for key, counts in values.items()
    }
