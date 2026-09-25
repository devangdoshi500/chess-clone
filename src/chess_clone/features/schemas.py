"""Normalized one-row-per-decision behavioral feature schema."""

from dataclasses import asdict, dataclass
from typing import Literal


IDENTIFIER_FIELDS = (
    "game_id",
    "player_username",
    "ply",
    "move_number",
    "player_color",
    "fen",
)

# These fields are available before the target player chooses a move. Engine
# fields here use only the current position, never a future game position.
INFERENCE_TIME_FEATURE_FIELDS = (
    "speed",
    "opening_eco",
    "opening_name",
    "player_rating",
    "opponent_rating",
    "rating_difference",
    "initial_time_seconds",
    "increment_seconds",
    "game_phase",
    "total_piece_count",
    "material_balance",
    "player_queen_present",
    "opponent_queen_present",
    "legal_move_count",
    "player_in_check",
    "castling_rights_available",
    "engine_best_move",
    "engine_eval_before",
    "engine_mate_before",
    "approximate_winning_chance_before",
)

# These fields describe the observed decision or clock state after that move.
OBSERVED_BEHAVIOR_FIELDS = (
    "actual_move_uci",
    "actual_move_san",
    "player_clock_seconds_after_move",
    "seconds_spent_on_move",
    "fraction_of_initial_time_remaining",
    "time_remaining_seconds",
    "time_remaining_fraction",
    "time_pressure",
    "low_time",
    "piece_moved",
    "is_capture",
    "is_check",
    "is_castle",
    "is_promotion",
    "is_en_passant",
    "is_queen_trade",
    "captured_piece_type",
)

# These are post-move labels/diagnostics. Future training code should not use
# them as inference-time model inputs.
DIAGNOSTIC_FIELDS = (
    "actual_move_eval",
    "actual_move_mate_in",
    "centipawn_loss",
    "actual_move_rank",
    "actual_move_in_top_1",
    "actual_move_in_top_3",
    "actual_move_in_top_5",
    "approximate_winning_chance_after_actual_move",
)


@dataclass(frozen=True, slots=True)
class BehaviorFeatureRecord:
    """One target-player decision with pre-move, behavior, and diagnostic fields."""

    game_id: str
    player_username: str
    ply: int
    move_number: int
    player_color: Literal["white", "black"]
    fen: str
    actual_move_uci: str
    actual_move_san: str

    speed: str | None
    opening_eco: str | None
    opening_name: str | None
    player_rating: int | None
    opponent_rating: int | None
    rating_difference: int | None

    player_clock_seconds_after_move: float | None
    seconds_spent_on_move: float | None
    initial_time_seconds: int | None
    increment_seconds: int | None
    fraction_of_initial_time_remaining: float | None
    time_pressure: bool | None
    time_remaining_seconds: float | None
    time_remaining_fraction: float | None
    low_time: bool | None

    game_phase: Literal["opening", "middlegame", "endgame"]
    total_piece_count: int
    material_balance: int
    player_queen_present: bool
    opponent_queen_present: bool
    legal_move_count: int
    player_in_check: bool
    castling_rights_available: bool

    piece_moved: str
    is_capture: bool
    is_check: bool
    is_castle: bool
    is_promotion: bool
    is_en_passant: bool
    is_queen_trade: bool
    captured_piece_type: str | None

    engine_best_move: str | None
    engine_eval_before: int | None
    engine_mate_before: int | None
    actual_move_eval: int | None
    actual_move_mate_in: int | None
    centipawn_loss: int | None
    actual_move_rank: int | None
    actual_move_in_top_1: bool | None
    actual_move_in_top_3: bool | None
    actual_move_in_top_5: bool | None
    approximate_winning_chance_before: float | None
    approximate_winning_chance_after_actual_move: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def inference_features(self) -> dict[str, object]:
        values = self.to_dict()
        fields = IDENTIFIER_FIELDS + INFERENCE_TIME_FEATURE_FIELDS
        return {field: values[field] for field in fields}

    def observed_behavior(self) -> dict[str, object]:
        values = self.to_dict()
        return {field: values[field] for field in OBSERVED_BEHAVIOR_FIELDS}

    def diagnostics(self) -> dict[str, object]:
        values = self.to_dict()
        return {field: values[field] for field in DIAGNOSTIC_FIELDS}
