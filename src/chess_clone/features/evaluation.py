"""Evaluation normalization and approximate winning-chance transforms."""

from dataclasses import dataclass
import math

import chess.engine

MATE_SCORE_CP = 100_000
_WINNING_CHANCE_SLOPE = 0.00368208


def evaluation_to_centipawns(
    score_cp: int | None, mate_in: int | None
) -> int | None:
    """Map separate centipawn/mate scores onto one deterministic scale."""

    if mate_in is not None:
        # Actual-move scores are converted back to the player-who-moved
        # perspective. At a terminal checkmate Stockfish reports mate 0 from
        # the now-mated side; after perspective conversion zero has no sign.
        # In this pipeline mate 0 therefore means the target player just
        # delivered mate. Pre-move positions always contain a legal move and
        # cannot use this terminal sentinel.
        if mate_in == 0:
            return MATE_SCORE_CP
        return chess.engine.Mate(mate_in).score(mate_score=MATE_SCORE_CP)
    return score_cp


def approximate_winning_chance(
    score_cp: int | None, mate_in: int | None = None
) -> float | None:
    """Map evaluation to an approximate player winning chance in [0, 1].

    This uses a logistic curve with slope 0.00368208, a commonly used
    Lichess-style centipawn normalization. It is explicitly approximate rather
    than a claim to reproduce current Lichess Insights behavior exactly.
    """

    normalized = evaluation_to_centipawns(score_cp, mate_in)
    if normalized is None:
        return None
    value = _WINNING_CHANCE_SLOPE * normalized
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True, slots=True)
class EngineRankFeatures:
    actual_move_rank: int | None
    actual_move_in_top_1: bool | None
    actual_move_in_top_3: bool | None
    actual_move_in_top_5: bool | None


def derive_engine_rank_features(
    actual_move_uci: str,
    analysis_lines: list[dict[str, object]],
) -> EngineRankFeatures:
    ranked = sorted(analysis_lines, key=lambda row: int(row["pv_rank"]))
    actual_rank = next(
        (
            int(row["pv_rank"])
            for row in ranked
            if row.get("best_move_uci") == actual_move_uci
        ),
        None,
    )
    requested = max(
        (int(row.get("multipv_requested") or 1) for row in ranked), default=0
    )
    available = len(ranked)

    def top_flag(limit: int) -> bool | None:
        if actual_rank is not None:
            return actual_rank <= limit
        if requested >= limit and available >= limit:
            return False
        return None

    return EngineRankFeatures(
        actual_move_rank=actual_rank,
        actual_move_in_top_1=top_flag(1),
        actual_move_in_top_3=top_flag(3),
        actual_move_in_top_5=top_flag(5),
    )
