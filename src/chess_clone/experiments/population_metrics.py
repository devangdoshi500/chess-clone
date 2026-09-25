"""Move-prediction and behavior-similarity metrics for all-legal policies."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, median

from chess_clone.modeling.legal_policy import BEHAVIOR_BOOLEAN_FIELDS


def legal_policy_metrics(
    rows: list[dict[str, object]], probabilities: list[float]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(rows) != len(probabilities):
        raise ValueError("Candidate row and probability counts differ")
    grouped: dict[str, list[tuple[dict[str, object], float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Probabilities must be finite and between zero and one")
        grouped[str(row["decision_id"])].append((row, probability))

    predictions: list[dict[str, object]] = []
    reciprocal_ranks: list[float] = []
    actual_ranks: list[int] = []
    normalized_rank_scores: list[float] = []
    chosen_probabilities: list[float] = []
    multiclass_brier_scores: list[float] = []
    for decision_id, candidates in grouped.items():
        if abs(sum(value for _, value in candidates) - 1.0) > 1e-6:
            raise ValueError(f"Probabilities do not sum to one for {decision_id}")
        ranked = sorted(
            candidates,
            key=lambda item: (-item[1], str(item[0]["candidate_move_uci"])),
        )
        positives = [item for item in ranked if bool(item[0]["chosen"])]
        if len(positives) != 1:
            raise ValueError(f"Decision {decision_id} does not have one positive")
        actual_row, actual_probability = positives[0]
        actual_rank = next(
            index for index, item in enumerate(ranked, start=1) if item[0] is actual_row
        )
        predicted_row = ranked[0][0]
        reciprocal_ranks.append(1.0 / actual_rank)
        actual_ranks.append(actual_rank)
        normalized_rank_scores.append(
            1.0
            if len(ranked) == 1
            else 1.0 - ((actual_rank - 1) / (len(ranked) - 1))
        )
        chosen_probabilities.append(max(actual_probability, 1e-15))
        multiclass_brier_scores.append(
            sum(
                (probability - int(row is actual_row)) ** 2
                for row, probability in ranked
            )
        )
        boolean_agreements = {
            field: bool(predicted_row[field]) == bool(actual_row[field])
            for field in BEHAVIOR_BOOLEAN_FIELDS
        }
        predictions.append(
            {
                "decision_id": decision_id,
                "game_id": str(actual_row["game_id"]),
                "player_username": str(actual_row["player_username"]),
                "actual_move_uci": str(actual_row["candidate_move_uci"]),
                "predicted_move_uci": str(predicted_row["candidate_move_uci"]),
                "exact_correct": actual_rank == 1,
                "top_3_correct": actual_rank <= 3,
                "top_5_correct": actual_rank <= 5,
                "actual_rank": actual_rank,
                "actual_move_probability": actual_probability,
                "top_1_confidence": ranked[0][1],
                "candidate_count": len(ranked),
                "brier_score": multiclass_brier_scores[-1],
                "normalized_rank_score": normalized_rank_scores[-1],
                "piece_match": predicted_row["candidate_piece_moved"]
                == actual_row["candidate_piece_moved"],
                "wing_match": predicted_row["candidate_destination_wing"]
                == actual_row["candidate_destination_wing"],
                "behavior_similarity": mean(boolean_agreements.values()),
                **{f"{field}_match": value for field, value in boolean_agreements.items()},
                **{
                    f"predicted_{field}": bool(predicted_row[field])
                    for field in BEHAVIOR_BOOLEAN_FIELDS
                },
                **{
                    f"actual_{field}": bool(actual_row[field])
                    for field in BEHAVIOR_BOOLEAN_FIELDS
                },
            }
        )

    count = len(predictions)
    if not count:
        raise ValueError("At least one decision is required")
    aggregate_rates = {}
    rate_errors = []
    for field in BEHAVIOR_BOOLEAN_FIELDS:
        predicted_rate = mean(bool(row[f"predicted_{field}"]) for row in predictions)
        actual_rate = mean(bool(row[f"actual_{field}"]) for row in predictions)
        error = abs(predicted_rate - actual_rate)
        rate_errors.append(error)
        aggregate_rates[field.removeprefix("candidate_")] = {
            "predicted": predicted_rate,
            "actual": actual_rate,
            "absolute_error": error,
        }
    metrics: dict[str, object] = {
        "decisions": count,
        "candidate_rows": len(rows),
        "mean_legal_moves": mean(row["candidate_count"] for row in predictions),
        "exact_move_accuracy": mean(row["exact_correct"] for row in predictions),
        "top_3_accuracy": mean(row["top_3_correct"] for row in predictions),
        "top_5_accuracy": mean(row["top_5_correct"] for row in predictions),
        "mean_reciprocal_rank": mean(reciprocal_ranks),
        "mean_actual_rank": mean(actual_ranks),
        "median_actual_rank": median(actual_ranks),
        "mean_normalized_rank_score": mean(normalized_rank_scores),
        "negative_log_likelihood": -mean(math.log(value) for value in chosen_probabilities),
        "multiclass_brier_score": mean(multiclass_brier_scores),
        "top_1_expected_calibration_error": _top_1_calibration_error(predictions),
        "piece_match_rate": mean(row["piece_match"] for row in predictions),
        "destination_wing_match_rate": mean(row["wing_match"] for row in predictions),
        "mean_behavior_similarity": mean(
            row["behavior_similarity"] for row in predictions
        ),
        "behavior_rate_mean_absolute_error": mean(rate_errors),
        "behavior_rates": aggregate_rates,
    }
    return metrics, predictions


def _top_1_calibration_error(
    predictions: list[dict[str, object]], *, bins: int = 10
) -> float:
    """Measure whether top-choice confidence matches top-choice accuracy."""

    buckets: list[list[dict[str, object]]] = [[] for _ in range(bins)]
    for row in predictions:
        confidence = float(row["top_1_confidence"])
        buckets[min(int(confidence * bins), bins - 1)].append(row)
    count = len(predictions)
    return sum(
        len(bucket)
        / count
        * abs(
            mean(float(row["top_1_confidence"]) for row in bucket)
            - mean(bool(row["exact_correct"]) for row in bucket)
        )
        for bucket in buckets
        if bucket
    )


def uniform_probabilities(rows: list[dict[str, object]]) -> list[float]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["decision_id"])] += 1
    return [1.0 / counts[str(row["decision_id"])] for row in rows]


def move_frequency_probabilities(
    train_rows: list[dict[str, object]], evaluation_rows: list[dict[str, object]], *, smoothing: float = 1.0
) -> list[float]:
    frequencies: dict[str, int] = defaultdict(int)
    for row in train_rows:
        if bool(row["chosen"]):
            frequencies[str(row["candidate_move_uci"])] += 1
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(evaluation_rows):
        grouped[str(row["decision_id"])].append(index)
    result = [0.0] * len(evaluation_rows)
    for indexes in grouped.values():
        weights = [
            frequencies[str(evaluation_rows[index]["candidate_move_uci"])] + smoothing
            for index in indexes
        ]
        total = sum(weights)
        for index, weight in zip(indexes, weights, strict=True):
            result[index] = weight / total
    return result
