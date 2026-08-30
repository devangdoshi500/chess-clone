"""CatBoost grouped ranking, classification, calibration, and diagnostics."""

from collections import defaultdict
import math

import numpy as np
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from scipy.optimize import minimize_scalar

from chess_clone.modeling.ranker import ranking_metrics, validate_candidate_labels

CATBOOST_CATEGORICAL_FEATURES = frozenset(
    {
        "candidate_move_uci",
        "candidate_piece_moved",
        "candidate_source_square",
        "candidate_destination_square",
        "candidate_captured_piece_type",
        "game_phase",
        "opening_eco",
        "opening_family",
        "player_color",
        "speed",
        "time_control",
    }
)


def train_grouped_ranker(
    train_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    feature_fields: tuple[str, ...],
) -> CatBoostRanker:
    """Fit QuerySoftMax with one chess decision per query group."""

    validate_candidate_labels(train_rows)
    validate_candidate_labels(validation_rows)
    train_pool = candidate_pool(train_rows, feature_fields, grouped=True)
    validation_pool = candidate_pool(validation_rows, feature_fields, grouped=True)
    model = CatBoostRanker(
        loss_function="QuerySoftMax",
        eval_metric="QuerySoftMax",
        iterations=350,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        random_seed=20260830,
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=50,
        verbose=False,
    )
    return model


def train_candidate_classifier(
    train_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    feature_fields: tuple[str, ...],
) -> CatBoostClassifier:
    """Fit a conventional binary scorer for comparison with grouped ranking."""

    validate_candidate_labels(train_rows)
    validate_candidate_labels(validation_rows)
    train_pool = candidate_pool(train_rows, feature_fields, grouped=False)
    validation_pool = candidate_pool(validation_rows, feature_fields, grouped=False)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=350,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        random_seed=20260830,
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        train_pool,
        eval_set=validation_pool,
        use_best_model=True,
        early_stopping_rounds=50,
        verbose=False,
    )
    return model


def candidate_pool(
    rows: list[dict[str, object]],
    feature_fields: tuple[str, ...],
    *,
    grouped: bool,
) -> Pool:
    values = [
        [
            _categorical(row.get(field))
            if field in CATBOOST_CATEGORICAL_FEATURES
            else _numeric(row.get(field))
            for field in feature_fields
        ]
        for row in rows
    ]
    labels = [int(bool(row["chosen"])) for row in rows]
    categorical_indexes = [
        index
        for index, field in enumerate(feature_fields)
        if field in CATBOOST_CATEGORICAL_FEATURES
    ]
    group_id = _group_ids(rows) if grouped else None
    return Pool(
        values,
        label=labels,
        group_id=group_id,
        cat_features=categorical_indexes,
        feature_names=list(feature_fields),
    )


def predict_relevance_scores(
    model: CatBoostRanker | CatBoostClassifier,
    rows: list[dict[str, object]],
    feature_fields: tuple[str, ...],
) -> list[float]:
    grouped = isinstance(model, CatBoostRanker)
    pool = candidate_pool(rows, feature_fields, grouped=grouped)
    if isinstance(model, CatBoostClassifier):
        values = model.predict(pool, prediction_type="RawFormulaVal")
    else:
        values = model.predict(pool)
    return [float(value) for value in np.asarray(values).reshape(-1)]


def groupwise_softmax(
    rows: list[dict[str, object]],
    scores: list[float],
    *,
    temperature: float = 1.0,
) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(rows) != len(scores):
        raise ValueError("Candidate row and score counts differ")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["decision_id"])].append(index)
    result = [0.0] * len(rows)
    for indexes in groups.values():
        scaled = [scores[index] / temperature for index in indexes]
        maximum = max(scaled)
        exponentials = [math.exp(value - maximum) for value in scaled]
        total = sum(exponentials)
        for index, value in zip(indexes, exponentials, strict=True):
            result[index] = value / total
    return result


def fit_temperature(
    validation_rows: list[dict[str, object]], validation_scores: list[float]
) -> float:
    """Fit one positive softmax temperature on validation NLL only."""

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        probabilities = groupwise_softmax(
            validation_rows, validation_scores, temperature=temperature
        )
        return _negative_log_likelihood(validation_rows, probabilities)

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"Temperature calibration failed: {result.message}")
    return math.exp(float(result.x))


def calibration_report(
    rows: list[dict[str, object]], probabilities: list[float], *, bins: int = 10
) -> dict[str, object]:
    if bins < 2:
        raise ValueError("At least two calibration bins are required")
    if len(rows) != len(probabilities):
        raise ValueError("Candidate row and probability counts differ")
    bucket_values: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for row, probability in zip(rows, probabilities, strict=True):
        index = min(int(probability * bins), bins - 1)
        bucket_values[index].append((probability, int(bool(row["chosen"]))))
    total = len(rows)
    expected_calibration_error = 0.0
    output_bins: list[dict[str, object]] = []
    for index, values in enumerate(bucket_values):
        if not values:
            continue
        mean_probability = sum(value[0] for value in values) / len(values)
        chosen_rate = sum(value[1] for value in values) / len(values)
        expected_calibration_error += (
            len(values) / total * abs(mean_probability - chosen_rate)
        )
        output_bins.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(values),
                "mean_probability": mean_probability,
                "chosen_rate": chosen_rate,
            }
        )
    decision_count = len({str(row["decision_id"]) for row in rows})
    brier = sum(
        (probability - int(bool(row["chosen"]))) ** 2
        for row, probability in zip(rows, probabilities, strict=True)
    ) / decision_count
    return {
        "expected_calibration_error": expected_calibration_error,
        "multiclass_brier_score": brier,
        "bins": output_bins,
    }


def catboost_feature_importance(
    model: CatBoostRanker | CatBoostClassifier,
    feature_fields: tuple[str, ...],
) -> list[dict[str, float | str]]:
    values = model.get_feature_importance(type="PredictionValuesChange")
    return [
        {"feature": feature, "importance": float(value)}
        for feature, value in sorted(
            zip(feature_fields, values, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def hybrid_probabilities(
    rows: list[dict[str, object]],
    model_probabilities: list[float],
    historical_counts: dict[str, dict[str, int]],
) -> list[float]:
    """Use exact-position frequencies when candidates have historical support."""

    by_decision: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_decision[str(row["decision_id"])].append(index)
    result = list(model_probabilities)
    for indexes in by_decision.values():
        position = str(rows[indexes[0]]["canonical_position"])
        counts = historical_counts.get(position)
        if not counts:
            continue
        candidate_counts = [
            counts.get(str(rows[index]["candidate_move_uci"]), 0) for index in indexes
        ]
        total = sum(candidate_counts)
        if total == 0:
            continue
        for index, count in zip(indexes, candidate_counts, strict=True):
            result[index] = count / total
    return result


def historical_coverage_diagnostics(
    rows: list[dict[str, object]],
    probabilities: list[float],
    historical_counts: dict[str, dict[str, int]],
) -> dict[str, object]:
    by_decision: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_decision[str(row["decision_id"])].append(index)
    buckets: dict[str, list[str]] = defaultdict(list)
    covered: list[str] = []
    uncovered: list[str] = []
    for decision_id, indexes in by_decision.items():
        position = str(rows[indexes[0]]["canonical_position"])
        observation_count = sum(historical_counts.get(position, {}).values())
        if observation_count == 0:
            uncovered.append(decision_id)
            continue
        covered.append(decision_id)
        buckets[_frequency_bucket(observation_count)].append(decision_id)
    return {
        "total_decisions": len(by_decision),
        "covered_decisions": len(covered),
        "coverage": len(covered) / len(by_decision) if by_decision else 0.0,
        "covered": _subset_metrics(rows, probabilities, set(covered)),
        "uncovered": _subset_metrics(rows, probabilities, set(uncovered)),
        "by_training_observation_frequency": {
            bucket: _subset_metrics(rows, probabilities, set(decision_ids))
            for bucket, decision_ids in sorted(buckets.items())
        },
    }


def temporal_slice_metrics(
    rows: list[dict[str, object]],
    probabilities: list[float],
    *,
    minimum_decisions: int = 100,
) -> dict[str, object]:
    decisions_by_year: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        played_at = row.get("played_at")
        year = str(getattr(played_at, "year", "unknown"))
        decisions_by_year[year].add(str(row["decision_id"]))
    return {
        year: _subset_metrics(rows, probabilities, decision_ids)
        for year, decision_ids in sorted(decisions_by_year.items())
        if len(decision_ids) >= minimum_decisions
    }


def _subset_metrics(
    rows: list[dict[str, object]],
    probabilities: list[float],
    decision_ids: set[str],
) -> dict[str, object] | None:
    if not decision_ids:
        return None
    selected = [
        (row, probability)
        for row, probability in zip(rows, probabilities, strict=True)
        if str(row["decision_id"]) in decision_ids
    ]
    return ranking_metrics(
        [row for row, _ in selected], [probability for _, probability in selected]
    )


def _negative_log_likelihood(
    rows: list[dict[str, object]], probabilities: list[float]
) -> float:
    chosen = [
        probability
        for row, probability in zip(rows, probabilities, strict=True)
        if bool(row["chosen"])
    ]
    return -sum(math.log(max(value, 1e-15)) for value in chosen) / len(chosen)


def _group_ids(rows: list[dict[str, object]]) -> list[int]:
    group_ids: list[int] = []
    seen: set[str] = set()
    current: str | None = None
    group_index = -1
    for row in rows:
        decision_id = str(row["decision_id"])
        if decision_id != current:
            if decision_id in seen:
                raise ValueError(f"Candidate decision group is not contiguous: {decision_id}")
            if current is not None:
                seen.add(current)
            current = decision_id
            group_index += 1
        group_ids.append(group_index)
    return group_ids


def _categorical(value: object) -> str:
    return "__MISSING__" if value is None else str(value)


def _numeric(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _frequency_bucket(observations: int) -> str:
    if observations == 1:
        return "1"
    if observations <= 3:
        return "2-3"
    if observations <= 10:
        return "4-10"
    return ">10"
