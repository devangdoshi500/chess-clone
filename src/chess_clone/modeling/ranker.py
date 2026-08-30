"""Sparse preprocessing, candidate scoring, baselines, and ranking metrics."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable, Protocol

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

from chess_clone.modeling.candidates import validate_no_leakage_fields

_MISSING = "__MISSING__"
_UNKNOWN = "__UNKNOWN__"

CATEGORICAL_FEATURES = frozenset(
    {
        "candidate_move_uci",
        "candidate_piece_moved",
        "candidate_captured_piece_type",
        "game_phase",
        "opening_eco",
        "opening_family",
        "player_color",
        "speed",
        "time_control",
    }
)


class ProbabilityModel(Protocol):
    def predict_proba(self, matrix: sparse.csr_matrix) -> np.ndarray: ...


@dataclass(slots=True)
class SparseOneHotPreprocessor:
    """A compact train-only median/one-hot transformer."""

    input_features: tuple[str, ...]
    numeric_features: tuple[str, ...] = ()
    categorical_features: tuple[str, ...] = ()
    medians: dict[str, float] | None = None
    categories: dict[str, tuple[str, ...]] | None = None
    expanded_feature_names: tuple[str, ...] = ()

    def fit(self, rows: list[dict[str, object]]) -> "SparseOneHotPreprocessor":
        if not rows:
            raise ValueError("Cannot fit preprocessing on an empty training set")
        validate_no_leakage_fields(self.input_features)
        categorical = tuple(
            feature for feature in self.input_features if feature in CATEGORICAL_FEATURES
        )
        numeric = tuple(
            feature for feature in self.input_features if feature not in CATEGORICAL_FEATURES
        )
        medians: dict[str, float] = {}
        for feature in numeric:
            values = [
                float(row[feature])
                for row in rows
                if row.get(feature) is not None
            ]
            medians[feature] = float(median(values)) if values else 0.0
        categories: dict[str, tuple[str, ...]] = {}
        for feature in categorical:
            observed = sorted({_category(row.get(feature)) for row in rows})
            categories[feature] = tuple(observed + [_UNKNOWN])
        expanded = list(numeric)
        for feature in categorical:
            expanded.extend(f"{feature}={value}" for value in categories[feature])
        self.numeric_features = numeric
        self.categorical_features = categorical
        self.medians = medians
        self.categories = categories
        self.expanded_feature_names = tuple(expanded)
        return self

    def transform(self, rows: list[dict[str, object]]) -> sparse.csr_matrix:
        if self.medians is None or self.categories is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        category_offsets: dict[str, int] = {}
        offset = len(self.numeric_features)
        category_indexes: dict[str, dict[str, int]] = {}
        for feature in self.categorical_features:
            category_offsets[feature] = offset
            values = self.categories[feature]
            category_indexes[feature] = {value: index for index, value in enumerate(values)}
            offset += len(values)

        data: list[float] = []
        row_indexes: list[int] = []
        column_indexes: list[int] = []
        for row_index, row in enumerate(rows):
            for column_index, feature in enumerate(self.numeric_features):
                raw = row.get(feature)
                value = self.medians[feature] if raw is None else float(raw)
                if value != 0.0:
                    row_indexes.append(row_index)
                    column_indexes.append(column_index)
                    data.append(value)
            for feature in self.categorical_features:
                value = _category(row.get(feature))
                index = category_indexes[feature].get(
                    value, category_indexes[feature][_UNKNOWN]
                )
                row_indexes.append(row_index)
                column_indexes.append(category_offsets[feature] + index)
                data.append(1.0)
        return sparse.csr_matrix(
            (data, (row_indexes, column_indexes)),
            shape=(len(rows), len(self.expanded_feature_names)),
            dtype=np.float32,
        )

    def fit_transform(self, rows: list[dict[str, object]]) -> sparse.csr_matrix:
        return self.fit(rows).transform(rows)

    def to_dict(self) -> dict[str, object]:
        if self.medians is None or self.categories is None:
            raise RuntimeError("Cannot serialize an unfitted preprocessor")
        return {
            "input_features": list(self.input_features),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "medians": self.medians,
            "categories": {key: list(value) for key, value in self.categories.items()},
            "expanded_feature_names": list(self.expanded_feature_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SparseOneHotPreprocessor":
        categories = payload["categories"]
        if not isinstance(categories, dict):
            raise ValueError("Invalid preprocessing categories")
        medians = payload["medians"]
        if not isinstance(medians, dict):
            raise ValueError("Invalid preprocessing medians")
        return cls(
            input_features=tuple(str(x) for x in payload["input_features"]),
            numeric_features=tuple(str(x) for x in payload["numeric_features"]),
            categorical_features=tuple(str(x) for x in payload["categorical_features"]),
            medians={str(key): float(value) for key, value in medians.items()},
            categories={
                str(key): tuple(str(item) for item in value)
                for key, value in categories.items()
            },
            expanded_feature_names=tuple(
                str(x) for x in payload["expanded_feature_names"]
            ),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "SparseOneHotPreprocessor":
        return cls.from_dict(json.loads(Path(path).read_text()))


def make_tree_classifier() -> RandomForestClassifier:
    """Return the small deterministic first-pass binary candidate scorer."""

    return RandomForestClassifier(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=20260830,
    )


def train_candidate_model(
    train_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    feature_fields: tuple[str, ...],
) -> tuple[RandomForestClassifier, SparseOneHotPreprocessor]:
    validate_candidate_labels(train_rows)
    validate_candidate_labels(validation_rows)
    preprocessor = SparseOneHotPreprocessor(feature_fields)
    train_matrix = preprocessor.fit_transform(train_rows)
    train_labels = np.asarray([int(row["chosen"]) for row in train_rows])
    # Validation rows are deliberately transformed only after train-only fitting;
    # fitting itself uses no validation or future-game information.
    preprocessor.transform(validation_rows)
    model = make_tree_classifier()
    model.fit(train_matrix, train_labels)
    return model, preprocessor


def predict_candidate_probabilities(
    model: ProbabilityModel,
    preprocessor: SparseOneHotPreprocessor,
    rows: list[dict[str, object]],
) -> list[float]:
    """Score candidates and normalize probabilities within each decision."""

    if not rows:
        return []
    raw = model.predict_proba(preprocessor.transform(rows))[:, 1]
    return normalize_candidate_scores(rows, [float(value) for value in raw])


def normalize_candidate_scores(
    rows: list[dict[str, object]], scores: list[float]
) -> list[float]:
    if len(rows) != len(scores):
        raise ValueError("Candidate row and score counts differ")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["decision_id"])].append(index)
    probabilities = [0.0] * len(rows)
    for indexes in grouped.values():
        nonnegative = [max(0.0, float(scores[index])) for index in indexes]
        total = sum(nonnegative)
        values = (
            [value / total for value in nonnegative]
            if total > 0
            else [1.0 / len(indexes)] * len(indexes)
        )
        for index, probability in zip(indexes, values, strict=True):
            probabilities[index] = probability
    return probabilities


def validate_candidate_labels(rows: list[dict[str, object]]) -> None:
    grouped: dict[str, int] = defaultdict(int)
    for row in rows:
        grouped[str(row["decision_id"])] += int(bool(row["chosen"]))
    invalid = {key: count for key, count in grouped.items() if count != 1}
    if invalid:
        sample = list(sorted(invalid.items()))[:3]
        raise ValueError(f"Candidate decisions without exactly one positive: {sample}")


def fit_global_rank_frequencies(
    train_rows: list[dict[str, object]],
) -> dict[int, float]:
    counts = Counter(
        int(row["engine_rank"]) for row in train_rows if bool(row["chosen"])
    )
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Training rows contain no chosen candidates")
    return {rank: counts[rank] / total for rank in range(1, 6)}


def fit_historical_move_counts(
    training_decisions: Iterable[dict[str, object]],
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for decision in training_decisions:
        result[str(decision["canonical_position"])][
            str(decision["actual_move_uci"])
        ] += 1
    return {position: dict(counts) for position, counts in result.items()}


def baseline_probabilities(
    rows: list[dict[str, object]],
    *,
    baseline: str,
    rank_frequencies: dict[int, float],
    historical_counts: dict[str, dict[str, int]] | None = None,
) -> list[float]:
    candidates_by_decision: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        candidates_by_decision[str(row["decision_id"])].append(row)
    use_history: dict[str, bool] = {}
    if baseline == "historical_exact_position":
        for decision_id, candidates in candidates_by_decision.items():
            counts = (historical_counts or {}).get(
                str(candidates[0]["canonical_position"])
            )
            use_history[decision_id] = bool(
                counts
                and any(
                    counts.get(str(candidate["candidate_move_uci"]), 0) > 0
                    for candidate in candidates
                )
            )
    scores: list[float] = []
    for row in rows:
        rank = int(row["engine_rank"])
        if baseline == "stockfish":
            score = 1.0 if rank == 1 else 0.0
        elif baseline == "global_rank_frequency":
            score = rank_frequencies.get(rank, 0.0)
        elif baseline == "historical_exact_position":
            counts = (historical_counts or {}).get(str(row["canonical_position"]))
            if counts and use_history[str(row["decision_id"])]:
                score = float(counts.get(str(row["candidate_move_uci"]), 0))
            else:
                score = rank_frequencies.get(rank, 0.0)
        else:
            raise ValueError(f"Unknown baseline: {baseline}")
        scores.append(score)
    return normalize_candidate_scores(rows, scores)


def ranking_metrics(
    rows: list[dict[str, object]], probabilities: list[float]
) -> dict[str, object]:
    """Compute ranking, NLL, and requested slice accuracies."""

    validate_candidate_labels(rows)
    if len(rows) != len(probabilities):
        raise ValueError("Candidate probability count differs from row count")
    grouped: dict[str, list[tuple[dict[str, object], float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        grouped[str(row["decision_id"])].append((row, float(probability)))

    results: list[dict[str, object]] = []
    max_sum_error = 0.0
    for decision_id, candidates in grouped.items():
        total = sum(probability for _, probability in candidates)
        max_sum_error = max(max_sum_error, abs(1.0 - total))
        ranked = sorted(
            candidates,
            key=lambda item: (
                -item[1],
                int(item[0]["engine_rank"]),
                str(item[0]["candidate_move_uci"]),
            ),
        )
        chosen_index = next(
            index for index, (row, _) in enumerate(ranked, start=1) if row["chosen"]
        )
        chosen_row, chosen_probability = next(
            (row, probability) for row, probability in candidates if row["chosen"]
        )
        predicted_engine_rank = int(ranked[0][0]["engine_rank"])
        results.append(
            {
                "decision_id": decision_id,
                "predicted_rank": chosen_index,
                "chosen_probability": chosen_probability,
                "actual_engine_rank": int(chosen_row["engine_rank"]),
                "predicted_engine_rank": predicted_engine_rank,
                "game_phase": str(chosen_row["game_phase"]),
                "pre_move_time_pressure": str(
                    chosen_row["pre_move_time_pressure"]
                ).lower(),
                "player_color": str(chosen_row["player_color"]),
                "time_control": str(chosen_row["time_control"]),
            }
        )
    if not results:
        raise ValueError("No decisions available for ranking metrics")
    non_rank_1 = [
        result for result in results if int(result["actual_engine_rank"]) > 1
    ]

    return {
        "decision_count": len(results),
        "exact_move_accuracy": _top_rate(results, 1),
        "top_2_accuracy": _top_rate(results, 2),
        "top_3_accuracy": _top_rate(results, 3),
        "mean_reciprocal_rank": sum(
            1.0 / int(result["predicted_rank"]) for result in results
        )
        / len(results),
        "negative_log_likelihood": -sum(
            math.log(max(float(result["chosen_probability"]), 1e-15))
            for result in results
        )
        / len(results),
        "maximum_probability_sum_error": max_sum_error,
        "non_rank_1": {
            "decision_count": len(non_rank_1),
            "exact_move_accuracy": (
                sum(int(result["predicted_rank"]) == 1 for result in non_rank_1)
                / len(non_rank_1)
                if non_rank_1
                else None
            ),
            "mean_reciprocal_rank": (
                sum(1.0 / int(result["predicted_rank"]) for result in non_rank_1)
                / len(non_rank_1)
                if non_rank_1
                else None
            ),
        },
        "confusion_matrix_actual_vs_predicted_engine_rank": _confusion_matrix(
            results
        ),
        "accuracy_by_actual_stockfish_rank": _accuracy_breakdown(
            results, "actual_engine_rank"
        ),
        "accuracy_by_game_phase": _accuracy_breakdown(results, "game_phase"),
        "accuracy_by_time_pressure": _accuracy_breakdown(
            results, "pre_move_time_pressure"
        ),
        "accuracy_by_player_color": _accuracy_breakdown(results, "player_color"),
        "accuracy_by_time_control": _accuracy_breakdown(results, "time_control"),
    }


def feature_importance(
    model: RandomForestClassifier, preprocessor: SparseOneHotPreprocessor
) -> dict[str, object]:
    expanded = [
        (name, float(value))
        for name, value in zip(
            preprocessor.expanded_feature_names,
            model.feature_importances_,
            strict=True,
        )
        if value > 0
    ]
    expanded.sort(key=lambda item: (-item[1], item[0]))
    aggregate: Counter[str] = Counter()
    for name, value in expanded:
        root = name.split("=", 1)[0]
        aggregate[root] += value
    total = sum(aggregate.values())
    aggregated = [
        {
            "feature": name,
            "importance": value,
            "importance_fraction": value / total if total else 0.0,
        }
        for name, value in sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "aggregated": aggregated,
        "top_expanded_features": [
            {"feature": name, "importance": value} for name, value in expanded[:30]
        ],
    }


def _category(value: object) -> str:
    return _MISSING if value is None else str(value)


def _top_rate(results: list[dict[str, object]], limit: int) -> float:
    return sum(int(result["predicted_rank"]) <= limit for result in results) / len(results)


def _accuracy_breakdown(
    results: list[dict[str, object]], field: str
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[str(result[field])].append(result)
    return {
        value: {
            "count": len(items),
            "accuracy": sum(int(item["predicted_rank"]) == 1 for item in items)
            / len(items),
        }
        for value, items in sorted(grouped.items())
    }


def _confusion_matrix(
    results: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        matrix[str(result["actual_engine_rank"])][
            str(result["predicted_engine_rank"])
        ] += 1
    return {
        actual: {predicted: counts.get(predicted, 0) for predicted in map(str, range(1, 6))}
        for actual, counts in sorted(matrix.items())
    }
