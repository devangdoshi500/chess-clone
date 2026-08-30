"""Train and evaluate grouped CatBoost personalization experiments."""

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq
from catboost import CatBoostClassifier, CatBoostRanker

from chess_clone.modeling.boosted import (
    calibration_report,
    catboost_feature_importance,
    fit_temperature,
    groupwise_softmax,
    historical_coverage_diagnostics,
    hybrid_probabilities,
    predict_relevance_scores,
    temporal_slice_metrics,
    train_candidate_classifier,
    train_grouped_ranker,
)
from chess_clone.modeling.candidates import (
    BOOSTED_CANDIDATE_FEATURE_FIELDS,
    BOOSTED_CONTEXT_FEATURE_FIELDS,
    ENGINE_FEATURE_FIELDS,
    CandidateDataset,
    build_candidate_dataset,
    chronological_game_split,
    validate_no_leakage_fields,
)
from chess_clone.modeling.player_history import (
    PLAYER_HISTORY_FEATURE_FIELDS,
    PlayerHistoryEncoder,
)
from chess_clone.modeling.ranker import (
    baseline_probabilities,
    fit_global_rank_frequencies,
    fit_historical_move_counts,
    predict_candidate_probabilities,
    ranking_metrics,
)
from chess_clone.modeling.training import load_ranker

BOOSTED_FEATURE_ABLATIONS = {
    "engine_only": ENGINE_FEATURE_FIELDS,
    "engine_and_candidate": (
        ENGINE_FEATURE_FIELDS + BOOSTED_CANDIDATE_FEATURE_FIELDS
    ),
    "engine_candidate_context": (
        ENGINE_FEATURE_FIELDS
        + BOOSTED_CANDIDATE_FEATURE_FIELDS
        + BOOSTED_CONTEXT_FEATURE_FIELDS
    ),
    "engine_candidate_context_history": (
        ENGINE_FEATURE_FIELDS
        + BOOSTED_CANDIDATE_FEATURE_FIELDS
        + BOOSTED_CONTEXT_FEATURE_FIELDS
        + PLAYER_HISTORY_FEATURE_FIELDS
    ),
}


@dataclass(frozen=True, slots=True)
class BoostedTrainingSummary:
    username: str
    artifact_dir: Path
    total_decisions: int
    usable_decisions: int
    outside_top_5_decisions: int
    runtime_seconds: float
    metrics: dict[str, object]


def default_boosted_artifact_dir(username: str, root: Path) -> Path:
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return root / f"{username.strip().lower()}_{batch_id}_catboost_ranker"


def train_boosted_rankers(
    username: str,
    *,
    features_path: str | Path,
    analysis_path: str | Path,
    games_path: str | Path,
    rf_artifact_dir: str | Path,
    artifact_dir: str | Path,
) -> BoostedTrainingSummary:
    started = perf_counter()
    feature_rows = pq.read_table(_path(features_path, "features")).to_pylist()
    analysis_rows = pq.read_table(_path(analysis_path, "analysis")).to_pylist()
    games_rows = pq.read_table(_path(games_path, "games")).to_pylist()
    requested = username.casefold()
    feature_rows = [
        row for row in feature_rows if str(row["player_username"]).casefold() == requested
    ]
    analysis_rows = [
        row for row in analysis_rows if str(row["player_username"]).casefold() == requested
    ]
    relevant_game_ids = {str(row["game_id"]) for row in feature_rows}
    games_rows = [
        row for row in games_rows if str(row["game_id"]) in relevant_game_ids
    ]
    split = chronological_game_split(games_rows)
    split_metadata = split.to_dict()
    _validate_preserved_split(split_metadata, Path(rf_artifact_dir))
    dataset = build_candidate_dataset(feature_rows, analysis_rows, split)

    raw_rows = {
        name: [row for row in dataset.candidate_rows if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    history_encoder = PlayerHistoryEncoder()
    augmented_rows = {
        "train": history_encoder.fit_transform_ordered(raw_rows["train"]),
        "validation": history_encoder.transform(raw_rows["validation"]),
        "test": history_encoder.transform(raw_rows["test"]),
    }
    for fields in BOOSTED_FEATURE_ABLATIONS.values():
        validate_no_leakage_fields(fields)

    rank_frequencies = fit_global_rank_frequencies(raw_rows["train"])
    training_decisions = [
        decision for decision in dataset.decisions if decision["split"] == "train"
    ]
    historical_counts = fit_historical_move_counts(training_decisions)

    probabilities: dict[str, dict[str, list[float]]] = {
        "validation": {},
        "test": {},
    }
    metrics: dict[str, object] = {
        "baselines": {"validation": {}, "test": {}},
        "random_forest": {"validation": {}, "test": {}},
        "catboost_ranker_ablations": {},
    }
    for split_name in ("validation", "test"):
        for baseline in (
            "stockfish",
            "global_rank_frequency",
            "historical_exact_position",
        ):
            values = baseline_probabilities(
                raw_rows[split_name],
                baseline=baseline,
                rank_frequencies=rank_frequencies,
                historical_counts=historical_counts,
            )
            probabilities[split_name][baseline] = values
            metrics["baselines"][split_name][baseline] = ranking_metrics(
                raw_rows[split_name], values
            )

    rf_source = Path(rf_artifact_dir)
    for variant in ("engine_only", "engine_and_context"):
        model, preprocessor = load_ranker(rf_source, variant=variant)
        for split_name in ("validation", "test"):
            values = predict_candidate_probabilities(
                model, preprocessor, raw_rows[split_name]
            )
            probabilities[split_name][f"random_forest_{variant}"] = values
            metrics["random_forest"][split_name][variant] = ranking_metrics(
                raw_rows[split_name], values
            )

    destination = Path(artifact_dir)
    if (destination / "manifest.json").is_file():
        raise FileExistsError(f"Completed artifact directory already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    trained_rankers = {}
    resumed_models: list[str] = []
    for ablation, fields in BOOSTED_FEATURE_ABLATIONS.items():
        source_rows = augmented_rows if ablation.endswith("history") else raw_rows
        model_path = destination / f"catboost_ranker_{ablation}.cbm"
        if model_path.is_file():
            model = CatBoostRanker()
            model.load_model(model_path)
            resumed_models.append(model_path.name)
        else:
            model = train_grouped_ranker(
                source_rows["train"], source_rows["validation"], fields
            )
        trained_rankers[ablation] = model
        model.save_model(model_path)
        metrics["catboost_ranker_ablations"][ablation] = {}
        for split_name in ("validation", "test"):
            scores = predict_relevance_scores(
                model, source_rows[split_name], fields
            )
            values = groupwise_softmax(source_rows[split_name], scores)
            probabilities[split_name][f"catboost_{ablation}"] = values
            metrics["catboost_ranker_ablations"][ablation][split_name] = (
                ranking_metrics(source_rows[split_name], values)
            )

    full_name = "engine_candidate_context_history"
    full_fields = BOOSTED_FEATURE_ABLATIONS[full_name]
    full_model = trained_rankers[full_name]
    validation_scores = predict_relevance_scores(
        full_model, augmented_rows["validation"], full_fields
    )
    temperature = fit_temperature(augmented_rows["validation"], validation_scores)
    calibration: dict[str, object] = {"temperature": temperature}
    calibrated_probabilities: dict[str, list[float]] = {}
    for split_name in ("validation", "test"):
        scores = (
            validation_scores
            if split_name == "validation"
            else predict_relevance_scores(
                full_model, augmented_rows[split_name], full_fields
            )
        )
        before = groupwise_softmax(augmented_rows[split_name], scores)
        after = groupwise_softmax(
            augmented_rows[split_name], scores, temperature=temperature
        )
        calibrated_probabilities[split_name] = after
        calibration[split_name] = {
            "before": {
                "ranking_metrics": ranking_metrics(
                    augmented_rows[split_name], before
                ),
                "calibration": calibration_report(
                    augmented_rows[split_name], before
                ),
            },
            "after": {
                "ranking_metrics": ranking_metrics(
                    augmented_rows[split_name], after
                ),
                "calibration": calibration_report(
                    augmented_rows[split_name], after
                ),
            },
        }
        probabilities[split_name]["catboost_grouped_calibrated"] = after

    classifier_path = destination / "catboost_classifier_full.cbm"
    if classifier_path.is_file():
        classifier = CatBoostClassifier()
        classifier.load_model(classifier_path)
        resumed_models.append(classifier_path.name)
    else:
        classifier = train_candidate_classifier(
            augmented_rows["train"], augmented_rows["validation"], full_fields
        )
        classifier.save_model(classifier_path)
    classifier_metrics: dict[str, object] = {}
    hybrid_metrics: dict[str, object] = {}
    historical_diagnostics: dict[str, object] = {}
    temporal: dict[str, object] = {}
    for split_name in ("validation", "test"):
        classifier_scores = predict_relevance_scores(
            classifier, augmented_rows[split_name], full_fields
        )
        classifier_values = groupwise_softmax(
            augmented_rows[split_name], classifier_scores
        )
        classifier_metrics[split_name] = ranking_metrics(
            augmented_rows[split_name], classifier_values
        )
        hybrid_values = hybrid_probabilities(
            augmented_rows[split_name],
            calibrated_probabilities[split_name],
            historical_counts,
        )
        probabilities[split_name]["historical_catboost_hybrid"] = hybrid_values
        hybrid_metrics[split_name] = ranking_metrics(
            augmented_rows[split_name], hybrid_values
        )
        historical_diagnostics[split_name] = historical_coverage_diagnostics(
            raw_rows[split_name],
            probabilities[split_name]["historical_exact_position"],
            historical_counts,
        )

    for method in (
        "stockfish",
        "historical_exact_position",
        "random_forest_engine_and_context",
        "catboost_grouped_calibrated",
        "historical_catboost_hybrid",
    ):
        temporal[method] = temporal_slice_metrics(
            augmented_rows["test"], probabilities["test"][method]
        )

    metrics["catboost_classifier"] = classifier_metrics
    metrics["catboost_grouped_calibration"] = calibration
    metrics["historical_catboost_hybrid"] = hybrid_metrics
    metrics["historical_diagnostics"] = historical_diagnostics
    metrics["temporal_test_slices"] = temporal
    metrics["feature_importance"] = {
        ablation: catboost_feature_importance(
            model, BOOSTED_FEATURE_ABLATIONS[ablation]
        )
        for ablation, model in trained_rankers.items()
    }
    metrics["global_rank_frequencies"] = {
        str(rank): value for rank, value in rank_frequencies.items()
    }

    history_encoder.save(destination / "player_history.json")
    _write_json(
        destination / "feature_list.json",
        {key: list(value) for key, value in BOOSTED_FEATURE_ABLATIONS.items()},
    )
    _write_json(destination / "evaluation_metrics.json", metrics)
    split_output = _split_output(dataset, split_metadata, rf_source)
    _write_json(destination / "split_metadata.json", split_output)
    _write_json(
        destination / "historical_lookup_metadata.json",
        {
            "training_game_ids": sorted(split.train_game_ids),
            "position_count": len(historical_counts),
            "source_split": "train",
        },
    )
    runtime = perf_counter() - started
    _write_json(
        destination / "manifest.json",
        {
            "format_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "username": username,
            "runtime_seconds": runtime,
            "runtime_scope": (
                "current finalization invocation; excludes earlier fitting time for "
                "resumed models"
                if resumed_models
                else "complete training and finalization invocation"
            ),
            "resumed_models": resumed_models,
            "model": {
                "library": "catboost",
                "ranking_objective": "QuerySoftMax",
                "target": "chosen candidate = 1; other candidates = 0",
                "group": "one (game_id, ply) decision with up to five candidates",
                "score": "candidate relevance; groupwise softmax gives probabilities",
                "inference": "rank candidates by relevance within their decision",
                "calibration_temperature": temperature,
            },
            "inputs": {
                "features": str(Path(features_path).resolve()),
                "analysis": str(Path(analysis_path).resolve()),
                "games": str(Path(games_path).resolve()),
                "random_forest_artifact": str(rf_source.resolve()),
            },
            "dataset": {
                "total_decisions": dataset.total_decisions,
                "usable_decisions": dataset.inside_top_5_decisions,
                "outside_top_5_decisions": dataset.outside_top_5_decisions,
                "candidate_rows": len(dataset.candidate_rows),
            },
        },
    )
    return BoostedTrainingSummary(
        username=username,
        artifact_dir=destination,
        total_decisions=dataset.total_decisions,
        usable_decisions=dataset.inside_top_5_decisions,
        outside_top_5_decisions=dataset.outside_top_5_decisions,
        runtime_seconds=runtime,
        metrics=metrics,
    )


def _validate_preserved_split(
    current: dict[str, object], rf_artifact_dir: Path
) -> None:
    path = rf_artifact_dir / "split_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"RF split metadata not found: {path}")
    previous = json.loads(path.read_text())
    for name in ("train", "validation", "test"):
        if set(current[name]["game_ids"]) != set(previous[name]["game_ids"]):
            raise ValueError(f"Chronological {name} split differs from RF baseline")


def _split_output(
    dataset: CandidateDataset,
    metadata: dict[str, object],
    rf_artifact_dir: Path,
) -> dict[str, object]:
    for name in ("train", "validation", "test"):
        decisions = [row for row in dataset.decisions if row["split"] == name]
        metadata[name]["decision_count"] = len(decisions)
        metadata[name]["usable_decision_count"] = sum(
            bool(row["usable"]) for row in decisions
        )
    metadata["integrity"] = {
        "matches_random_forest_split": True,
        "random_forest_artifact": str(rf_artifact_dir.resolve()),
        "game_overlap_count": 0,
        "history_fit_split": "train",
        "calibration_fit_split": "validation",
    }
    return metadata


def _path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} dataset not found: {path}")
    return path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
