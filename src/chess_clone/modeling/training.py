"""End-to-end personalized candidate-ranking training and artifact persistence."""

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq
import joblib
from sklearn.ensemble import RandomForestClassifier

from chess_clone.modeling.candidates import (
    CandidateDataset,
    ENGINE_FEATURE_FIELDS,
    FULL_FEATURE_FIELDS,
    build_candidate_dataset,
    chronological_game_split,
    validate_no_leakage_fields,
)
from chess_clone.modeling.ranker import (
    SparseOneHotPreprocessor,
    baseline_probabilities,
    feature_importance,
    fit_global_rank_frequencies,
    fit_historical_move_counts,
    predict_candidate_probabilities,
    ranking_metrics,
    train_candidate_model,
    validate_candidate_labels,
)


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    username: str
    artifact_dir: Path
    total_decisions: int
    inside_top_5_decisions: int
    outside_top_5_decisions: int
    runtime_seconds: float
    metrics: dict[str, object]
    split_metadata: dict[str, object]


def default_model_artifact_dir(username: str, root: Path) -> Path:
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return root / f"{username.strip().lower()}_{batch_id}"


def train_personalized_ranker(
    username: str,
    *,
    features_path: str | Path,
    analysis_path: str | Path,
    games_path: str | Path,
    artifact_dir: str | Path,
) -> TrainingRunSummary:
    """Build the candidate task, train the ablation pair, evaluate, and save."""

    started = perf_counter()
    feature_source = _existing_path(features_path, "BehaviorFeature")
    analysis_source = _existing_path(analysis_path, "analysis")
    games_source = _existing_path(games_path, "game metadata")
    try:
        feature_rows = pq.read_table(feature_source).to_pylist()
        analysis_rows = pq.read_table(analysis_source).to_pylist()
        game_rows = pq.read_table(games_source).to_pylist()
    except Exception as exc:
        raise ValueError(f"Could not read ranking input Parquet: {exc}") from exc

    requested = username.casefold()
    feature_rows = [
        row for row in feature_rows if str(row["player_username"]).casefold() == requested
    ]
    analysis_rows = [
        row for row in analysis_rows if str(row["player_username"]).casefold() == requested
    ]
    game_ids = {str(row["game_id"]) for row in feature_rows}
    game_rows = [row for row in game_rows if str(row["game_id"]) in game_ids]
    if not feature_rows or not analysis_rows or not game_rows:
        raise ValueError(f"No complete ranking inputs found for player '{username}'")

    split = chronological_game_split(game_rows)
    dataset = build_candidate_dataset(feature_rows, analysis_rows, split)
    validate_no_leakage_fields(ENGINE_FEATURE_FIELDS)
    validate_no_leakage_fields(FULL_FEATURE_FIELDS)

    rows_by_split = {
        name: [row for row in dataset.candidate_rows if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    for rows in rows_by_split.values():
        validate_candidate_labels(rows)
    if any(not rows for rows in rows_by_split.values()):
        raise ValueError("At least one chronological candidate split is empty")

    rank_frequencies = fit_global_rank_frequencies(rows_by_split["train"])
    training_decisions = [
        decision for decision in dataset.decisions if decision["split"] == "train"
    ]
    historical_counts = fit_historical_move_counts(training_decisions)

    baseline_metrics: dict[str, object] = {}
    for split_name in ("validation", "test"):
        split_rows = rows_by_split[split_name]
        baseline_metrics[split_name] = {}
        for name in (
            "stockfish",
            "global_rank_frequency",
            "historical_exact_position",
        ):
            probabilities = baseline_probabilities(
                split_rows,
                baseline=name,
                rank_frequencies=rank_frequencies,
                historical_counts=historical_counts,
            )
            baseline_metrics[split_name][name] = ranking_metrics(
                split_rows, probabilities
            )

    model_results: dict[str, object] = {}
    trained: dict[str, tuple[RandomForestClassifier, SparseOneHotPreprocessor]] = {}
    for name, fields in (
        ("engine_only", ENGINE_FEATURE_FIELDS),
        ("engine_and_context", FULL_FEATURE_FIELDS),
    ):
        model, preprocessor = train_candidate_model(
            rows_by_split["train"], rows_by_split["validation"], fields
        )
        trained[name] = (model, preprocessor)
        model_results[name] = {
            split_name: ranking_metrics(
                rows_by_split[split_name],
                predict_candidate_probabilities(
                    model, preprocessor, rows_by_split[split_name]
                ),
            )
            for split_name in ("validation", "test")
        }
        model_results[name]["feature_importance"] = feature_importance(
            model, preprocessor
        )

    split_metadata = _split_metadata(dataset, split.to_dict())
    metrics: dict[str, object] = {
        "baselines": baseline_metrics,
        "models": model_results,
        "global_rank_frequencies": {
            str(rank): probability for rank, probability in rank_frequencies.items()
        },
        "historical_coverage": _historical_coverage(dataset, historical_counts),
    }
    destination = Path(artifact_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Artifact directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, (model, preprocessor) in trained.items():
        joblib.dump(model, destination / f"{name}_model.joblib")
        preprocessor.save(destination / f"{name}_preprocessing.json")

    runtime = perf_counter() - started
    dataset_summary = {
        "total_decisions": dataset.total_decisions,
        "inside_top_5_decisions": dataset.inside_top_5_decisions,
        "outside_top_5_decisions": dataset.outside_top_5_decisions,
        "usable_candidate_ranking_decisions": (
            dataset.usable_candidate_ranking_decisions
        ),
        "candidate_rows": len(dataset.candidate_rows),
    }
    _write_json(destination / "split_metadata.json", split_metadata)
    _write_json(destination / "evaluation_metrics.json", metrics)
    _write_json(
        destination / "feature_list.json",
        {
            "engine_only": list(ENGINE_FEATURE_FIELDS),
            "engine_and_context": list(FULL_FEATURE_FIELDS),
        },
    )
    _write_json(
        destination / "manifest.json",
        {
            "format_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "username": username,
            "runtime_seconds": runtime,
            "model": {
                "type": "sklearn.ensemble.RandomForestClassifier",
                "hyperparameters": trained["engine_and_context"][0].get_params(),
                "candidate_objective": "binary chosen/not-chosen scoring",
                "decision_probability_normalization": "sum-to-one",
            },
            "dataset_summary": dataset_summary,
            "inputs": {
                "features": str(feature_source.resolve()),
                "analysis": str(analysis_source.resolve()),
                "games": str(games_source.resolve()),
            },
            "artifacts": {
                "engine_only_model": "engine_only_model.joblib",
                "engine_only_preprocessing": "engine_only_preprocessing.json",
                "engine_and_context_model": "engine_and_context_model.joblib",
                "engine_and_context_preprocessing": (
                    "engine_and_context_preprocessing.json"
                ),
                "split_metadata": "split_metadata.json",
                "evaluation_metrics": "evaluation_metrics.json",
                "feature_list": "feature_list.json",
            },
        },
    )
    return TrainingRunSummary(
        username=username,
        artifact_dir=destination,
        total_decisions=dataset.total_decisions,
        inside_top_5_decisions=dataset.inside_top_5_decisions,
        outside_top_5_decisions=dataset.outside_top_5_decisions,
        runtime_seconds=runtime,
        metrics=metrics,
        split_metadata=split_metadata,
    )


def load_ranker(
    artifact_dir: str | Path, *, variant: str = "engine_and_context"
) -> tuple[RandomForestClassifier, SparseOneHotPreprocessor]:
    source = Path(artifact_dir)
    model_path = source / f"{variant}_model.joblib"
    preprocessing_path = source / f"{variant}_preprocessing.json"
    if not model_path.is_file() or not preprocessing_path.is_file():
        raise FileNotFoundError(f"Incomplete model artifact for variant '{variant}'")
    model = joblib.load(model_path)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError(f"Unsupported saved model type: {type(model).__name__}")
    return model, SparseOneHotPreprocessor.load(preprocessing_path)


def evaluate_saved_artifact(artifact_dir: str | Path) -> dict[str, object]:
    source = Path(artifact_dir)
    metrics_path = source / "evaluation_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluation metrics not found: {metrics_path}")
    return json.loads(metrics_path.read_text())


def predict_one_decision(
    artifact_dir: str | Path,
    candidate_rows: list[dict[str, object]],
    *,
    variant: str = "engine_and_context",
) -> list[dict[str, object]]:
    if not candidate_rows:
        raise ValueError("A decision must contain at least one candidate")
    decision_ids = {str(row["decision_id"]) for row in candidate_rows}
    if len(decision_ids) != 1:
        raise ValueError("Inference rows must belong to exactly one decision")
    model, preprocessor = load_ranker(artifact_dir, variant=variant)
    probabilities = predict_candidate_probabilities(model, preprocessor, candidate_rows)
    ranked = sorted(
        zip(candidate_rows, probabilities, strict=True),
        key=lambda item: (-item[1], int(item[0]["engine_rank"])),
    )
    return [
        {
            "candidate_move_uci": row["candidate_move_uci"],
            "engine_rank": int(row["engine_rank"]),
            "probability": probability,
        }
        for row, probability in ranked
    ]


def _split_metadata(
    dataset: CandidateDataset, metadata: dict[str, object]
) -> dict[str, object]:
    decision_counts = Counter(str(item["split"]) for item in dataset.decisions)
    usable_counts = Counter(
        str(item["split"]) for item in dataset.decisions if bool(item["usable"])
    )
    candidate_counts = Counter(str(item["split"]) for item in dataset.candidate_rows)
    for name in ("train", "validation", "test"):
        values = metadata[name]
        if not isinstance(values, dict):
            raise RuntimeError("Invalid split metadata")
        values["decision_count"] = decision_counts[name]
        values["usable_decision_count"] = usable_counts[name]
        values["outside_top_5_decision_count"] = (
            decision_counts[name] - usable_counts[name]
        )
        values["candidate_row_count"] = candidate_counts[name]
    train_ids = set(metadata["train"]["game_ids"])
    validation_ids = set(metadata["validation"]["game_ids"])
    test_ids = set(metadata["test"]["game_ids"])
    metadata["integrity"] = {
        "game_overlap_count": len(
            (train_ids & validation_ids) | (train_ids & test_ids) | (validation_ids & test_ids)
        ),
        "preprocessing_fit_split": "train",
        "chronological": True,
    }
    return metadata


def _historical_coverage(
    dataset: CandidateDataset, historical_counts: dict[str, dict[str, int]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for split_name in ("validation", "test"):
        decisions = [
            row
            for row in dataset.decisions
            if row["split"] == split_name and bool(row["usable"])
        ]
        covered = sum(
            str(row["canonical_position"]) in historical_counts for row in decisions
        )
        result[split_name] = {
            "usable_decisions": len(decisions),
            "previously_seen_positions": covered,
            "coverage": covered / len(decisions) if decisions else 0.0,
        }
    return result


def _existing_path(path: str | Path, label: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"{label} dataset not found: {result}")
    return result


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
