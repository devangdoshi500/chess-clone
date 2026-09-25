"""Reconstruct validation predictions from frozen population artifacts without fitting."""

import hashlib
import json
import math
from pathlib import Path

from catboost import CatBoostRanker
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.population_policy import load_population_sources, _cap_player_positions
from chess_clone.experiments.population_metrics import (
    legal_policy_metrics, move_frequency_probabilities, uniform_probabilities,
)
from chess_clone.modeling.candidates import chronological_game_split
from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows, PlayerTendencyEncoder
from chess_clone.modeling.boosted import predict_relevance_scores, groupwise_softmax
from chess_clone.experiments.move_quality import _write_json


def replay_validation(cohort: Path, artifact: Path, output: Path, *, cap: int = 1500) -> dict:
    if output.exists():
        raise FileExistsError(output)
    metrics = json.loads((artifact / "metrics.json").read_text())
    fields = json.loads((artifact / "feature_sets.json").read_text())
    protocol = metrics["protocol"]
    positions, dates, splits = [], {}, {}
    input_paths = [cohort, artifact / "metrics.json", artifact / "feature_sets.json"]
    for source in load_population_sources(cohort):
        input_paths.extend([source.positions, source.games])
        rows = [r for r in pq.read_table(source.positions).to_pylist()
                if str(r["player_username"]).casefold() == source.username.casefold()
                and r.get("player_rating") is not None
                and protocol["rating_min"] <= int(r["player_rating"]) <= protocol["rating_max"]
                and str(r.get("speed", "")).casefold() == "blitz"]
        ids = {r["game_id"] for r in rows}
        games = [r for r in pq.read_table(source.games).to_pylist() if r["game_id"] in ids]
        split = chronological_game_split(games)
        names = {key: split.name_for(key) for key in split.game_dates}
        # Filter out test before feature construction; no test predictions are loaded.
        kept = _cap_player_positions(rows, split.game_dates, names, cap)
        expected = next(p for p in metrics["players"] if p["username"].casefold() == source.username.casefold())
        for name in ("train", "validation"):
            count = sum(names[r["game_id"]] == name for r in kept)
            if count != expected["split_decisions"][name]:
                raise ValueError(f"Reconstructed {source.username} {name} count differs")
        positions.extend(r for r in kept if names[r["game_id"]] != "test")
        dates.update(split.game_dates)
        splits.update(names)
    # Historical replay only: saved v2/v3 models used archived full-game tags.
    candidates = build_all_legal_candidate_rows(positions, dates, splits, include_archived_openings=True)
    train = [r for r in candidates if r["split"] == "train"]
    validation = [r for r in candidates if r["split"] == "validation"]
    if {r["game_id"] for r in train} & {r["game_id"] for r in validation}:
        raise ValueError("Reconstructed train/validation games overlap")
    history = PlayerTendencyEncoder()
    history.fit_transform_ordered(train)
    personalized = history.transform(validation)
    predictions, checks = {}, {}
    for name, rows in (("population", validation), ("personalized", personalized)):
        path = artifact / f"{name}.cbm"
        input_paths.append(path)
        model = CatBoostRanker()
        model.load_model(path)
        scores = predict_relevance_scores(model, rows, tuple(fields[name]))
        probabilities = groupwise_softmax(rows, scores, temperature=metrics["models"][name]["calibration_temperature"])
        actual, predictions[name] = legal_policy_metrics(rows, probabilities)
        recorded = metrics["models"][name]["validation"]
        for key in ("decisions", "exact_move_accuracy", "top_3_accuracy", "negative_log_likelihood", "multiclass_brier_score"):
            if not math.isclose(actual[key], recorded[key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Frozen validation replay differs: {name}/{key}")
        checks[name] = {key: actual[key] for key in recorded if isinstance(actual.get(key), (int, float))}
    for name, probabilities in (("uniform_legal", uniform_probabilities(validation)),
                                ("global_move_frequency", move_frequency_probabilities(train, validation))):
        _, predictions[name] = legal_policy_metrics(validation, probabilities)
    output.mkdir(parents=True)
    for name, rows in predictions.items():
        pq.write_table(pa.Table.from_pylist(rows), output / f"validation_predictions_{name}.parquet")
    manifest = {"split": "validation", "status": "complete", "cap": cap,
                "reproduced_metrics": checks,
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
                "train_game_ids": sorted({r["game_id"] for r in train}),
                "validation_game_ids": sorted({r["game_id"] for r in validation})}
    _write_json(output / "manifest.json", manifest)
    return manifest
