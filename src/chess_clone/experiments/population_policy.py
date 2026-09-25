"""Train a rating-centered, all-legal human move policy and personalization layer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.modeling.boosted import (
    fit_temperature,
    groupwise_softmax,
    predict_relevance_scores,
    train_grouped_ranker,
)
from chess_clone.modeling.candidates import chronological_game_split
from chess_clone.modeling.legal_policy import (
    LEGAL_POLICY_FEATURE_FIELDS,
    PLAYER_TENDENCY_FIELDS,
    PlayerTendencyEncoder,
    build_all_legal_candidate_rows,
)
from chess_clone.experiments.population_metrics import (
    legal_policy_metrics,
    move_frequency_probabilities,
    uniform_probabilities,
)


@dataclass(frozen=True, slots=True)
class PopulationPolicySummary:
    artifact_dir: Path
    players: int
    decisions: int
    runtime_seconds: float
    metrics: dict[str, object]


@dataclass(frozen=True, slots=True)
class PopulationSource:
    username: str
    positions: Path
    games: Path


def train_population_policy(
    cohort_path: str | Path,
    artifact_dir: str | Path,
    *,
    rating_min: int = 1300,
    rating_max: int = 1700,
    max_decisions_per_player: int = 1500,
) -> PopulationPolicySummary:
    """Train population and history-personalized policies over every legal move."""

    if rating_min >= rating_max:
        raise ValueError("rating_min must be less than rating_max")
    if max_decisions_per_player < 30:
        raise ValueError("max_decisions_per_player must be at least 30")
    destination = Path(artifact_dir)
    if destination.exists():
        raise FileExistsError(f"Artifact directory already exists: {destination}")
    started = perf_counter()
    sources = load_population_sources(cohort_path)

    positions: list[dict[str, object]] = []
    game_dates: dict[str, datetime] = {}
    split_names: dict[str, str] = {}
    player_metadata: list[dict[str, object]] = []
    for source in sources:
        player_positions = pq.read_table(source.positions).to_pylist()
        player_games = pq.read_table(source.games).to_pylist()
        requested = source.username.casefold()
        player_positions = [
            row
            for row in player_positions
            if str(row["player_username"]).casefold() == requested
            and row.get("player_rating") is not None
            and rating_min <= int(row["player_rating"]) <= rating_max
            and str(row.get("speed", "")).casefold() == "blitz"
        ]
        relevant_ids = {str(row["game_id"]) for row in player_positions}
        player_games = [row for row in player_games if str(row["game_id"]) in relevant_ids]
        if len(player_games) < 10:
            raise ValueError(f"{source.username} has fewer than 10 eligible games")
        split = chronological_game_split(player_games)
        local_split_names = {
            game_id: split.name_for(game_id) for game_id in split.game_dates
        }
        player_positions = _cap_player_positions(
            player_positions,
            split.game_dates,
            local_split_names,
            max_decisions_per_player,
        )
        kept_ids = {str(row["game_id"]) for row in player_positions}
        positions.extend(player_positions)
        game_dates.update(
            {game_id: date for game_id, date in split.game_dates.items() if game_id in kept_ids}
        )
        split_names.update(
            {game_id: name for game_id, name in local_split_names.items() if game_id in kept_ids}
        )
        ratings = [int(row["player_rating"]) for row in player_positions]
        player_metadata.append(
            {
                "username": source.username,
                "decisions": len(player_positions),
                "games": len(kept_ids),
                "rating_min": min(ratings),
                "rating_median": median(ratings),
                "rating_max": max(ratings),
                "split_decisions": {
                    name: sum(
                        split_names[str(row["game_id"])] == name
                        for row in player_positions
                    )
                    for name in ("train", "validation", "test")
                },
            }
        )

    candidate_rows = build_all_legal_candidate_rows(positions, game_dates, split_names)
    raw = {
        name: [row for row in candidate_rows if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    for name, rows in raw.items():
        if not rows:
            raise ValueError(f"No {name} candidate rows remain after filtering")

    history = PlayerTendencyEncoder()
    personalized = {
        "train": history.fit_transform_ordered(raw["train"]),
        "validation": history.transform(raw["validation"]),
        "test": history.transform(raw["test"]),
    }
    feature_sets = {
        "population": LEGAL_POLICY_FEATURE_FIELDS,
        "personalized": LEGAL_POLICY_FEATURE_FIELDS + PLAYER_TENDENCY_FIELDS,
    }
    destination.mkdir(parents=True)
    metrics: dict[str, object] = {
        "protocol": {
            "task": "predict the human move from every legal move",
            "rating_min": rating_min,
            "rating_max": rating_max,
            "split": "per-player chronological whole-game 70/15/15",
            "selection": "validation only; test reported once after both fixed variants",
        },
        "players": player_metadata,
        "baselines": {},
        "models": {},
    }
    predictions: dict[str, list[dict[str, object]]] = {}
    for baseline, values in (
        ("uniform_legal", uniform_probabilities(raw["test"])),
        (
            "global_move_frequency",
            move_frequency_probabilities(raw["train"], raw["test"]),
        ),
    ):
        result, prediction_rows = legal_policy_metrics(raw["test"], values)
        metrics["baselines"][baseline] = result
        predictions[baseline] = prediction_rows

    for name, fields in feature_sets.items():
        source_rows = personalized if name == "personalized" else raw
        model = train_grouped_ranker(
            source_rows["train"], source_rows["validation"], fields
        )
        model.save_model(destination / f"{name}.cbm")
        metrics["models"][name] = {}
        validation_scores = predict_relevance_scores(
            model, source_rows["validation"], fields
        )
        temperature = fit_temperature(source_rows["validation"], validation_scores)
        metrics["models"][name]["calibration_temperature"] = temperature
        for split_name in ("validation", "test"):
            scores = (
                validation_scores
                if split_name == "validation"
                else predict_relevance_scores(model, source_rows[split_name], fields)
            )
            probabilities = groupwise_softmax(
                source_rows[split_name], scores, temperature=temperature
            )
            result, prediction_rows = legal_policy_metrics(
                source_rows[split_name], probabilities
            )
            result["by_player"] = _metrics_by_player(
                source_rows[split_name], probabilities
            )
            metrics["models"][name][split_name] = result
            if split_name == "test":
                predictions[name] = prediction_rows

    metrics["comparisons"] = {
        "population_minus_global_move_frequency": _paired_bootstrap(
            predictions["global_move_frequency"], predictions["population"]
        ),
        "personalized_minus_population": _paired_bootstrap(
            predictions["population"], predictions["personalized"]
        )
    }
    all_ratings = [int(row["player_rating"]) for row in positions]
    metrics["cohort"] = {
        "players": len(sources),
        "games": len({str(row["game_id"]) for row in positions}),
        "decisions": len(positions),
        "candidate_rows": len(candidate_rows),
        "observed_rating_min": min(all_ratings),
        "observed_rating_median": median(all_ratings),
        "observed_rating_mean": mean(all_ratings),
        "observed_rating_max": max(all_ratings),
    }
    _write_json(destination / "metrics.json", metrics)
    _write_json(
        destination / "feature_sets.json",
        {name: list(fields) for name, fields in feature_sets.items()},
    )
    _write_predictions(destination, predictions)
    (destination / "REPORT.md").write_text(_render_report(metrics))
    runtime = perf_counter() - started
    _write_json(
        destination / "manifest.json",
        {
            "format_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "runtime_seconds": runtime,
            "cohort_config": str(Path(cohort_path).resolve()),
            "models": {
                "population": "CatBoost QuerySoftMax over all legal moves",
                "personalized": "same policy plus train-only per-player tendency priors",
            },
        },
    )
    return PopulationPolicySummary(
        artifact_dir=destination,
        players=len(sources),
        decisions=len(positions),
        runtime_seconds=runtime,
        metrics=metrics,
    )


def load_population_sources(path: str | Path) -> list[PopulationSource]:
    path = Path(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "players"}:
        raise ValueError("population cohort must contain schema_version and players")
    if payload["schema_version"] != 1 or not isinstance(payload["players"], list):
        raise ValueError("unsupported population cohort schema")
    result = []
    for item in payload["players"]:
        if not isinstance(item, dict) or set(item) != {"username", "positions", "games"}:
            raise ValueError("each population player needs username, positions, and games")
        positions = (path.parent / str(item["positions"])).resolve()
        games = (path.parent / str(item["games"])).resolve()
        if not positions.is_file() or not games.is_file():
            raise FileNotFoundError(f"Missing population source for {item['username']}")
        result.append(PopulationSource(str(item["username"]), positions, games))
    if len(result) < 2:
        raise ValueError("population training requires at least two players")
    if len({item.username.casefold() for item in result}) != len(result):
        raise ValueError("population usernames must be unique")
    return result


def _cap_player_positions(
    rows: list[dict[str, object]],
    dates: dict[str, datetime],
    splits: dict[str, str],
    maximum: int,
) -> list[dict[str, object]]:
    fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
    selected: list[dict[str, object]] = []
    for name, fraction in fractions.items():
        limit = max(1, math.floor(maximum * fraction))
        by_game: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            game_id = str(row["game_id"])
            if splits[game_id] == name:
                by_game[game_id].append(row)
        game_ids = sorted(by_game, key=lambda game_id: (dates[game_id], game_id))
        chosen: list[str] = []
        decision_count = 0
        for game_id in reversed(game_ids):
            game_count = len(by_game[game_id])
            if chosen and decision_count + game_count > limit:
                continue
            chosen.append(game_id)
            decision_count += game_count
            if decision_count >= limit:
                break
        for game_id in reversed(chosen):
            selected.extend(sorted(by_game[game_id], key=lambda row: int(row["ply"])))
    return selected


def _metrics_by_player(
    rows: list[dict[str, object]], probabilities: list[float]
) -> dict[str, object]:
    indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexes[str(row["player_username"])].append(index)
    result = {}
    for player, selected in sorted(indexes.items()):
        player_rows = [rows[index] for index in selected]
        player_probabilities = [probabilities[index] for index in selected]
        result[player] = legal_policy_metrics(player_rows, player_probabilities)[0]
    return result


def _paired_bootstrap(
    baseline: list[dict[str, object]], challenger: list[dict[str, object]], *, resamples: int = 2000
) -> dict[str, object]:
    left = {str(row["decision_id"]): row for row in baseline}
    right = {str(row["decision_id"]): row for row in challenger}
    if set(left) != set(right):
        raise ValueError("Paired methods must score identical decisions")
    games: dict[str, list[str]] = defaultdict(list)
    for key, row in left.items():
        games[str(row["game_id"])].append(key)
    game_ids = sorted(games)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(game_ids), size=(resamples, len(game_ids)))
    output: dict[str, object] = {"games": len(game_ids), "resamples": resamples, "seed": 42}
    for field in ("exact_correct", "top_3_correct", "behavior_similarity"):
        differences = np.array(
            [
                sum(float(right[key][field]) - float(left[key][field]) for key in games[game])
                for game in game_ids
            ]
        )
        counts = np.array([len(games[game]) for game in game_ids])
        samples = differences[draws].sum(axis=1) / counts[draws].sum(axis=1)
        output[field] = {
            "delta": float(differences.sum() / counts.sum()),
            "paired_game_bootstrap_95_interval": [
                float(value) for value in np.quantile(samples, [0.025, 0.975])
            ],
        }
    return output


def _write_predictions(
    destination: Path, predictions: dict[str, list[dict[str, object]]]
) -> None:
    for name, rows in predictions.items():
        pq.write_table(pa.Table.from_pylist(rows), destination / f"test_predictions_{name}.parquet")


def _render_report(metrics: dict[str, object]) -> str:
    cohort = metrics["cohort"]
    lines = [
        "# Rating-centered all-legal human policy",
        "",
        "This experiment predicts the observed human move from every legal move; Stockfish does not gate the candidate set.",
        "",
        "## Cohort",
        "",
        f"- {cohort['players']} players, {cohort['games']} games, {cohort['decisions']} decisions",
        f"- observed rating mean {cohort['observed_rating_mean']:.1f}, median {cohort['observed_rating_median']:.1f}, range {cohort['observed_rating_min']}–{cohort['observed_rating_max']}",
        f"- {cohort['candidate_rows']} legal candidate rows",
        "",
        "## Held-out test results",
        "",
        "| Method | Exact | Top 3 | Top 5 | MRR | Normalized rank | NLL | Brier | Calibration error | Behavior-rate MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    methods = {
        "Uniform legal": metrics["baselines"]["uniform_legal"],
        "Global move frequency": metrics["baselines"]["global_move_frequency"],
        "Population CatBoost": metrics["models"]["population"]["test"],
        "Personalized CatBoost": metrics["models"]["personalized"]["test"],
    }
    for name, values in methods.items():
        lines.append(
            f"| {name} | {values['exact_move_accuracy']:.2%} | {values['top_3_accuracy']:.2%} | "
            f"{values['top_5_accuracy']:.2%} | {values['mean_reciprocal_rank']:.3f} | "
            f"{values['mean_normalized_rank_score']:.2%} | {values['negative_log_likelihood']:.3f} | "
            f"{values['multiclass_brier_score']:.3f} | {values['top_1_expected_calibration_error']:.2%} | "
            f"{values['behavior_rate_mean_absolute_error']:.2%} |"
        )
    comparison = metrics["comparisons"]["personalized_minus_population"]
    baseline_comparison = metrics["comparisons"]["population_minus_global_move_frequency"]
    baseline_exact = baseline_comparison["exact_correct"]
    baseline_top3 = baseline_comparison["top_3_correct"]
    exact = comparison["exact_correct"]
    behavior = comparison["behavior_similarity"]
    lines += [
        "",
        "## Personalization effect",
        "",
        f"- Population versus global-frequency exact delta: {baseline_exact['delta']:+.2%} (paired-game 95% interval {baseline_exact['paired_game_bootstrap_95_interval'][0]:+.2%} to {baseline_exact['paired_game_bootstrap_95_interval'][1]:+.2%}).",
        f"- Population versus global-frequency top-3 delta: {baseline_top3['delta']:+.2%} (paired-game 95% interval {baseline_top3['paired_game_bootstrap_95_interval'][0]:+.2%} to {baseline_top3['paired_game_bootstrap_95_interval'][1]:+.2%}).",
        f"- Exact-move delta: {exact['delta']:+.2%} (paired-game 95% interval {exact['paired_game_bootstrap_95_interval'][0]:+.2%} to {exact['paired_game_bootstrap_95_interval'][1]:+.2%}).",
        f"- Behavioral-similarity delta: {behavior['delta']:+.2%} (paired-game 95% interval {behavior['paired_game_bootstrap_95_interval'][0]:+.2%} to {behavior['paired_game_bootstrap_95_interval'][1]:+.2%}).",
        "",
        "Behavior similarity averages agreement on nine interpretable move attributes. It complements rather than replaces exact accuracy and likelihood.",
        "",
    ]
    return "\n".join(lines)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
