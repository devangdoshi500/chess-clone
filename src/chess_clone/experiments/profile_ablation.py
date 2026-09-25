"""Validation-controlled removal of opening leakage and opportunity-profile study."""

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.population_policy import load_population_sources, _cap_player_positions, _paired_bootstrap
from chess_clone.experiments.population_metrics import legal_policy_metrics, move_frequency_probabilities
from chess_clone.experiments.move_quality import _write_json, run_move_quality
from chess_clone.experiments.winning_chance import build_report, read_quality
from chess_clone.modeling.candidates import chronological_game_split
from chess_clone.modeling.legal_policy import LEGAL_POLICY_FEATURE_FIELDS, build_all_legal_candidate_rows
from chess_clone.modeling.opportunity_profile import OPPORTUNITY_FIELDS, OpportunityProfile
from chess_clone.modeling.boosted import train_grouped_ranker, predict_relevance_scores, groupwise_softmax, fit_temperature


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(cohort: Path, original: Path):
    metrics = json.loads((original / "metrics.json").read_text())
    positions, dates, splits, audit, hashes = [], {}, {}, {}, {}
    old_games, all_dates = set(), []
    for source in load_population_sources(cohort):
        for path in (source.positions, source.games):
            hashes[str(path)] = digest(path)
        raw = pq.read_table(source.positions).to_pylist()
        games = pq.read_table(source.games).to_pylist()
        old_games.update(g["game_id"] for g in games)
        all_dates.extend(g["played_at"] for g in games if g["played_at"] is not None)
        first = [r for r in raw if r["ply"] == 1]
        audit[source.username] = {
            "white_first_moves": len(first),
            "first_moves_with_final_opening": sum(r.get("opening_name") is not None for r in first),
            "distinct_opening_tags_on_initial_board": len({r.get("opening_name") for r in first}),
            "examples": Counter(str(r.get("opening_name")) for r in first).most_common(5),
        }
        rows = [r for r in raw if str(r["player_username"]).casefold() == source.username.casefold()
                and r.get("player_rating") is not None
                and metrics["protocol"]["rating_min"] <= r["player_rating"] <= metrics["protocol"]["rating_max"]
                and str(r.get("speed", "")).casefold() == "blitz"]
        relevant = {r["game_id"] for r in rows}
        split = chronological_game_split([g for g in games if g["game_id"] in relevant])
        names = {key: split.name_for(key) for key in split.game_dates}
        kept = _cap_player_positions(rows, split.game_dates, names, 1500)
        expected = next(p for p in metrics["players"] if p["username"].casefold() == source.username.casefold())
        for name in ("train", "validation"):
            if sum(names[r["game_id"]] == name for r in kept) != expected["split_decisions"][name]:
                raise ValueError(f"Original split count differs: {source.username}/{name}")
        positions.extend(r for r in kept if names[r["game_id"]] != "test")
        for key, date in split.game_dates.items():
            if key in splits and splits[key] != names[key]:
                raise ValueError("Shared game assigned conflicting splits across players")
            dates[key], splits[key] = date, names[key]
    candidates = build_all_legal_candidate_rows(positions, dates, splits)
    raw = {name: [r for r in candidates if r["split"] == name] for name in ("train", "validation")}
    if {r["game_id"] for r in raw["train"]} & {r["game_id"] for r in raw["validation"]}:
        raise ValueError("Train/validation overlap")
    return raw, {"players": audit, "source_sha256": hashes, "old_game_ids": sorted(old_games),
                 "fresh_after": max(all_dates).isoformat(),
                 "train_game_ids": sorted({r["game_id"] for r in raw["train"]}),
                 "validation_game_ids": sorted({r["game_id"] for r in raw["validation"]})}


def select_profile(baseline, profile, ranking_pair, quality_pair):
    """Fixed gates: no post-result relaxation or selection on old test scores."""
    gates = {
        "exact_noninferiority_1pp": profile["exact_move_accuracy"] >= baseline["exact_move_accuracy"] - .01,
        "top3_noninferiority_1pp": profile["top_3_accuracy"] >= baseline["top_3_accuracy"] - .01,
        "top5_noninferiority_1pp": profile["top_5_accuracy"] >= baseline["top_5_accuracy"] - .01,
        "nll_no_more_than_0.02_worse": profile["negative_log_likelihood"] <= baseline["negative_log_likelihood"] + .02,
        "brier_no_more_than_0.01_worse": profile["multiclass_brier_score"] <= baseline["multiclass_brier_score"] + .01,
        "ece_no_more_than_1pp_worse": profile["top_1_expected_calibration_error"] <= baseline["top_1_expected_calibration_error"] + .01,
        "behavior_mae_no_more_than_1pp_worse": profile["behavior_rate_mean_absolute_error"] <= baseline["behavior_rate_mean_absolute_error"] + .01,
    }
    for field in ("human_gap_pp", "predicted_loss_pp"):
        paired = quality_pair[f"both_different_common_finite/{field}"]["threshold_rate_delta"]
        gates[f"{field}_rate_no_more_than_1pp_worse"] = paired is not None and paired["estimate"] >= -.01
    benefits = {"exact": ranking_pair["exact_correct"]["paired_game_bootstrap_95_interval"][0] > 0}
    for field in ("human_gap_pp", "predicted_loss_pp"):
        paired = quality_pair[f"both_different_common_finite/{field}"]["threshold_rate_delta"]
        benefits[field] = paired is not None and paired["paired_game_95_interval"][0] > 0
    selected = "opportunity_profile" if all(gates.values()) and any(benefits.values()) else "safe_population"
    return {"selected": selected, "gates": gates, "positive_paired_evidence": benefits,
            "selected_at": datetime.now(UTC).isoformat(),
            "scope": "validation-selected candidate; fresh test confirmation required"}


def run(cohort: Path, original: Path, threshold_path: Path, output: Path, *, threads: int = 4):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol = {
        "status": "running", "created_at": datetime.now(UTC).isoformat(),
        "cohort": str(cohort), "original": str(original),
        "feature_policy": "exclude archived opening ECO and family from both arms",
        "arms": ["safe_population", "opportunity_profile"],
        "profile": "capture/check discretionary opportunity rates; earlier training games; smoothing=20; population fallback",
        "selection": "all noninferiority point gates AND positive paired-game lower bound for exact or common-miss similarity/soundness; else population",
        "tolerances": {"exact_top3_top5_ece_behavior_quality_rates": .01, "nll": .02, "brier": .01},
        "threads": threads, "hyperparameters": "existing QuerySoftMax 350 depth6 lr0.05 seed20260830 early_stop50",
        "input_sha256": {str(cohort): digest(cohort), str(original / "metrics.json"): digest(original / "metrics.json"),
                         str(threshold_path): digest(threshold_path)},
        "fresh_protocol": {"max_games_per_player": 100, "rating_min": 1300, "rating_max": 1700,
                           "speed": "blitz", "minimum_players": 3, "minimum_games": 30,
                           "minimum_decisions": 500, "until": datetime.now(UTC).isoformat()},
    }
    _write_json(output / "protocol.json", protocol)
    started = perf_counter()
    try:
        raw, audit = prepare(cohort, original)
        _write_json(output / "audit.json", audit)
        profile = OpportunityProfile()
        encoded = {"train": profile.fit_transform_ordered(raw["train"]), "validation": profile.transform(raw["validation"])}
        _write_json(output / "profile.json", profile.to_dict())
        predictions, metrics, schema = {}, {}, {}
        for name, data, fields in (
            ("safe_population", raw, LEGAL_POLICY_FEATURE_FIELDS),
            ("opportunity_profile", encoded, LEGAL_POLICY_FEATURE_FIELDS + OPPORTUNITY_FIELDS),
        ):
            print(f"Fitting {name}: {len(data['train'])} training candidate rows", flush=True)
            model = train_grouped_ranker(data["train"], data["validation"], fields, thread_count=threads, verbose=25)
            model.save_model(output / f"{name}.cbm")
            scores = predict_relevance_scores(model, data["validation"], fields)
            temperature = fit_temperature(data["validation"], scores)
            probabilities = groupwise_softmax(data["validation"], scores, temperature=temperature)
            values, predictions[name] = legal_policy_metrics(data["validation"], probabilities)
            metrics[name] = {"validation": values, "temperature": temperature, "trees": model.tree_count_}
            schema[name] = list(fields)
            pq.write_table(pa.Table.from_pylist(predictions[name]), output / f"validation_predictions_{name}.parquet")
            _write_json(output / "metrics.json", metrics)
            _write_json(output / "feature_sets.json", schema)
        # Preserve a human-frequency comparator for fresh evaluation and diagnostics.
        frequency = move_frequency_probabilities(raw["train"], raw["validation"])
        _, baseline_predictions = legal_policy_metrics(raw["validation"], frequency)
        pq.write_table(pa.Table.from_pylist(baseline_predictions), output / "validation_predictions_global_move_frequency.parquet")
        counts = Counter(r["candidate_move_uci"] for r in raw["train"] if r["chosen"])
        _write_json(output / "frequency_counts.json", dict(counts))
        _write_json(output / "ranking_comparison.json", _paired_bootstrap(predictions["safe_population"], predictions["opportunity_profile"]))
        run_move_quality(cohort, output, output / "validation_quality", split="validation")
        quality_rows, _ = read_quality(output / "validation_quality", expected_split="validation")
        thresholds = json.loads(threshold_path.read_text())
        if thresholds["split"] != "validation":
            raise ValueError("Expected frozen validation thresholds")
        quality = build_report(quality_rows, thresholds)
        from chess_clone.experiments.winning_chance import paired_comparison
        paired = paired_comparison([r for r in quality_rows if r["method"] == "safe_population"],
                                   [r for r in quality_rows if r["method"] == "opportunity_profile"], thresholds)
        quality["comparisons"]["profile_minus_safe_population"] = paired
        _write_json(output / "winning_chance.json", quality)
        ranking_pair = json.loads((output / "ranking_comparison.json").read_text())
        selection = select_profile(metrics["safe_population"]["validation"], metrics["opportunity_profile"]["validation"], ranking_pair, paired)
        selection["frozen_sha256"] = {str(output / name): digest(output / name) for name in (
            "protocol.json", "audit.json", "safe_population.cbm", "opportunity_profile.cbm", "profile.json",
            "feature_sets.json", "metrics.json", "winning_chance.json", "ranking_comparison.json", "frequency_counts.json")}
        _write_json(output / "selection.json", selection)
        _write_json(output / "manifest.json", {"status": "complete", "runtime_seconds": perf_counter() - started,
                                              "selection_sha256": digest(output / "selection.json")})
        print(json.dumps(selection, indent=2), flush=True)
        return selection
    except Exception as exc:
        _write_json(output / "manifest.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("configs/population_1500_mvp.json"))
    parser.add_argument("--original", type=Path, default=Path("artifacts/models/population_1500_tactical_v3"))
    parser.add_argument("--thresholds", type=Path, default=Path("artifacts/benchmarks/tactical-v3-winning-chance/thresholds.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.cohort, args.original, args.thresholds, args.output, threads=args.threads)
