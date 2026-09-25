"""Ranking/probability/behavior panels from sufficient per-decision statistics."""

from collections import defaultdict
import math
from statistics import mean, median

import numpy as np

from chess_clone.experiments.population_metrics import _top_1_calibration_error
from chess_clone.modeling.legal_policy import BEHAVIOR_BOOLEAN_FIELDS


def summarize_predictions(rows):
    if not rows:
        return {"decisions": 0, "games": 0, "players": 0, "status": "no_data"}
    rates = {}
    for field in BEHAVIOR_BOOLEAN_FIELDS:
        p = mean(bool(r[f"predicted_{field}"]) for r in rows)
        a = mean(bool(r[f"actual_{field}"]) for r in rows)
        rates[field.removeprefix("candidate_")] = {"predicted": p, "actual": a, "absolute_error": abs(p-a)}
    return {
        "status": "complete", "decisions": len(rows), "games": len({r["game_id"] for r in rows}),
        "players": len({r["player_username"].casefold() for r in rows}),
        "candidate_rows": sum(r["candidate_count"] for r in rows),
        "mean_legal_moves": mean(r["candidate_count"] for r in rows),
        "exact_move_accuracy": mean(r["exact_correct"] for r in rows),
        "top_3_accuracy": mean(r["top_3_correct"] for r in rows),
        "top_5_accuracy": mean(r["top_5_correct"] for r in rows),
        "mean_reciprocal_rank": mean(1/r["actual_rank"] for r in rows),
        "mean_actual_rank": mean(r["actual_rank"] for r in rows),
        "median_actual_rank": median(r["actual_rank"] for r in rows),
        "mean_normalized_rank_score": mean(r["normalized_rank_score"] for r in rows),
        "negative_log_likelihood": -mean(math.log(max(r["actual_move_probability"], 1e-15)) for r in rows),
        "multiclass_brier_score": mean(r["brier_score"] for r in rows),
        "top_1_expected_calibration_error": _top_1_calibration_error(rows),
        "piece_match_rate": mean(r["piece_match"] for r in rows),
        "destination_wing_match_rate": mean(r["wing_match"] for r in rows),
        "mean_behavior_similarity": mean(r["behavior_similarity"] for r in rows),
        "behavior_rate_mean_absolute_error": mean(r["absolute_error"] for r in rates.values()),
        "behavior_rates": rates,
    }


def ranking_panel(rows):
    result = summarize_predictions(rows)
    result["breakdowns"] = {}
    for field in ("player_username", "game_phase", "rating_band", "player_color", "time_control"):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r[field])].append(r)
        result["breakdowns"][field] = {k: summarize_predictions(v) for k, v in sorted(groups.items())}
    players = list(result["breakdowns"]["player_username"].values())
    excluded = {"decisions", "games", "players", "candidate_rows"}
    result["player_macro"] = {key: {"players": len(players), "mean": mean(p[key] for p in players)}
                              for key, value in result.items() if isinstance(value, (int, float)) and key not in excluded} if players else {}
    return result


def paired_uncertainty(left, right, *, unit="game_id", resamples=2000, seed=42):
    a, b = ({r["decision_id"]: r for r in rows} for rows in (left, right))
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError("Paired predictions must have unique identical decisions")
    groups = defaultdict(list)
    for key in sorted(a):
        if any(a[key][f] != b[key][f] for f in ("game_id", "player_username", "actual_move_uci")):
            raise ValueError("Paired identity mismatch")
        groups[a[key][unit]].append(key)
    output = {"unit": unit, "units": len(groups), "decisions": len(a), "seed": seed, "resamples": resamples}
    if not groups:
        return output
    counts = np.array([len(keys) for keys in groups.values()])
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(resamples, len(groups)))
    fields = {
        "exact": lambda r: float(r["exact_correct"]), "top3": lambda r: float(r["top_3_correct"]),
        "top5": lambda r: float(r["top_5_correct"]), "mrr": lambda r: 1/r["actual_rank"],
        "nll": lambda r: -math.log(max(r["actual_move_probability"], 1e-15)),
        "brier": lambda r: r["brier_score"], "behavior_agreement": lambda r: r["behavior_similarity"],
    }
    for field, value in fields.items():
        totals = np.array([sum(value(b[k])-value(a[k]) for k in keys) for keys in groups.values()])
        samples = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
        item = {"delta": float(totals.sum()/counts.sum()), "95_interval": np.quantile(samples, [.025, .975]).tolist()}
        if unit == "player_username":
            macro_samples = (totals/counts)[draws].mean(axis=1)
            item["equal_player_macro"] = {"delta": float((totals/counts).mean()), "95_interval": np.quantile(macro_samples, [.025, .975]).tolist()}
        output[field] = item
    return output
