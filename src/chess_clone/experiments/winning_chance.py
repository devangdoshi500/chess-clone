"""Versioned winning-chance diagnostics and validation-only tolerance selection."""

from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.move_quality import _write_json, run_move_quality
from chess_clone.experiments.validation_replay import replay_validation

FORMULA = {
    "version": "lichess-win-percent-0.00368208-v1",
    "source": "https://lichess.org/page/accuracy",
    "verified_on": "2026-09-19",
    "formula": "100 / (1 + exp(-0.00368208 * cp))",
    "units": "percentage points (0 to 100)",
    "mate_policy": "exclude from continuous arithmetic; report categorical outcome agreement",
}
GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
TARGET = 0.75
RULE = "Smallest fixed-grid tolerance covering >=75% of finite validation human reference losses; fail if unresolved."


def win_percent(cp: float | None) -> float | None:
    if cp is None:
        return None
    if not math.isfinite(cp):
        raise ValueError("Centipawn score must be finite")
    # Stable for extreme finite scores, without mate sentinels or CP clipping.
    z = math.exp(-abs(0.00368208 * cp))
    return 100 / (1 + z) if cp >= 0 else 100 * z / (1 + z)


def convert_row(row: dict) -> dict:
    values = {name: win_percent(row[f"{name}_cp"]) for name in ("actual", "predicted", "reference")}
    a, p, r = (values[name] for name in ("actual", "predicted", "reference"))
    return {**row, **{f"{key}_win_percent": value for key, value in values.items()},
            "human_gap_pp": abs(p - a) if p is not None and a is not None else None,
            "signed_gap_pp": p - a if p is not None and a is not None else None,
            "predicted_loss_pp": max(0., r - p) if r is not None and p is not None else None,
            "actual_loss_pp": max(0., r - a) if r is not None and a is not None else None}


def select_thresholds(rows: list[dict], *, split: str) -> dict:
    if split != "validation":
        raise ValueError("Thresholds may only be selected on validation")
    # One human observation per decision, independent of method and its mistakes.
    unique = {}
    for row in rows:
        key = row["decision_id"]
        signature = (row["game_id"], row["actual_cp"], row["reference_cp"],
                     row["actual_outcome"], row["reference_outcome"])
        if key in unique and unique[key][0] != signature:
            raise ValueError("Inconsistent human/reference scores between methods")
        unique[key] = (signature, row["actual_loss_pp"])
    losses = [loss for _, loss in unique.values() if loss is not None]
    if not losses:
        raise ValueError("No finite validation human/reference pairs")
    curve = [{"tolerance_pp": t, "human_coverage": mean(v <= t for v in losses)} for t in GRID]
    selected = next((r["tolerance_pp"] for r in curve if r["human_coverage"] >= TARGET), None)
    if selected is None:
        raise ValueError("Validation threshold unresolved on frozen grid")
    return {"schema_version": 1, "split": split, "formula": FORMULA, "rule": RULE,
            "target_human_coverage": TARGET, "grid": curve,
            "human_similarity_pp": selected, "reference_soundness_pp": selected,
            "interpretation": "Cohort-relative empirical tolerance; not an official Lichess reasonable-move label.",
            "decisions": len(unique), "finite_human_reference_pairs": len(losses),
            "excluded_human_reference_pairs": len(unique) - len(losses),
            "validation_game_ids": sorted({r["game_id"] for r in rows}),
            "frozen_at": datetime.now(UTC).isoformat()}


def summary(rows: list[dict], thresholds: dict) -> dict:
    result = {"decisions": len(rows), "games": len({r["game_id"] for r in rows}),
              "exact_accuracy": mean(r["exact_correct"] for r in rows) if rows else None}
    for field in ("human_gap_pp", "signed_gap_pp", "predicted_loss_pp", "actual_loss_pp"):
        values = [r[field] for r in rows if r[field] is not None]
        result[field] = {"count": len(values), "excluded": len(rows) - len(values),
                         "mean": mean(values) if values else None, "median": median(values) if values else None}
    for name, subset in (("all", rows), ("different_move", [r for r in rows if not r["exact_correct"]])):
        finite = [r for r in subset if r["human_gap_pp"] is not None]
        sound = [r for r in subset if r["predicted_loss_pp"] is not None]
        joint = [r for r in finite if r["predicted_loss_pp"] is not None]
        result[name] = {
            "decisions": len(subset), "similarity_pairs": len(finite), "soundness_pairs": len(sound),
            "joint_pairs": len(joint),
            "similar_rate": mean(r["human_gap_pp"] <= thresholds["human_similarity_pp"] for r in finite) if finite else None,
            "sound_rate": mean(r["predicted_loss_pp"] <= thresholds["reference_soundness_pp"] for r in sound) if sound else None,
            "similar_and_sound_rate": mean(r["human_gap_pp"] <= thresholds["human_similarity_pp"] and
                                           r["predicted_loss_pp"] <= thresholds["reference_soundness_pp"] for r in joint) if joint else None,
            "sensitivity": {str(t): {"similar": mean(r["human_gap_pp"] <= t for r in finite) if finite else None,
                                     "sound": mean(r["predicted_loss_pp"] <= t for r in sound) if sound else None} for t in GRID}}
    mates = [r for r in rows if "mate" in r["actual_outcome"] or "mate" in r["predicted_outcome"]]
    result["mate_pairs"] = len(mates)
    result["mate_outcome_agreement"] = mean(r["actual_outcome"] == r["predicted_outcome"] for r in mates) if mates else None
    result["missing_pairs"] = sum("missing" in (r["actual_outcome"], r["predicted_outcome"]) for r in rows)
    result["predicted_exceeds_reference_count"] = sum(r["predicted_cp"] is not None and r["reference_cp"] is not None
                                                     and r["predicted_cp"] > r["reference_cp"] for r in rows)
    return result


def paired_comparison(left: list[dict], right: list[dict], thresholds: dict, *, resamples: int = 2000,
                      unit: str = "game_id") -> dict:
    a, b = ({r["decision_id"]: r for r in rows} for rows in (left, right))
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError("Paired methods require unique identical decisions")
    if unit not in ("game_id", "player_username"):
        raise ValueError("Unsupported resampling unit")
    output = {"resamples": resamples, "seed": 42, "direction": "right minus left", "resampling_unit": unit}
    for subset in ("all_common_finite", "both_different_common_finite"):
        for field, threshold in (("human_gap_pp", thresholds["human_similarity_pp"]),
                                 ("predicted_loss_pp", thresholds["reference_soundness_pp"])):
            groups = defaultdict(list)
            eligible_games = set()
            for key in sorted(a):
                x, y = a[key], b[key]
                if x["game_id"] != y["game_id"] or x["actual_move_uci"] != y["actual_move_uci"]:
                    raise ValueError("Paired decision identity mismatch")
                if x[field] is None or y[field] is None:
                    continue
                if subset.startswith("both_different") and (x["exact_correct"] or y["exact_correct"]):
                    continue
                if x[unit] != y[unit]:
                    raise ValueError("Paired resampling identity mismatch")
                eligible_games.add(x["game_id"])
                groups[x[unit]].append((y[field] - x[field], int(y[field] <= threshold) - int(x[field] <= threshold)))
            counts = np.array([len(v) for v in groups.values()])
            result = {"decisions": int(counts.sum()), "games": len(eligible_games), "resampling_units": len(groups)}
            if groups:
                totals = np.array([np.sum(v, axis=0) for v in groups.values()])
                draws = np.random.default_rng(42).integers(0, len(groups), size=(resamples, len(groups)))
                samples = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
                for i, name in enumerate(("mean_pp_delta", "threshold_rate_delta")):
                    result[name] = {"estimate": float(totals[:, i].sum() / counts.sum()),
                                    f"paired_{'game' if unit == 'game_id' else 'player'}_95_interval": np.quantile(samples[:, i], [.025, .975]).tolist()}
            else:
                result.update(mean_pp_delta=None, threshold_rate_delta=None)
            output[f"{subset}/{field}"] = result
    return output


def build_report(rows: list[dict], thresholds: dict) -> dict:
    methods = defaultdict(list)
    for row in rows:
        methods[row["method"]].append(row)
    result = {"formula": FORMULA, "thresholds": thresholds, "methods": {}, "comparisons": {}}
    for name, values in sorted(methods.items()):
        report = summary(values, thresholds)
        report["breakdowns"] = {}
        for field in ("player_username", "game_phase", "rating_band", "player_color", "time_control"):
            groups = defaultdict(list)
            for row in values:
                groups[row[field]].append(row)
            report["breakdowns"][field] = {k: summary(v, thresholds) for k, v in sorted(groups.items())}
        players = list(report["breakdowns"]["player_username"].values())
        report["player_macro"] = {}
        for subset in ("all", "different_move"):
            report["player_macro"][subset] = {}
            for metric in ("similar_rate", "sound_rate", "similar_and_sound_rate"):
                eligible = [p[subset][metric] for p in players if p[subset][metric] is not None]
                report["player_macro"][subset][metric] = {"players": len(eligible), "mean": mean(eligible) if eligible else None}
        result["methods"][name] = report
    for left, right in (("global_move_frequency", "population"), ("population", "personalized")):
        if left in methods and right in methods:
            result["comparisons"][f"{right}_minus_{left}"] = paired_comparison(methods[left], methods[right], thresholds)
    return result


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_quality(path: Path, *, expected_split: str) -> tuple[list[dict], dict]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("split", "test") != expected_split:
        raise ValueError("Quality manifest status/split mismatch")
    for source, digest in manifest["input_sha256"].items():
        if _hash(Path(source)) != digest:
            raise ValueError(f"Quality source hash changed: {source}")
    rows = [convert_row(r) for r in pq.read_table(path / "decisions.parquet").to_pylist()]
    seen = set()
    for row in rows:
        key = (row["method"], row["decision_id"])
        if key in seen:
            raise ValueError("Duplicate quality decision")
        seen.add(key)
    if not rows:
        raise ValueError("Empty quality data")
    methods = defaultdict(set)
    for row in rows:
        methods[row["method"]].add(row["decision_id"])
    if any(ids != next(iter(methods.values())) for ids in methods.values()):
        raise ValueError("Quality methods must score identical decisions")
    return rows, manifest


def run_winning_chance(cohort: Path, artifact_dir: Path, test_quality_dir: Path, output_dir: Path,
                       *, stockfish_path: str = "stockfish", cache_dir: Path = Path("data/cache/candidate-coverage")) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    manifest = {"status": "running", "formula": FORMULA, "selection_rule": RULE,
                "grid_pp": GRID, "target_human_coverage": TARGET,
                "started_at": datetime.now(UTC).isoformat()}
    _write_json(output_dir / "manifest.json", manifest)
    try:
        replay_validation(cohort, artifact_dir, output_dir / "validation_predictions")
        run_move_quality(cohort, output_dir / "validation_predictions", output_dir / "validation_quality",
                         split="validation", stockfish_path=stockfish_path, cache_dir=cache_dir)
        validation, quality_manifest = read_quality(output_dir / "validation_quality", expected_split="validation")
        thresholds = select_thresholds(validation, split="validation")
        thresholds["validation_decisions_sha256"] = _hash(output_dir / "validation_quality" / "decisions.parquet")
        threshold_path = output_dir / "thresholds.json"
        _write_json(threshold_path, thresholds)
        manifest["thresholds_sha256"] = _hash(threshold_path)
        _write_json(output_dir / "manifest.json", manifest)
        # The frozen selection is written before loading the previously inspected test diagnostics.
        test, test_manifest = read_quality(test_quality_dir, expected_split="test")
        if quality_manifest["engine_identity"] != test_manifest["engine_identity"] or quality_manifest["engine_settings"] != test_manifest["engine_settings"]:
            raise ValueError("Validation and test engine protocols differ")
        if set(thresholds["validation_game_ids"]) & {r["game_id"] for r in test}:
            raise ValueError("Validation/test games overlap")
        manifest["test_loaded_at"] = datetime.now(UTC).isoformat()
        manifest["test_decisions_sha256"] = _hash(test_quality_dir / "decisions.parquet")
        report = {"validation": build_report(validation, thresholds), "inspected_test": build_report(test, thresholds)}
        for name, rows in (("validation", validation), ("inspected_test", test)):
            pq.write_table(pa.Table.from_pylist(rows), output_dir / f"{name}_decisions.parquet")
        _write_json(output_dir / "report.json", report)
        (output_dir / "REPORT.md").write_text(render_report(report, thresholds))
        manifest["status"] = "complete"
        _write_json(output_dir / "manifest.json", manifest)
        return report
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _write_json(output_dir / "manifest.json", manifest)
        raise


def render_report(report: dict, thresholds: dict) -> str:
    lines = ["# Winning-chance benchmark", "", f"Formula: {FORMULA['version']}.", "",
             f"Validation-selected tolerance: {thresholds['human_similarity_pp']:g} percentage points for both human similarity and reference soundness.",
             RULE, "", "This is an empirical cohort-relative tolerance, not an official Lichess label.",
             "Mate pairs are categorical and excluded from continuous calculations. Uniform legal uses deterministic tie-breaking.", ""]
    def pct(v):
        return "n/a" if v is None else f"{v:.2%}"
    for split, values in report.items():
        lines += [f"## {split}", "", "| Method | Exact | Similar | Sound | Different pairs | Different similar | Different sound |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, m in values["methods"].items():
            lines.append(f"| {name} | {pct(m['exact_accuracy'])} | {pct(m['all']['similar_rate'])} | {pct(m['all']['sound_rate'])} | {m['different_move']['similarity_pairs']} | {pct(m['different_move']['similar_rate'])} | {pct(m['different_move']['sound_rate'])} |")
        lines.append("")
    lines += ["Full JSON includes separate denominators, continuous differences, grid sensitivity, context/player summaries and paired whole-game intervals on common eligible decisions.",
              "The inspected test is descriptive. No model promotion is justified until confirmation on fresh games.", ""]
    return "\n".join(lines)
