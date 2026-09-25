"""History-only residual fitting, validation selection, then sealed confirmation."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
import tempfile
import time

from catboost import CatBoostRanker
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.deep_style_cohort import load_cohort
from chess_clone.experiments.expanded_evaluation import verify_hashes
from chess_clone.experiments.move_quality import _write_json
from chess_clone.experiments.population_metrics import legal_policy_metrics
from chess_clone.experiments.profile_ablation import digest
from chess_clone.experiments.style_adaptation import paired_player_interval
from chess_clone.experiments.style_metrics import STYLE_FIELDS, style_metrics
from chess_clone.modeling.boosted import predict_relevance_scores
from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows
from chess_clone.modeling.style_residual import (
    BOOLEAN_FIELDS, NUMERIC_FIELDS, VERSION, fit_residual, predict_residual, shared_residual,
)
from chess_clone.modeling import legal_policy, style_residual

KEEP_FIELDS = tuple(dict.fromkeys((
    "decision_id", "game_id", "player_username", "game_phase", "move_number", "split",
    "candidate_move_uci", "chosen", *STYLE_FIELDS, *BOOLEAN_FIELDS, *NUMERIC_FIELDS,
)))


def cache_rows(root, player, record, split, artifact):
    """Bounded feature generation; atomic completed cache with input hashes."""
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{player}_{split}.parquet"
    metadata_path = path.with_suffix(".json")
    inputs = {record["paths"][split]: digest(record["paths"][split]),
              str(artifact / "safe_population.cbm"): digest(artifact / "safe_population.cbm"),
              str(artifact / "metrics.json"): digest(artifact / "metrics.json"),
              str(artifact / "feature_sets.json"): digest(artifact / "feature_sets.json"),
              str(Path(legal_policy.__file__)): digest(legal_policy.__file__)}
    if path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["inputs"] != inputs or metadata["sha256"] != digest(path):
            raise ValueError("Candidate cache changed")
    else:
        positions = pq.read_table(record["paths"][split]).to_pylist()
        dates = {g: datetime.fromisoformat(d) for g, d in record["selected_dates"].items()}
        positions.sort(key=lambda r: (dates[r["game_id"]], r["game_id"], r["ply"]))
        model = CatBoostRanker().load_model(artifact / "safe_population.cbm")
        fields = tuple(json.loads((artifact / "feature_sets.json").read_text())["safe_population"])
        temperature = json.loads((artifact / "metrics.json").read_text())["safe_population"]["temperature"]
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
        writer = None
        try:
            for start in range(0, len(positions), 200):
                batch = positions[start:start+200]
                keys = {f"{p['game_id']}:{p['ply']}": " ".join(p["fen"].split()[:4]) for p in batch}
                rows = build_all_legal_candidate_rows(batch, dates, {g: split for g in dates})
                scores = predict_relevance_scores(model, rows, fields)
                compact = [{**{f: row[f] for f in KEEP_FIELDS},
                            "position_key": keys[row["decision_id"]], "base_logit": score / temperature}
                           for row, score in zip(rows, scores, strict=True)]
                table = pa.Table.from_pylist(compact)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
            if writer is None:
                raise ValueError("Empty candidate split")
        finally:
            if writer is not None:
                writer.close()
        metadata = {"inputs": inputs, "sha256": digest(temporary), "version": VERSION}
        _write_json(metadata_path, metadata)
        temporary.rename(path)
    return pq.read_table(path).to_pylist()


def fit_player(root, player, record, penalties, artifact):
    path = root / "models" / f"{player}.json"
    cache_meta = root / "cache" / f"{player}_history.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["history_sha256"] != digest(record["paths"]["history"])
                or set(saved["models"]) != {str(p) for p in penalties}
                or saved["model_code_sha256"] != digest(style_residual.__file__)
                or saved["cache_metadata_sha256"] != digest(cache_meta)):
            raise ValueError("History model inputs changed")
        return saved["models"]
    print(f"Build history features and fit: {player}", flush=True)
    rows = cache_rows(root, player, record, "history", artifact)
    models = {str(p): fit_residual(rows, penalty=p) for p in penalties}
    stability = history_stability(rows)
    path.parent.mkdir(exist_ok=True)
    _write_json(path, {"models": models, "history_sha256": digest(record["paths"]["history"]),
                      "model_code_sha256": digest(style_residual.__file__), "history_stability": stability,
                      "cache_metadata_sha256": digest(cache_meta)})
    return models


def history_stability(rows):
    """Descriptive human-vs-human phase-conditioned drift; not a noise ceiling."""
    games = list(dict.fromkeys(r["game_id"] for r in rows))
    early = set(games[:len(games) // 2])
    halves = [[r for r in rows if (r["game_id"] in early) == flag] for flag in (True, False)]
    reports = [style_metrics(part, [float(r["chosen"]) for r in part]) for part in halves]
    indexed = [{(c["phase"], c["attribute"]): c for c in report["cells"] if c["supported"]}
               for report in reports]
    cells = []
    for key in sorted(set(indexed[0]) & set(indexed[1])):
        left, right = (index[key] for index in indexed)
        a, b = ({str(r["category"]): r["human"] for r in cell["rates"]} for cell in (left, right))
        cells.append({"phase": key[0], "attribute": key[1],
                      "early_opportunities": left["opportunities"], "late_opportunities": right["opportunities"],
                      "human_tv": sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)) / 2})
    return {"early_games": len(early), "late_games": len(games) - len(early), "cells": cells,
            "mean_human_tv": mean(c["human_tv"] for c in cells) if cells else None,
            "scope": "Descriptive early-vs-late history drift; includes sampling noise and position-mix changes"}


def donor_for(player, cohort, available, split):
    """Use same-size donor histories ending strictly before target evaluation."""
    target = cohort[player]
    n_history = 300  # v1 declared fixed support, checked by caller/cohort configuration
    ordered = sorted(target["game_ids"], key=lambda g: (target["selected_dates"][g], g))
    first = ordered[n_history if split == "validation" else n_history + 100]
    cutoff = target["selected_dates"][first]
    names = sorted(available)
    after = [p for p in names if p > player] + [p for p in names if p <= player]
    for donor in after:
        if donor == player:
            continue
        record = cohort[donor]
        games = sorted(record["game_ids"], key=lambda g: (record["selected_dates"][g], g))[:n_history]
        if len(games) == n_history and max(record["selected_dates"][g] for g in games) < cutoff:
            return donor
    return None


def evaluate(rows, model):
    probabilities = predict_residual(rows, model)
    ranking, predictions = legal_policy_metrics(rows, probabilities)
    return {"ranking": ranking, "style": style_metrics(rows, probabilities)}, predictions


def style_delta(records, left, right):
    pairs = [r for r in records.values() if left in r and right in r
             and r[left]["style"]["macro_policy_tv"] is not None and r[right]["style"]["macro_policy_tv"] is not None]
    return paired_player_interval([r[left]["style"]["macro_policy_tv"] - r[right]["style"]["macro_policy_tv"] for r in pairs])


def macro(records, arm, metric):
    values = [r[arm]["style"]["macro_policy_tv"] if metric == "style_tv" else r[arm]["ranking"][metric]
              for r in records.values() if arm in r]
    return mean(v for v in values if v is not None) if any(v is not None for v in values) else None


def select_penalty(results):
    candidates = []
    for penalty, records in results.items():
        gates = (macro(records, "personal", "negative_log_likelihood") <= macro(records, "population", "negative_log_likelihood")
                 and macro(records, "personal", "exact_move_accuracy") >= macro(records, "population", "exact_move_accuracy") - .01
                 and macro(records, "personal", "top_3_accuracy") >= macro(records, "population", "top_3_accuracy") - .01)
        tv = macro(records, "personal", "style_tv")
        if gates and tv is not None:
            candidates.append((tv, macro(records, "personal", "negative_log_likelihood"), -float(penalty), penalty))
    return min(candidates)[-1] if candidates else None


def prepare(source, *, follow=False):
    """Fit only accepted development histories while serial acquisition runs.

    No validation or confirmation labels are opened. Membership is determined
    exclusively by the already declared eligibility rule, never model results.
    """
    declaration = json.loads((source / "declaration.json").read_text())
    verify_hashes(declaration["input_sha256"])
    root = source / "residual"
    root.mkdir(exist_ok=True)
    config = declaration["config"]
    done = set()
    while True:
        try:
            state = json.loads((source / "acquisition.json").read_text())
        except json.JSONDecodeError:
            # Acquisition checkpoints may be in the middle of a short write.
            if not follow:
                raise
            time.sleep(1)
            continue
        if state["declaration_sha256"] != digest(source / "declaration.json"):
            raise ValueError("Declaration changed")
        for player, record in state["players"].items():
            if player not in done and record["accepted"] and record["role"] == "development":
                verify_hashes(record["sha256"])
                fit_player(root, player, record, config["regularization_grid"], Path(config["artifact_dir"]))
                done.add(player)
                print(f"Prepared {len(done)} development histories; no held-out scoring", flush=True)
        if not follow or state["status"] in {"complete", "failed", "insufficient_cohort"}:
            print(f"Preparation stopped with acquisition status: {state['status']}", flush=True)
            return
        time.sleep(10)


def validate(source):
    declaration, cohort = load_cohort(source)
    config = declaration["config"]
    if (config["history_games"], config["validation_games"], config["confirmation_games"]) != (300, 100, 100):
        raise ValueError("Runner v1 requires declared 300/100/100 games")
    root = source / "residual"
    root.mkdir(exist_ok=True)
    if (root / "selection.json").exists():
        raise FileExistsError("Selection already sealed")
    artifact = Path(config["artifact_dir"])
    development = {p: r for p, r in cohort.items() if r["role"] == "development"}
    models = {p: fit_player(root, p, r, config["regularization_grid"], artifact) for p, r in development.items()}
    shared = {str(p): shared_residual([m[str(p)] for m in models.values()]) for p in config["regularization_grid"]}
    results = {str(p): {} for p in config["regularization_grid"]}
    for player, record in development.items():
        print(f"Validation: {player}", flush=True)
        rows = cache_rows(root, player, record, "validation", artifact)
        population, _ = evaluate(rows, None)
        donor = donor_for(player, cohort, models, "validation")
        for penalty in results:
            arms = {"population": population, "donor": donor}
            for name, model in (("personal", models[player][penalty]), ("shared", shared[penalty])):
                arms[name], _ = evaluate(rows, model)
            if donor:
                arms["wrong"], _ = evaluate(rows, models[donor][penalty])
            results[penalty][player] = arms
        _write_json(root / "validation.json", results)
    selected = select_penalty(results)
    _write_json(root / "shared.json", shared)
    selection = {"selected_penalty": selected, "status": "selected" if selected else "no_eligible_candidate",
                 "declaration_sha256": digest(source / "declaration.json"),
                 "acquisition_sha256": digest(source / "acquisition.json"),
                 "frozen_sha256": {str(p): digest(p) for p in [Path(__file__), Path(style_residual.__file__), root / "validation.json", root / "shared.json", *sorted((root / "models").glob("*.json"))]},
                 "scope": "development validation selection only; confirmation not opened"}
    _write_json(root / "selection.json", selection)
    _write_json(root / "selection_seal.json", {"sha256": digest(root / "selection.json")})
    print(json.dumps({k: v for k, v in selection.items() if k != "frozen_sha256"}, indent=2), flush=True)


def confirm(source):
    declaration, cohort = load_cohort(source)
    config = declaration["config"]
    root = source / "residual"
    selection = json.loads((root / "selection.json").read_text())
    if digest(root / "selection.json") != json.loads((root / "selection_seal.json").read_text())["sha256"]:
        raise ValueError("Selection seal changed")
    if selection["acquisition_sha256"] != digest(source / "acquisition.json") or selection["declaration_sha256"] != digest(source / "declaration.json"):
        raise ValueError("Selected cohort changed")
    verify_hashes(selection["frozen_sha256"])
    if selection["status"] != "selected":
        raise ValueError("No validation-selected candidate; confirmation stays unopened")
    destination = root / "confirmation"
    if destination.exists():
        raise FileExistsError("Confirmation already opened; do not tune and rerun")
    destination.mkdir()
    manifest = {"status": "running", "selection_sha256": digest(root / "selection.json")}
    _write_json(destination / "manifest.json", manifest)
    try:
        penalty = selection["selected_penalty"]
        artifact = Path(config["artifact_dir"])
        shared = json.loads((root / "shared.json").read_text())[penalty]
        models = {p: json.loads((root / "models" / f"{p}.json").read_text())["models"][penalty]
                  for p, r in cohort.items() if r["role"] == "development"}
        targets = {p: r for p, r in cohort.items() if r["role"] == "confirmation"}
        for player, record in targets.items():
            models[player] = fit_player(root, player, record, [float(penalty)], artifact)[penalty]
        records, predictions = {}, {}
        for player, record in targets.items():
            print(f"Confirmation: {player}", flush=True)
            rows = cache_rows(root, player, record, "confirmation", artifact)
            donor = donor_for(player, cohort, models, "confirmation")
            arms = {"population": None, "personal": models[player], "shared": shared}
            if donor:
                arms["wrong"] = models[donor]
            records[player] = {"donor": donor}
            for name, model in arms.items():
                records[player][name], local = evaluate(rows, model)
                predictions.setdefault(name, []).extend(local)
            _write_json(destination / "players.json", records)
        summary = {"players": len(records), "selected_penalty": penalty,
                   "personal_minus_shared_style": style_delta(records, "personal", "shared"),
                   "personal_minus_wrong_style": style_delta(records, "personal", "wrong"),
                   "macro": {arm: {metric: macro(records, arm, metric) for metric in
                                    ("style_tv", "negative_log_likelihood", "exact_move_accuracy", "top_3_accuracy")}
                             for arm in ("population", "shared", "personal", "wrong")}}
        intervals = [summary[f"personal_minus_{arm}_style"] for arm in ("shared", "wrong")]
        m = summary["macro"]
        summary["replay_gates_pass"] = (all(i and i["players"] >= 8 and i["paired_player_95_interval"][1] < 0 for i in intervals)
            and m["personal"]["negative_log_likelihood"] <= m["shared"]["negative_log_likelihood"]
            and all(m["personal"][metric] >= m["shared"][metric] - .01 for metric in ("exact_move_accuracy", "top_3_accuracy")))
        summary["scope"] = "Retrospective unseen-player replay confirmation; engine quality and generated-game behavior still required"
        for arm, values in predictions.items():
            pq.write_table(pa.Table.from_pylist(values), destination / f"test_predictions_{arm}.parquet")
        _write_json(destination / "summary.json", summary)
        manifest.update(status="complete")
        _write_json(destination / "manifest.json", manifest)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _write_json(destination / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "validate", "confirm"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--follow", action="store_true", help="prepare accepted histories as acquisition progresses")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.source, follow=args.follow)
    elif args.phase == "validate":
        validate(args.source)
    else:
        confirm(args.source)
