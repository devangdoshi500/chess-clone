"""One sealed chronological confirmation after validation selects a safe model."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

from catboost import CatBoostRanker
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.profile_ablation import digest
from chess_clone.experiments.population_policy import load_population_sources, _paired_bootstrap
from chess_clone.experiments.population_metrics import legal_policy_metrics
from chess_clone.experiments.move_quality import _write_json, run_move_quality
from chess_clone.experiments.winning_chance import read_quality, build_report, paired_comparison
from chess_clone.ingestion import ingest_games
from chess_clone.providers import LichessProvider
from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows
from chess_clone.modeling.opportunity_profile import OpportunityProfile
from chess_clone.modeling.boosted import predict_relevance_scores, groupwise_softmax


def cap_complete_games(rows, dates, cap=500):
    """Earliest complete games, stopping at the cap; never cherry-pick to pack it."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["game_id"]].append(row)
    kept = []
    for key in sorted(groups, key=lambda g: (dates[g], g)):
        if len(kept) + len(groups[key]) > cap:
            break
        kept.extend(groups[key])
    return kept


def eligible_fresh_positions(positions, games, *, player, after, until, old_ids, rating_min=1300, rating_max=1700):
    eligible = {g["game_id"]: g["played_at"] for g in games
                if g["played_at"] is not None and after < g["played_at"] <= until
                and g["game_id"] not in old_ids and g["rated"]
                and g["variant"].casefold() == "standard" and g["speed"] == "blitz"}
    rows = [r for r in positions if r["game_id"] in eligible
            and str(r["player_username"]).casefold() == player.casefold()
            and r.get("player_rating") is not None and rating_min <= r["player_rating"] <= rating_max]
    return cap_complete_games(rows, eligible), eligible


def frequency_probabilities(rows, counts):
    totals = defaultdict(float)
    weights = [counts.get(r["candidate_move_uci"], 0) + 1 for r in rows]
    for row, weight in zip(rows, weights, strict=True):
        totals[row["decision_id"]] += weight
    return [weight / totals[row["decision_id"]] for row, weight in zip(rows, weights, strict=True)]


def run(artifact: Path, output: Path, threshold_path: Path, *, provider=None):
    if output.exists():
        raise FileExistsError(output)
    selection_path = artifact / "selection.json"
    selection = json.loads(selection_path.read_text())
    completed = json.loads((artifact / "manifest.json").read_text())
    if completed["status"] != "complete" or completed["selection_sha256"] != digest(selection_path):
        raise ValueError("Validation selection seal mismatch")
    for path, expected in selection["frozen_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
    protocol = json.loads((artifact / "protocol.json").read_text())
    if digest(threshold_path) != protocol["input_sha256"][str(threshold_path)]:
        raise ValueError("Winning-chance thresholds changed")
    audit = json.loads((artifact / "audit.json").read_text())
    settings = protocol["fresh_protocol"]
    after = datetime.fromisoformat(audit["fresh_after"])
    until = datetime.fromisoformat(settings["until"])
    output.mkdir(parents=True)
    manifest = {"status": "acquiring", "selection_sha256": digest(selection_path),
                "selected": selection["selected"], "after": after.isoformat(), "until": until.isoformat(),
                "sampling": "latest 100 exported games per player, then earliest complete eligible games up to 500 decisions; stop at cap",
                "max_decisions_per_player": 500, "players": {}, "input_sha256": {}}
    _write_json(output / "manifest.json", manifest)
    positions, dates, cohort = [], {}, []
    try:
        for source in load_population_sources(Path(protocol["cohort"])):
            print(f"Downloading fresh blitz games: {source.username}", flush=True)
            downloaded = ingest_games(provider or LichessProvider(), source.username, max_games=settings["max_games_per_player"],
                                      since=int(after.timestamp() * 1000) + 1000, until=until,
                                      perf_type="blitz", raw_dir=output / "raw", processed_dir=output / "normalized")
            games = pq.read_table(downloaded.games_path).to_pylist()
            raw = pq.read_table(downloaded.positions_path).to_pylist()
            kept, local_dates = eligible_fresh_positions(raw, games, player=source.username, after=after, until=until,
                                                       old_ids=set(audit["old_game_ids"]))
            for path in (downloaded.raw_path, downloaded.games_path, downloaded.positions_path):
                manifest["input_sha256"][str(path)] = digest(path)
            manifest["players"][source.username] = {"downloaded_games": downloaded.games, "skipped_parser_games": downloaded.skipped_games,
                                                   "selected_games": len({r["game_id"] for r in kept}), "selected_decisions": len(kept)}
            print(f"Selected {len(kept)} decisions for {source.username}", flush=True)
            positions.extend(kept)
            dates.update(local_dates)
            cohort.append({"username": source.username, "positions": str(downloaded.positions_path.resolve()),
                           "games": str(downloaded.games_path.resolve())})
            _write_json(output / "manifest.json", manifest)
        _write_json(output / "cohort.json", {"schema_version": 1, "players": cohort})
        selected_games = {r["game_id"] for r in positions}
        selected_players = {r["player_username"].casefold() for r in positions}
        manifest["selected_game_ids"] = sorted(selected_games)
        manifest["decisions"] = len(positions)
        if (len(selected_games) < settings["minimum_games"] or len(selected_players) < settings["minimum_players"]
                or len(positions) < settings["minimum_decisions"]):
            manifest["status"] = "insufficient_fresh_data"
            _write_json(output / "manifest.json", manifest)
            print(json.dumps({"status": manifest["status"], "decisions": len(positions),
                              "players": len(selected_players), "games": len(selected_games)}, indent=2), flush=True)
            return manifest
        rows = build_all_legal_candidate_rows(positions, dates, {g: "test" for g in selected_games})
        profile = OpportunityProfile.from_dict(json.loads((artifact / "profile.json").read_text()))
        schemas = json.loads((artifact / "feature_sets.json").read_text())
        fitted_metrics = json.loads((artifact / "metrics.json").read_text())
        predictions, metrics = {}, {}
        for name, features in schemas.items():
            data = profile.transform(rows) if name == "opportunity_profile" else rows
            model = CatBoostRanker()
            model.load_model(artifact / f"{name}.cbm")
            scores = predict_relevance_scores(model, data, tuple(features))
            probabilities = groupwise_softmax(data, scores, temperature=fitted_metrics[name]["temperature"])
            metrics[name], predictions[name] = legal_policy_metrics(data, probabilities)
        probabilities = frequency_probabilities(rows, json.loads((artifact / "frequency_counts.json").read_text()))
        metrics["global_move_frequency"], predictions["global_move_frequency"] = legal_policy_metrics(rows, probabilities)
        for name, values in predictions.items():
            pq.write_table(pa.Table.from_pylist(values), output / f"test_predictions_{name}.parquet")
        _write_json(output / "metrics.json", metrics)
        _write_json(output / "ranking_comparisons.json", {
            "profile_minus_safe_population": _paired_bootstrap(predictions["safe_population"], predictions["opportunity_profile"]),
            "selected_minus_frequency": _paired_bootstrap(predictions["global_move_frequency"], predictions[selection["selected"]])})
        run_move_quality(output / "cohort.json", output, output / "quality")
        quality_rows, _ = read_quality(output / "quality", expected_split="test")
        thresholds = json.loads(threshold_path.read_text())
        quality = build_report(quality_rows, thresholds)
        quality["comparisons"]["profile_minus_safe_population"] = paired_comparison(
            [r for r in quality_rows if r["method"] == "safe_population"],
            [r for r in quality_rows if r["method"] == "opportunity_profile"], thresholds)
        _write_json(output / "winning_chance.json", quality)
        manifest["status"] = "complete"
        _write_json(output / "manifest.json", manifest)
        print(json.dumps({"selected": selection["selected"], "metrics": metrics}, indent=2), flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=Path("artifacts/benchmarks/tactical-v3-winning-chance/thresholds.json"))
    args = parser.parse_args()
    run(args.artifact, args.output, args.thresholds)
