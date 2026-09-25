"""Frozen-model expanded player evaluation, with acquisition sealed before scoring."""

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path

from catboost import CatBoostRanker
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.profile_ablation import digest
from chess_clone.experiments.population_policy import load_population_sources
from chess_clone.experiments.population_metrics import legal_policy_metrics
from chess_clone.experiments.evaluation_panel import ranking_panel, paired_uncertainty
from chess_clone.experiments.fresh_profile_test import frequency_probabilities
from chess_clone.experiments.move_quality import _write_json, run_move_quality
from chess_clone.experiments.winning_chance import build_report, read_quality, paired_comparison
from chess_clone.ingestion import ingest_games
from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows
from chess_clone.modeling.opportunity_profile import OpportunityProfile
from chess_clone.modeling.boosted import predict_relevance_scores, groupwise_softmax
from chess_clone.providers import LichessProvider
from chess_clone.providers.base import InvalidUsernameError
import hashlib


def seeded_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def discover_players(sources, known, config):
    counts, seen = Counter(), set()
    for source in sources:
        for game in pq.read_table(source.games).to_pylist():
            if game["game_id"] in seen:
                continue
            seen.add(game["game_id"])
            for color in ("white", "black"):
                name = game[f"{color}_username"].casefold()
                rating = game[f"{color}_rating"]
                if name not in known and rating is not None and config["rating_min"] <= rating <= config["rating_max"]:
                    counts[name] += 1
    ordered = sorted(counts, key=lambda name: (-counts[name], seeded_key(config["seed"], name)))
    return [{"username": name, "eligible_opponent_appearances": counts[name]} for name in ordered[:config["max_candidate_players"]]]


def initialize(config_path, output):
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    if config["schema_version"] != 1 or config["target_unseen_players"] > config["max_candidate_players"]:
        raise ValueError("Invalid evaluation configuration")
    artifact = Path(config["artifact_dir"])
    selection = json.loads((artifact / "selection.json").read_text())
    completed = json.loads((artifact / "manifest.json").read_text())
    if completed["status"] != "complete" or completed["selection_sha256"] != digest(artifact / "selection.json"):
        raise ValueError("Model selection seal mismatch")
    verify_hashes(selection["frozen_sha256"])
    old_protocol = json.loads((artifact / "protocol.json").read_text())
    if digest(config["thresholds"]) != old_protocol["input_sha256"][config["thresholds"]]:
        raise ValueError("Thresholds differ from model selection")
    profile = json.loads((artifact / "profile.json").read_text())
    known = set(profile["players"])
    sources = load_population_sources(config["discovery_cohort"])
    if known != {s.username.casefold() for s in sources}:
        raise ValueError("Discovery cohort must match known training targets")
    audit = json.loads((artifact / "audit.json").read_text())
    candidates = discover_players(sources, known, config)
    hashes = {str(config_path): digest(config_path), str(artifact / "selection.json"): digest(artifact / "selection.json"),
              config["thresholds"]: digest(config["thresholds"]), **selection["frozen_sha256"]}
    for source in sources:
        hashes[str(source.games)] = digest(source.games)
        hashes[str(source.positions)] = digest(source.positions)
    declaration = {"created_at": datetime.now(UTC).isoformat(), "config": config, "input_sha256": hashes,
                   "known_players": sorted(known), "unseen_candidates": candidates,
                   "original_game_ids": audit["old_game_ids"], "fresh_after": audit["fresh_after"],
                   "selected_model": selection["selected"]}
    output.mkdir(parents=True)
    _write_json(output / "declaration.json", declaration)
    _write_json(output / "acquisition.json", {"status": "pending", "declaration_sha256": digest(output / "declaration.json"), "players": {}})
    print(f"Declared {len(candidates)} candidate players; target {config['target_unseen_players']}", flush=True)


def whole_game_prefix(rows, dates, cap, *, seed=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row["game_id"]].append(row)
    order = (sorted(groups, key=lambda g: seeded_key(seed, g)) if seed is not None
             else sorted(groups, key=lambda g: (dates[g], g), reverse=True))
    selected = []
    for game in order:
        if len(selected) + len(groups[game]) > cap:
            break
        selected.extend(sorted(groups[game], key=lambda r: r["ply"]))
    return selected


def select_positions(raw, games, *, username, since, until, excluded_games, config):
    """Filter and cap complete target-player games, never individual labels."""
    grouped = defaultdict(list)
    seen = set()
    for row in raw:
        if row["player_username"].casefold() != username.casefold():
            continue
        key = (row["game_id"], row["ply"])
        if key in seen:
            raise ValueError("Duplicate target decision")
        seen.add(key)
        grouped[row["game_id"]].append(row)
    eligible, dates, reasons = [], {}, Counter()
    seen_games = set()
    for game in games:
        key = game["game_id"]
        if key in seen_games:
            raise ValueError("Duplicate source game")
        seen_games.add(key)
        rows = grouped[key]
        if key in excluded_games:
            reasons["previous_or_shared_game"] += 1
        elif game["played_at"] is None or not since < game["played_at"] <= until:
            reasons["outside_window"] += 1
        elif not game["rated"] or game["variant"].casefold() != "standard" or game["speed"] != "blitz":
            reasons["ineligible_type"] += 1
        elif not rows or any(r.get("player_rating") is None or not config["rating_min"] <= r["player_rating"] <= config["rating_max"] for r in rows):
            reasons["rating_or_no_target_decisions"] += 1
        else:
            eligible.extend(rows)
            dates[key] = game["played_at"]
    selected = whole_game_prefix(eligible, dates, config["max_decisions_per_player"])
    reasons["eligible_games_outside_cap"] = len(dates) - len({r["game_id"] for r in selected})
    return selected, dates, dict(reasons)


def acquire(output, *, provider=None):
    declaration = json.loads((output / "declaration.json").read_text())
    state = json.loads((output / "acquisition.json").read_text())
    if state["declaration_sha256"] != digest(output / "declaration.json"):
        raise ValueError("Declaration seal mismatch")
    verify_hashes(declaration["input_sha256"])
    if state["status"] == "complete":
        raise ValueError("Acquisition is already sealed")
    config = declaration["config"]
    entries = [(name, "known") for name in declaration["known_players"]]
    entries += [(r["username"], "unseen") for r in declaration["unseen_candidates"]]
    used = set(declaration["original_game_ids"])
    accepted_unseen = 0
    state["status"] = "acquiring"
    state.pop("error", None)
    try:
        for username, kind in entries:
            if kind == "unseen" and accepted_unseen >= config["target_unseen_players"]:
                break
            if username in state["players"]:
                record = state["players"][username]
                verify_hashes(record.get("sha256", {}))
            else:
                since = datetime.fromisoformat(declaration["fresh_after"] if kind == "known" else config["unseen_since"])
                print(f"Download {kind} player {username}", flush=True)
                try:
                    data = ingest_games(provider or LichessProvider(), username, max_games=config["max_games_per_player"],
                                        since=int(since.timestamp()*1000)+1000, until=config["until"], perf_type="blitz",
                                        raw_dir=output / "raw", processed_dir=output / "normalized")
                except InvalidUsernameError as exc:
                    record = {"kind": kind, "accepted": False, "error": str(exc), "game_ids": [], "decisions": 0}
                else:
                    games = pq.read_table(data.games_path).to_pylist()
                    rows = pq.read_table(data.positions_path).to_pylist()
                    selected, dates, reasons = select_positions(rows, games, username=username, since=since,
                        until=datetime.fromisoformat(config["until"]), excluded_games=used, config=config)
                    game_ids = sorted({r["game_id"] for r in selected})
                    accepted = bool(selected) if kind == "known" else (len(game_ids) >= config["minimum_games_per_unseen_player"] and len(selected) >= config["minimum_decisions_per_unseen_player"])
                    selected_path = output / "selected" / f"{username}.parquet"
                    if accepted:
                        selected_path.parent.mkdir(exist_ok=True)
                        pq.write_table(pa.Table.from_pylist(selected), selected_path)
                    paths = [data.raw_path, data.games_path, data.positions_path] + ([selected_path] if accepted else [])
                    record = {"kind": kind, "accepted": accepted, "downloaded_games": data.games, "parser_skips": data.skipped_games,
                              "decisions": len(selected), "game_ids": game_ids, "exclusions": reasons,
                              "positions": str(selected_path.resolve()) if accepted else None,
                              "games": str(data.games_path.resolve()), "sha256": {str(p): digest(p) for p in paths},
                              "selected_dates": {g: dates[g].isoformat() for g in game_ids}}
                state["players"][username] = record
                _write_json(output / "acquisition.json", state)
            if record["accepted"]:
                used.update(record["game_ids"])
                accepted_unseen += int(kind == "unseen")
            print(f"  accepted={record['accepted']} decisions={record['decisions']} unseen={accepted_unseen}", flush=True)
        cohort = [{"username": name, "positions": r["positions"], "games": r["games"]}
                  for name, r in state["players"].items() if r["accepted"]]
        if len(cohort) < 2:
            raise ValueError("Insufficient eligible players to evaluate")
        _write_json(output / "cohort.json", {"schema_version": 1, "players": cohort})
        state.update(status="complete", accepted_unseen=accepted_unseen,
                     target_reached=accepted_unseen == config["target_unseen_players"], cohort_sha256=digest(output / "cohort.json"))
        _write_json(output / "acquisition.json", state)
        _write_json(output / "acquisition_seal.json", {"sha256": digest(output / "acquisition.json")})
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        _write_json(output / "acquisition.json", state)
        raise


def load_sealed(output):
    declaration = json.loads((output / "declaration.json").read_text())
    state = json.loads((output / "acquisition.json").read_text())
    if state["status"] != "complete" or json.loads((output / "acquisition_seal.json").read_text())["sha256"] != digest(output / "acquisition.json"):
        raise ValueError("Acquisition seal mismatch")
    if state["declaration_sha256"] != digest(output / "declaration.json") or state["cohort_sha256"] != digest(output / "cohort.json"):
        raise ValueError("Declaration/cohort changed after acquisition")
    verify_hashes(declaration["input_sha256"])
    for r in state["players"].values():
        verify_hashes(r.get("sha256", {}))
    return declaration, state


def evaluate(output):
    declaration, state = load_sealed(output)
    destination = output / "evaluation"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir()
    manifest = {"status": "running", "acquisition_sha256": digest(output / "acquisition.json"),
                "selected_model": declaration["selected_model"], "first_scored_at": datetime.now(UTC).isoformat()}
    _write_json(destination / "manifest.json", manifest)
    try:
        artifact = Path(declaration["config"]["artifact_dir"])
        schemas = json.loads((artifact / "feature_sets.json").read_text())
        fitted = json.loads((artifact / "metrics.json").read_text())
        counts = json.loads((artifact / "frequency_counts.json").read_text())
        profile = OpportunityProfile.from_dict(json.loads((artifact / "profile.json").read_text()))
        models = {name: CatBoostRanker().load_model(artifact / f"{name}.cbm") for name in schemas}
        predictions = {name: [] for name in [*schemas, "global_move_frequency"]}
        quality_ids = set()
        for username, record in state["players"].items():
            if not record["accepted"]:
                continue
            if record["kind"] == "unseen" and username in profile.players:
                raise ValueError("Unseen-player overlap with training profile")
            positions = pq.read_table(record["positions"]).to_pylist()
            dates = {g: datetime.fromisoformat(d) for g, d in record["selected_dates"].items()}
            rows = build_all_legal_candidate_rows(positions, dates, {g: "test" for g in dates})
            quality_rows = whole_game_prefix(positions, dates, declaration["config"]["quality_decisions_per_player"], seed=declaration["config"]["seed"])
            quality_ids.update(f"{r['game_id']}:{r['ply']}" for r in quality_rows)
            context = {r["decision_id"]: r for r in rows}
            for name in predictions:
                if name == "global_move_frequency":
                    probabilities = frequency_probabilities(rows, counts)
                else:
                    data = profile.transform(rows) if name == "opportunity_profile" else rows
                    scores = predict_relevance_scores(models[name], data, tuple(schemas[name]))
                    probabilities = groupwise_softmax(data, scores, temperature=fitted[name]["temperature"])
                _, values = legal_policy_metrics(rows, probabilities)
                for r in values:
                    c = context[r["decision_id"]]
                    r.update(player_username=username, cohort_kind=record["kind"], played_at=dates[r["game_id"]],
                             fresh=dates[r["game_id"]] > datetime.fromisoformat(declaration["fresh_after"]),
                             game_phase=c["game_phase"], rating_band=str(int(c["player_rating"])//200*200),
                             player_color=c["player_color"], time_control=str(c["time_control"] or "unknown"))
                predictions[name].extend(values)
            print(f"Scored {username}: {len(positions)} decisions", flush=True)
        for name, rows in predictions.items():
            pq.write_table(pa.Table.from_pylist(rows), destination / f"test_predictions_{name}.parquet")
        quality_input = destination / "quality_predictions"
        quality_input.mkdir()
        for name, rows in predictions.items():
            pq.write_table(pa.Table.from_pylist([r for r in rows if r["decision_id"] in quality_ids]), quality_input / f"test_predictions_{name}.parquet")
        _write_json(destination / "quality_sampling.json", {"seed": declaration["config"]["seed"],
            "cap_per_player": declaration["config"]["quality_decisions_per_player"], "decision_ids": sorted(quality_ids),
            "note": "Separate fixed whole-game sample; quality denominators do not equal full ranking denominators."})
        report = {}
        for group, predicate in SUBSETS.items():
            subset = {name: [r for r in rows if predicate(r)] for name, rows in predictions.items()}
            result = {"methods": {name: ranking_panel(rows) for name, rows in subset.items()}, "comparisons": {}}
            for left, right in (("safe_population", "opportunity_profile"), ("global_move_frequency", declaration["selected_model"])):
                result["comparisons"][f"{right}_minus_{left}"] = {unit: paired_uncertainty(subset[left], subset[right], unit=unit)
                                                                 for unit in ("game_id", "player_username")}
            report[group] = result
        _write_json(destination / "ranking_report.json", report)
        manifest.update(status="complete", predictions_sha256={str(destination / f"test_predictions_{name}.parquet"): digest(destination / f"test_predictions_{name}.parquet") for name in predictions},
                        quality_inputs_sha256={str(p): digest(p) for p in [destination / "quality_sampling.json", *sorted(quality_input.glob("*.parquet"))]})
        _write_json(destination / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _write_json(destination / "manifest.json", manifest)
        raise


SUBSETS = {
    "unseen_players": lambda r: r["cohort_kind"] == "unseen",
    "unseen_players_fresh": lambda r: r["cohort_kind"] == "unseen" and r["fresh"],
    "known_players_fresh": lambda r: r["cohort_kind"] == "known" and r["fresh"],
}


def quality(output):
    declaration, state = load_sealed(output)
    destination = output / "evaluation"
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Ranking evaluation must complete before quality")
    verify_hashes(manifest["predictions_sha256"])
    verify_hashes(manifest["quality_inputs_sha256"])
    run_move_quality(output / "cohort.json", destination / "quality_predictions", destination / "quality")
    rows, _ = read_quality(destination / "quality", expected_split="test")
    thresholds = json.loads(Path(declaration["config"]["thresholds"]).read_text())
    for r in rows:
        info = state["players"][r["player_username"].casefold()]
        r.update(cohort_kind=info["kind"], fresh=datetime.fromisoformat(info["selected_dates"][r["game_id"]]) > datetime.fromisoformat(declaration["fresh_after"]))
    reports = {}
    for name, predicate in SUBSETS.items():
        selected = [r for r in rows if predicate(r)]
        report = build_report(selected, thresholds)
        for left, right in (("safe_population", "opportunity_profile"), ("global_move_frequency", declaration["selected_model"])):
            a, b = ([r for r in selected if r["method"] == method] for method in (left, right))
            if a and b:
                report["comparisons"][f"{right}_minus_{left}"] = paired_comparison(a, b, thresholds)
                report["comparisons"][f"{right}_minus_{left}_player_clustered"] = paired_comparison(a, b, thresholds, unit="player_username")
        reports[name] = report
    _write_json(destination / "winning_chance_report.json", reports)


def render(output):
    declaration, acquisition = load_sealed(output)
    destination = output / "evaluation"
    ranking = json.loads((destination / "ranking_report.json").read_text())
    chances = json.loads((destination / "winning_chance_report.json").read_text())
    lines = ["# Expanded frozen-v4 results", "",
             f"Accepted unseen players: {acquisition['accepted_unseen']}; target reached: {acquisition['target_reached']}.",
             "Frozen models and validation-selected 5-point tolerance; no test-based selection.",
             "Unseen players use population profile fallback, not individualized adaptation.", ""]
    for subset, panel in ranking.items():
        lines += [f"## {subset}", "",
                  "| Method | Players | Games | Decisions | Exact | Top 3 | Top 5 | NLL | Brier | ECE | Behavior MAE |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, m in panel["methods"].items():
            if not m["decisions"]:
                lines.append(f"| {name} | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(f"| {name} | {m['players']} | {m['games']} | {m['decisions']} | {m['exact_move_accuracy']:.2%} | {m['top_3_accuracy']:.2%} | {m['top_5_accuracy']:.2%} | {m['negative_log_likelihood']:.3f} | {m['multiclass_brier_score']:.3f} | {m['top_1_expected_calibration_error']:.2%} | {m['behavior_rate_mean_absolute_error']:.2%} |")
        lines += ["", "Quality is a separate seeded whole-game sample; finite denominators differ from ranking.", "",
                  "| Method | Sample decisions | Similarity pairs | Similar | Different pairs | Different similar | Soundness pairs | Sound |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, m in chances[subset]["methods"].items():
            pct = lambda value: "n/a" if value is None else f"{value:.2%}"
            lines.append(f"| {name} | {m['decisions']} | {m['all']['similarity_pairs']} | {pct(m['all']['similar_rate'])} | {m['different_move']['similarity_pairs']} | {pct(m['different_move']['similar_rate'])} | {m['all']['soundness_pairs']} | {pct(m['all']['sound_rate'])} |")
        lines.append("")
        for comparison, paired in panel["comparisons"].items():
            result = paired["player_username"].get("exact")
            if result is not None:
                lower, upper = result["95_interval"]
                lines.append(f"{comparison}: exact delta {result['delta']*100:+.2f} pp; paired-player 95% interval [{lower*100:+.2f}, {upper*100:+.2f}] pp.")
        lines.append("")
    lines += ["The cohort was selected from previous opponents and availability, not sampled randomly from Lichess.",
              "Intervals condition on this cohort and frozen fitted models. New-player outcomes do not confirm known-player personalization.", ""]
    (destination / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("init", "acquire", "evaluate", "quality", "report"))
    parser.add_argument("--config", type=Path, default=Path("configs/expanded_evaluation_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "init":
        initialize(args.config, args.output)
    else:
        {"acquire": acquire, "evaluate": evaluate, "quality": quality, "report": render}[args.phase](args.output)
