"""Declare and acquire a bounded, deep-history style cohort before modeling."""

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.expanded_evaluation import discover_players, load_sealed, verify_hashes
from chess_clone.experiments.move_quality import _write_json
from chess_clone.experiments.population_policy import load_population_sources
from chess_clone.experiments.profile_ablation import digest
from chess_clone.ingestion import ingest_games
from chess_clone.providers import LichessProvider
from chess_clone.providers.base import InvalidUsernameError


def initialize(config_path, output):
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    if config["schema_version"] != 1 or not 0 < config["development_players"] < config["target_players"] <= config["max_candidate_players"]:
        raise ValueError("Invalid cohort sizes")
    prior_decl, prior = load_sealed(Path(config["prior_evaluation"]))
    excluded = set(prior_decl["known_players"]) | set(prior["players"])
    old_games = set(prior_decl["original_game_ids"])
    for record in prior["players"].values():
        if record.get("games"):
            old_games.update(g["game_id"] for g in pq.read_table(record["games"]).to_pylist())
    sources = load_population_sources(config["discovery_cohort"])
    candidates = discover_players(sources, excluded, config)
    artifact = Path(config["artifact_dir"])
    selection = json.loads((artifact / "selection.json").read_text())
    verify_hashes(selection["frozen_sha256"])
    hashes = {str(config_path): digest(config_path), **selection["frozen_sha256"],
              str(artifact / "selection.json"): digest(artifact / "selection.json"),
              str(Path(config["prior_evaluation"]) / "acquisition.json"): digest(Path(config["prior_evaluation"]) / "acquisition.json")}
    for source in sources:
        hashes[str(source.games)] = digest(source.games)
    declaration = {"created_at": datetime.now(UTC).isoformat(), "config": config,
                   "candidates": candidates, "excluded_players": sorted(excluded),
                   "excluded_game_ids": sorted(old_games), "input_sha256": hashes}
    output.mkdir(parents=True)
    _write_json(output / "declaration.json", declaration)
    _write_json(output / "acquisition.json", {"status": "pending", "players": {},
                "declaration_sha256": digest(output / "declaration.json")})
    print(f"Declared {len(candidates)} candidates for {config['target_players']} targets", flush=True)


def select_games(positions, games, player, config, excluded):
    groups = defaultdict(list)
    seen = set()
    for row in positions:
        if row["player_username"].casefold() == player.casefold():
            key = (row["game_id"], row["ply"])
            if key in seen:
                raise ValueError("Duplicate decision")
            seen.add(key)
            groups[row["game_id"]].append(row)
    since, until = (datetime.fromisoformat(config[k]) for k in ("since", "until"))
    eligible = {}
    seen_games = set()
    for game in games:
        key = game["game_id"]
        if key in seen_games:
            raise ValueError("Duplicate game")
        seen_games.add(key)
        rows = groups[key]
        if (key not in excluded and game["played_at"] is not None and since < game["played_at"] <= until
                and game["rated"] and game["variant"].casefold() == "standard" and game["speed"] == "blitz"
                and rows and all(r.get("player_rating") is not None and
                    config["rating_min"] <= r["player_rating"] <= config["rating_max"] for r in rows)):
            eligible[key] = game["played_at"]
    counts = [config[k] for k in ("history_games", "validation_games", "confirmation_games")]
    if len(eligible) < sum(counts):
        return None, {"eligible_games": len(eligible), "reason": "insufficient_games"}
    ordered = sorted(eligible, key=lambda g: (eligible[g], g))[-sum(counts):]
    for boundary in (counts[0], counts[0] + counts[1]):
        if eligible[ordered[boundary-1]] >= eligible[ordered[boundary]]:
            return None, {"eligible_games": len(eligible), "reason": "tied_split_boundary"}
    parts, start = {}, 0
    for split, count in zip(("history", "validation", "confirmation"), counts, strict=True):
        selected = ordered[start:start+count]
        parts[split] = [r for g in selected for r in sorted(groups[g], key=lambda r: r["ply"])]
        start += count
    return parts, {"eligible_games": len(eligible), "selected_dates": {g: eligible[g].isoformat() for g in ordered},
                   "game_ids": ordered}


def acquire(output, *, provider=None):
    declaration = json.loads((output / "declaration.json").read_text())
    state = json.loads((output / "acquisition.json").read_text())
    if state["declaration_sha256"] != digest(output / "declaration.json"):
        raise ValueError("Declaration changed")
    verify_hashes(declaration["input_sha256"])
    if state["status"] == "complete":
        raise ValueError("Already sealed")
    config = declaration["config"]
    used = set(declaration["excluded_game_ids"])
    accepted = 0
    state.update(status="acquiring")
    state.pop("error", None)
    try:
        # Publish resumed state before a potentially slow request, so a history-
        # preparation follower does not mistake the old failure for a new stop.
        _write_json(output / "acquisition.json", state)
        for candidate in declaration["candidates"]:
            if accepted == config["target_players"]:
                break
            player = candidate["username"]
            if player in state["players"]:
                record = state["players"][player]
                verify_hashes(record.get("sha256", {}))
            else:
                print(f"Download {player}: accepted {accepted}/{config['target_players']}", flush=True)
                try:
                    data = ingest_games(provider or LichessProvider(), player,
                        max_games=config["max_games_per_player"], since=config["since"], until=config["until"],
                        perf_type="blitz", raw_dir=output / "raw", processed_dir=output / "normalized")
                except InvalidUsernameError as exc:
                    record = {"accepted": False, "reason": str(exc)}
                else:
                    games = pq.read_table(data.games_path).to_pylist()
                    positions = pq.read_table(data.positions_path).to_pylist()
                    parts, record = select_games(positions, games, player, config, used)
                    record.update(accepted=parts is not None, downloaded_games=data.games,
                                  parser_skips=data.skipped_games, paths={})
                    paths = [data.raw_path, data.games_path, data.positions_path]
                    if parts is not None:
                        record["role"] = "development" if accepted < config["development_players"] else "confirmation"
                        for split, rows in parts.items():
                            path = output / split / f"{player}.parquet"
                            path.parent.mkdir(exist_ok=True)
                            pq.write_table(pa.Table.from_pylist(rows), path)
                            record["paths"][split] = str(path.resolve())
                            paths.append(path)
                        record["split_decisions"] = {s: len(r) for s, r in parts.items()}
                    record["sha256"] = {str(p): digest(p) for p in paths}
                state["players"][player] = record
                _write_json(output / "acquisition.json", state)
            if record["accepted"]:
                used.update(record["game_ids"])
                accepted += 1
            print(f"  eligible={record.get('eligible_games', 0)} accepted={record['accepted']}", flush=True)
        state.update(status="complete" if accepted == config["target_players"] else "insufficient_cohort",
                     accepted_players=accepted)
        _write_json(output / "acquisition.json", state)
        if state["status"] == "complete":
            _write_json(output / "acquisition_seal.json", {"sha256": digest(output / "acquisition.json")})
        print(f"Cohort {state['status']}: {accepted} players", flush=True)
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        _write_json(output / "acquisition.json", state)
        raise


def load_cohort(output):
    declaration = json.loads((output / "declaration.json").read_text())
    state = json.loads((output / "acquisition.json").read_text())
    if state["status"] != "complete" or digest(output / "acquisition.json") != json.loads((output / "acquisition_seal.json").read_text())["sha256"]:
        raise ValueError("Cohort not sealed and complete")
    if digest(output / "declaration.json") != state["declaration_sha256"]:
        raise ValueError("Declaration changed")
    verify_hashes(declaration["input_sha256"])
    for record in state["players"].values():
        verify_hashes(record.get("sha256", {}))
    return declaration, {p: r for p, r in state["players"].items() if r["accepted"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("init", "acquire"))
    parser.add_argument("--config", type=Path, default=Path("configs/deep_style_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initialize(args.config, args.output) if args.phase == "init" else acquire(args.output)
