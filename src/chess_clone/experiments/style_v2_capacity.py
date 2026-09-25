"""Read-only capacity audit for the predeclared Style Model v2 cohort."""

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import pyarrow.parquet as pq


_GAME_FILE = re.compile(r"^games_(.+)_\d{8}T\d+Z\.parquet$")
_COLUMNS = (
    "game_id",
    "played_at",
    "white_username",
    "black_username",
    "white_rating",
    "black_rating",
    "rated",
    "variant",
    "speed",
    "time_control",
    "total_plies",
)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _player_from_path(path: Path) -> str:
    match = _GAME_FILE.match(path.name)
    if not match:
        raise ValueError(f"Cannot infer target player from game filename: {path}")
    return match.group(1).casefold()


def _parse_control(value: object) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)\+(\d+)", str(value or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _matches_scenario(time_control: object, scenario: dict[str, object]) -> bool:
    if scenario["kind"] == "exact":
        return str(time_control) in {str(v) for v in scenario["time_controls"]}
    if scenario["kind"] == "initial_seconds_range":
        parsed = _parse_control(time_control)
        return bool(
            parsed
            and int(scenario["minimum_initial_seconds"]) <= parsed[0]
            <= int(scenario["maximum_initial_seconds"])
        )
    raise ValueError(f"Unknown capacity scenario kind: {scenario['kind']}")


def validate_config(config: dict[str, object]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Style v2 config requires schema_version 1")
    if not isinstance(config.get("seed"), int):
        raise ValueError("Style v2 config requires an integer seed")
    scope = config.get("scope")
    if not isinstance(scope, dict) or scope.get("no_live_clock_inputs") is not True:
        raise ValueError("Style v2 must explicitly exclude live clock inputs")
    prototype = config.get("prototype")
    if not isinstance(prototype, dict):
        raise ValueError("Missing prototype declaration")
    counts = [
        int(prototype[key])
        for key in ("history_games", "validation_games", "evaluation_games")
    ]
    if sum(counts) != int(prototype["games_per_player"]):
        raise ValueError("Prototype split does not equal games_per_player")
    if int(prototype["development_players"]) + int(
        prototype["evaluation_players"]
    ) != int(prototype["target_players"]):
        raise ValueError("Prototype player roles do not equal target_players")
    scenarios = config.get("capacity_scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("At least one capacity scenario is required")
    names = [str(item.get("name")) for item in scenarios if isinstance(item, dict)]
    if len(names) != len(scenarios) or len(set(names)) != len(names):
        raise ValueError("Capacity scenario names must be unique")
    for scenario in scenarios:
        _matches_scenario("180+0", scenario)
    thresholds = config.get("capacity_thresholds")
    if not isinstance(thresholds, list) or any(int(value) <= 0 for value in thresholds):
        raise ValueError("Capacity thresholds must be positive")


def discover_game_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.rglob("games_*.parquet") if _GAME_FILE.match(path.name))


def _normalized_record(row: dict[str, object], player: str) -> dict[str, object] | None:
    white = str(row["white_username"]).casefold()
    black = str(row["black_username"]).casefold()
    if player not in {white, black}:
        return None
    rating = row["white_rating"] if player == white else row["black_rating"]
    moves = (
        (int(row["total_plies"]) + 1) // 2
        if player == white
        else int(row["total_plies"]) // 2
    )
    played_at = row["played_at"]
    return {
        "game_id": str(row["game_id"]),
        "played_at": played_at.isoformat() if played_at is not None else None,
        "rating": int(rating) if rating is not None else None,
        "rated": bool(row["rated"]),
        "variant": str(row["variant"]).casefold(),
        "speed": str(row["speed"]).casefold(),
        "time_control": str(row["time_control"] or "unknown"),
        "estimated_decisions": moves,
    }


def audit_capacity(config_path: Path, game_files: Iterable[Path]) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    validate_config(config)
    scope = config["scope"]
    since = datetime.fromisoformat(scope["since"])
    until = datetime.fromisoformat(scope["until"])
    by_player: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    manifest = []
    for path in sorted(set(game_files)):
        player = _player_from_path(path)
        table = pq.read_table(path, columns=list(_COLUMNS))
        manifest.append(
            {"path": str(path.resolve()), "rows": table.num_rows, "sha256": _digest(path)}
        )
        for raw in table.to_pylist():
            row = _normalized_record(raw, player)
            if row is None:
                continue
            existing = by_player[player].get(row["game_id"])
            if existing is not None and existing != row:
                raise ValueError(f"Conflicting duplicate game {row['game_id']} for {player}")
            by_player[player][row["game_id"]] = row

    scenarios = config["capacity_scenarios"]
    thresholds = [int(value) for value in config["capacity_thresholds"]]
    players = {}
    for player, games_by_id in sorted(by_player.items()):
        eligible = []
        rejected = Counter()
        for row in games_by_id.values():
            played_at = datetime.fromisoformat(row["played_at"]) if row["played_at"] else None
            reason = None
            if not row["rated"]:
                reason = "unrated"
            elif row["variant"] != str(scope["variant"]).casefold():
                reason = "variant"
            elif row["speed"] != str(scope["speed"]).casefold():
                reason = "speed"
            elif row["rating"] is None or not int(scope["rating_min"]) <= row["rating"] <= int(scope["rating_max"]):
                reason = "rating"
            elif played_at is None or not since < played_at <= until:
                reason = "date"
            if reason:
                rejected[reason] += 1
            else:
                eligible.append(row)
        controls = Counter(str(row["time_control"]) for row in eligible)
        scenario_counts = {
            scenario["name"]: sum(
                _matches_scenario(row["time_control"], scenario) for row in eligible
            )
            for scenario in scenarios
        }
        scenario_decisions = {
            scenario["name"]: sum(
                int(row["estimated_decisions"])
                for row in eligible
                if _matches_scenario(row["time_control"], scenario)
            )
            for scenario in scenarios
        }
        players[player] = {
            "unique_games_seen": len(games_by_id),
            "eligible_blitz_games": len(eligible),
            "eligible_estimated_decisions": sum(int(row["estimated_decisions"]) for row in eligible),
            "scenario_games": scenario_counts,
            "scenario_estimated_decisions": scenario_decisions,
            "time_controls": dict(sorted(controls.items(), key=lambda item: (-item[1], item[0]))),
            "rejected": dict(sorted(rejected.items())),
        }

    support = {
        scenario["name"]: {
            str(threshold): sum(
                int(player["scenario_games"][scenario["name"]]) >= threshold
                for player in players.values()
            )
            for threshold in thresholds
        }
        for scenario in scenarios
    }
    prototype = config["prototype"]
    required_players = int(prototype["target_players"])
    required_games = int(prototype["games_per_player"])
    selected = next(
        (
            scenario["name"]
            for scenario in scenarios
            if support[scenario["name"]][str(required_games)] >= required_players
        ),
        None,
    )
    qualified = (
        sorted(
            (
                player
                for player, values in players.items()
                if int(values["scenario_games"][selected]) >= required_games
            ),
            key=lambda player: hashlib.sha256(
                f"{config['seed']}:{player}".encode()
            ).hexdigest(),
        )
        if selected is not None
        else []
    )
    selected_players = qualified[:required_players]
    development_count = int(prototype["development_players"])
    selected_roles = {
        "development": selected_players[:development_count],
        "evaluation": selected_players[development_count:],
    }
    return {
        "schema_version": 1,
        "audit_kind": "read_only_label_free_capacity",
        "config_path": str(config_path.resolve()),
        "config_sha256": _digest(config_path),
        "input_files": manifest,
        "summary": {
            "players_seen": len(players),
            "unique_player_games_seen": sum(player["unique_games_seen"] for player in players.values()),
            "support_by_scenario_and_games": support,
            "prototype_required_players": required_players,
            "prototype_required_games_per_player": required_games,
            "selected_prototype_scenario": selected,
            "prototype_capacity_passed": selected is not None,
            "qualified_prototype_players": qualified,
            "selected_prototype_players": selected_players,
            "selected_prototype_roles": selected_roles,
            "eligible_games_by_scenario": {
                scenario["name"]: sum(
                    int(player["scenario_games"][scenario["name"]])
                    for player in players.values()
                )
                for scenario in scenarios
            },
            "estimated_decisions_by_scenario": {
                scenario["name"]: sum(
                    int(player["scenario_estimated_decisions"][scenario["name"]])
                    for player in players.values()
                )
                for scenario in scenarios
            },
            "labels_or_position_splits_read": False,
        },
        "players": players,
    }


def run(config_path: Path, input_root: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    report = audit_capacity(config_path, discover_game_files(input_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
