import json
from datetime import UTC, datetime
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chess_clone.experiments.style_v2_capacity import (
    _player_from_path,
    audit_capacity,
    validate_config,
)


def config(tmp_path):
    payload = {
        "schema_version": 1,
        "seed": 42,
        "scope": {
            "variant": "standard",
            "speed": "blitz",
            "rating_min": 1300,
            "rating_max": 1700,
            "since": "2025-01-01T00:00:00+00:00",
            "until": "2027-01-01T00:00:00+00:00",
            "no_live_clock_inputs": True,
        },
        "capacity_scenarios": [
            {"name": "three", "kind": "exact", "time_controls": ["180+0"]},
            {"name": "narrow", "kind": "initial_seconds_range",
             "minimum_initial_seconds": 180, "maximum_initial_seconds": 300},
        ],
        "prototype": {"target_players": 1, "development_players": 1,
                      "evaluation_players": 0, "games_per_player": 2,
                      "history_games": 1, "validation_games": 1,
                      "evaluation_games": 0},
        "capacity_thresholds": [1, 2],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path, payload


def game(game_id, player, control, *, rating=1500, played_at=None):
    return {
        "game_id": game_id,
        "played_at": played_at or datetime(2026, 1, 1, tzinfo=UTC),
        "white_username": player,
        "black_username": "opponent",
        "white_rating": rating,
        "black_rating": 1500,
        "rated": True,
        "variant": "Standard",
        "speed": "blitz",
        "time_control": control,
        "total_plies": 41,
    }


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_player_filename_keeps_underscores(tmp_path):
    path = tmp_path / "games_player_name_20260922T123456789Z.parquet"
    assert _player_from_path(path) == "player_name"


def test_read_only_audit_deduplicates_snapshots_and_selects_narrow_scope(tmp_path):
    config_path, _ = config(tmp_path)
    first = tmp_path / "one/normalized/games_target_20260922T100000000Z.parquet"
    second = tmp_path / "two/normalized/games_target_20260922T110000000Z.parquet"
    rows = [game("a", "Target", "180+0"), game("b", "Target", "300+3")]
    write(first, rows)
    write(second, [rows[0]])
    report = audit_capacity(config_path, [first, second])
    player = report["players"]["target"]
    assert player["unique_games_seen"] == 2
    assert player["scenario_games"] == {"three": 1, "narrow": 2}
    assert player["scenario_estimated_decisions"] == {"three": 21, "narrow": 42}
    assert report["summary"]["selected_prototype_scenario"] == "narrow"
    assert report["summary"]["selected_prototype_players"] == ["target"]
    assert report["summary"]["selected_prototype_roles"] == {
        "development": ["target"], "evaluation": []
    }
    assert report["summary"]["eligible_games_by_scenario"] == {
        "three": 1, "narrow": 2
    }
    assert report["summary"]["labels_or_position_splits_read"] is False


def test_audit_applies_date_rating_and_game_filters(tmp_path):
    config_path, _ = config(tmp_path)
    path = tmp_path / "normalized/games_target_20260922T100000000Z.parquet"
    rows = [
        game("ok", "target", "180+0"),
        game("old", "target", "180+0", played_at=datetime(2024, 1, 1, tzinfo=UTC)),
        game("rating", "target", "180+0", rating=1800),
    ]
    rows.append(dict(game("rapid", "target", "300+0"), speed="rapid"))
    write(path, rows)
    report = audit_capacity(config_path, [path])
    player = report["players"]["target"]
    assert player["eligible_blitz_games"] == 1
    assert player["rejected"] == {"date": 1, "rating": 1, "speed": 1}


def test_config_rejects_clock_inputs_and_inconsistent_splits(tmp_path):
    _, payload = config(tmp_path)
    payload["scope"]["no_live_clock_inputs"] = False
    with pytest.raises(ValueError, match="clock"):
        validate_config(payload)
    payload["scope"]["no_live_clock_inputs"] = True
    payload["prototype"]["history_games"] = 2
    with pytest.raises(ValueError, match="split"):
        validate_config(payload)


def test_checked_in_declaration_and_capacity_selected_cohort_are_consistent():
    root = Path(__file__).parents[1]
    declaration_path = root / "configs/style_model_v2.json"
    declaration = json.loads(declaration_path.read_text())
    validate_config(declaration)
    cohort = json.loads(
        (root / "configs/style_model_v2_prototype_cohort.json").read_text()
    )
    assert cohort["time_controls"] == ["180+0"]
    assert len(cohort["development_players"]) == 14
    assert len(cohort["evaluation_players"]) == 6
    assert not set(cohort["development_players"]) & set(cohort["evaluation_players"])
    assert cohort["declaration_sha256"] == hashlib.sha256(
        declaration_path.read_bytes()
    ).hexdigest()
