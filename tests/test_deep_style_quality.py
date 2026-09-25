"""No live engine or network: exercise the declared quality-sampling boundary."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chess_clone.experiments import deep_style_quality as quality
from chess_clone.experiments.expanded_evaluation import seeded_key
from chess_clone.experiments.profile_ablation import digest


def test_quality_declaration_cannot_follow_confirmation(tmp_path):
    (tmp_path / "residual/confirmation").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="before confirmation"):
        quality.declare(tmp_path, tmp_path / "unused-config.json")


def test_quality_requires_completed_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(quality, "load_cohort", lambda source: ({}, {}))
    (tmp_path / "quality_declaration.json").write_text(json.dumps({"input_sha256": {}, "config": {}}))
    confirmation = tmp_path / "residual/confirmation"
    confirmation.mkdir(parents=True)
    (confirmation / "manifest.json").write_text('{"status": "failed"}')
    with pytest.raises(ValueError, match="must be complete"):
        quality.run(tmp_path)
    assert not (tmp_path / "quality_sample").exists()


def test_quality_samples_whole_games_on_common_decisions_and_hashes_inputs(monkeypatch, tmp_path):
    confirmation = tmp_path / "residual/confirmation"
    confirmation.mkdir(parents=True)
    (confirmation / "manifest.json").write_text('{"status": "complete"}')
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text("{}")
    config = {"seed": 17, "games_per_player": 2, "thresholds": str(thresholds)}
    (tmp_path / "quality_declaration.json").write_text(json.dumps({"config": config, "input_sha256": {
        str(thresholds): digest(thresholds)}}))
    cohort, all_predictions = {"development": {"role": "development"}}, []
    expected = set()
    for player in ("a", "b", "no_donor"):
        games = [f"{player}-{i}" for i in range(5)]
        positions = [{"game_id": g, "ply": ply} for g in games for ply in (1, 3, 5)]
        path = tmp_path / f"{player}.parquet"
        pq.write_table(pa.Table.from_pylist(positions), path)
        games_path = tmp_path / f"games_{player}_fixture.parquet"
        pq.write_table(pa.Table.from_pylist([{"game_id": g} for g in games]), games_path)
        cohort[player] = {"role": "confirmation", "paths": {"confirmation": str(path)},
                          "sha256": {str(games_path): digest(games_path)}}
        all_predictions.extend({"decision_id": f"{r['game_id']}:{r['ply']}",
                                "player_username": player} for r in positions)
        if player != "no_donor":
            selected = sorted(games, key=lambda g: seeded_key(17, g))[:2]
            expected.update(f"{g}:{ply}" for g in selected for ply in (1, 3, 5))
    for arm in ("population", "shared", "personal", "wrong"):
        rows = [r for r in all_predictions if arm != "wrong" or r["player_username"] != "no_donor"]
        pq.write_table(pa.Table.from_pylist(rows), confirmation / f"test_predictions_{arm}.parquet")
    monkeypatch.setattr(quality, "load_cohort", lambda source: ({}, cohort))
    calls = []
    monkeypatch.setattr(quality, "run_move_quality", lambda *args: calls.append(args))
    monkeypatch.setattr(quality, "read_quality", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(quality, "build_report", lambda *args: {})
    quality.run(tmp_path)
    output = tmp_path / "quality_sample"
    assert len(calls) == 1
    for arm in ("population", "shared", "personal", "wrong"):
        rows = pq.read_table(output / f"test_predictions_{arm}.parquet").to_pylist()
        assert {r["decision_id"] for r in rows} == expected
    sampling = json.loads((output / "sampling.json").read_text())
    assert sampling["decisions"] == 12
    assert len(sampling["confirmation_prediction_sha256"]) == 4
    sources = json.loads((output / "cohort.json").read_text())["players"]
    assert {s["username"] for s in sources} == {"a", "b", "no_donor"}
    assert (output / "winning_chance.json").exists()
    with pytest.raises(FileExistsError):
        quality.run(tmp_path)
