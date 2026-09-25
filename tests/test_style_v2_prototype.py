from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from chess_clone.experiments.style_v2_dataset import (
    _select_player_positions, prepare_candidate_shards,
)
from chess_clone.experiments.style_v2_prototype import (
    _train, fit_baseline, load_json_sealed, replay_gate, residual_from_embedding,
    save_json_sealed, select_role_manifest, split_audit,
)
from chess_clone.modeling.player_embedding import PlayerConditionedPolicy, ShardDecisionDataset
from chess_clone.modeling.style_residual import DIMENSION, feature_matrix, predict_residual
from test_player_embedding import candidate_rows


def sources(tmp_path, *, incomplete=False, tie=False):
    games, positions = [], []
    for index in range(5):
        date = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)
        if tie and index == 3:
            date -= timedelta(days=1)
        game_id = f"g{index}"
        games.append({"game_id": game_id, "played_at": date, "white_username": "alpha",
                      "black_username": "opponent", "white_rating": 1500, "black_rating": 1500,
                      "rated": True, "variant": "standard", "speed": "blitz",
                      "time_control": "180+0", "total_plies": 0 if index == 1 else 4})
        board = chess.Board()
        for ply, move in enumerate(("e2e4", "e7e5", "g1f3", "b8c6"), 1):
            if index != 1 and ply % 2:
                positions.append({"game_id": game_id, "ply": ply, "move_number": (ply + 1) // 2,
                                  "fen": board.fen(), "player_username": "alpha", "player_color": "white",
                                  "actual_move_uci": move, "time_control": "180+0"})
            board.push_uci(move)
    if incomplete:
        positions.pop()
    gp, pp = tmp_path / "games.parquet", tmp_path / "positions.parquet"
    pq.write_table(pa.Table.from_pylist(games), gp)
    pq.write_table(pa.Table.from_pylist(positions), pp)
    return [(gp, pp)]


SCOPE = {"since": "2025-01-01T00:00:00+00:00", "until": "2027-01-01T00:00:00+00:00",
         "variant": "standard", "speed": "blitz", "rating_min": 1300, "rating_max": 1700}


def test_whole_games_deduplicated_and_zero_decision_games_retained(tmp_path):
    pairs = sources(tmp_path)
    splits, dates, summary = _select_player_positions(
        pairs + pairs, "alpha", SCOPE, {"180+0"}, {"history": 2, "validation": 1, "evaluation": 1},
    )
    assert summary["game_ids"] == {"history": ["g1", "g2"], "validation": ["g3"], "evaluation": ["g4"]}
    assert summary["zero_decision_games"] == ["g1"]
    assert [len(splits[key]) for key in splits] == [2, 2, 2]
    assert max(dates[g] for g in summary["game_ids"]["history"]) < dates["g3"]


@pytest.mark.parametrize("kwargs,match", [({"incomplete": True}, "Incomplete whole-game"),
                                         ({"tie": True}, "Tied chronological")])
def test_invalid_whole_game_splits_fail_closed(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _select_player_positions(sources(tmp_path, **kwargs), "alpha", SCOPE, {"180+0"},
                                 {"history": 2, "validation": 1, "evaluation": 1})


def test_cross_player_overlap_is_quarantined_or_rejected():
    selection = {
        "a": {"game_ids": {"history": ["a1"], "validation": ["shared"], "evaluation": ["a3"]},
              "zero_decision_games": []},
        "b": {"game_ids": {"history": ["b1"], "validation": ["b2"], "evaluation": ["shared"]},
              "zero_decision_games": []},
    }
    assert split_audit(selection, ["a"], ["b"])["development_validation_quarantine"] == {"a": ["shared"]}
    selection["b"]["game_ids"]["history"] = ["a3"]
    with pytest.raises(ValueError, match="cross history"):
        split_audit(selection, ["a"], ["b"])


def test_linear_collapse_matches_tensor_policy_and_shared_mean():
    torch.manual_seed(42)
    model = PlayerConditionedPolicy(3, DIMENSION, 32)
    rows = candidate_rows("alpha", "e2e4", count=1)
    features = torch.tensor(feature_matrix(rows).toarray(), dtype=torch.float32).unsqueeze(0)
    base = torch.tensor([[row["base_logit"] for row in rows]], dtype=torch.float32)
    with torch.no_grad():
        expected = torch.softmax(model(features, base, torch.tensor([1])), dim=1).flatten().numpy()
    assert np.allclose(predict_residual(rows, residual_from_embedding(model, 1)), expected, atol=1e-7)
    coefficients = [residual_from_embedding(model, index)["coefficients"] for index in range(3)]
    assert np.allclose(residual_from_embedding(model, None)["coefficients"], np.mean(coefficients, axis=0), atol=1e-7)


def test_history_roles_isolated_and_adaptation_freezes_representation(tmp_path):
    source = {}
    for player, move in (("alpha", "e2e4"), ("beta", "d2d4"), ("gamma", "b1c3"), ("omega", "g1f3")):
        path = tmp_path / f"{player}.parquet"
        pq.write_table(pa.Table.from_pylist(candidate_rows(player, move)), path)
        source[player] = path
    manifest = prepare_candidate_shards(source, tmp_path / "shards", decisions_per_shard=4)
    selected = select_role_manifest(manifest, ["alpha", "beta"])
    assert selected["decisions"] == 16
    config = json.loads(Path("configs/style_model_v2_prototype_run.json").read_text())
    config.update(epochs=1, adaptation_epochs=1, batch_decisions=8)
    torch.set_num_threads(1)
    model, _ = _train(manifest, ["alpha", "beta"], config, tmp_path / "dev")
    adapted, _ = _train(manifest, ["gamma", "omega"], config, tmp_path / "eval", representation=model)
    assert torch.equal(model.move_projection.weight, adapted.move_projection.weight)
    assert torch.equal(model.shared_adjustment.weight, adapted.shared_adjustment.weight)
    resumed, _ = _train(manifest, ["alpha", "beta"], config, tmp_path / "dev")
    for key, value in model.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])


def test_resume_rejects_modified_result(tmp_path):
    path = tmp_path / "result.json"
    save_json_sealed(path, {"value": 1})
    assert load_json_sealed(path) == {"value": 1}
    path.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="Changed sealed input"):
        load_json_sealed(path)


def test_interrupted_training_resumes_identically(tmp_path, monkeypatch):
    paths = {}
    for player, move in (("alpha", "e2e4"), ("beta", "d2d4")):
        path = tmp_path / f"{player}.parquet"
        pq.write_table(pa.Table.from_pylist(candidate_rows(player, move)), path)
        paths[player] = path
    manifest = prepare_candidate_shards(paths, tmp_path / "data", decisions_per_shard=4)
    config = json.loads(Path("configs/style_model_v2_prototype_run.json").read_text())
    config.update(epochs=2, batch_decisions=8)
    complete, _ = _train(manifest, list(paths), config, tmp_path / "complete")
    original = ShardDecisionDataset.__iter__

    def interrupted(dataset):
        if dataset.seed == config["seed"] + 1:
            raise RuntimeError("simulated interruption")
        return original(dataset)

    with monkeypatch.context() as patch:
        patch.setattr(ShardDecisionDataset, "__iter__", interrupted)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            _train(manifest, list(paths), config, tmp_path / "interrupted")
    resumed, history = _train(manifest, list(paths), config, tmp_path / "interrupted")
    assert len(history) == 2
    for key, value in complete.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])


def test_baseline_resume_refuses_a_changed_penalty(tmp_path):
    from chess_clone.experiments.style_v2_dataset import digest

    history = tmp_path / "data/candidate-cache/history/alpha.parquet"
    history.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(candidate_rows("alpha", "e2e4")), history)
    history.with_suffix(".json").write_text(json.dumps({"sha256": digest(history)}))
    model = fit_baseline(tmp_path / "data", tmp_path / "results", "alpha", .001)
    assert fit_baseline(tmp_path / "data", tmp_path / "results", "alpha", .001) == model
    with pytest.raises(ValueError, match="penalty changed"):
        fit_baseline(tmp_path / "data", tmp_path / "results", "alpha", .01)


def test_v1_replay_failure_cannot_promote_candidate():
    metrics = {"negative_log_likelihood": 2., "exact_move_accuracy": .3, "top_3_accuracy": .5,
               "top_5_accuracy": .6, "mean_reciprocal_rank": .4, "multiclass_brier_score": .8,
               "top_1_expected_calibration_error": .03}
    records = {"alpha": {arm: {"style": {"macro_policy_tv": tv}, "ranking": dict(metrics)}
                         for arm, tv in (("personal", .04), ("shared", .05), ("wrong", .06),
                                         ("population", .06), ("v1_residual", .03))}}
    report = replay_gate(records)
    assert report["checks"]["style_beats_shared"]
    assert not report["checks"]["style_beats_v1_residual"]
    assert not report["passed"]
