from datetime import UTC, datetime, timedelta
import json

import chess
import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.optimize import check_grad

from chess_clone.experiments.deep_style_cohort import select_games
from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows
from chess_clone.modeling.style_residual import (
    DIMENSION, feature_matrix, fit_residual, group_structure, objective, predict_residual,
)
from chess_clone.experiments.deep_style import donor_for, history_stability, select_penalty
from chess_clone.experiments import deep_style
from chess_clone.experiments import deep_style_cohort
from chess_clone.experiments.profile_ablation import digest


def rows(count=12):
    positions = [{"game_id": f"g{i}", "ply": 1, "move_number": 1, "fen": chess.STARTING_FEN,
                  "player_username": "a", "player_color": "white", "actual_move_uci": "e2e4"}
                 for i in range(count)]
    dates = {r["game_id"]: datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i)
             for i, r in enumerate(positions)}
    result = build_all_legal_candidate_rows(positions, dates, {g: "history" for g in dates})
    return [dict(r, position_key=" ".join(chess.STARTING_FEN.split()[:4]), base_logit=0.) for r in result]


def test_residual_gradient_matches_finite_difference():
    data = rows()[:40]
    matrix = feature_matrix(data)
    starts, lengths, labels = group_structure(data, require_labels=True)
    base = np.zeros(len(data))
    args = matrix, base, starts, lengths, labels, .01
    beta = np.random.default_rng(1).normal(0, .01, DIMENSION)
    assert check_grad(lambda b: objective(b, *args)[0], lambda b: objective(b, *args)[1], beta) < 1e-5


def test_residual_learns_preferences_and_inference_does_not_read_labels():
    data = rows()
    model = fit_residual(data, penalty=.01)
    probabilities = predict_residual(data, model)
    assert probabilities[next(i for i, r in enumerate(data) if r["chosen"])] > .5
    altered = [{k: v for k, v in r.items() if k != "chosen"} for r in data]
    assert predict_residual(altered, model) == probabilities
    for start in range(0, len(data), 20):
        assert sum(probabilities[start:start+20]) == pytest.approx(1)


def test_residual_rejects_nonhistory_and_invalid_schema():
    data = rows()
    with pytest.raises(ValueError, match="history"):
        fit_residual([dict(r, split="confirmation") for r in data], penalty=.1)
    with pytest.raises(ValueError, match="schema"):
        predict_residual(data, {"version": "wrong", "coefficients": [0.] * DIMENSION})
    with pytest.raises(ValueError, match="contiguous"):
        group_structure(data + data[:1], require_labels=False)


def test_features_ignore_labels_opening_tags_and_player_identity():
    data = rows()
    other = [dict(r, chosen=not r["chosen"], player_username="other", opening_eco="C99") for r in data]
    assert (feature_matrix(data) != feature_matrix(other)).nnz == 0


def cohort_fixture():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    games = [{"game_id": str(i), "played_at": start + timedelta(days=i), "rated": True,
              "variant": "Standard", "speed": "blitz"} for i in range(8)]
    positions = [{"game_id": str(i), "ply": j, "player_username": "a", "player_rating": 1500}
                 for i in range(8) for j in (1, 3)]
    config = {"since": start.isoformat(), "until": (start + timedelta(days=10)).isoformat(),
              "rating_min": 1300, "rating_max": 1700, "history_games": 3,
              "validation_games": 1, "confirmation_games": 1}
    return positions, games, config


def test_deep_cohort_whole_game_selection_and_old_game_exclusion():
    positions, games, config = cohort_fixture()
    parts, metadata = select_games(positions, games, "a", config, {"7"})
    assert metadata["game_ids"] == ["2", "3", "4", "5", "6"]
    assert {r["game_id"] for r in parts["history"]} == {"2", "3", "4"}
    assert len(parts["validation"]) == len(parts["confirmation"]) == 2
    parts, metadata = select_games(positions, games, "a", config, {"3", "4", "5", "6", "7"})
    assert parts is None and metadata["reason"] == "insufficient_games"


def test_deep_cohort_rejects_tied_boundary_and_duplicate_labels():
    positions, games, config = cohort_fixture()
    games[6]["played_at"] = games[5]["played_at"]
    parts, metadata = select_games(positions, games, "a", config, set())
    assert parts is None and metadata["reason"] == "tied_split_boundary"
    with pytest.raises(ValueError, match="Duplicate"):
        select_games(positions + positions[:1], games, "a", config, set())


def test_donor_support_and_dates_exclude_self_and_future_history():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    cohort = {}
    for player, offset in (("a", 0), ("b", 1000), ("c", -1000)):
        cohort[player] = {"game_ids": [str(i) for i in range(500)],
                          "selected_dates": {str(i): (start + timedelta(days=i+offset)).isoformat() for i in range(500)}}
    assert donor_for("a", cohort, cohort, "validation") == "c"
    assert donor_for("c", cohort, cohort, "validation") is None


def test_selection_style_primary_but_likelihood_and_plausibility_gate():
    def arm(tv, nll=2., exact=.3, top3=.5):
        return {"style": {"macro_policy_tv": tv}, "ranking": {
            "negative_log_likelihood": nll, "exact_move_accuracy": exact, "top_3_accuracy": top3}}
    results = {"0.001": {"p": {"personal": arm(.01, nll=3.), "population": arm(.1)}},
               "0.01": {"p": {"personal": arm(.05), "population": arm(.1)}},
               "0.1": {"p": {"personal": arm(.05), "population": arm(.1)}}}
    assert select_penalty(results) == "0.1"
    assert select_penalty({"0.001": results["0.001"]}) is None


def test_human_history_stability_reports_zero_for_identical_repeated_preferences():
    # Twenty-four decisions per half meet the fixed cell support threshold.
    data = rows()
    expanded = [dict(r, game_id=f"{r['game_id']}-{i}", decision_id=f"{r['decision_id']}-{i}")
                for i in range(4) for r in data]
    result = history_stability(expanded)
    assert result["mean_human_tv"] == 0
    assert result["early_games"] == result["late_games"] == 24


def test_validation_never_opens_confirmation_player_or_split(monkeypatch, tmp_path):
    root = tmp_path / "residual"
    (root / "models").mkdir(parents=True)
    (tmp_path / "declaration.json").write_text("{}")
    (tmp_path / "acquisition.json").write_text("{}")
    config = {"history_games": 300, "validation_games": 100, "confirmation_games": 100,
              "artifact_dir": str(tmp_path), "regularization_grid": [.01]}
    cohort = {"dev": {"role": "development"}, "reserved": {"role": "confirmation"}}
    monkeypatch.setattr(deep_style, "load_cohort", lambda source: ({"config": config}, cohort))
    calls = []
    def fit(root, player, record, penalties, artifact):
        calls.append((player, "history"))
        return {"0.01": {"version": "conditional-style-residual-v1", "coefficients": [0.] * DIMENSION}}
    monkeypatch.setattr(deep_style, "fit_player", fit)
    def cache(root, player, record, split, artifact):
        calls.append((player, split))
        return rows(24)
    monkeypatch.setattr(deep_style, "cache_rows", cache)
    monkeypatch.setattr(deep_style, "donor_for", lambda *args: None)
    deep_style.validate(tmp_path)
    assert calls == [("dev", "history"), ("dev", "validation")]
    selection = json.loads((root / "selection.json").read_text())
    assert selection["status"] == "selected"
    assert selection["scope"].endswith("confirmation not opened")


def test_confirmation_refuses_unselected_candidate_before_opening_labels(monkeypatch, tmp_path):
    root = tmp_path / "residual"
    root.mkdir()
    for name in ("acquisition.json", "declaration.json"):
        (tmp_path / name).write_text("{}")
    selection = {"status": "no_eligible_candidate", "frozen_sha256": {},
                 **{name.removesuffix(".json") + "_sha256": digest(tmp_path / name)
                    for name in ("acquisition.json", "declaration.json")}}
    (root / "selection.json").write_text(json.dumps(selection))
    (root / "selection_seal.json").write_text(json.dumps({"sha256": digest(root / "selection.json")}))
    monkeypatch.setattr(deep_style, "load_cohort", lambda source: ({"config": {}}, {}))
    with pytest.raises(ValueError, match="stays unopened"):
        deep_style.confirm(tmp_path)
    assert not (root / "confirmation").exists()


def acquisition_fixture(tmp_path):
    """One completed target and one pending target, without live network access."""
    history = tmp_path / "completed.parquet"
    history.write_bytes(b"frozen history fixture")
    declaration = {"config": {"target_players": 2, "development_players": 1,
                              "max_games_per_player": 1000, "since": "2024-09-20",
                              "until": "2026-09-20"},
                   "input_sha256": {}, "excluded_game_ids": ["old"],
                   "candidates": [{"username": "completed"}, {"username": "pending"}]}
    (tmp_path / "declaration.json").write_text(json.dumps(declaration))
    state = {"status": "failed", "error": "previous interrupted response",
             "declaration_sha256": digest(tmp_path / "declaration.json"),
             "players": {"completed": {"accepted": True, "game_ids": ["used"],
                                       "sha256": {str(history): digest(history)}}}}
    (tmp_path / "acquisition.json").write_text(json.dumps(state))
    return history


def test_acquisition_resume_reuses_completed_records_and_preserves_failure(monkeypatch, tmp_path):
    from chess_clone.providers.base import ProviderError
    acquisition_fixture(tmp_path)
    calls = []
    def interrupted(provider, player, **kwargs):
        resumed = json.loads((tmp_path / "acquisition.json").read_text())
        assert resumed["status"] == "acquiring"
        assert "error" not in resumed
        calls.append(player)
        raise ProviderError("incomplete response")
    monkeypatch.setattr(deep_style_cohort, "ingest_games", interrupted)
    with pytest.raises(ProviderError, match="incomplete response"):
        deep_style_cohort.acquire(tmp_path)
    state = json.loads((tmp_path / "acquisition.json").read_text())
    assert calls == ["pending"]
    assert state["status"] == "failed"
    assert state["error"] == "incomplete response"
    assert list(state["players"]) == ["completed"]
    assert not (tmp_path / "acquisition_seal.json").exists()


def test_acquisition_detects_changed_completed_artifact_before_download(monkeypatch, tmp_path):
    history = acquisition_fixture(tmp_path)
    history.write_bytes(b"modified")
    monkeypatch.setattr(deep_style_cohort, "ingest_games",
                        lambda *a, **kw: pytest.fail("must not download after a hash failure"))
    with pytest.raises(ValueError, match="changed"):
        deep_style_cohort.acquire(tmp_path)


def test_confirmation_refuses_changed_sealed_inputs_before_opening(monkeypatch, tmp_path):
    root = tmp_path / "residual"
    root.mkdir()
    for name in ("acquisition.json", "declaration.json"):
        (tmp_path / name).write_text("{}")
    frozen = root / "shared.json"
    frozen.write_text("{}")
    selection = {"status": "selected", "frozen_sha256": {str(frozen): digest(frozen)},
                 **{name.removesuffix(".json") + "_sha256": digest(tmp_path / name)
                    for name in ("acquisition.json", "declaration.json")}}
    (root / "selection.json").write_text(json.dumps(selection))
    (root / "selection_seal.json").write_text(json.dumps({"sha256": digest(root / "selection.json")}))
    frozen.write_text('{"modified": true}')
    monkeypatch.setattr(deep_style, "load_cohort", lambda source: ({"config": {}}, {}))
    with pytest.raises(ValueError, match="changed"):
        deep_style.confirm(tmp_path)
    assert not (root / "confirmation").exists()


def test_confirmation_workflow_keeps_targets_reserved_and_does_not_promote_identical_policies(monkeypatch, tmp_path):
    root = tmp_path / "residual"
    (root / "models").mkdir(parents=True)
    for name in ("acquisition.json", "declaration.json"):
        (tmp_path / name).write_text("{}")
    model = {"version": "conditional-style-residual-v1", "coefficients": [0.] * DIMENSION}
    (root / "shared.json").write_text(json.dumps({"0.01": model}))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    cohort = {}
    for i in range(10):
        player = f"player{i}"
        role = "development" if i < 2 else "confirmation"
        cohort[player] = {"role": role, "game_ids": [str(g) for g in range(500)],
                          "selected_dates": {str(g): (start + timedelta(days=g)).isoformat()
                                             for g in range(500)}}
        if role == "development":
            (root / "models" / f"{player}.json").write_text(json.dumps({"models": {"0.01": model}}))
    selection = {"status": "selected", "selected_penalty": "0.01",
                 "frozen_sha256": {str(p): digest(p) for p in
                                   [root / "shared.json", *sorted((root / "models").glob("*.json"))]},
                 **{name.removesuffix(".json") + "_sha256": digest(tmp_path / name)
                    for name in ("acquisition.json", "declaration.json")}}
    (root / "selection.json").write_text(json.dumps(selection))
    (root / "selection_seal.json").write_text(json.dumps({"sha256": digest(root / "selection.json")}))
    monkeypatch.setattr(deep_style, "load_cohort", lambda source: (
        {"config": {"artifact_dir": str(tmp_path)}}, cohort))
    fits, scored = [], []
    def fit(root, player, record, penalties, artifact):
        fits.append((player, penalties))
        assert record["role"] == "confirmation"
        return {"0.01": model}
    def cache(root, player, record, split, artifact):
        scored.append((player, split))
        return [dict(r, player_username=player, game_id=f"{player}-{r['game_id']}",
                     decision_id=f"{player}-{r['decision_id']}") for r in rows(24)]
    monkeypatch.setattr(deep_style, "fit_player", fit)
    monkeypatch.setattr(deep_style, "cache_rows", cache)
    deep_style.confirm(tmp_path)
    destination = root / "confirmation"
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["players"] == 8
    assert summary["replay_gates_pass"] is False
    assert summary["personal_minus_wrong_style"]["paired_player_95_interval"] == [0., 0.]
    assert fits == [(f"player{i}", [.01]) for i in range(2, 10)]
    assert scored == [(f"player{i}", "confirmation") for i in range(2, 10)]
    assert json.loads((destination / "manifest.json").read_text())["status"] == "complete"
    for arm in ("population", "shared", "personal", "wrong"):
        predictions = pq.read_table(destination / f"test_predictions_{arm}.parquet").to_pylist()
        assert len(predictions) == len({r["decision_id"] for r in predictions}) == 192
    with pytest.raises(FileExistsError, match="already opened"):
        deep_style.confirm(tmp_path)
