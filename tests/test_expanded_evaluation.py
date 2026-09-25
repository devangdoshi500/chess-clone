from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chess_clone.experiments.expanded_evaluation import (
    discover_players, whole_game_prefix, select_positions, verify_hashes, load_sealed, SUBSETS,
)
from chess_clone.experiments.profile_ablation import digest
from chess_clone.experiments.population_metrics import legal_policy_metrics, uniform_probabilities
from chess_clone.experiments.evaluation_panel import summarize_predictions, ranking_panel, paired_uncertainty
from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows


@pytest.fixture
def predictions():
    positions = [{"game_id": f"g{i}", "ply": 1, "move_number": 1, "player_username": name,
                  "player_color": "white", "fen": chess.STARTING_FEN, "actual_move_uci": "e2e4"}
                 for i, name in enumerate(("A", "A", "B"))]
    dates = {p["game_id"]: datetime(2026, 1, i+1, tzinfo=UTC) for i, p in enumerate(positions)}
    rows = build_all_legal_candidate_rows(positions, dates, {g: "test" for g in dates})
    metrics, predictions = legal_policy_metrics(rows, uniform_probabilities(rows))
    for row in predictions:
        row.update(game_phase="opening", rating_band="1400", player_color="white", time_control="180+0")
    return metrics, predictions


def test_sufficient_statistics_reproduce_every_original_metric(predictions):
    expected, rows = predictions
    actual = summarize_predictions(rows)
    for key in expected:
        if key == "behavior_rates":
            assert actual[key] == expected[key]
        else:
            assert actual[key] == pytest.approx(expected[key]), key


def test_full_panel_macro_weights_players_not_decisions(predictions):
    _, rows = predictions
    for r in rows:
        r["exact_correct"] = r["player_username"] == "A"
    result = ranking_panel(rows)
    assert result["exact_move_accuracy"] == pytest.approx(2/3)
    assert result["player_macro"]["exact_move_accuracy"]["mean"] == .5
    assert result["player_macro"]["exact_move_accuracy"]["players"] == 2
    for field in ("game_phase", "rating_band", "player_color", "time_control"):
        subgroup = next(iter(result["breakdowns"][field].values()))
        assert "multiclass_brier_score" in subgroup
        assert "top_1_expected_calibration_error" in subgroup
        assert "median_actual_rank" in subgroup
    assert ranking_panel([])["status"] == "no_data"
    assert ranking_panel([])["player_macro"] == {}


def test_paired_player_bootstrap_and_identity_checks(predictions):
    _, left = predictions
    right = [dict(r, exact_correct=r["player_username"] == "A") for r in left]
    result = paired_uncertainty(left, right, unit="player_username", resamples=100)
    assert result["units"] == 2
    assert result["decisions"] == 3
    assert result["exact"]["delta"] == pytest.approx(2/3)
    assert result["exact"]["equal_player_macro"]["delta"] == .5
    assert result["exact"]["95_interval"] == [0, 1]
    with pytest.raises(ValueError, match="unique identical"):
        paired_uncertainty(left, right[:-1])
    with pytest.raises(ValueError, match="identity"):
        paired_uncertainty(left, [dict(right[0], actual_move_uci="d2d4"), *right[1:]])


def test_deterministic_discovery_excludes_training_targets_and_deduplicates_games(tmp_path):
    games = [{"game_id": name, "white_username": "Known", "black_username": other,
              "white_rating": 1500, "black_rating": rating}
             for name, other, rating in (("a", "Alpha", 1500), ("b", "Alpha", 1500),
                                         ("c", "Beta", 1400), ("d", "Elite", 2500))]
    one, two = tmp_path / "one.parquet", tmp_path / "two.parquet"
    pq.write_table(pa.Table.from_pylist(games), one)
    pq.write_table(pa.Table.from_pylist(list(reversed(games))), two)
    config = {"rating_min": 1300, "rating_max": 1700, "seed": 42, "max_candidate_players": 10}
    found = discover_players([SimpleNamespace(games=one), SimpleNamespace(games=two)], {"known"}, config)
    assert found == [{"username": "alpha", "eligible_opponent_appearances": 2}, {"username": "beta", "eligible_opponent_appearances": 1}]


def test_whole_game_caps_and_seeded_quality_sample_do_not_split_or_pack():
    dates = {g: datetime(2026, 1, day, tzinfo=UTC) for g, day in (("a", 3), ("b", 2), ("c", 1))}
    rows = [{"game_id": g, "ply": i} for g, count in (("a", 3), ("b", 5), ("c", 1)) for i in range(count)]
    selected = whole_game_prefix(rows, dates, 5)
    assert len(selected) == 3
    assert {r["game_id"] for r in selected} == {"a"}
    assert whole_game_prefix(rows, dates, 6, seed=42) == whole_game_prefix(list(reversed(rows)), dates, 6, seed=42)


def test_selection_rejects_prior_shared_and_partial_rating_games():
    since = datetime(2026, 1, 1, tzinfo=UTC)
    until = since + timedelta(days=10)
    config = {"rating_min": 1300, "rating_max": 1700, "max_decisions_per_player": 100}
    games = [{"game_id": g, "played_at": since + timedelta(days=1), "rated": True, "speed": "blitz", "variant": "Standard"}
             for g in ("prior", "shared", "partial", "good")]
    rows = [{"game_id": g["game_id"], "ply": i, "player_username": "A", "player_rating": 1500}
            for g in games for i in (1, 3)]
    rows[5]["player_rating"] = 1295
    selected, _, reasons = select_positions(rows, games, username="a", since=since, until=until,
                                            excluded_games={"prior", "shared"}, config=config)
    assert {r["game_id"] for r in selected} == {"good"}
    assert len(selected) == 2
    assert reasons["previous_or_shared_game"] == 2
    assert reasons["rating_or_no_target_decisions"] == 1
    with pytest.raises(ValueError, match="Duplicate target"):
        select_positions(rows + rows[:1], games, username="a", since=since, until=until, excluded_games=set(), config=config)


def test_fresh_and_unseen_questions_are_separate():
    rows = [{"cohort_kind": "known", "fresh": True}, {"cohort_kind": "unseen", "fresh": False},
            {"cohort_kind": "unseen", "fresh": True}]
    assert [sum(predicate(r) for r in rows) for predicate in SUBSETS.values()] == [2, 1, 1]


def test_changed_inputs_and_broken_acquisition_seals_fail(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}")
    hashes = {str(source): digest(source)}
    verify_hashes(hashes)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_hashes(hashes)
    (tmp_path / "declaration.json").write_text("{}")
    (tmp_path / "acquisition.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "acquisition_seal.json").write_text(json.dumps({"sha256": "wrong"}))
    with pytest.raises(ValueError, match="seal"):
        load_sealed(tmp_path)


def test_evaluation_pipeline_writes_full_and_quality_panels_without_fitting(tmp_path, monkeypatch):
    from chess_clone.experiments import expanded_evaluation as experiment
    from chess_clone.modeling.opportunity_profile import OpportunityProfile
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "feature_sets.json").write_text(json.dumps({"safe_population": [], "opportunity_profile": []}))
    (artifact / "metrics.json").write_text(json.dumps({k: {"temperature": 1} for k in ("safe_population", "opportunity_profile")}))
    (artifact / "frequency_counts.json").write_text(json.dumps({"e2e4": 20}))
    profile = OpportunityProfile()
    profile.fit_transform_ordered([])
    (artifact / "profile.json").write_text(json.dumps(profile.to_dict()))
    output = tmp_path / "evaluation-run"
    output.mkdir()
    state = {"players": {}}
    for i, name in enumerate(("a", "b")):
        path = tmp_path / f"{name}.parquet"
        row = {"game_id": f"game{i}", "ply": 1, "move_number": 1, "player_username": name,
               "player_color": "white", "fen": chess.STARTING_FEN, "actual_move_uci": "e2e4", "player_rating": 1500}
        pq.write_table(pa.Table.from_pylist([row]), path)
        state["players"][name] = {"accepted": True, "kind": "unseen", "positions": str(path),
                                   "selected_dates": {f"game{i}": "2026-01-01T00:00:00+00:00"}}
    (output / "acquisition.json").write_text(json.dumps(state))
    declaration = {"selected_model": "opportunity_profile", "fresh_after": "2025-12-01T00:00:00+00:00",
                   "config": {"artifact_dir": str(artifact), "quality_decisions_per_player": 200, "seed": 42}}
    monkeypatch.setattr(experiment, "load_sealed", lambda _: (declaration, state))
    class FakeModel:
        def load_model(self, path):
            return self
    monkeypatch.setattr(experiment, "CatBoostRanker", FakeModel)
    monkeypatch.setattr(experiment, "predict_relevance_scores", lambda model, rows, fields: [0.] * len(rows))
    experiment.evaluate(output)
    report = json.loads((output / "evaluation/ranking_report.json").read_text())
    assert report["unseen_players"]["methods"]["opportunity_profile"]["players"] == 2
    assert report["known_players_fresh"]["methods"]["safe_population"]["status"] == "no_data"
    manifest = json.loads((output / "evaluation/manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["quality_inputs_sha256"]) == 4
    verify_hashes(manifest["predictions_sha256"])
    verify_hashes(manifest["quality_inputs_sha256"])
    with pytest.raises(FileExistsError):
        experiment.evaluate(output)
