from datetime import UTC, datetime, timedelta

import pytest

from chess_clone.modeling.legal_policy import LEGAL_POLICY_FEATURE_FIELDS, build_all_legal_candidate_rows
from chess_clone.modeling.opportunity_profile import OpportunityProfile, OPPORTUNITY_FIELDS
from chess_clone.experiments.profile_ablation import select_profile


def group(game, day, *, player="A", chosen_capture=True, ply=1, split="train", forced=False):
    return [{"decision_id": f"{game}:{ply}", "game_id": game, "player_username": player,
             "played_at": datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day), "split": split,
             "candidate_move_uci": str(i), "candidate_is_capture": forced or i == 0,
             "candidate_gives_check": False, "chosen": (i == 0) == chosen_capture} for i in range(2)]


def values(rows):
    return [[r[f] for f in OPPORTUNITY_FIELDS] for r in rows]


def test_profiles_exclude_current_game_future_games_and_timestamp_ties():
    first = group("a", 0)
    same_game = group("a", 0, ply=3)
    simultaneous = group("b", 0)
    later = group("c", 1)
    rows = first + same_game + simultaneous + later
    result = OpportunityProfile().fit_transform_ordered(rows)
    assert all(r["profile_capture_observations"] == 0 for r in result[:6])
    assert all(r["profile_capture_observations"] == 3 for r in result[6:])
    changed = first + group("a", 0, ply=3, chosen_capture=False) + simultaneous + group("c", 1, chosen_capture=False)
    other = OpportunityProfile().fit_transform_ordered(changed)
    assert values(result[:6]) == values(other[:6])


def test_forced_moves_not_counted_and_transform_does_not_read_labels_or_update():
    model = OpportunityProfile()
    model.fit_transform_ordered(group("a", 0) + group("b", 1, forced=True))
    held = group("c", 2, split="validation")
    before = model.to_dict()
    result = model.transform(held)
    assert result[0]["profile_capture_observations"] == 1
    assert result[0]["profile_check_observations"] == 0
    assert not result[0]["profile_check_available"]
    altered = [dict(r, chosen=not r["chosen"]) for r in held]
    assert values(result) == values(model.transform(altered))
    assert before == model.to_dict()
    restored = OpportunityProfile.from_dict(before)
    assert values(restored.transform(held)) == values(result)
    unknown = model.transform(group("d", 2, player="unknown", split="test"))
    assert unknown[0]["profile_capture_observations"] == 0
    assert unknown[0]["profile_capture_match_probability"] == pytest.approx(2/3)
    with pytest.raises(ValueError, match="training"):
        model.fit_transform_ordered(held)


def test_training_input_order_does_not_change_profiles():
    rows = group("b", 2) + group("a", 0) + group("c", 3)
    left = OpportunityProfile().fit_transform_ordered(rows)
    right = OpportunityProfile().fit_transform_ordered(list(reversed(rows)))
    key = lambda r: (r["decision_id"], r["candidate_move_uci"])
    assert values(sorted(left, key=key)) == values(sorted(right, key=key))


def test_archived_tags_cannot_change_safe_features():
    import chess
    position = {"game_id": "g", "ply": 1, "move_number": 1, "player_username": "A", "player_color": "white",
                "fen": chess.STARTING_FEN, "actual_move_uci": "e2e4", "eco": "C44", "opening_name": "Scotch Game"}
    dates = {"g": datetime(2026, 1, 1, tzinfo=UTC)}
    split = {"g": "train"}
    left = build_all_legal_candidate_rows([position], dates, split)
    right = build_all_legal_candidate_rows([dict(position, eco="A00", opening_name="Different future")], dates, split)
    assert left == right
    assert not {"opening_eco", "opening_family"} & set(LEGAL_POLICY_FEATURE_FIELDS)
    historical = build_all_legal_candidate_rows([position], dates, split, include_archived_openings=True)
    assert historical[0]["opening_eco"] == "C44"
    assert left[0]["opening_eco"] is None


def test_selection_requires_all_gates_and_positive_paired_evidence():
    baseline = {"exact_move_accuracy": .3, "top_3_accuracy": .5, "top_5_accuracy": .6,
                "negative_log_likelihood": 2.4, "multiclass_brier_score": .8,
                "top_1_expected_calibration_error": .04, "behavior_rate_mean_absolute_error": .05}
    ranking = {"exact_correct": {"paired_game_bootstrap_95_interval": [-.01, .02]}}
    quality = {f"both_different_common_finite/{f}": {"threshold_rate_delta": {
        "estimate": .005, "paired_game_95_interval": [-.01, .02]}} for f in ("human_gap_pp", "predicted_loss_pp")}
    assert select_profile(baseline, baseline, ranking, quality)["selected"] == "safe_population"
    ranking["exact_correct"]["paired_game_bootstrap_95_interval"] = [.001, .02]
    assert select_profile(baseline, baseline, ranking, quality)["selected"] == "opportunity_profile"
    worse = dict(baseline, negative_log_likelihood=2.5)
    assert select_profile(baseline, worse, ranking, quality)["selected"] == "safe_population"


def test_adaptation_preserves_frozen_prior_and_original_and_replaces_history():
    model = OpportunityProfile()
    model.fit_transform_ordered(group("a", 0))
    before = model.to_dict()
    rows = group("b", 1, player="new", chosen_capture=False, split="history")
    cutoff = datetime(2026, 1, 3, tzinfo=UTC)
    adapted = model.with_player_history(rows, before=cutoff)
    assert model.to_dict() == before
    assert adapted.population == model.population
    assert adapted.players["a"] == model.players["a"]
    assert adapted.players["new"]["capture"] == {"opportunities": 1, "chosen": 0}
    assert adapted.with_player_history(rows, before=cutoff).to_dict() == adapted.to_dict()


def test_adaptation_rejects_cutoff_labels_and_duplicates():
    model = OpportunityProfile()
    model.fit_transform_ordered(group("a", 0))
    cutoff = datetime(2026, 1, 3, tzinfo=UTC)
    invalid = [group("b", 2, split="history"), group("b", 1, split="test"),
               group("b", 1, split="history") * 2]
    for rows in invalid:
        with pytest.raises(ValueError):
            model.with_player_history(rows, before=cutoff)
