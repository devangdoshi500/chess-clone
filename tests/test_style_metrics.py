from datetime import UTC, datetime, timedelta

import pytest

from chess_clone.experiments.style_metrics import STYLE_FIELDS, style_metrics
from chess_clone.experiments.style_adaptation import history_candidates, split_history, paired_player_interval


def group(decision="d", player="p", chosen=0):
    return [{"decision_id": decision, "game_id": decision, "game_phase": "opening",
             "player_username": player, "candidate_move_uci": str(i), "chosen": i == chosen,
             **{field: bool(i) for field in STYLE_FIELDS}} for i in range(2)]


def test_full_distribution_can_match_style_when_argmax_does_not():
    result = style_metrics(group("a") + group("b", chosen=1), [.5] * 4, min_opportunities=2)
    assert result["macro_policy_tv"] == 0
    assert result["players"]["p"]["argmax_tv"] == .5
    assert result["players"]["p"]["attribute_brier"] == .5


def test_forced_attributes_excluded_not_credited_as_style_success():
    rows = group()
    for row in rows:
        for field in STYLE_FIELDS:
            row[field] = False
    result = style_metrics(rows, [.5, .5], min_opportunities=1)
    assert result["macro_policy_tv"] is None
    assert all(c["forced_excluded"] == 1 and c["opportunities"] == 0 for c in result["cells"])


def test_player_macro_does_not_weight_prolific_player_more():
    rows = group(player="a") + sum((group(str(i), player="b") for i in range(9)), [])
    result = style_metrics(rows, [0, 1] + [1, 0] * 9, min_opportunities=1)
    assert result["macro_policy_tv"] == .5


@pytest.mark.parametrize("probabilities", [[.2, .2], [float("nan"), .5], [-.1, 1.1], [1]])
def test_invalid_probabilities_rejected(probabilities):
    with pytest.raises(ValueError):
        style_metrics(group(), probabilities)


def test_inconsistent_metadata_and_duplicate_candidates_rejected():
    for field, value in [("player_username", "other"), ("candidate_move_uci", "0")]:
        rows = group()
        rows[1][field] = value
        with pytest.raises(ValueError):
            style_metrics(rows, [.5, .5])


def test_whole_game_split_and_timestamp_ties():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    dates = {str(i): start + timedelta(days=i // 2) for i in range(10)}
    positions = [{"game_id": str(i)} for i in range(10)] * 2
    history, later, cutoff = split_history(positions, dates)
    assert len(history) == 8 and len(later) == 12
    assert all(dates[r["game_id"]] < cutoff for r in history)
    assert not {r["game_id"] for r in history} & {r["game_id"] for r in later}


def test_player_bootstrap_reproducible():
    assert paired_player_interval([-.1, .2]) == paired_player_interval([-.1, .2])
    assert paired_player_interval([.1] * 4)["paired_player_95_interval"] == pytest.approx([.1, .1])


def test_lightweight_history_attributes_match_training_builder():
    import chess
    from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows
    board = chess.Board()
    for uci in ["e2e4", "d7d5"]:
        board.push_uci(uci)
    position = {"game_id": "g", "ply": 3, "move_number": 2, "player_username": "p",
                "player_color": "white", "fen": board.fen(), "actual_move_uci": "e4d5"}
    dates = {"g": datetime(2026, 1, 1, tzinfo=UTC)}
    lightweight = history_candidates([position], dates)
    full = build_all_legal_candidate_rows([position], dates, {"g": "history"})
    expected = {r["candidate_move_uci"]: r for r in full}
    assert len(lightweight) == len(full)
    for row in lightweight:
        assert all(value == expected[row["candidate_move_uci"]][key] for key, value in row.items())
