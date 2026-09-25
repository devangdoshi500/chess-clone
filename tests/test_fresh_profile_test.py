from datetime import UTC, datetime, timedelta

import pytest

from chess_clone.experiments.fresh_profile_test import cap_complete_games, eligible_fresh_positions, frequency_probabilities


def test_cap_preserves_complete_games_and_does_not_pack_smaller_later_games():
    dates = {g: datetime(2026, 1, d, tzinfo=UTC) for g, d in (("a", 1), ("b", 2), ("c", 3))}
    rows = [{"game_id": g, "ply": i} for g, n in (("a", 3), ("b", 5), ("c", 1)) for i in range(n)]
    kept = cap_complete_games(list(reversed(rows)), dates, cap=5)
    assert len(kept) == 3
    assert {r["game_id"] for r in kept} == {"a"}


def test_fresh_filters_reject_old_future_out_of_rating_and_wrong_player():
    after = datetime(2026, 1, 1, tzinfo=UTC)
    until = after + timedelta(days=2)
    games = [{"game_id": name, "played_at": date, "rated": True, "variant": "Standard", "speed": "blitz"}
             for name, date in (("boundary", after), ("old", after + timedelta(days=1)),
                                ("valid", after + timedelta(days=1)), ("future", until + timedelta(days=1)),
                                ("low", until), ("other", until))]
    positions = [{"game_id": g["game_id"], "player_username": "A", "player_rating": 1500} for g in games]
    positions[-2]["player_rating"] = 1295
    positions[-1]["player_username"] = "B"
    rows, _ = eligible_fresh_positions(positions, games, player="a", after=after, until=until, old_ids={"old"})
    assert [r["game_id"] for r in rows] == ["valid"]


def test_saved_frequency_probabilities_normalize_per_decision():
    rows = [{"decision_id": "a", "candidate_move_uci": m} for m in ("e2e4", "d2d4")]
    rows += [{"decision_id": "b", "candidate_move_uci": "a2a3"}]
    assert frequency_probabilities(rows, {"e2e4": 3}) == pytest.approx([.8, .2, 1.])
