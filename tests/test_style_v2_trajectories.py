"""Focused invariants for the frozen matched-trajectory study."""

import chess
import pytest

from chess_clone.experiments.style_v2_trajectories import (
    FAMILIES, behavior_distance, color_for, identity_score, paired_interval,
    style_vector, uniform,
)


def _game(key, color, uci, *, piece="pawn", wing="center", capture=False):
    return {"game_key": key, "color": color,
            "last_target_pawns": {"remaining_pawns": 8, "doubled_pawns": 0,
                                  "isolated_pawns": 0, "pawn_islands": 1},
            "moves": [{"move_uci": uci, "piece": piece, "wing": wing,
                       "phase": "opening", "is_capture": capture, "is_check": False,
                       "is_castle": False, "is_queen_trade": False,
                       "is_pawn_push": piece == "pawn", "is_queen_move": piece == "queen",
                       "is_opponent_territory": False, "is_king_zone": False}]}


def test_matched_randomness_and_color_balance():
    assert uniform(123, "alpha", 0, 1, "target") == uniform(123, "alpha", 0, 1, "target")
    assert uniform(123, "alpha", 0, 1, "target") != uniform(123, "alpha", 0, 1, "opponent")
    colors = [color_for(123, "alpha", index) for index in range(25)]
    assert sorted((colors.count(chess.WHITE), colors.count(chess.BLACK))) == [12, 13]


def test_style_vector_uses_color_and_transition_denominators():
    white = _game("a", "white", "e2e4")
    black = _game("b", "black", "c7c5", wing="queenside")
    black["moves"].append({**black["moves"][0], "move_uci": "g8f6", "piece": "knight",
                           "wing": "kingside", "is_capture": True})
    vector, support = style_vector([white, black])
    assert support == {"assigned_games": 2, "games_with_target_moves": 2, "target_moves": 3,
                       "target_transitions": 1, "post_target_pawn_states": 2,
                       "white_openings": 1, "black_openings": 1}
    assert vector["opening_choice"]["white_e4"] == 1
    assert vector["opening_choice"]["black_c5"] == 1
    assert vector["behavior"]["capture"] == pytest.approx(1 / 3)
    assert vector["sequence"]["repeat_piece"] == 0


def test_family_balanced_distance_and_identity_rank():
    target, _ = style_vector([_game("a", "white", "e2e4")])
    other, _ = style_vector([_game("b", "white", "d2d4", piece="queen")])
    scale = {family: {field: 1.0 for field in fields} for family, fields in FAMILIES.items()}
    reference = {"scale": scale, "centroids": {"alpha": target, "beta": other}}
    assert identity_score(target, "alpha", reference)["target_rank_error"] == 0
    assert identity_score(target, "beta", reference)["target_rank_error"] == 1
    distance = behavior_distance(target, other, scale)
    assert distance["mean"] == pytest.approx(sum(distance["families"].values()) / 7)


def test_paired_interval_uses_players_as_unit():
    result = paired_interval([0.1] * 14, 12)
    assert result["players"] == 14
    assert result["mean_delta"] == pytest.approx(0.1)
    assert result["paired_player_95_interval"] == pytest.approx([0.1, 0.1])
