import chess
import pytest

from chess_clone.experiments.generated_rollout import (
    compare_to_human,
    pawn_structure,
    rollout_uniform,
    trajectory_summary,
)


def test_rollout_uniform_is_stable_and_bounded():
    value = rollout_uniform(4, "Player", 1, 8, "target")
    assert value == rollout_uniform(4, "player", 1, 8, "target")
    assert 0 <= value < 1


def test_pawn_structure_counts_doubled_isolated_and_islands():
    board = chess.Board("8/8/8/8/8/P7/P1P5/4K2k w - - 0 1")
    assert pawn_structure(board, chess.WHITE) == {
        "remaining_pawns": 3,
        "doubled_pawns": 1,
        "isolated_pawns": 3,
        "pawn_islands": 2,
    }


def test_trajectory_summary_and_comparison():
    records = [
        {"game_key": "g", "ply": 1, "phase": "opening", "piece": "pawn", "wing": "center",
         "is_capture": False, "is_check": False, "is_castle": False, "is_promotion": False, "is_queen_trade": False},
        {"game_key": "g", "ply": 3, "phase": "opening", "piece": "pawn", "wing": "kingside",
         "is_capture": True, "is_check": False, "is_castle": False, "is_promotion": False, "is_queen_trade": False},
    ]
    games = [{"terminal": True, "truncated": False, "plies": 20, "result": "1-0",
              "pawn_structure": {"remaining_pawns": 7, "doubled_pawns": 1, "isolated_pawns": 0, "pawn_islands": 1}}]
    report = trajectory_summary(records, games)
    assert report["boolean_rates"]["is_capture"] == 0.5
    assert report["repeated_piece_transition_rate"] == 1
    comparison = compare_to_human(report, report)
    assert all(value == pytest.approx(0) for value in comparison.values())
