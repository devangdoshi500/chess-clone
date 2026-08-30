from datetime import UTC, datetime, timedelta

import chess
import numpy as np
import pytest

from chess_clone.features.board import extract_board_state_features
from chess_clone.modeling.candidates import (
    ENGINE_FEATURE_FIELDS,
    FULL_FEATURE_FIELDS,
    LEAKAGE_FIELDS,
    build_candidate_dataset,
    chronological_game_split,
    validate_no_leakage_fields,
)
from chess_clone.modeling.ranker import (
    SparseOneHotPreprocessor,
    baseline_probabilities,
    fit_global_rank_frequencies,
    normalize_candidate_scores,
    predict_candidate_probabilities,
    ranking_metrics,
    validate_candidate_labels,
)


MOVES = ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3"]


def _inputs(count: int = 10, *, outside_last: bool = False):
    games = []
    features = []
    analysis = []
    board = chess.Board()
    board_features = extract_board_state_features(board, chess.WHITE)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for index in range(count):
        game_id = f"game-{index:02d}"
        games.append({"game_id": game_id, "played_at": start + timedelta(days=index)})
        actual = "a2a3" if outside_last and index == count - 1 else MOVES[index % 5]
        features.append(
            {
                "game_id": game_id,
                "ply": 1,
                "move_number": 1,
                "player_username": "Target",
                "player_color": "white",
                "fen": chess.STARTING_FEN,
                "actual_move_uci": actual,
                "actual_move_san": board.san(chess.Move.from_uci(actual)),
                "speed": "blitz",
                "opening_eco": "A00",
                "opening_name": "Uncommon Opening: Test",
                "player_rating": 2500 + index,
                "opponent_rating": 2450,
                "rating_difference": 50 + index,
                "player_clock_seconds_after_move": 179.0,
                "initial_time_seconds": 180,
                "increment_seconds": 0,
                "game_phase": board_features.game_phase,
                "material_balance": board_features.material_balance,
                "legal_move_count": board_features.legal_move_count,
                "player_in_check": board_features.player_in_check,
                "castling_rights_available": board_features.castling_rights_available,
                "approximate_winning_chance_before": 0.5,
            }
        )
        for rank, move in enumerate(MOVES, start=1):
            analysis.append(
                {
                    "game_id": game_id,
                    "ply": 1,
                    "player_username": "Target",
                    "canonical_position": "canonical start",
                    "pv_rank": rank,
                    "score_cp": 50 - rank * 10,
                    "mate_in": None,
                    "best_move_uci": move,
                }
            )
    return games, features, analysis


def test_chronological_split_uses_whole_games_without_leakage() -> None:
    games, _, _ = _inputs()
    split = chronological_game_split(reversed(games))

    assert len(split.train_game_ids) == 7
    assert len(split.validation_game_ids) == 1
    assert len(split.test_game_ids) == 2
    assert not split.train_game_ids & split.validation_game_ids
    assert not split.train_game_ids & split.test_game_ids
    assert not split.validation_game_ids & split.test_game_ids
    assert max(split.game_dates[x] for x in split.train_game_ids) <= min(
        split.game_dates[x] for x in split.validation_game_ids
    )
    assert max(split.game_dates[x] for x in split.validation_game_ids) <= min(
        split.game_dates[x] for x in split.test_game_ids
    )


def test_candidate_generation_has_one_positive_and_keeps_outside_for_reporting() -> None:
    games, features, analysis = _inputs(outside_last=True)
    dataset = build_candidate_dataset(
        features, analysis, chronological_game_split(games)
    )

    assert dataset.total_decisions == 10
    assert dataset.inside_top_5_decisions == 9
    assert dataset.outside_top_5_decisions == 1
    assert len(dataset.decisions) == 10
    assert len(dataset.candidate_rows) == 45
    assert sum(not row["usable"] for row in dataset.decisions) == 1
    validate_candidate_labels(dataset.candidate_rows)
    positives = {}
    for row in dataset.candidate_rows:
        positives.setdefault(row["decision_id"], 0)
        positives[row["decision_id"]] += int(row["chosen"])
        assert row["candidate_move_uci"] in MOVES
        assert row["candidate_piece_moved"] in {"pawn", "knight"}
        assert row["candidate_source_square"] in {"e2", "d2", "g1", "c2", "b1"}
        assert row["candidate_destination_square"] in {"e4", "d4", "f3", "c4", "c3"}
        assert row["candidate_manhattan_displacement"] >= 1
    assert set(positives.values()) == {1}


def test_model_feature_lists_exclude_every_declared_leakage_field() -> None:
    assert not set(ENGINE_FEATURE_FIELDS) & LEAKAGE_FIELDS
    assert not set(FULL_FEATURE_FIELDS) & LEAKAGE_FIELDS
    validate_no_leakage_fields(FULL_FEATURE_FIELDS)
    with pytest.raises(ValueError, match="leakage fields"):
        validate_no_leakage_fields((*FULL_FEATURE_FIELDS, "centipawn_loss"))


def test_candidate_probability_normalization_and_one_decision_inference() -> None:
    games, features, analysis = _inputs()
    rows = build_candidate_dataset(
        features, analysis, chronological_game_split(games)
    ).candidate_rows[:5]
    preprocessor = SparseOneHotPreprocessor(("engine_rank",)).fit(rows)

    class FakeModel:
        def predict_proba(self, matrix):
            positive = np.arange(1, matrix.shape[0] + 1, dtype=float)
            positive /= positive.max() + 1
            return np.column_stack((1 - positive, positive))

    probabilities = predict_candidate_probabilities(FakeModel(), preprocessor, rows)
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(0 <= probability <= 1 for probability in probabilities)

    two_decisions = rows + [dict(row, decision_id="other") for row in rows]
    normalized = normalize_candidate_scores(two_decisions, [1.0] * 10)
    assert sum(normalized[:5]) == pytest.approx(1.0)
    assert sum(normalized[5:]) == pytest.approx(1.0)


def test_stockfish_and_global_rank_baselines_follow_rank_order() -> None:
    games, features, analysis = _inputs()
    rows = build_candidate_dataset(
        features, analysis, chronological_game_split(games)
    ).candidate_rows[:15]
    frequencies = fit_global_rank_frequencies(rows)

    stockfish = ranking_metrics(
        rows,
        baseline_probabilities(
            rows, baseline="stockfish", rank_frequencies=frequencies
        ),
    )
    global_rank = ranking_metrics(
        rows,
        baseline_probabilities(
            rows,
            baseline="global_rank_frequency",
            rank_frequencies=frequencies,
        ),
    )

    assert stockfish["exact_move_accuracy"] == pytest.approx(1 / 3)
    assert stockfish["top_3_accuracy"] == pytest.approx(1.0)
    assert global_rank["maximum_probability_sum_error"] < 1e-12

    historical_counts = {"canonical start": {"d2d4": 10}}
    historical = ranking_metrics(
        rows,
        baseline_probabilities(
            rows,
            baseline="historical_exact_position",
            rank_frequencies=frequencies,
            historical_counts=historical_counts,
        ),
    )
    assert historical["accuracy_by_actual_stockfish_rank"]["2"]["accuracy"] == 1.0


def test_duplicate_or_missing_positive_candidate_is_rejected() -> None:
    rows = [
        {"decision_id": "one", "chosen": False},
        {"decision_id": "one", "chosen": False},
    ]
    with pytest.raises(ValueError, match="exactly one positive"):
        validate_candidate_labels(rows)
