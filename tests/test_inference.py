from datetime import UTC, datetime
import copy

import chess
import pytest

from chess_clone.modeling.inference import DeepStylePredictor, PredictionContext
from chess_clone.modeling.legal_policy import (
    LEGAL_POLICY_FEATURE_FIELDS,
    build_all_legal_candidate_rows,
    build_all_legal_inference_rows,
)
from chess_clone.modeling.style_residual import DIMENSION, VERSION


def context(player="known"):
    return PredictionContext(
        player_username=player,
        player_rating=1500,
        opponent_rating=1520,
        speed="blitz",
        time_control="180+0",
    )


def predictor(monkeypatch):
    instance = object.__new__(DeepStylePredictor)
    instance.source = None
    instance.feature_fields = ("candidate_is_capture",)
    instance.temperature = 1.0
    instance.base_model = object()
    zero = {"version": VERSION, "coefficients": [0.0] * DIMENSION}
    instance.shared_model = zero
    instance.personal_models = {"known": zero}
    instance.display_names = {"known": "known"}
    instance.load_latency_ms = 0.0
    monkeypatch.setattr(
        "chess_clone.modeling.inference.predict_relevance_scores",
        lambda model, rows, fields: [float(row["candidate_is_capture"]) for row in rows],
    )
    return instance


def test_inference_rows_are_label_free_and_cover_special_legal_moves():
    rows = build_all_legal_inference_rows(
        "r3k2r/1P3ppp/8/3pP3/8/8/P4PPP/R3K2R w KQkq d6 0 20",
        player_username="p",
        player_rating=1500,
        opponent_rating=1510,
        speed="blitz",
        time_control="180+0",
    )
    moves = {row["candidate_move_uci"] for row in rows}
    assert not any("chosen" in row or "actual_move_uci" in row for row in rows)
    assert "e1g1" in moves
    assert "e1c1" in moves
    assert "e5d6" in moves
    assert {"b7b8q", "b7b8r", "b7b8b", "b7b8n"} <= moves


def test_inference_features_match_replay_construction_without_labels():
    position = {
        "game_id": "g",
        "ply": 1,
        "move_number": 1,
        "fen": chess.STARTING_FEN,
        "player_username": "known",
        "player_color": "white",
        "actual_move_uci": "e2e4",
        "player_rating": 1500,
        "opponent_rating": 1520,
        "speed": "blitz",
        "time_control": "180+0",
    }
    date = datetime(2026, 1, 1, tzinfo=UTC)
    replay = build_all_legal_candidate_rows([position], {"g": date}, {"g": "test"})
    inference = build_all_legal_inference_rows(
        chess.STARTING_FEN,
        player_username="known",
        player_rating=1500,
        opponent_rating=1520,
        speed="blitz",
        time_control="180+0",
        decision_id="g:1",
    )
    assert [row["candidate_move_uci"] for row in inference] == [
        row["candidate_move_uci"] for row in replay
    ]
    for left, right in zip(inference, replay, strict=True):
        assert {field: left[field] for field in LEGAL_POLICY_FEATURE_FIELDS} == {
            field: right[field] for field in LEGAL_POLICY_FEATURE_FIELDS
        }


def test_predict_is_deterministic_normalized_and_preserves_full_probabilities(monkeypatch):
    model = predictor(monkeypatch)
    before = copy.deepcopy(model.personal_models)
    first = model.predict(chess.STARTING_FEN, context(), policy="auto", top_k=3)
    second = model.predict(chess.STARTING_FEN, context(), policy="auto", top_k=3)
    assert first.policy == "personal"
    assert first.personal_model_available
    assert first.legal_move_count == 20
    assert first.probability_sum == pytest.approx(1.0)
    assert [(m.uci, m.probability) for m in first.moves] == [
        (m.uci, m.probability) for m in second.moves
    ]
    assert len(first.moves) == 3
    assert sum(move.probability for move in first.moves) < 1.0
    assert model.personal_models == before


def test_unknown_player_falls_back_and_explicit_personal_fails(monkeypatch):
    model = predictor(monkeypatch)
    result = model.predict(chess.STARTING_FEN, context("new"), policy="auto")
    assert result.policy == "population"
    assert not result.personal_model_available
    with pytest.raises(ValueError, match="No personal residual"):
        model.predict(chess.STARTING_FEN, context("new"), policy="personal")


def test_terminal_and_invalid_positions(monkeypatch):
    model = predictor(monkeypatch)
    terminal = model.predict(
        "7k/5Q2/7K/8/8/8/8/8 b - - 0 1", context(), top_k=5
    )
    assert terminal.terminal
    assert terminal.terminal_reason == "stalemate"
    assert terminal.moves == ()
    with pytest.raises(ValueError, match="Invalid"):
        model.predict("not a fen", context())


def test_missing_context_uses_explicit_missing_values(monkeypatch):
    model = predictor(monkeypatch)
    result = model.predict(
        chess.STARTING_FEN,
        PredictionContext(None, None, None, None, None),
        policy="population",
    )
    assert result.legal_move_count == 20
    assert result.probability_sum == pytest.approx(1.0)


def test_loader_reports_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        DeepStylePredictor(tmp_path)


def test_check_evasion_rows_only_include_legal_moves():
    fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
    rows = build_all_legal_inference_rows(
        fen,
        player_username=None,
        player_rating=None,
        opponent_rating=None,
        speed=None,
        time_control=None,
    )
    board = chess.Board(fen)
    assert board.is_check()
    assert {row["candidate_move_uci"] for row in rows} == {
        move.uci() for move in board.legal_moves
    }
