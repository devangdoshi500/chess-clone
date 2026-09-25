from datetime import UTC, datetime

import pytest

from chess_clone.experiments.population_metrics import (
    legal_policy_metrics,
    move_frequency_probabilities,
    uniform_probabilities,
)
from chess_clone.modeling.legal_policy import (
    PLAYER_TENDENCY_FIELDS,
    PlayerTendencyEncoder,
    build_all_legal_candidate_rows,
)
from chess_clone.experiments.population_policy import _cap_player_positions


def _position(
    game_id: str = "game-a", player: str = "Human", actual: str = "e2e4"
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "ply": 1,
        "move_number": 1,
        "player_username": player,
        "player_color": "white",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "actual_move_uci": actual,
        "player_rating": 1500,
        "opponent_rating": 1510,
        "speed": "blitz",
        "time_control": "180+0",
        "eco": "B00",
        "opening_name": "King's Pawn",
    }


def _rows(*positions: dict[str, object]) -> list[dict[str, object]]:
    dates = {
        str(position["game_id"]): datetime(2026, 1, index + 1, tzinfo=UTC)
        for index, position in enumerate(positions)
    }
    splits = {game_id: "train" for game_id in dates}
    return build_all_legal_candidate_rows(positions, dates, splits)


def test_all_legal_rows_include_actual_and_do_not_use_engine_gate() -> None:
    rows = _rows(_position())

    assert len(rows) == 20
    assert sum(bool(row["chosen"]) for row in rows) == 1
    assert {str(row["candidate_move_uci"]) for row in rows} == {
        "a2a3", "a2a4", "b1a3", "b1c3", "b2b3", "b2b4", "c2c3", "c2c4",
        "d2d3", "d2d4", "e2e3", "e2e4", "f2f3", "f2f4", "g1f3", "g1h3",
        "g2g3", "g2g4", "h2h3", "h2h4",
    }
    assert all("engine_rank" not in row for row in rows)


def test_player_tendencies_are_ordered_and_label_safe() -> None:
    training = _rows(
        _position("game-a", actual="e2e4"),
        _position("game-b", actual="g1f3"),
    )
    encoder = PlayerTendencyEncoder(smoothing_strength=6)
    encoded = encoder.fit_transform_ordered(training)
    first = [row for row in encoded if row["decision_id"] == "game-a:1"]
    second = [row for row in encoded if row["decision_id"] == "game-b:1"]

    assert {row["history_observations"] for row in first} == {0}
    assert {row["history_observations"] for row in second} == {1}
    validation = _rows(_position("game-c", actual="d2d4"))
    before = encoder.transform(validation)
    changed = [dict(row, chosen=not bool(row["chosen"])) for row in validation]
    after = encoder.transform(changed)
    for left, right in zip(before, after, strict=True):
        assert [left[field] for field in PLAYER_TENDENCY_FIELDS] == [
            right[field] for field in PLAYER_TENDENCY_FIELDS
        ]


def test_behavior_metrics_give_partial_credit_to_similar_moves() -> None:
    rows = _rows(_position(actual="e2e4"))
    probabilities = [0.0] * len(rows)
    predicted_index = next(
        index for index, row in enumerate(rows) if row["candidate_move_uci"] == "d2d4"
    )
    actual_index = next(
        index for index, row in enumerate(rows) if row["candidate_move_uci"] == "e2e4"
    )
    probabilities[predicted_index] = 0.6
    probabilities[actual_index] = 0.4
    metrics, predictions = legal_policy_metrics(rows, probabilities)

    assert metrics["exact_move_accuracy"] == 0
    assert metrics["top_5_accuracy"] == 1
    assert metrics["mean_actual_rank"] > 1
    assert 0 < metrics["mean_normalized_rank_score"] < 1
    assert metrics["multiclass_brier_score"] == pytest.approx(0.72)
    assert metrics["top_1_expected_calibration_error"] == pytest.approx(0.6)
    assert metrics["piece_match_rate"] == 1
    assert metrics["destination_wing_match_rate"] == 1
    assert metrics["mean_behavior_similarity"] > 0.8
    assert predictions[0]["actual_move_uci"] == "e2e4"
    assert predictions[0]["predicted_move_uci"] == "d2d4"


def test_candidate_rows_include_post_move_tactical_features() -> None:
    position = _position(actual="e2e4")
    position["fen"] = "4k3/8/8/8/8/8/4p3/4R1K1 w - - 0 1"
    position["actual_move_uci"] = "e1e2"
    position["move_number"] = 1
    rows = _rows(position)
    capture = next(row for row in rows if row["candidate_move_uci"] == "e1e2")

    assert capture["candidate_captured_piece_type"] == "pawn"
    assert capture["candidate_capture_value"] == 1
    assert capture["candidate_material_gain"] == 1
    assert isinstance(capture["candidate_opponent_mobility_after"], int)
    assert isinstance(capture["candidate_attackers_after"], int)


def test_uniform_and_frequency_baselines_normalize_by_decision() -> None:
    train = _rows(_position("game-a", actual="e2e4"))
    evaluation = _rows(_position("game-b", actual="d2d4"))
    for values in (
        uniform_probabilities(evaluation),
        move_frequency_probabilities(train, evaluation),
    ):
        assert sum(values) == pytest.approx(1.0)
    frequency = move_frequency_probabilities(train, evaluation)
    e4 = next(
        value
        for row, value in zip(evaluation, frequency, strict=True)
        if row["candidate_move_uci"] == "e2e4"
    )
    d4 = next(
        value
        for row, value in zip(evaluation, frequency, strict=True)
        if row["candidate_move_uci"] == "d2d4"
    )
    assert e4 > d4


def test_decision_cap_never_splits_a_game() -> None:
    dates = {
        "old": datetime(2026, 1, 1, tzinfo=UTC),
        "new": datetime(2026, 1, 2, tzinfo=UTC),
    }
    splits = {"old": "test", "new": "test"}
    positions = [
        {"game_id": game_id, "ply": ply}
        for game_id in ("old", "new")
        for ply in range(1, 11, 2)
    ]

    selected = _cap_player_positions(positions, dates, splits, maximum=34)

    selected_ids = {str(row["game_id"]) for row in selected}
    assert selected_ids == {"new"}
    assert len(selected) == 5
