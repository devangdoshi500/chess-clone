import json

import pytest

from chess_clone.modeling.boosted import (
    candidate_pool,
    fit_temperature,
    groupwise_softmax,
)
from chess_clone.modeling.boosted_training import BOOSTED_FEATURE_ABLATIONS
from chess_clone.modeling.candidates import LEAKAGE_FIELDS, validate_no_leakage_fields
from chess_clone.modeling.player_history import (
    PLAYER_HISTORY_FEATURE_FIELDS,
    PlayerHistoryEncoder,
)
from chess_clone.modeling.ranker import (
    fit_historical_move_counts,
    ranking_metrics,
)


def _decision(
    decision_id: str,
    chosen_rank: int,
    *,
    phase: str = "opening",
    pressure: bool = False,
    opening: str = "C20",
) -> list[dict[str, object]]:
    rows = []
    for rank in range(1, 4):
        rows.append(
            {
                "decision_id": decision_id,
                "game_id": decision_id.split(":")[0],
                "chosen": rank == chosen_rank,
                "engine_rank": rank,
                "engine_evaluation": 50 - rank * 10,
                "evaluation_difference_from_best": (rank - 1) * 10,
                "candidate_piece_moved": ("pawn", "knight", "bishop")[rank - 1],
                "candidate_move_uci": ("e2e4", "g1f3", "f1c4")[rank - 1],
                "candidate_is_capture": rank == 2,
                "candidate_gives_check": rank == 3,
                "candidate_trades_queens": False,
                "candidate_is_castle": False,
                "game_phase": phase,
                "pre_move_time_pressure": pressure,
                "opening_eco": opening,
                "player_color": "white",
                "time_control": "180+0",
            }
        )
    return rows


def test_player_history_is_ordered_and_frozen_before_validation(tmp_path) -> None:
    train = _decision("train-a:1", 1) + _decision("train-b:1", 2)
    validation = _decision("validation:1", 3)
    encoder = PlayerHistoryEncoder(smoothing_strength=5, opening_min_samples=2)
    ordered = encoder.fit_transform_ordered(train)

    assert ordered[0]["history_total_observations"] == 0
    assert ordered[3]["history_total_observations"] == 1
    assert encoder.fit_decision_ids == {"train-a:1", "train-b:1"}

    first = encoder.transform(validation)
    labels_changed = [dict(row, chosen=not bool(row["chosen"])) for row in validation]
    second = encoder.transform(labels_changed)
    for left, right in zip(first, second, strict=True):
        assert [left[field] for field in PLAYER_HISTORY_FEATURE_FIELDS] == [
            right[field] for field in PLAYER_HISTORY_FEATURE_FIELDS
        ]

    path = tmp_path / "history.json"
    encoder.save(path)
    payload = json.loads(path.read_text())
    assert payload["fit_decision_ids"] == ["train-a:1", "train-b:1"]
    assert "validation:1" not in payload["fit_decision_ids"]


def test_grouped_pool_preserves_decisions_and_requires_one_positive() -> None:
    rows = _decision("game-a:1", 1) + _decision("game-b:1", 2)
    pool = candidate_pool(
        rows,
        ("engine_rank", "candidate_piece_moved"),
        grouped=True,
    )
    hashes = list(pool.get_group_id_hash())
    assert len(set(hashes[:3])) == 1
    assert len(set(hashes[3:])) == 1
    assert hashes[0] != hashes[3]

    invalid = [dict(row, chosen=False) for row in rows]
    with pytest.raises(ValueError, match="exactly one positive"):
        PlayerHistoryEncoder().fit(invalid)


def test_historical_lookup_contains_only_passed_training_decisions() -> None:
    training = [
        {"canonical_position": "train-position", "actual_move_uci": "e2e4"},
        {"canonical_position": "train-position", "actual_move_uci": "d2d4"},
    ]
    validation = {
        "canonical_position": "validation-position",
        "actual_move_uci": "g1f3",
    }
    lookup = fit_historical_move_counts(training)

    assert lookup["train-position"] == {"e2e4": 1, "d2d4": 1}
    assert validation["canonical_position"] not in lookup


def test_boosted_ablation_features_exclude_leakage() -> None:
    for fields in BOOSTED_FEATURE_ABLATIONS.values():
        validate_no_leakage_fields(fields)
        assert not set(fields) & LEAKAGE_FIELDS


def test_grouped_inference_and_temperature_calibration_are_normalized() -> None:
    rows = _decision("one:1", 2) + _decision("two:1", 3)
    scores = [3.0, 2.0, 1.0, 0.5, 1.0, 2.0]
    before = groupwise_softmax(rows, scores)
    temperature = fit_temperature(rows, scores)
    after = groupwise_softmax(rows, scores, temperature=temperature)

    assert temperature > 0
    assert sum(before[:3]) == pytest.approx(1.0)
    assert sum(before[3:]) == pytest.approx(1.0)
    assert sum(after[:3]) == pytest.approx(1.0)
    assert sum(after[3:]) == pytest.approx(1.0)
    assert max(range(3), key=before.__getitem__) == max(
        range(3), key=after.__getitem__
    )
    metrics = ranking_metrics(rows, after)
    assert metrics["decision_count"] == 2
    assert metrics["maximum_probability_sum_error"] < 1e-12
