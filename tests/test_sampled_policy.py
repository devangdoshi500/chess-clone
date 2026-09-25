import pytest

from chess_clone.experiments.sampled_policy import (
    sample_move,
    selected_game_ids,
    shared_uniform,
)


def test_game_selection_is_stable_and_whole_game():
    rows = [{"game_id": game} for game in ("a", "b", "c") for _ in range(3)]
    first = selected_game_ids(rows, player="P", seed=7, count=2)
    assert first == selected_game_ids(list(reversed(rows)), player="p", seed=7, count=2)
    assert len(first) == 2


def test_shared_uniform_and_inverse_cdf_are_deterministic():
    value = shared_uniform(42, "game:1", 0)
    assert value == shared_uniform(42, "game:1", 0)
    assert 0 <= value < 1
    distribution = {"b": 0.6, "a": 0.4}
    assert sample_move(distribution, 0.0) == "a"
    assert sample_move(distribution, 0.399) == "a"
    assert sample_move(distribution, 0.4) == "b"


def test_sampling_rejects_invalid_distributions():
    with pytest.raises(ValueError, match="sum"):
        sample_move({"a": 0.5}, 0.2)
    with pytest.raises(ValueError, match="required"):
        sample_move({}, 0.2)
