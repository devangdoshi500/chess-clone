from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chess_clone.modeling import HistoricalMoveModel, canonical_position_key

STARTING_FEN = chess.STARTING_FEN
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
AFTER_D4 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"


def test_canonical_fen_ignores_only_move_counters() -> None:
    changed_counters = STARTING_FEN.rsplit(" ", 2)[0] + " 37 99"

    assert canonical_position_key(STARTING_FEN) == canonical_position_key(
        changed_counters
    )
    assert canonical_position_key(STARTING_FEN).split() == [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        "w",
        "KQkq",
        "-",
    ]


@pytest.mark.parametrize(
    "changed_fen",
    [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",
        AFTER_E4,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1",
    ],
)
def test_canonical_fen_preserves_position_state(changed_fen: str) -> None:
    assert canonical_position_key(STARTING_FEN) != canonical_position_key(changed_fen)


def test_canonical_fen_preserves_non_legal_en_passant_target() -> None:
    assert canonical_position_key(AFTER_E4).endswith(" b KQkq e3")
    assert canonical_position_key(AFTER_D4).endswith(" b KQkq d3")
    assert canonical_position_key(AFTER_E4) != canonical_position_key(
        AFTER_E4.replace(" e3 ", " - ")
    )


def test_distribution_probabilities_normalize_and_sort() -> None:
    model = HistoricalMoveModel.from_observations(
        [(STARTING_FEN, "e2e4")] * 3 + [(STARTING_FEN, "d2d4")]
    )

    distribution = model.get_move_distribution(STARTING_FEN)

    assert distribution == [
        {"move_uci": "e2e4", "count": 3, "probability": 0.75},
        {"move_uci": "d2d4", "count": 1, "probability": 0.25},
    ]
    assert sum(float(record["probability"]) for record in distribution) == 1.0


def test_summary_counts_repeated_and_single_observation_positions() -> None:
    model = HistoricalMoveModel.from_observations(
        [(STARTING_FEN, "e2e4")] * 2
        + [(STARTING_FEN, "d2d4")]
        + [(AFTER_E4, "c7c5")]
    )

    summary = model.summary
    assert summary.total_observations == 4
    assert summary.unique_positions == 2
    assert summary.repeated_positions == 1
    assert summary.average_observations_per_position == 2.0
    assert summary.max_observations_for_one_position == 3
    assert summary.repeated_position_records_percentage == 75.0
    assert model.get_move_distribution(AFTER_E4) == [
        {"move_uci": "c7c5", "count": 1, "probability": 1.0}
    ]


def test_seeded_sampling_is_deterministic() -> None:
    model = HistoricalMoveModel.from_observations(
        [(STARTING_FEN, "e2e4")] * 3 + [(STARTING_FEN, "d2d4")]
    )

    assert model.sample_move(STARTING_FEN, seed=12) == model.sample_move(
        STARTING_FEN, seed=12
    )
    assert model.sample_move(STARTING_FEN, seed=12) in {"e2e4", "d2d4"}


def test_unknown_position_returns_empty_distribution_and_no_sample() -> None:
    model = HistoricalMoveModel.from_observations([(STARTING_FEN, "e2e4")])

    assert model.get_move_distribution(AFTER_E4) == []
    assert model.sample_move(AFTER_E4, seed=1) is None


def test_loads_only_requested_player_from_parquet_case_insensitively(
    tmp_path: Path,
) -> None:
    path = tmp_path / "positions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "fen": STARTING_FEN,
                    "actual_move_uci": "e2e4",
                    "player_username": "TargetPlayer",
                },
                {
                    "fen": STARTING_FEN,
                    "actual_move_uci": "d2d4",
                    "player_username": "other",
                },
            ]
        ),
        path,
    )

    model = HistoricalMoveModel.from_parquet(path, "targetplayer")

    assert model.summary.total_observations == 1
    assert model.get_move_distribution(STARTING_FEN) == [
        {"move_uci": "e2e4", "count": 1, "probability": 1.0}
    ]
