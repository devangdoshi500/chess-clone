from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chess_clone.features.board import (
    classify_game_phase,
    extract_board_state_features,
    extract_move_behavior_features,
)
from chess_clone.features.evaluation import (
    approximate_winning_chance,
    derive_engine_rank_features,
    evaluation_to_centipawns,
)
from chess_clone.features.pipeline import build_behavior_features
from chess_clone.features.summary import summarize_behavior_features
from chess_clone.features.time import (
    TimePressureThresholds,
    derive_time_features,
)
from chess_clone.ingestion.storage import save_positions_parquet
from chess_clone.models import PositionRecord


def test_board_features_use_target_player_perspective_for_both_colors() -> None:
    white_board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    black_board = chess.Board("4k3/8/8/8/8/8/8/R3K3 b - - 0 1")

    white = extract_board_state_features(white_board, chess.WHITE)
    black = extract_board_state_features(black_board, chess.BLACK)

    assert white.material_balance == 5
    assert black.material_balance == -5


@pytest.mark.parametrize(
    ("fen", "move", "expected"),
    [
        (
            "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",
            "e2e7",
            {"piece_moved": "rook", "is_check": True},
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "e1g1",
            {"piece_moved": "king", "is_castle": True},
        ),
        (
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "a7a8q",
            {"piece_moved": "pawn", "is_promotion": True},
        ),
        (
            "4k3/8/8/8/8/8/3q4/3QK3 w - - 0 1",
            "d1d2",
            {
                "piece_moved": "queen",
                "is_capture": True,
                "is_queen_trade": True,
                "captured_piece_type": "queen",
            },
        ),
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "e5d6",
            {
                "piece_moved": "pawn",
                "is_capture": True,
                "is_en_passant": True,
                "captured_piece_type": "pawn",
            },
        ),
    ],
)
def test_move_behavior_is_derived_from_the_pre_move_board(
    fen: str, move: str, expected: dict[str, object]
) -> None:
    features = extract_move_behavior_features(chess.Board(fen), move)
    for field, value in expected.items():
        assert getattr(features, field) == value


def test_legal_move_count_and_game_phase_rules() -> None:
    board = chess.Board()
    assert extract_board_state_features(board, chess.WHITE).legal_move_count == 20
    assert classify_game_phase(board) == "opening"

    middlegame = chess.Board()
    for square in (chess.A2, chess.B2, chess.C2, chess.D2, chess.A7, chess.B7):
        middlegame.remove_piece_at(square)
    assert classify_game_phase(middlegame) == "middlegame"
    assert classify_game_phase(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")) == "endgame"


def test_increment_aware_time_and_first_move_null_behavior() -> None:
    thresholds = TimePressureThresholds(fraction=0.10, seconds=30)
    first = derive_time_features(
        time_control="300+3",
        clock_after_move=299,
        previous_player_clock_after_move=None,
        thresholds=thresholds,
    )
    second = derive_time_features(
        time_control="300+3",
        clock_after_move=297,
        previous_player_clock_after_move=299,
        thresholds=thresholds,
    )
    low = derive_time_features(
        time_control="300+3",
        clock_after_move=20,
        previous_player_clock_after_move=25,
        thresholds=thresholds,
    )

    assert first.seconds_spent_on_move is None
    assert second.seconds_spent_on_move == 5
    assert second.time_remaining_fraction == pytest.approx(0.99)
    assert second.low_time is False
    assert low.low_time is True


def test_unreliable_time_control_and_negative_elapsed_time_are_null() -> None:
    features = derive_time_features(
        time_control="-",
        clock_after_move=100,
        previous_player_clock_after_move=90,
        thresholds=TimePressureThresholds(),
    )
    assert features.initial_time_seconds is None
    assert features.increment_seconds is None
    assert features.seconds_spent_on_move is None
    assert features.time_remaining_fraction is None

    impossible = derive_time_features(
        time_control="60+0",
        clock_after_move=61,
        previous_player_clock_after_move=60,
        thresholds=TimePressureThresholds(),
    )
    assert impossible.seconds_spent_on_move is None


def test_winning_chance_is_monotonic_bounded_and_centered() -> None:
    losing = approximate_winning_chance(-200)
    equal = approximate_winning_chance(0)
    winning = approximate_winning_chance(200)
    assert losing is not None and winning is not None
    assert 0 < losing < equal == 0.5 < winning < 1
    assert approximate_winning_chance(None, 1) == pytest.approx(1.0)
    assert approximate_winning_chance(None, -1) == pytest.approx(0.0)


def test_post_move_mate_zero_means_the_target_player_delivered_mate() -> None:
    assert evaluation_to_centipawns(None, 0) == 100_000
    assert approximate_winning_chance(None, 0) == pytest.approx(1.0)


def test_engine_rank_flags_respect_available_multipv() -> None:
    lines = [
        {"pv_rank": rank, "best_move_uci": move, "multipv_requested": 5}
        for rank, move in enumerate(
            ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3"], start=1
        )
    ]
    ranked = derive_engine_rank_features("g1f3", lines)
    assert ranked.actual_move_rank == 3
    assert ranked.actual_move_in_top_1 is False
    assert ranked.actual_move_in_top_3 is True
    assert ranked.actual_move_in_top_5 is True

    one_line = derive_engine_rank_features(
        "d2d4",
        [{"pv_rank": 1, "best_move_uci": "e2e4", "multipv_requested": 1}],
    )
    assert one_line.actual_move_in_top_1 is False
    assert one_line.actual_move_in_top_3 is None
    assert one_line.actual_move_in_top_5 is None


def _position(
    game_id: str,
    ply: int,
    fen: str,
    move_uci: str,
    move_san: str,
    clock: float | None,
) -> PositionRecord:
    return PositionRecord(
        game_id=game_id,
        ply=ply,
        move_number=(ply + 1) // 2,
        player_username="TargetPlayer",
        player_color="white",
        fen=fen,
        actual_move_uci=move_uci,
        actual_move_san=move_san,
        player_rating=2500,
        opponent_rating=2450,
        white_rating=2500,
        black_rating=2450,
        speed="blitz",
        time_control="300+3",
        eco="C20",
        opening_name="King's Pawn Game",
        opening_variation=None,
        clock_seconds_after_move=clock,
    )


def _analysis_lines(position: PositionRecord, actual_rank: int = 2) -> list[dict[str, object]]:
    moves = ["d2d4", position.actual_move_uci, "c2c4", "g1f3", "b1c3"]
    if actual_rank == 1:
        moves[0], moves[1] = moves[1], moves[0]
    return [
        {
            "game_id": position.game_id,
            "ply": position.ply,
            "player_username": position.player_username,
            "source_fen": position.fen,
            "actual_move_uci": position.actual_move_uci,
            "score_perspective": "side_to_move",
            "pv_rank": rank,
            "multipv_requested": 5,
            "score_cp": 80 - rank * 10,
            "mate_in": None,
            "actual_move_score_cp": 25,
            "actual_move_mate_in": None,
            "best_move_uci": move,
        }
        for rank, move in enumerate(moves, start=1)
    ]


def _write_feature_inputs(tmp_path: Path) -> tuple[Path, Path]:
    first = _position("game", 1, chess.STARTING_FEN, "e2e4", "e4", 299)
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    second = _position("game", 3, board.fen(), "g1f3", "Nf3", 297)
    positions = tmp_path / "positions.parquet"
    analysis = tmp_path / "analysis.parquet"
    save_positions_parquet([first, second], positions)
    pq.write_table(
        pa.Table.from_pylist(_analysis_lines(first) + _analysis_lines(second, 1)),
        analysis,
    )
    return positions, analysis


def test_feature_pipeline_has_one_row_per_analyzed_decision_and_summary(
    tmp_path: Path,
) -> None:
    positions, analysis = _write_feature_inputs(tmp_path)
    output = tmp_path / "features.parquet"

    build = build_behavior_features(
        positions, analysis, "targetplayer", output_path=output
    )
    rows = pq.read_table(output).to_pylist()

    assert build.available_position_records == 2
    assert build.analyzed_position_records == build.feature_rows == len(rows) == 2
    assert build.unanalyzed_position_records == 0
    assert rows[0]["seconds_spent_on_move"] is None
    assert rows[1]["seconds_spent_on_move"] == 5
    assert rows[0]["rating_difference"] == 50
    assert rows[0]["engine_eval_before"] == 70
    assert rows[0]["actual_move_eval"] == 25
    assert rows[0]["centipawn_loss"] == 45
    assert rows[0]["actual_move_rank"] == 2
    assert rows[1]["actual_move_rank"] == 1

    summary = summarize_behavior_features(output)
    assert summary.move_count == 2
    assert summary.average_centipawn_loss == 45
    assert summary.top_1_move_rate == 50
    assert summary.top_3_move_rate == 100
    assert summary.average_move_time_seconds == 5


def test_feature_pipeline_reports_unanalyzed_rows_without_silent_row_loss(
    tmp_path: Path,
) -> None:
    positions, analysis = _write_feature_inputs(tmp_path)
    table = pq.read_table(analysis)
    pq.write_table(table.filter(pa.compute.equal(table["ply"], 1)), analysis)

    summary = build_behavior_features(
        positions, analysis, "TARGETPLAYER", output_path=tmp_path / "features.parquet"
    )
    assert summary.available_position_records == 2
    assert summary.feature_rows == 1
    assert summary.unanalyzed_position_records == 1


def test_feature_pipeline_preserves_missing_optional_engine_values(
    tmp_path: Path,
) -> None:
    positions, analysis = _write_feature_inputs(tmp_path)
    position_table = pq.read_table(positions)
    position_rows = position_table.to_pylist()
    for row in position_rows:
        row["eco"] = None
        row["opening_name"] = None
        row["clock_seconds_after_move"] = None
    pq.write_table(
        pa.Table.from_pylist(position_rows, schema=position_table.schema), positions
    )
    rows = pq.read_table(analysis).to_pylist()
    for row in rows:
        row["actual_move_score_cp"] = None
        row["actual_move_mate_in"] = None
    pq.write_table(pa.Table.from_pylist(rows), analysis)

    output = tmp_path / "features.parquet"
    build_behavior_features(positions, analysis, "targetplayer", output_path=output)
    features = pq.read_table(output).to_pylist()
    assert all(row["opening_eco"] is None for row in features)
    assert all(row["opening_name"] is None for row in features)
    assert all(row["seconds_spent_on_move"] is None for row in features)
    assert all(row["time_remaining_seconds"] is None for row in features)
    assert all(row["actual_move_eval"] is None for row in features)
    assert all(row["centipawn_loss"] is None for row in features)
    assert all(
        row["approximate_winning_chance_after_actual_move"] is None
        for row in features
    )


def test_feature_pipeline_rejects_duplicate_position_decisions(tmp_path: Path) -> None:
    positions, analysis = _write_feature_inputs(tmp_path)
    table = pq.read_table(positions)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), positions)

    with pytest.raises(ValueError, match="Duplicate PositionRecord"):
        build_behavior_features(
            positions, analysis, "targetplayer", output_path=tmp_path / "out.parquet"
        )


def test_feature_pipeline_rejects_duplicate_analysis_ranks(tmp_path: Path) -> None:
    positions, analysis = _write_feature_inputs(tmp_path)
    table = pq.read_table(analysis)
    duplicate = pa.concat_tables([table, table.slice(0, 1)])
    pq.write_table(duplicate, analysis)

    with pytest.raises(ValueError, match="Duplicate analysis PV ranks"):
        build_behavior_features(
            positions, analysis, "targetplayer", output_path=tmp_path / "out.parquet"
        )
