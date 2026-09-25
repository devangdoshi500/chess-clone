from pathlib import Path

import chess

from chess_clone.ingestion.pgn_parser import parse_pgn

FIXTURES = Path(__file__).parent / "fixtures"


def test_parses_games_and_only_target_turns() -> None:
    parsed = parse_pgn((FIXTURES / "sample_games.pgn").read_bytes(), "TARGETPLAYER")

    assert len(parsed.games) == 2
    assert len(parsed.positions) == 5
    assert parsed.games[0].rated is True
    assert parsed.games[0].speed == "blitz"
    assert [position.player_color for position in parsed.positions] == [
        "white",
        "white",
        "white",
        "black",
        "black",
    ]
    assert all(position.player_username.casefold() == "targetplayer" for position in parsed.positions)


def test_position_contains_pre_move_board_move_metadata_and_clock() -> None:
    parsed = parse_pgn((FIXTURES / "sample_games.pgn").read_bytes(), "TargetPlayer")
    first = parsed.positions[0]

    assert first.fen == chess.Board().fen()
    assert first.actual_move_uci == "e2e4"
    assert first.actual_move_san == "e4"
    assert first.move_number == 1
    assert first.ply == 1
    assert first.player_rating == 1800
    assert first.opponent_rating == 1750
    assert first.speed == "blitz"
    assert first.eco == "C20"
    assert first.opening_name == "King's Pawn Game"
    assert first.clock_seconds_after_move == 181.0


def test_black_positions_have_correct_pre_move_fen_and_variation() -> None:
    parsed = parse_pgn((FIXTURES / "sample_games.pgn").read_bytes(), "targetplayer")
    first_black = parsed.positions[3]
    expected = chess.Board()
    expected.push_uci("d2d4")

    assert first_black.fen == expected.fen()
    assert first_black.actual_move_uci == "d7d5"
    assert first_black.move_number == 1
    assert first_black.opening_variation == "Chigorin Variation"
    assert first_black.clock_seconds_after_move == 597.0


def test_skips_variants_and_games_without_target() -> None:
    parsed = parse_pgn(
        (FIXTURES / "non_target_and_variant.pgn").read_bytes(), "TargetPlayer"
    )

    assert parsed.games == []
    assert parsed.positions == []
    assert parsed.skipped_games == 2
