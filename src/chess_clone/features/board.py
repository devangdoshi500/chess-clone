"""Deterministic pre-move board and actual-move feature extraction."""

from dataclasses import dataclass

import chess

_MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass(frozen=True, slots=True)
class BoardStateFeatures:
    game_phase: str
    total_piece_count: int
    material_balance: int
    player_queen_present: bool
    opponent_queen_present: bool
    legal_move_count: int
    player_in_check: bool
    castling_rights_available: bool


@dataclass(frozen=True, slots=True)
class MoveBehaviorFeatures:
    piece_moved: str
    is_capture: bool
    is_check: bool
    is_castle: bool
    is_promotion: bool
    is_en_passant: bool
    is_queen_trade: bool
    captured_piece_type: str | None


def player_color_from_name(player_color: str) -> chess.Color:
    normalized = player_color.casefold()
    if normalized == "white":
        return chess.WHITE
    if normalized == "black":
        return chess.BLACK
    raise ValueError(f"Invalid player color: {player_color}")


def classify_game_phase(board: chess.Board) -> str:
    """Classify a position with a deliberately small deterministic rule.

    - Endgame: no queens remain, or combined non-pawn/non-king material is at
      most 20 points.
    - Opening: otherwise, at least 28 pieces remain.
    - Middlegame: everything else.
    """

    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    non_pawn_material = sum(
        _MATERIAL_VALUES[piece_type]
        * (
            len(board.pieces(piece_type, chess.WHITE))
            + len(board.pieces(piece_type, chess.BLACK))
        )
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    if queens == 0 or non_pawn_material <= 20:
        return "endgame"
    if len(board.piece_map()) >= 28:
        return "opening"
    return "middlegame"


def extract_board_state_features(
    board: chess.Board, player_color: chess.Color
) -> BoardStateFeatures:
    if board.turn != player_color:
        raise ValueError("Position side to move does not match player color")

    opponent_color = not player_color
    player_material = sum(
        _MATERIAL_VALUES[piece_type] * len(board.pieces(piece_type, player_color))
        for piece_type in _MATERIAL_VALUES
    )
    opponent_material = sum(
        _MATERIAL_VALUES[piece_type] * len(board.pieces(piece_type, opponent_color))
        for piece_type in _MATERIAL_VALUES
    )
    return BoardStateFeatures(
        game_phase=classify_game_phase(board),
        total_piece_count=len(board.piece_map()),
        material_balance=player_material - opponent_material,
        player_queen_present=bool(board.pieces(chess.QUEEN, player_color)),
        opponent_queen_present=bool(board.pieces(chess.QUEEN, opponent_color)),
        legal_move_count=board.legal_moves.count(),
        player_in_check=board.is_check(),
        castling_rights_available=(
            board.has_kingside_castling_rights(player_color)
            or board.has_queenside_castling_rights(player_color)
        ),
    )


def extract_move_behavior_features(
    board: chess.Board, move_uci: str
) -> MoveBehaviorFeatures:
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {move_uci} for position {board.fen()}")

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError(f"No piece on move origin for {move_uci}")

    is_en_passant = board.is_en_passant(move)
    captured_piece = (
        chess.Piece(chess.PAWN, not board.turn)
        if is_en_passant
        else board.piece_at(move.to_square)
    )
    captured_piece_type = (
        chess.piece_name(captured_piece.piece_type)
        if captured_piece is not None
        else None
    )
    return MoveBehaviorFeatures(
        piece_moved=chess.piece_name(moving_piece.piece_type),
        is_capture=board.is_capture(move),
        is_check=board.gives_check(move),
        is_castle=board.is_castling(move),
        is_promotion=move.promotion is not None,
        is_en_passant=is_en_passant,
        is_queen_trade=(
            moving_piece.piece_type == chess.QUEEN
            and captured_piece is not None
            and captured_piece.piece_type == chess.QUEEN
        ),
        captured_piece_type=captured_piece_type,
    )
