"""All-legal-move rows for human policy modeling without an engine gate."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Iterable

import chess

from chess_clone.features.board import (
    extract_board_state_features,
    extract_move_behavior_features,
    player_color_from_name,
)


LEGAL_POLICY_CANDIDATE_FIELDS = (
    "candidate_move_uci",
    "candidate_piece_moved",
    "candidate_source_square",
    "candidate_destination_square",
    "candidate_is_capture",
    "candidate_gives_check",
    "candidate_is_castle",
    "candidate_is_promotion",
    "candidate_is_en_passant",
    "candidate_is_queen_trade",
    "candidate_captured_piece_type",
    "candidate_capture_value",
    "candidate_material_gain",
    "candidate_is_pawn_push",
    "candidate_moves_queen",
    "candidate_is_development",
    "candidate_gives_mate",
    "candidate_enters_opponent_territory",
    "candidate_targets_king_zone",
    "candidate_attackers_after",
    "candidate_defenders_after",
    "candidate_is_hanging_after",
    "candidate_center_control_after",
    "candidate_opponent_mobility_after",
    "candidate_destination_wing",
    "candidate_file_displacement",
    "candidate_rank_displacement",
    "candidate_manhattan_displacement",
)

_PIECE_VALUES = {
    None: 0,
    "pawn": 1,
    "knight": 3,
    "bishop": 3,
    "rook": 5,
    "queen": 9,
    "king": 0,
}

_DEVELOPMENT_SQUARES = {
    chess.WHITE: {
        chess.B1: chess.KNIGHT,
        chess.G1: chess.KNIGHT,
        chess.C1: chess.BISHOP,
        chess.F1: chess.BISHOP,
    },
    chess.BLACK: {
        chess.B8: chess.KNIGHT,
        chess.G8: chess.KNIGHT,
        chess.C8: chess.BISHOP,
        chess.F8: chess.BISHOP,
    },
}

_CENTER = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5

LEGAL_POLICY_CONTEXT_FIELDS = (
    "move_number",
    "game_phase",
    "material_balance",
    "legal_move_count",
    "player_in_check",
    "castling_rights_available",
    "player_rating",
    "opponent_rating",
    "rating_difference",
    "player_color",
    "speed",
    "time_control",
)

LEGAL_POLICY_FEATURE_FIELDS = LEGAL_POLICY_CANDIDATE_FIELDS + LEGAL_POLICY_CONTEXT_FIELDS

PLAYER_TENDENCY_FIELDS = (
    "history_piece_probability",
    "history_capture_match_probability",
    "history_check_match_probability",
    "history_castle_match_probability",
    "history_queen_trade_match_probability",
    "history_pawn_push_match_probability",
    "history_king_zone_match_probability",
    "history_wing_probability",
    "history_observations",
)

BEHAVIOR_BOOLEAN_FIELDS = (
    "candidate_is_capture",
    "candidate_gives_check",
    "candidate_is_castle",
    "candidate_is_promotion",
    "candidate_is_queen_trade",
    "candidate_is_pawn_push",
    "candidate_moves_queen",
    "candidate_enters_opponent_territory",
    "candidate_targets_king_zone",
)


def build_all_legal_candidate_rows(
    positions: Iterable[dict[str, object]],
    game_dates: dict[str, datetime],
    split_names: dict[str, str],
    *,
    include_archived_openings: bool = False,
) -> list[dict[str, object]]:
    """Emit one row for every legal move and exactly one positive per position."""

    rows: list[dict[str, object]] = []
    ordered = sorted(
        positions,
        key=lambda row: (
            str(row["player_username"]).casefold(),
            game_dates[str(row["game_id"])],
            int(row["ply"]),
        ),
    )
    seen: set[str] = set()
    for position in ordered:
        if not include_archived_openings:
            # PGN opening tags describe the completed game, including future moves.
            position = dict(position, eco=None, opening_name=None, opening_variation=None)
        game_id = str(position["game_id"])
        decision_id = f"{game_id}:{int(position['ply'])}"
        if decision_id in seen:
            raise ValueError(f"Duplicate decision: {decision_id}")
        seen.add(decision_id)
        board = chess.Board(str(position["fen"]))
        color = player_color_from_name(str(position["player_color"]))
        state = extract_board_state_features(board, color)
        actual = str(position["actual_move_uci"])
        legal_moves = list(board.legal_moves)
        if chess.Move.from_uci(actual) not in legal_moves:
            raise ValueError(f"Actual move is illegal for {decision_id}: {actual}")
        for move in legal_moves:
            rows.append(
                _candidate_row(
                    position,
                    board,
                    move,
                    state=asdict(state),
                    split=split_names[game_id],
                    played_at=game_dates[game_id],
                )
            )
    return rows


def build_all_legal_inference_rows(
    fen: str,
    *,
    player_username: str | None,
    player_rating: int | None,
    opponent_rating: int | None,
    speed: str | None,
    time_control: str | None,
    decision_id: str = "inference:1",
) -> list[dict[str, object]]:
    """Build label-free rows for every legal move in one FEN position.

    The caller supplies game-level context that cannot be derived from FEN. No
    observed move, result, clock value, or completed-game opening tag is added.
    Terminal positions return an empty list.
    """

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {exc}") from exc
    if not board.is_valid():
        raise ValueError(f"Invalid chess position: {fen}")
    if player_rating is not None and player_rating <= 0:
        raise ValueError("player_rating must be positive when provided")
    if opponent_rating is not None and opponent_rating <= 0:
        raise ValueError("opponent_rating must be positive when provided")

    game_id, separator, ply_text = decision_id.rpartition(":")
    if not separator or not game_id:
        raise ValueError("decision_id must have the form '<game>:<ply>'")
    try:
        ply = int(ply_text)
    except ValueError as exc:
        raise ValueError("decision_id ply must be an integer") from exc

    color_name = "white" if board.turn == chess.WHITE else "black"
    state = asdict(extract_board_state_features(board, board.turn))
    position = {
        "game_id": game_id,
        "ply": ply,
        "move_number": board.fullmove_number,
        "fen": board.fen(),
        "player_username": player_username or "",
        "player_color": color_name,
        "player_rating": player_rating,
        "opponent_rating": opponent_rating,
        "speed": speed,
        "time_control": time_control,
        "eco": None,
        "opening_name": None,
        "opening_variation": None,
    }
    return [
        _candidate_row(
            position,
            board,
            move,
            state=state,
            split="inference",
            played_at=None,
        )
        for move in board.legal_moves
    ]


def _candidate_row(
    position: dict[str, object],
    board: chess.Board,
    move: chess.Move,
    *,
    state: dict[str, object],
    split: str,
    played_at: datetime | None,
) -> dict[str, object]:
    move_uci = move.uci()
    behavior = extract_move_behavior_features(board, move_uci)
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError(f"Candidate has no moving piece: {move_uci}")
    source_file = chess.square_file(move.from_square)
    source_rank = chess.square_rank(move.from_square)
    destination_file = chess.square_file(move.to_square)
    destination_rank = chess.square_rank(move.to_square)
    player = board.turn
    opponent = not player
    opponent_king = board.king(opponent)
    post = board.copy(stack=False)
    post.push(move)
    attackers_after = len(post.attackers(opponent, move.to_square))
    defenders_after = len(post.attackers(player, move.to_square))
    captured_value = _PIECE_VALUES[behavior.captured_piece_type]
    promotion_gain = (
        _PIECE_VALUES[chess.piece_name(move.promotion)] - _PIECE_VALUES["pawn"]
        if move.promotion is not None
        else 0
    )
    is_development = (
        _DEVELOPMENT_SQUARES[player].get(move.from_square) == piece.piece_type
        and move.to_square not in _DEVELOPMENT_SQUARES[player]
    )
    targets_king_zone = False
    if opponent_king is not None:
        king_zone = post.attacks(opponent_king) | chess.BB_SQUARES[opponent_king]
        targets_king_zone = bool(post.attacks(move.to_square) & king_zone)
    enters_opponent_territory = (
        destination_rank >= 4 if board.turn == chess.WHITE else destination_rank <= 3
    )
    player_rating = position.get("player_rating")
    opponent_rating = position.get("opponent_rating")
    rating_difference = (
        int(player_rating) - int(opponent_rating)
        if player_rating is not None and opponent_rating is not None
        else None
    )
    game_id = str(position["game_id"])
    row = {
        "decision_id": f"{game_id}:{int(position['ply'])}",
        "game_id": game_id,
        "ply": int(position["ply"]),
        "player_username": str(position["player_username"]),
        "split": split,
        "played_at": played_at,
        "candidate_move_uci": move_uci,
        "candidate_piece_moved": behavior.piece_moved,
        "candidate_source_square": chess.square_name(move.from_square),
        "candidate_destination_square": chess.square_name(move.to_square),
        "candidate_is_capture": behavior.is_capture,
        "candidate_gives_check": behavior.is_check,
        "candidate_is_castle": behavior.is_castle,
        "candidate_is_promotion": behavior.is_promotion,
        "candidate_is_en_passant": behavior.is_en_passant,
        "candidate_is_queen_trade": behavior.is_queen_trade,
        "candidate_captured_piece_type": behavior.captured_piece_type,
        "candidate_capture_value": captured_value,
        "candidate_material_gain": captured_value + promotion_gain,
        "candidate_is_pawn_push": piece.piece_type == chess.PAWN,
        "candidate_moves_queen": piece.piece_type == chess.QUEEN,
        "candidate_is_development": is_development,
        "candidate_gives_mate": post.is_checkmate(),
        "candidate_enters_opponent_territory": enters_opponent_territory,
        "candidate_targets_king_zone": targets_king_zone,
        "candidate_attackers_after": attackers_after,
        "candidate_defenders_after": defenders_after,
        "candidate_is_hanging_after": attackers_after > 0 and defenders_after == 0,
        "candidate_center_control_after": int(
            post.attacks(move.to_square) & _CENTER
        ).bit_count(),
        "candidate_opponent_mobility_after": post.legal_moves.count(),
        "candidate_destination_wing": _wing(destination_file),
        "candidate_file_displacement": abs(destination_file - source_file),
        "candidate_rank_displacement": abs(destination_rank - source_rank),
        "candidate_manhattan_displacement": abs(destination_file - source_file)
        + abs(destination_rank - source_rank),
        "move_number": int(position["move_number"]),
        "game_phase": state["game_phase"],
        "material_balance": state["material_balance"],
        "legal_move_count": state["legal_move_count"],
        "player_in_check": state["player_in_check"],
        "castling_rights_available": state["castling_rights_available"],
        "opening_eco": position.get("eco"),
        "opening_family": _opening_family(position.get("opening_name")),
        "player_rating": player_rating,
        "opponent_rating": opponent_rating,
        "rating_difference": rating_difference,
        "player_color": position.get("player_color"),
        "speed": position.get("speed"),
        "time_control": position.get("time_control"),
    }
    actual_move = position.get("actual_move_uci")
    if actual_move is not None:
        row["actual_move_uci"] = str(actual_move)
        row["chosen"] = move_uci == str(actual_move)
    return row


class PlayerTendencyEncoder:
    """Leakage-safe per-player behavioral priors for candidate moves."""

    def __init__(self, smoothing_strength: float = 20.0) -> None:
        if smoothing_strength <= 0:
            raise ValueError("smoothing_strength must be positive")
        self.smoothing_strength = float(smoothing_strength)
        self._states: dict[str, _TendencyState] = {}
        self._fitted = False

    def fit_transform_ordered(
        self, rows: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        groups = _groups(rows)
        states: dict[str, _TendencyState] = defaultdict(_TendencyState)
        output: list[dict[str, object]] = []
        for group in groups:
            player = str(group[0]["player_username"]).casefold()
            output.extend(self._encode(group, states[player]))
            states[player].update(group)
        self._states = dict(states)
        self._fitted = True
        return output

    def transform(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        if not self._fitted:
            raise RuntimeError("PlayerTendencyEncoder must be fitted first")
        output: list[dict[str, object]] = []
        for group in _groups(rows):
            state = self._states.get(
                str(group[0]["player_username"]).casefold(), _TendencyState()
            )
            output.extend(self._encode(group, state))
        return output

    def _encode(
        self, group: list[dict[str, object]], state: "_TendencyState"
    ) -> list[dict[str, object]]:
        strength = self.smoothing_strength
        denominator = state.total + strength
        result = []
        for row in group:
            item = dict(row)
            item.update(
                {
                    "history_piece_probability": (
                        state.pieces[str(row["candidate_piece_moved"])] + strength / 6
                    )
                    / denominator,
                    "history_wing_probability": (
                        state.wings[str(row["candidate_destination_wing"])] + strength / 3
                    )
                    / denominator,
                    "history_observations": state.total,
                }
            )
            for output_name, candidate_field in (
                ("history_capture_match_probability", "candidate_is_capture"),
                ("history_check_match_probability", "candidate_gives_check"),
                ("history_castle_match_probability", "candidate_is_castle"),
                ("history_queen_trade_match_probability", "candidate_is_queen_trade"),
                ("history_pawn_push_match_probability", "candidate_is_pawn_push"),
                ("history_king_zone_match_probability", "candidate_targets_king_zone"),
            ):
                probability = (state.true_counts[candidate_field] + strength * 0.5) / denominator
                item[output_name] = probability if bool(row[candidate_field]) else 1 - probability
            result.append(item)
        return result


class _TendencyState:
    def __init__(self) -> None:
        self.total = 0
        self.pieces: Counter[str] = Counter()
        self.wings: Counter[str] = Counter()
        self.true_counts: Counter[str] = Counter()

    def update(self, group: list[dict[str, object]]) -> None:
        chosen = [row for row in group if bool(row["chosen"])]
        if len(chosen) != 1:
            raise ValueError("Every legal-move decision must have exactly one positive")
        row = chosen[0]
        self.total += 1
        self.pieces[str(row["candidate_piece_moved"])] += 1
        self.wings[str(row["candidate_destination_wing"])] += 1
        for field in BEHAVIOR_BOOLEAN_FIELDS:
            self.true_counts[field] += int(bool(row[field]))


def _groups(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    order: list[str] = []
    for row in rows:
        key = str(row["decision_id"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    return [grouped[key] for key in order]


def _wing(file_index: int) -> str:
    if file_index <= 2:
        return "queenside"
    if file_index >= 5:
        return "kingside"
    return "center"


def _opening_family(value: object) -> str | None:
    if value is None:
        return None
    return str(value).split(":", 1)[0].strip() or None
