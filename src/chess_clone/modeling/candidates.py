"""Leakage-safe candidate rows and chronological game splits."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import chess

from chess_clone.features.evaluation import evaluation_to_centipawns
from chess_clone.modeling.historical import canonical_position_key

DecisionKey = tuple[str, int]

_MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

LEAKAGE_FIELDS = frozenset(
    {
        "actual_move_eval",
        "actual_move_mate_in",
        "centipawn_loss",
        "actual_move_rank",
        "actual_move_in_top_1",
        "actual_move_in_top_3",
        "actual_move_in_top_5",
        "approximate_winning_chance_after_actual_move",
        "actual_move_san",
        "seconds_spent_on_move",
        "piece_moved",
        "is_capture",
        "is_check",
        "is_castle",
        "is_promotion",
        "is_en_passant",
        "is_queen_trade",
        "captured_piece_type",
        "player_clock_seconds_after_move",
        "fraction_of_initial_time_remaining",
        "time_remaining_seconds",
        "time_remaining_fraction",
        "time_pressure",
        "low_time",
    }
)

ENGINE_FEATURE_FIELDS = (
    "engine_rank",
    "engine_evaluation",
    "evaluation_difference_from_best",
)

CONTEXT_FEATURE_FIELDS = (
    "candidate_move_uci",
    "candidate_piece_moved",
    "candidate_is_capture",
    "candidate_gives_check",
    "candidate_is_castle",
    "candidate_is_promotion",
    "candidate_captured_piece_type",
    "candidate_material_change",
    "move_number",
    "game_phase",
    "material_balance",
    "legal_move_count",
    "player_in_check",
    "castling_rights_available",
    "opening_eco",
    "opening_family",
    "player_rating",
    "opponent_rating",
    "rating_difference",
    "pre_move_clock_seconds",
    "pre_move_time_fraction",
    "pre_move_time_pressure",
    "approximate_winning_chance_before",
    "player_color",
    "speed",
    "initial_time_seconds",
    "increment_seconds",
    "time_control",
)

FULL_FEATURE_FIELDS = ENGINE_FEATURE_FIELDS + CONTEXT_FEATURE_FIELDS

BOOSTED_CANDIDATE_FEATURE_FIELDS = (
    "candidate_move_uci",
    "candidate_piece_moved",
    "candidate_source_square",
    "candidate_destination_square",
    "candidate_is_capture",
    "candidate_gives_check",
    "candidate_is_castle",
    "candidate_is_promotion",
    "candidate_captured_piece_type",
    "candidate_material_change",
    "candidate_develops_minor_piece",
    "candidate_moves_queen",
    "candidate_trades_queens",
    "candidate_creates_queen_trade_opportunity",
    "candidate_moves_toward_opponent",
    "candidate_enters_opponent_territory",
    "candidate_is_pawn_push",
    "candidate_is_passed_pawn_push",
    "candidate_repeats_previous_player_post_position",
    "candidate_file_displacement",
    "candidate_rank_displacement",
    "candidate_manhattan_displacement",
)

BOOSTED_CONTEXT_FEATURE_FIELDS = (
    "move_number",
    "game_phase",
    "material_balance",
    "legal_move_count",
    "player_in_check",
    "castling_rights_available",
    "opening_eco",
    "opening_family",
    "player_rating",
    "opponent_rating",
    "rating_difference",
    "pre_move_clock_seconds",
    "pre_move_time_fraction",
    "pre_move_time_pressure",
    "approximate_winning_chance_before",
    "player_color",
    "speed",
    "initial_time_seconds",
    "increment_seconds",
    "time_control",
)


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    train_game_ids: frozenset[str]
    validation_game_ids: frozenset[str]
    test_game_ids: frozenset[str]
    game_dates: dict[str, datetime]

    def name_for(self, game_id: str) -> str:
        if game_id in self.train_game_ids:
            return "train"
        if game_id in self.validation_game_ids:
            return "validation"
        if game_id in self.test_game_ids:
            return "test"
        raise KeyError(f"Game is absent from chronological split: {game_id}")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, ids in (
            ("train", self.train_game_ids),
            ("validation", self.validation_game_ids),
            ("test", self.test_game_ids),
        ):
            dates = [self.game_dates[game_id] for game_id in ids]
            result[name] = {
                "game_count": len(ids),
                "date_min": min(dates).isoformat() if dates else None,
                "date_max": max(dates).isoformat() if dates else None,
                "game_ids": sorted(ids),
            }
        return result


@dataclass(frozen=True, slots=True)
class CandidateDataset:
    candidate_rows: list[dict[str, object]]
    decisions: list[dict[str, object]]
    total_decisions: int
    inside_top_5_decisions: int
    outside_top_5_decisions: int

    @property
    def usable_candidate_ranking_decisions(self) -> int:
        return self.inside_top_5_decisions


def chronological_game_split(
    game_rows: Iterable[dict[str, object]],
) -> ChronologicalSplit:
    """Assign entire games to an earliest-70/next-15/latest-15 split."""

    dated: list[tuple[datetime, str]] = []
    seen: set[str] = set()
    for row in game_rows:
        game_id = str(row["game_id"])
        if game_id in seen:
            raise ValueError(f"Duplicate game metadata row: {game_id}")
        played_at = row.get("played_at")
        if not isinstance(played_at, datetime):
            raise ValueError(f"Game {game_id} has no usable played_at timestamp")
        seen.add(game_id)
        dated.append((played_at, game_id))
    if len(dated) < 3:
        raise ValueError("At least three dated games are required for splitting")

    dated.sort(key=lambda item: (item[0], item[1]))
    train_end = int(len(dated) * 0.70)
    validation_end = int(len(dated) * 0.85)
    if train_end < 1 or validation_end <= train_end or validation_end >= len(dated):
        raise ValueError("Chronological fractions produced an empty split")

    train = frozenset(game_id for _, game_id in dated[:train_end])
    validation = frozenset(game_id for _, game_id in dated[train_end:validation_end])
    test = frozenset(game_id for _, game_id in dated[validation_end:])
    if train & validation or train & test or validation & test:
        raise RuntimeError("A game was assigned to multiple chronological splits")

    dates = {game_id: played_at for played_at, game_id in dated}
    if max(dates[x] for x in train) > min(dates[x] for x in validation):
        raise RuntimeError("Validation data precedes training data")
    if max(dates[x] for x in validation) > min(dates[x] for x in test):
        raise RuntimeError("Test data precedes validation data")
    return ChronologicalSplit(train, validation, test, dates)


def build_candidate_dataset(
    feature_rows: Iterable[dict[str, object]],
    analysis_rows: Iterable[dict[str, object]],
    split: ChronologicalSplit,
) -> CandidateDataset:
    """Create one row per Stockfish candidate for each usable decision."""

    features = list(feature_rows)
    analyses: dict[DecisionKey, list[dict[str, object]]] = defaultdict(list)
    for row in analysis_rows:
        analyses[(str(row["game_id"]), int(row["ply"]))].append(row)

    pre_move_clocks = _pre_move_clocks(features)
    candidate_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    seen_decisions: set[DecisionKey] = set()
    previous_player_post_by_game: dict[str, str] = {}
    inside = 0

    for feature in sorted(
        features, key=lambda row: (split.game_dates[str(row["game_id"])], int(row["ply"]))
    ):
        key = (str(feature["game_id"]), int(feature["ply"]))
        if key in seen_decisions:
            raise ValueError(f"Duplicate BehaviorFeature decision: {key}")
        seen_decisions.add(key)
        lines = sorted(analyses.get(key, []), key=lambda row: int(row["pv_rank"]))
        if not lines:
            raise ValueError(f"Missing candidate analysis for decision: {key}")
        ranks = [int(row["pv_rank"]) for row in lines]
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"Duplicate candidate rank for decision: {key}")

        actual_move = str(feature["actual_move_uci"])
        candidate_moves = [str(row["best_move_uci"]) for row in lines]
        usable = actual_move in candidate_moves
        split_name = split.name_for(key[0])
        canonical = str(lines[0]["canonical_position"])
        decision = {
            "game_id": key[0],
            "ply": key[1],
            "decision_id": f"{key[0]}:{key[1]}",
            "split": split_name,
            "played_at": split.game_dates[key[0]],
            "canonical_position": canonical,
            "actual_move_uci": actual_move,
            "usable": usable,
            "game_phase": feature["game_phase"],
            "pre_move_time_pressure": _pre_move_time_pressure(
                pre_move_clocks[key], feature.get("initial_time_seconds")
            ),
            "player_color": feature["player_color"],
            "time_control": _time_control(feature),
        }
        decisions.append(decision)
        if not usable:
            continue

        inside += 1
        best_eval = _line_evaluation(lines[0])
        if best_eval is None:
            raise ValueError(f"Best line has no evaluation for decision: {key}")
        generated = [
            _candidate_row(
                feature,
                line,
                split_name=split_name,
                canonical_position=canonical,
                actual_move=actual_move,
                best_evaluation=best_eval,
                pre_move_clock=pre_move_clocks[key],
                played_at=split.game_dates[key[0]],
                previous_player_post_position=previous_player_post_by_game.get(
                    key[0]
                ),
            )
            for line in lines
        ]
        positives = sum(int(row["chosen"]) for row in generated)
        if positives != 1:
            raise ValueError(
                f"Decision {key} has {positives} positive candidate rows; expected 1"
            )
        candidate_rows.extend(generated)
        actual_board = chess.Board(str(feature["fen"]))
        actual_board.push_uci(actual_move)
        previous_player_post_by_game[key[0]] = canonical_position_key(
            actual_board.fen(en_passant="fen")
        )

    unmatched = set(analyses) - seen_decisions
    if unmatched:
        raise ValueError(f"Analysis has {len(unmatched)} unmatched decisions")
    return CandidateDataset(
        candidate_rows=candidate_rows,
        decisions=decisions,
        total_decisions=len(decisions),
        inside_top_5_decisions=inside,
        outside_top_5_decisions=len(decisions) - inside,
    )


def validate_no_leakage_fields(feature_fields: Iterable[str]) -> None:
    leaked = sorted(set(feature_fields) & LEAKAGE_FIELDS)
    if leaked:
        raise ValueError(f"Post-decision leakage fields are forbidden: {leaked}")


def _pre_move_clocks(
    feature_rows: list[dict[str, object]],
) -> dict[DecisionKey, float | None]:
    by_game: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in feature_rows:
        by_game[str(row["game_id"])].append(row)

    result: dict[DecisionKey, float | None] = {}
    for game_id, rows in by_game.items():
        previous_post_move_clock: float | None = None
        for row in sorted(rows, key=lambda item: int(item["ply"])):
            initial = row.get("initial_time_seconds")
            pre_move = (
                float(initial)
                if previous_post_move_clock is None and initial is not None
                else previous_post_move_clock
            )
            result[(game_id, int(row["ply"]))] = pre_move
            current = row.get("player_clock_seconds_after_move")
            previous_post_move_clock = float(current) if current is not None else None
    return result


def _candidate_row(
    feature: dict[str, object],
    line: dict[str, object],
    *,
    split_name: str,
    canonical_position: str,
    actual_move: str,
    best_evaluation: int,
    pre_move_clock: float | None,
    played_at: datetime,
    previous_player_post_position: str | None,
) -> dict[str, object]:
    board = chess.Board(str(feature["fen"]))
    move_text = str(line["best_move_uci"])
    move = chess.Move.from_uci(move_text)
    if move not in board.legal_moves:
        raise ValueError(
            f"Illegal engine candidate {move_text} for {feature['game_id']}:{feature['ply']}"
        )
    player_color = board.turn
    piece = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        captured = chess.Piece(chess.PAWN, not player_color)
    is_capture = board.is_capture(move)
    is_castle = board.is_castling(move)
    is_promotion = move.promotion is not None
    source_file = chess.square_file(move.from_square)
    source_rank = chess.square_rank(move.from_square)
    destination_file = chess.square_file(move.to_square)
    destination_rank = chess.square_rank(move.to_square)
    develops_minor = _develops_minor_piece(piece, move)
    moves_queen = piece is not None and piece.piece_type == chess.QUEEN
    trades_queens = (
        moves_queen and captured is not None and captured.piece_type == chess.QUEEN
    )
    moves_toward_opponent = (
        destination_rank > source_rank
        if player_color == chess.WHITE
        else destination_rank < source_rank
    )
    enters_opponent_territory = (
        destination_rank >= 4 if player_color == chess.WHITE else destination_rank <= 3
    )
    is_pawn_push = piece is not None and piece.piece_type == chess.PAWN
    is_passed_pawn_push = is_pawn_push and _is_passed_pawn(board, move.from_square)
    queen_trade_opportunity_before = _queens_attack_each_other(board)
    board.push(move)
    material_change = (
        (_MATERIAL_VALUES[captured.piece_type] if captured else 0)
        + (
            _MATERIAL_VALUES[move.promotion] - _MATERIAL_VALUES[chess.PAWN]
            if move.promotion is not None
            else 0
        )
    )
    candidate_eval = _line_evaluation(line)
    if candidate_eval is None:
        raise ValueError("Candidate line is missing both centipawn and mate evaluation")
    initial = feature.get("initial_time_seconds")
    pre_fraction = (
        pre_move_clock / float(initial)
        if pre_move_clock is not None and initial is not None and float(initial) > 0
        else None
    )
    decision_id = f"{feature['game_id']}:{feature['ply']}"
    candidate_position = canonical_position_key(board.fen(en_passant="fen"))
    return {
        "decision_id": decision_id,
        "game_id": str(feature["game_id"]),
        "ply": int(feature["ply"]),
        "split": split_name,
        "played_at": played_at,
        "canonical_position": canonical_position,
        "actual_move_uci": actual_move,
        "chosen": move_text == actual_move,
        "engine_rank": int(line["pv_rank"]),
        "engine_evaluation": candidate_eval,
        "evaluation_difference_from_best": best_evaluation - candidate_eval,
        "candidate_move_uci": move_text,
        "candidate_piece_moved": chess.piece_name(piece.piece_type) if piece else None,
        "candidate_source_square": chess.square_name(move.from_square),
        "candidate_destination_square": chess.square_name(move.to_square),
        "candidate_is_capture": is_capture,
        "candidate_gives_check": board.is_check(),
        "candidate_is_castle": is_castle,
        "candidate_is_promotion": is_promotion,
        "candidate_captured_piece_type": (
            chess.piece_name(captured.piece_type) if captured else None
        ),
        "candidate_material_change": material_change,
        "candidate_develops_minor_piece": develops_minor,
        "candidate_moves_queen": moves_queen,
        "candidate_trades_queens": trades_queens,
        "candidate_creates_queen_trade_opportunity": (
            _queens_attack_each_other(board) and not queen_trade_opportunity_before
        ),
        "candidate_moves_toward_opponent": moves_toward_opponent,
        "candidate_enters_opponent_territory": enters_opponent_territory,
        "candidate_is_pawn_push": is_pawn_push,
        "candidate_is_passed_pawn_push": is_passed_pawn_push,
        "candidate_repeats_previous_player_post_position": (
            previous_player_post_position is not None
            and candidate_position == previous_player_post_position
        ),
        "candidate_file_displacement": abs(destination_file - source_file),
        "candidate_rank_displacement": abs(destination_rank - source_rank),
        "candidate_manhattan_displacement": (
            abs(destination_file - source_file) + abs(destination_rank - source_rank)
        ),
        "move_number": int(feature["move_number"]),
        "game_phase": feature.get("game_phase"),
        "material_balance": feature.get("material_balance"),
        "legal_move_count": feature.get("legal_move_count"),
        "player_in_check": feature.get("player_in_check"),
        "castling_rights_available": feature.get("castling_rights_available"),
        "opening_eco": feature.get("opening_eco"),
        "opening_family": _opening_family(feature.get("opening_name")),
        "player_rating": feature.get("player_rating"),
        "opponent_rating": feature.get("opponent_rating"),
        "rating_difference": feature.get("rating_difference"),
        "pre_move_clock_seconds": pre_move_clock,
        "pre_move_time_fraction": pre_fraction,
        "pre_move_time_pressure": _pre_move_time_pressure(pre_move_clock, initial),
        "approximate_winning_chance_before": feature.get(
            "approximate_winning_chance_before"
        ),
        "player_color": feature.get("player_color"),
        "speed": feature.get("speed"),
        "initial_time_seconds": initial,
        "increment_seconds": feature.get("increment_seconds"),
        "time_control": _time_control(feature),
    }


def _line_evaluation(line: dict[str, object]) -> int | None:
    score = line.get("score_cp")
    mate = line.get("mate_in")
    return evaluation_to_centipawns(
        int(score) if score is not None else None,
        int(mate) if mate is not None else None,
    )


def _opening_family(value: object) -> str | None:
    if value is None:
        return None
    return str(value).split(":", 1)[0].strip() or None


def _time_control(feature: dict[str, object]) -> str | None:
    initial = feature.get("initial_time_seconds")
    increment = feature.get("increment_seconds")
    if initial is None or increment is None:
        return None
    return f"{int(initial)}+{int(increment)}"


def _pre_move_time_pressure(clock: float | None, initial: object) -> bool | None:
    if clock is None:
        return None
    fraction_low = initial is not None and float(initial) > 0 and clock / float(initial) < 0.10
    return clock < 30.0 or fraction_low


def _develops_minor_piece(piece: chess.Piece | None, move: chess.Move) -> bool:
    if piece is None or piece.piece_type not in {chess.KNIGHT, chess.BISHOP}:
        return False
    starting_squares = (
        {chess.B1, chess.G1, chess.C1, chess.F1}
        if piece.color == chess.WHITE
        else {chess.B8, chess.G8, chess.C8, chess.F8}
    )
    return move.from_square in starting_squares


def _is_passed_pawn(board: chess.Board, square: chess.Square) -> bool:
    pawn = board.piece_at(square)
    if pawn is None or pawn.piece_type != chess.PAWN:
        return False
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    enemy_pawns = board.pieces(chess.PAWN, not pawn.color)
    for enemy_square in enemy_pawns:
        enemy_file = chess.square_file(enemy_square)
        enemy_rank = chess.square_rank(enemy_square)
        if abs(enemy_file - file_index) > 1:
            continue
        if pawn.color == chess.WHITE and enemy_rank > rank_index:
            return False
        if pawn.color == chess.BLACK and enemy_rank < rank_index:
            return False
    return True


def _queens_attack_each_other(board: chess.Board) -> bool:
    white_queens = board.pieces(chess.QUEEN, chess.WHITE)
    black_queens = board.pieces(chess.QUEEN, chess.BLACK)
    return any(
        black_square in board.attacks(white_square)
        for white_square in white_queens
        for black_square in black_queens
    )
