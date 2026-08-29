"""Join PositionRecords with engine analysis and build behavioral features."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.features.board import (
    extract_board_state_features,
    extract_move_behavior_features,
    player_color_from_name,
)
from chess_clone.features.evaluation import (
    approximate_winning_chance,
    derive_engine_rank_features,
    evaluation_to_centipawns,
)
from chess_clone.features.schemas import BehaviorFeatureRecord
from chess_clone.features.time import (
    TimeFeatures,
    TimePressureThresholds,
    derive_time_features,
)
from chess_clone.modeling import canonical_position_key

DecisionKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class FeatureBuildSummary:
    username: str
    available_position_records: int
    analyzed_position_records: int
    feature_rows: int
    unanalyzed_position_records: int
    output_path: Path


FEATURE_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("player_username", pa.string()),
        ("ply", pa.int64()),
        ("move_number", pa.int64()),
        ("player_color", pa.string()),
        ("fen", pa.string()),
        ("actual_move_uci", pa.string()),
        ("actual_move_san", pa.string()),
        ("speed", pa.string()),
        ("opening_eco", pa.string()),
        ("opening_name", pa.string()),
        ("player_rating", pa.int64()),
        ("opponent_rating", pa.int64()),
        ("rating_difference", pa.int64()),
        ("player_clock_seconds_after_move", pa.float64()),
        ("seconds_spent_on_move", pa.float64()),
        ("initial_time_seconds", pa.int64()),
        ("increment_seconds", pa.int64()),
        ("fraction_of_initial_time_remaining", pa.float64()),
        ("time_pressure", pa.bool_()),
        ("time_remaining_seconds", pa.float64()),
        ("time_remaining_fraction", pa.float64()),
        ("low_time", pa.bool_()),
        ("game_phase", pa.string()),
        ("total_piece_count", pa.int64()),
        ("material_balance", pa.int64()),
        ("player_queen_present", pa.bool_()),
        ("opponent_queen_present", pa.bool_()),
        ("legal_move_count", pa.int64()),
        ("player_in_check", pa.bool_()),
        ("castling_rights_available", pa.bool_()),
        ("piece_moved", pa.string()),
        ("is_capture", pa.bool_()),
        ("is_check", pa.bool_()),
        ("is_castle", pa.bool_()),
        ("is_promotion", pa.bool_()),
        ("is_en_passant", pa.bool_()),
        ("is_queen_trade", pa.bool_()),
        ("captured_piece_type", pa.string()),
        ("engine_best_move", pa.string()),
        ("engine_eval_before", pa.int64()),
        ("engine_mate_before", pa.int64()),
        ("actual_move_eval", pa.int64()),
        ("actual_move_mate_in", pa.int64()),
        ("centipawn_loss", pa.int64()),
        ("actual_move_rank", pa.int64()),
        ("actual_move_in_top_1", pa.bool_()),
        ("actual_move_in_top_3", pa.bool_()),
        ("actual_move_in_top_5", pa.bool_()),
        ("approximate_winning_chance_before", pa.float64()),
        ("approximate_winning_chance_after_actual_move", pa.float64()),
    ]
)


def default_feature_output_path(username: str, output_dir: Path) -> Path:
    safe_username = username.strip().lower()
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return output_dir / f"features_{safe_username}_{batch_id}.parquet"


def build_behavior_features(
    position_path: str | Path,
    analysis_path: str | Path,
    username: str,
    *,
    output_path: str | Path,
    thresholds: TimePressureThresholds = TimePressureThresholds(),
) -> FeatureBuildSummary:
    """Build exactly one feature row for each analyzed player decision."""

    positions_source = Path(position_path)
    analysis_source = Path(analysis_path)
    if not positions_source.is_file():
        raise FileNotFoundError(f"Position dataset not found: {positions_source}")
    if not analysis_source.is_file():
        raise FileNotFoundError(f"Analysis dataset not found: {analysis_source}")

    try:
        position_rows = pq.read_table(positions_source).to_pylist()
        analysis_rows = pq.read_table(analysis_source).to_pylist()
    except Exception as exc:
        raise ValueError(f"Could not read feature input Parquet: {exc}") from exc

    requested = username.casefold()
    position_rows = [
        row
        for row in position_rows
        if str(row["player_username"]).casefold() == requested
    ]
    analysis_rows = [
        row
        for row in analysis_rows
        if str(row["player_username"]).casefold() == requested
    ]
    if not position_rows:
        raise ValueError(f"No PositionRecords found for player '{username}'")
    if not analysis_rows:
        raise ValueError(f"No analysis rows found for player '{username}'")

    positions_by_key: dict[DecisionKey, dict[str, object]] = {}
    for row in position_rows:
        key = _decision_key(row)
        if key in positions_by_key:
            raise ValueError(f"Duplicate PositionRecord decision: {key}")
        positions_by_key[key] = row

    analysis_by_key: dict[DecisionKey, list[dict[str, object]]] = defaultdict(list)
    for row in analysis_rows:
        analysis_by_key[_decision_key(row)].append(row)

    missing_positions = sorted(set(analysis_by_key) - set(positions_by_key))
    if missing_positions:
        raise ValueError(
            f"Analysis contains {len(missing_positions)} unmatched decision rows: "
            f"{missing_positions[:3]}"
        )

    time_by_key = _derive_all_time_features(position_rows, thresholds)
    records: list[BehaviorFeatureRecord] = []
    for key in sorted(analysis_by_key, key=lambda item: (item[0], item[1])):
        position = positions_by_key[key]
        lines = analysis_by_key[key]
        _validate_join(position, lines, key)

        board = chess.Board(str(position["fen"]))
        player_color_name = str(position["player_color"]).casefold()
        player_color = player_color_from_name(player_color_name)
        board_features = extract_board_state_features(board, player_color)
        move_features = extract_move_behavior_features(
            board, str(position["actual_move_uci"])
        )
        time_features = time_by_key[key]

        ranked = sorted(lines, key=lambda row: int(row["pv_rank"]))
        best = ranked[0]
        rank_features = derive_engine_rank_features(
            str(position["actual_move_uci"]), ranked
        )
        before_score = _optional_int(best.get("score_cp"))
        before_mate = _optional_int(best.get("mate_in"))
        actual_score = _optional_int(best.get("actual_move_score_cp"))
        actual_mate = _optional_int(best.get("actual_move_mate_in"))
        before_eval = evaluation_to_centipawns(before_score, before_mate)
        actual_eval = evaluation_to_centipawns(actual_score, actual_mate)
        centipawn_loss = (
            max(0, before_eval - actual_eval)
            if before_eval is not None and actual_eval is not None
            else None
        )
        player_rating = _optional_int(position.get("player_rating"))
        opponent_rating = _optional_int(position.get("opponent_rating"))
        rating_difference = (
            player_rating - opponent_rating
            if player_rating is not None and opponent_rating is not None
            else None
        )

        records.append(
            BehaviorFeatureRecord(
                game_id=str(position["game_id"]),
                player_username=str(position["player_username"]),
                ply=int(position["ply"]),
                move_number=int(position["move_number"]),
                player_color=player_color_name,
                fen=str(position["fen"]),
                actual_move_uci=str(position["actual_move_uci"]),
                actual_move_san=str(position["actual_move_san"]),
                speed=_optional_str(position.get("speed")),
                opening_eco=_optional_str(position.get("eco")),
                opening_name=_optional_str(position.get("opening_name")),
                player_rating=player_rating,
                opponent_rating=opponent_rating,
                rating_difference=rating_difference,
                **asdict(time_features),
                **asdict(board_features),
                **asdict(move_features),
                engine_best_move=_optional_str(best.get("best_move_uci")),
                engine_eval_before=before_eval,
                engine_mate_before=before_mate,
                actual_move_eval=actual_eval,
                actual_move_mate_in=actual_mate,
                centipawn_loss=centipawn_loss,
                actual_move_rank=rank_features.actual_move_rank,
                actual_move_in_top_1=rank_features.actual_move_in_top_1,
                actual_move_in_top_3=rank_features.actual_move_in_top_3,
                actual_move_in_top_5=rank_features.actual_move_in_top_5,
                approximate_winning_chance_before=approximate_winning_chance(
                    before_score, before_mate
                ),
                approximate_winning_chance_after_actual_move=(
                    approximate_winning_chance(actual_score, actual_mate)
                ),
            )
        )

    if len(records) != len(analysis_by_key):
        raise RuntimeError("Feature generation lost or duplicated decision rows")
    output_keys = {(record.game_id, record.ply) for record in records}
    if len(output_keys) != len(records):
        raise RuntimeError("Duplicate decision rows generated")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [record.to_dict() for record in records], schema=FEATURE_SCHEMA
    )
    pq.write_table(table, destination)
    return FeatureBuildSummary(
        username=username,
        available_position_records=len(position_rows),
        analyzed_position_records=len(analysis_by_key),
        feature_rows=len(records),
        unanalyzed_position_records=len(position_rows) - len(analysis_by_key),
        output_path=destination,
    )


def _derive_all_time_features(
    position_rows: list[dict[str, object]], thresholds: TimePressureThresholds
) -> dict[DecisionKey, TimeFeatures]:
    games: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in position_rows:
        games[str(row["game_id"])].append(row)

    result: dict[DecisionKey, TimeFeatures] = {}
    for rows in games.values():
        previous_clock: float | None = None
        for row in sorted(rows, key=lambda item: int(item["ply"])):
            clock = _optional_float(row.get("clock_seconds_after_move"))
            result[_decision_key(row)] = derive_time_features(
                time_control=_optional_str(row.get("time_control")),
                clock_after_move=clock,
                previous_player_clock_after_move=previous_clock,
                thresholds=thresholds,
            )
            previous_clock = clock
    return result


def _validate_join(
    position: dict[str, object],
    lines: list[dict[str, object]],
    key: DecisionKey,
) -> None:
    ranks = [int(line["pv_rank"]) for line in lines]
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"Duplicate analysis PV ranks for decision {key}")
    if 1 not in ranks:
        raise ValueError(f"Analysis is missing rank 1 for decision {key}")

    expected_position = canonical_position_key(str(position["fen"]))
    expected_move = str(position["actual_move_uci"])
    for line in lines:
        if canonical_position_key(str(line["source_fen"])) != expected_position:
            raise ValueError(f"FEN mismatch while joining decision {key}")
        if str(line["actual_move_uci"]) != expected_move:
            raise ValueError(f"Actual-move mismatch while joining decision {key}")
        perspective = line.get("score_perspective")
        if perspective is not None and perspective != "side_to_move":
            raise ValueError(f"Unsupported engine score perspective for decision {key}")


def _decision_key(row: dict[str, object]) -> DecisionKey:
    return str(row["game_id"]), int(row["ply"])


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
