"""Sequential, cache-aware PositionRecord engine analysis pipeline."""

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol, Self

import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.analysis.cache import FileAnalysisCache, build_analysis_cache_key
from chess_clone.analysis.schemas import EngineLine, EngineSettings
from chess_clone.modeling import canonical_position_key


class PositionAnalyzer(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...

    @property
    def engine_identity(self) -> str: ...

    def analyze(self, fen: str, settings: EngineSettings) -> list[EngineLine]: ...


@dataclass(frozen=True, slots=True)
class AnalysisRunSummary:
    username: str
    position_records: int
    unique_positions: int
    engine_calls: int
    cache_hits: int
    output_rows: int
    engine_identity: str
    output_path: Path


ANALYSIS_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("ply", pa.int64()),
        ("player_username", pa.string()),
        ("player_color", pa.string()),
        ("source_fen", pa.string()),
        ("canonical_position", pa.string()),
        ("actual_move_uci", pa.string()),
        ("cache_key", pa.string()),
        ("cache_hit", pa.bool_()),
        ("engine_identity", pa.string()),
        ("nodes_requested", pa.int64()),
        ("multipv_requested", pa.int64()),
        ("threads_requested", pa.int64()),
        ("hash_mb_requested", pa.int64()),
        ("engine_options_json", pa.string()),
        ("score_perspective", pa.string()),
        ("pv_rank", pa.int64()),
        ("score_cp", pa.int64()),
        ("mate_in", pa.int64()),
        ("best_move_uci", pa.string()),
        ("pv_uci", pa.string()),
        ("depth", pa.int64()),
        ("seldepth", pa.int64()),
        ("nodes_searched", pa.int64()),
        ("time_seconds", pa.float64()),
    ]
)


def default_analysis_output_path(username: str, output_dir: Path) -> Path:
    safe_username = username.strip().lower()
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return output_dir / f"analysis_{safe_username}_{batch_id}.parquet"


def analyze_position_dataset(
    position_path: str | Path,
    username: str,
    *,
    analyzer: PositionAnalyzer,
    settings: EngineSettings,
    cache: FileAnalysisCache,
    output_path: str | Path,
    max_positions: int | None = None,
) -> AnalysisRunSummary:
    """Analyze one player's PositionRecords sequentially with exact caching."""

    if max_positions is not None and max_positions < 1:
        raise ValueError("max_positions must be at least 1")
    source_path = Path(position_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Position dataset not found: {source_path}")

    columns = [
        "game_id",
        "ply",
        "player_username",
        "player_color",
        "fen",
        "actual_move_uci",
    ]
    try:
        records = pq.read_table(source_path, columns=columns).to_pylist()
    except Exception as exc:
        raise ValueError(
            f"Could not read PositionRecords from {source_path}: {exc}"
        ) from exc

    requested = username.casefold()
    records = [
        record
        for record in records
        if record["player_username"].casefold() == requested
    ]
    if max_positions is not None:
        records = records[:max_positions]
    if not records:
        raise ValueError(f"No PositionRecords found for player '{username}'")

    output_rows: list[dict[str, object]] = []
    unique_positions: set[str] = set()
    engine_calls = 0
    cache_hits = 0

    with analyzer as active_analyzer:
        engine_identity = active_analyzer.engine_identity
        for record in records:
            fen = str(record["fen"])
            position_key = canonical_position_key(fen)
            unique_positions.add(position_key)
            cache_key = build_analysis_cache_key(fen, settings, engine_identity)
            cached = cache.get(cache_key)
            cache_hit = cached is not None
            if cached is None:
                analysis_fen = f"{position_key} 0 1"
                lines = active_analyzer.analyze(analysis_fen, settings)
                engine_calls += 1
                cache.put(
                    cache_key,
                    position_key=position_key,
                    engine_identity=engine_identity,
                    settings=settings,
                    lines=lines,
                )
            else:
                lines = list(cached.lines)
                cache_hits += 1

            for line in lines:
                output_rows.append(
                    {
                        "game_id": record["game_id"],
                        "ply": record["ply"],
                        "player_username": record["player_username"],
                        "player_color": record["player_color"],
                        "source_fen": fen,
                        "canonical_position": position_key,
                        "actual_move_uci": record["actual_move_uci"],
                        "cache_key": cache_key,
                        "cache_hit": cache_hit,
                        "engine_identity": engine_identity,
                        "nodes_requested": settings.nodes,
                        "multipv_requested": settings.multipv,
                        "threads_requested": settings.threads,
                        "hash_mb_requested": settings.hash_mb,
                        "engine_options_json": json.dumps(
                            dict(settings.options), sort_keys=True
                        ),
                        "score_perspective": "side_to_move",
                        "pv_rank": line.rank,
                        "score_cp": line.score_cp,
                        "mate_in": line.mate_in,
                        "best_move_uci": line.best_move_uci,
                        "pv_uci": line.pv_uci,
                        "depth": line.depth,
                        "seldepth": line.seldepth,
                        "nodes_searched": line.nodes_searched,
                        "time_seconds": line.time_seconds,
                    }
                )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(output_rows, schema=ANALYSIS_SCHEMA)
    pq.write_table(table, destination)

    return AnalysisRunSummary(
        username=username,
        position_records=len(records),
        unique_positions=len(unique_positions),
        engine_calls=engine_calls,
        cache_hits=cache_hits,
        output_rows=len(output_rows),
        engine_identity=engine_identity,
        output_path=destination,
    )
