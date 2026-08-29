from dataclasses import replace
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chess_clone.analysis import (
    EngineLine,
    EngineSettings,
    FileAnalysisCache,
    StockfishAnalyzer,
    StockfishNotFoundError,
    analyze_position_dataset,
    build_analysis_cache_key,
)


class FakeAnalyzer:
    engine_identity = "Fakefish 1.0|sha256:test"

    def __init__(self) -> None:
        self.calls: list[tuple[str, EngineSettings]] = []

    def __enter__(self) -> "FakeAnalyzer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def analyze(self, fen: str, settings: EngineSettings) -> list[EngineLine]:
        self.calls.append((fen, settings))
        return [
            EngineLine(
                rank=1,
                score_cp=31,
                mate_in=None,
                best_move_uci="e2e4",
                pv_uci="e2e4 e7e5",
                depth=5,
                seldepth=7,
                nodes_searched=settings.nodes,
                time_seconds=0.01,
            )
        ]


def test_cache_key_uses_canonical_position_and_all_settings() -> None:
    same_position_different_counters = chess.STARTING_FEN.rsplit(" ", 2)[0] + " 12 42"
    settings = EngineSettings(nodes=500)

    first = build_analysis_cache_key(chess.STARTING_FEN, settings, "engine-a")
    assert first == build_analysis_cache_key(
        same_position_different_counters, settings, "engine-a"
    )
    assert first != build_analysis_cache_key(
        chess.STARTING_FEN, EngineSettings(nodes=501), "engine-a"
    )
    assert first != build_analysis_cache_key(
        chess.STARTING_FEN, settings, "engine-b"
    )
    for changed_settings in [
        replace(settings, multipv=2),
        replace(settings, threads=2),
        replace(settings, hash_mb=32),
        replace(settings, options=(("Skill Level", 10),)),
    ]:
        assert first != build_analysis_cache_key(
            chess.STARTING_FEN, changed_settings, "engine-a"
        )


def test_file_cache_round_trip(tmp_path: Path) -> None:
    cache = FileAnalysisCache(tmp_path / "cache")
    settings = EngineSettings(nodes=100)
    line = FakeAnalyzer().analyze(chess.STARTING_FEN, settings)[0]
    key = build_analysis_cache_key(chess.STARTING_FEN, settings, "fake")

    assert cache.get(key) is None
    cache.put(
        key,
        position_key=" ".join(chess.STARTING_FEN.split()[:4]),
        engine_identity="fake",
        settings=settings,
        lines=[line],
    )

    cached = cache.get(key)
    assert cached is not None
    assert cached.engine_identity == "fake"
    assert cached.settings == settings.cache_payload()
    assert cached.lines == (line,)


def test_pipeline_reuses_cache_for_canonical_duplicate_positions(
    tmp_path: Path,
) -> None:
    position_path = tmp_path / "positions.parquet"
    output_path = tmp_path / "analysis.parquet"
    cache = FileAnalysisCache(tmp_path / "cache")
    settings = EngineSettings(nodes=100)
    changed_counters = chess.STARTING_FEN.rsplit(" ", 2)[0] + " 8 25"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "game_id": "game1",
                    "ply": 1,
                    "player_username": "TargetPlayer",
                    "player_color": "white",
                    "fen": chess.STARTING_FEN,
                    "actual_move_uci": "e2e4",
                },
                {
                    "game_id": "game2",
                    "ply": 9,
                    "player_username": "targetplayer",
                    "player_color": "white",
                    "fen": changed_counters,
                    "actual_move_uci": "d2d4",
                },
                {
                    "game_id": "other",
                    "ply": 1,
                    "player_username": "someone_else",
                    "player_color": "white",
                    "fen": chess.STARTING_FEN,
                    "actual_move_uci": "c2c4",
                },
            ]
        ),
        position_path,
    )
    analyzer = FakeAnalyzer()

    summary = analyze_position_dataset(
        position_path,
        "TARGETPLAYER",
        analyzer=analyzer,
        settings=settings,
        cache=cache,
        output_path=output_path,
    )

    assert summary.position_records == 2
    assert summary.unique_positions == 1
    assert summary.engine_calls == 1
    assert summary.cache_hits == 1
    assert len(analyzer.calls) == 1
    assert analyzer.calls[0][0] == " ".join(chess.STARTING_FEN.split()[:4]) + " 0 1"
    rows = pq.read_table(output_path).to_pylist()
    assert len(rows) == 2
    assert [row["cache_hit"] for row in rows] == [False, True]
    assert all(row["score_cp"] == 31 for row in rows)
    assert all(row["nodes_requested"] == 100 for row in rows)
    assert all(row["threads_requested"] == 1 for row in rows)
    assert all(row["hash_mb_requested"] == 16 for row in rows)
    assert all(row["engine_options_json"] == "{}" for row in rows)
    assert all(row["score_perspective"] == "side_to_move" for row in rows)

    second_analyzer = FakeAnalyzer()
    second_summary = analyze_position_dataset(
        position_path,
        "targetplayer",
        analyzer=second_analyzer,
        settings=settings,
        cache=cache,
        output_path=tmp_path / "analysis_second.parquet",
    )
    assert second_summary.engine_calls == 0
    assert second_summary.cache_hits == 2
    assert second_analyzer.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nodes": 0},
        {"multipv": 0},
        {"threads": 0},
        {"hash_mb": 0},
        {"options": (("Skill Level", 1), ("Skill Level", 2))},
    ],
)
def test_engine_settings_reject_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EngineSettings(**kwargs)


def test_missing_stockfish_has_clean_error(tmp_path: Path) -> None:
    with pytest.raises(StockfishNotFoundError, match="executable not found"):
        StockfishAnalyzer(tmp_path / "not-stockfish")
