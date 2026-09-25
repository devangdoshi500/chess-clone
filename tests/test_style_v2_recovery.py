"""Crash recovery and provenance checks for Style v2 data preparation."""

from datetime import UTC, datetime
import json
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chess_clone.experiments import style_v2_dataset as dataset
from test_player_embedding import candidate_rows


def _sources(tmp_path):
    sources = {}
    for player, move in (("alpha", "e2e4"), ("beta", "d2d4")):
        path = tmp_path / f"{player}.parquet"
        pq.write_table(pa.Table.from_pylist(candidate_rows(player, move, count=2)), path)
        sources[player] = path
    return sources


def test_interrupted_shards_leave_no_published_directory_and_retry(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    output = tmp_path / "shards"
    original = dataset._write_shard
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted shard build")
        return original(*args)

    with monkeypatch.context() as patch:
        patch.setattr(dataset, "_write_shard", interrupted)
        with pytest.raises(RuntimeError, match="interrupted shard"):
            dataset.prepare_candidate_shards(sources, output, decisions_per_shard=1)
    assert not output.exists()
    assert not list(tmp_path.glob(".shards.partial-*"))

    manifest = dataset.prepare_candidate_shards(sources, output, decisions_per_shard=1)
    assert manifest["decisions"] == 4
    assert all(dataset.digest(Path(item["path"])) == item["sha256"] for item in manifest["shards"])
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_legacy_incomplete_shards_are_preserved_and_completed_inputs_are_checked(tmp_path):
    sources = _sources(tmp_path)
    output = tmp_path / "shards"
    output.mkdir()
    (output / "alpha-00000.npz").write_bytes(b"unfinished")
    manifest = dataset._load_or_prepare_shards(sources, output, "test evidence")
    quarantined = list(tmp_path.glob("shards.incomplete-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "alpha-00000.npz").read_bytes() == b"unfinished"
    assert dataset._load_or_prepare_shards(sources, output, "test evidence") == manifest

    pq.write_table(pa.Table.from_pylist(candidate_rows("alpha", "d2d4", count=2)), sources["alpha"])
    with pytest.raises(ValueError, match="Changed inputs"):
        dataset._load_or_prepare_shards(sources, output, "test evidence")


def test_candidate_cache_without_metadata_is_recovered_only_if_identical(tmp_path, monkeypatch):
    position = {"game_id": "g", "ply": 1, "fen": chess.STARTING_FEN,
                "move_number": 1, "player_username": "alpha", "player_color": "white",
                "actual_move_uci": "e2e4"}
    dates = {"g": datetime(2026, 1, 1, tzinfo=UTC)}
    path = tmp_path / "alpha.parquet"
    inputs = {"source_sha256": "fixed"}
    monkeypatch.setattr(dataset, "predict_relevance_scores",
                        lambda model, rows, fields: np.zeros(len(rows)))

    dataset._atomic_candidate_cache(path, [position], dates, "history", None, (), 1.0)
    original_hash = dataset.digest(path)
    dataset._ensure_candidate_cache(path, [position], dates, "history", None, (), 1.0, inputs)
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata == {"inputs": inputs, "sha256": original_hash, "decisions": 1}
    assert dataset.digest(path) == original_hash

    path.with_suffix(".json").unlink()
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="Unsealed prototype cache differs"):
        dataset._ensure_candidate_cache(path, [position], dates, "history", None, (), 1.0, inputs)
    assert not path.with_suffix(".json").exists()


def test_candidate_cache_exception_removes_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "alpha.parquet"

    def interrupted(*args):
        raise RuntimeError("candidate generation failed")

    monkeypatch.setattr(dataset, "build_all_legal_candidate_rows", interrupted)
    with pytest.raises(RuntimeError, match="candidate generation"):
        dataset._atomic_candidate_cache(path, [{"game_id": "g", "ply": 1,
                                               "fen": chess.STARTING_FEN}], {},
                                        "history", None, (), 1.0)
    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))
