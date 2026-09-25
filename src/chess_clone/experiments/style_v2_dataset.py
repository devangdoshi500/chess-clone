"""Create exact, bounded sparse candidate shards for Style Model v2."""

from collections.abc import Iterator
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from catboost import CatBoostRanker
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.style_v2_capacity import _player_from_path
from chess_clone.modeling.boosted import predict_relevance_scores
from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows
from chess_clone.modeling.style_residual import DIMENSION, VERSION, feature_matrix


REQUIRED_FIELDS = (
    "decision_id",
    "game_id",
    "split",
    "player_username",
    "game_phase",
    "move_number",
    "candidate_move_uci",
    "chosen",
    "candidate_is_capture",
    "candidate_gives_check",
    "candidate_is_castle",
    "candidate_is_promotion",
    "candidate_is_queen_trade",
    "candidate_is_pawn_push",
    "candidate_moves_queen",
    "candidate_enters_opponent_territory",
    "candidate_targets_king_zone",
    "candidate_piece_moved",
    "candidate_destination_wing",
    "candidate_is_development",
    "candidate_is_hanging_after",
    "candidate_gives_mate",
    "candidate_material_gain",
    "candidate_attackers_after",
    "candidate_defenders_after",
    "candidate_center_control_after",
    "candidate_opponent_mobility_after",
    "position_key",
    "base_logit",
)

GAME_COLUMNS = (
    "game_id", "played_at", "white_username", "black_username", "white_rating",
    "black_rating", "rated", "variant", "speed", "time_control", "total_plies",
)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    """Publish metadata only after the complete JSON has reached a sibling file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_candidate_cache(
    path: Path,
    positions: list[dict[str, object]],
    dates: dict[str, datetime],
    split: str,
    model: CatBoostRanker,
    fields: tuple[str, ...],
    temperature: float,
) -> None:
    """Write a complete candidate cache without exposing a partial Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as handle:
        temporary = Path(handle.name)
    writer = None
    try:
        try:
            for start in range(0, len(positions), 200):
                batch = positions[start:start + 200]
                keys = {
                    f"{row['game_id']}:{row['ply']}": " ".join(str(row["fen"]).split()[:4])
                    for row in batch
                }
                rows = build_all_legal_candidate_rows(
                    batch, dates, {game_id: split for game_id in dates}
                )
                scores = predict_relevance_scores(model, rows, fields)
                compact = [
                    {
                        **{field: row[field] for field in REQUIRED_FIELDS if field not in {"position_key", "base_logit"}},
                        "position_key": keys[row["decision_id"]],
                        "base_logit": float(score) / temperature,
                    }
                    for row, score in zip(rows, scores, strict=True)
                ]
                table = pa.Table.from_pylist(compact)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
            if writer is None:
                raise ValueError(f"No positions for {path}")
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_candidate_cache(
    path: Path,
    positions: list[dict[str, object]],
    dates: dict[str, datetime],
    split: str,
    model: CatBoostRanker,
    fields: tuple[str, ...],
    temperature: float,
    expected_inputs: dict[str, object],
) -> None:
    """Recover a cache missing metadata only when a fresh build is byte-identical."""

    metadata_path = path.with_suffix(".json")
    if metadata_path.exists() and not path.exists():
        raise ValueError(f"Prototype cache metadata has no Parquet: {metadata_path}")
    if path.exists():
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if (metadata.get("inputs") != expected_inputs
                    or metadata.get("sha256") != digest(path)
                    or metadata.get("decisions") != len(positions)):
                raise ValueError(f"Changed prototype cache: {path}")
            return
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".cache-recovery-") as name:
            rebuilt = Path(name) / path.name
            _atomic_candidate_cache(rebuilt, positions, dates, split, model, fields, temperature)
            if digest(rebuilt) != digest(path):
                raise ValueError(f"Unsealed prototype cache differs from fresh build: {path}")
    else:
        _atomic_candidate_cache(path, positions, dates, split, model, fields, temperature)
    _atomic_json(metadata_path, {"inputs": expected_inputs, "sha256": digest(path),
                                 "decisions": len(positions)})


def _ensure_position_cache(path: Path, positions: list[dict[str, object]]) -> None:
    if path.exists():
        if pq.read_table(path).to_pylist() != positions:
            raise ValueError(f"Changed selected position cache: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(pa.Table.from_pylist(positions), temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _eligible_game(
    row: dict[str, object], player: str, scope: dict[str, object], time_controls: set[str]
) -> bool:
    white = str(row["white_username"]).casefold()
    black = str(row["black_username"]).casefold()
    if player not in {white, black}:
        return False
    rating = row["white_rating"] if player == white else row["black_rating"]
    played_at = row["played_at"]
    since = datetime.fromisoformat(str(scope["since"]))
    until = datetime.fromisoformat(str(scope["until"]))
    return bool(
        row["rated"]
        and str(row["variant"]).casefold() == str(scope["variant"]).casefold()
        and str(row["speed"]).casefold() == str(scope["speed"]).casefold()
        and rating is not None
        and int(scope["rating_min"]) <= int(rating) <= int(scope["rating_max"])
        and played_at is not None
        and since < played_at <= until
        and str(row["time_control"]) in time_controls
    )


def _player_source_pairs(capacity: dict[str, object], player: str) -> list[tuple[Path, Path]]:
    pairs = []
    for item in capacity["input_files"]:
        games = Path(item["path"])
        if _player_from_path(games) == player:
            if digest(games) != item["sha256"]:
                raise ValueError(f"Capacity input changed: {games}")
            positions = games.with_name("positions_" + games.name.removeprefix("games_"))
            if not positions.exists():
                raise FileNotFoundError(positions)
            pairs.append((games, positions))
    if not pairs:
        raise FileNotFoundError(f"No normalized sources for {player}")
    return sorted(pairs)


def _select_player_positions(
    pairs: list[tuple[Path, Path]],
    player: str,
    scope: dict[str, object],
    time_controls: set[str],
    split_counts: dict[str, int],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, datetime], dict[str, object]]:
    """Select the declared newest unique games and their target-player decisions."""

    games: dict[str, dict[str, object]] = {}
    game_sources: dict[str, set[Path]] = {}
    for game_path, position_path in pairs:
        for row in pq.read_table(game_path, columns=list(GAME_COLUMNS)).to_pylist():
            if not _eligible_game(row, player, scope, time_controls):
                continue
            game_id = str(row["game_id"])
            normalized = {key: row[key] for key in GAME_COLUMNS}
            if game_id in games and games[game_id] != normalized:
                raise ValueError(f"Conflicting duplicate game metadata: {player}/{game_id}")
            games[game_id] = normalized
            game_sources.setdefault(game_id, set()).add(position_path)
    required = sum(split_counts.values())
    if len(games) < required:
        raise ValueError(f"{player} has {len(games)} eligible games; {required} required")
    ordered = sorted(games, key=lambda game_id: (games[game_id]["played_at"], game_id))[-required:]
    boundaries = np.cumsum([split_counts[name] for name in split_counts]).tolist()
    for boundary in boundaries[:-1]:
        if games[ordered[boundary - 1]]["played_at"] == games[ordered[boundary]]["played_at"]:
            raise ValueError(f"Tied chronological split boundary for {player}")
    game_split = {}
    start = 0
    for name, count in split_counts.items():
        for game_id in ordered[start:start + count]:
            game_split[game_id] = name
        start += count
    selected = set(ordered)
    rows_by_key: dict[tuple[str, int], dict[str, object]] = {}
    source_paths = sorted({path for game_id in selected for path in game_sources[game_id]})
    for position_path in source_paths:
        table = pq.read_table(position_path)
        for row in table.to_pylist():
            game_id = str(row["game_id"])
            if game_id not in selected or str(row["player_username"]).casefold() != player:
                continue
            key = (game_id, int(row["ply"]))
            if key in rows_by_key and rows_by_key[key] != row:
                raise ValueError(f"Conflicting duplicate position: {player}/{game_id}/{row['ply']}")
            rows_by_key[key] = row
    empty_games = []
    for game_id in ordered:
        game = games[game_id]
        white = str(game["white_username"]).casefold() == player
        expected_plies = set(range(1 if white else 2, int(game["total_plies"]) + 1, 2))
        actual_plies = {ply for gid, ply in rows_by_key if gid == game_id}
        if actual_plies != expected_plies:
            raise ValueError(f"Incomplete whole-game positions: {player}/{game_id}")
        if not expected_plies:
            empty_games.append(game_id)
        for ply in expected_plies:
            row = rows_by_key[game_id, ply]
            if row["player_color"] != ("white" if white else "black") or row["time_control"] not in time_controls:
                raise ValueError(f"Position context disagrees with game: {player}/{game_id}")
    dates = {game_id: games[game_id]["played_at"] for game_id in ordered}
    splits = {name: [] for name in split_counts}
    for (game_id, _), row in sorted(
        rows_by_key.items(), key=lambda item: (dates[item[0][0]], item[0][0], item[0][1])
    ):
        splits[game_split[game_id]].append(row)
    summary = {
        "games": {name: split_counts[name] for name in split_counts},
        "decisions": {name: len(splits[name]) for name in split_counts},
        "first_game_at": dates[ordered[0]].isoformat(),
        "last_game_at": dates[ordered[-1]].isoformat(),
        "game_ids": {name: [game_id for game_id in ordered if game_split[game_id] == name]
                     for name in split_counts},
        "selected_dates": {g: dates[g].isoformat() for g in ordered},
        "zero_decision_games": empty_games,
        "sources": [{"games": str(g.resolve()), "games_sha256": digest(g),
                     "positions": str(p.resolve()), "positions_sha256": digest(p)}
                    for g, p in pairs],
    }
    return splits, dates, summary


def prepare_prototype_candidate_caches(
    config_path: Path,
    cohort_path: Path,
    capacity_report_path: Path,
    artifact_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build the frozen strict-3+0 400/50/50 candidate caches and sparse shards."""

    config = json.loads(config_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    capacity = json.loads(capacity_report_path.read_text())
    players = cohort["development_players"] + cohort["evaluation_players"]
    split_counts = {
        "history": int(config["prototype"]["history_games"]),
        "validation": int(config["prototype"]["validation_games"]),
        "evaluation": int(config["prototype"]["evaluation_games"]),
    }
    if cohort["game_splits"] != split_counts or cohort["time_controls"] != ["180+0"]:
        raise ValueError("Prototype cohort does not match the declared strict-3+0 split")
    if digest(config_path) != cohort["declaration_sha256"]:
        raise ValueError("Style v2 declaration hash changed")
    if digest(capacity_report_path) != cohort["capacity_report_sha256"]:
        raise ValueError("Capacity report hash changed")
    if output_dir.exists() and (output_dir / "bundle.json").exists():
        raise FileExistsError(f"Completed prototype data already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = CatBoostRanker().load_model(artifact_dir / "safe_population.cbm")
    fields = tuple(json.loads((artifact_dir / "feature_sets.json").read_text())["safe_population"])
    temperature = float(json.loads((artifact_dir / "metrics.json").read_text())["safe_population"]["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid frozen population temperature")
    selection_path = output_dir / "selection.json"
    previous_selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    selection = {}
    cache_sources = {name: {} for name in split_counts}
    for index, player in enumerate(players, start=1):
        pairs = _player_source_pairs(capacity, player)
        splits, dates, summary = _select_player_positions(
            pairs, player, config["scope"], set(cohort["time_controls"]), split_counts
        )
        if player in previous_selection and previous_selection[player] != summary:
            raise ValueError(f"Selected source positions changed for {player}")
        selection[player] = summary
        for split, positions in splits.items():
            path = output_dir / "candidate-cache" / split / f"{player}.parquet"
            expected = {
                "player": player,
                "split": split,
                "game_ids": summary["game_ids"][split],
                "config_sha256": digest(config_path),
                "cohort_sha256": digest(cohort_path),
                "population_model_sha256": digest(artifact_dir / "safe_population.cbm"),
                "population_features_sha256": digest(artifact_dir / "feature_sets.json"),
                "population_metrics_sha256": digest(artifact_dir / "metrics.json"),
                "source_positions": summary["sources"],
                "candidate_code_sha256": digest(Path(__file__)),
            }
            _ensure_candidate_cache(path, positions, dates, split, model, fields, temperature, expected)
            cache_sources[split][player] = path
            position_path = output_dir / "positions" / split / f"{player}.parquet"
            _ensure_position_cache(position_path, positions)
        _atomic_json(output_dir / "selection.json", selection)
        print(f"Style v2 exact caches: {index}/{len(players)} players", flush=True)
    shard_manifests = {}
    status = "frozen strict-3+0 Style Model v2 prototype; development-only architecture comparison"
    for split, sources in cache_sources.items():
        shard_dir = output_dir / "shards" / split
        manifest = _load_or_prepare_shards(sources, shard_dir, status)
        shard_manifests[split] = {
            "path": str((shard_dir / "manifest.json").resolve()),
            "sha256": digest(shard_dir / "manifest.json"),
            "decisions": manifest["decisions"],
            "candidate_rows": manifest["candidate_rows"],
        }
    bundle = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": digest(config_path),
        "cohort_sha256": digest(cohort_path),
        "capacity_report_sha256": digest(capacity_report_path),
        "population_artifacts": {str(path.name): digest(path) for path in (
            artifact_dir / "safe_population.cbm", artifact_dir / "feature_sets.json",
            artifact_dir / "metrics.json")},
        "players": players,
        "development_players": cohort["development_players"],
        "evaluation_players": cohort["evaluation_players"],
        "splits": shard_manifests,
        "selection_sha256": digest(output_dir / "selection.json"),
        "evidence_status": status,
    }
    _atomic_json(output_dir / "bundle.json", bundle)
    return bundle


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def iter_decision_groups(path: Path) -> Iterator[list[dict[str, object]]]:
    """Stream complete contiguous decisions without loading a Parquet into RAM."""

    parquet = pq.ParquetFile(path)
    missing = set(REQUIRED_FIELDS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Candidate cache is missing fields: {sorted(missing)}")
    current: list[dict[str, object]] = []
    current_id = None
    seen: set[str] = set()
    for batch in parquet.iter_batches(batch_size=65536, columns=list(REQUIRED_FIELDS)):
        for row in batch.to_pylist():
            decision_id = str(row["decision_id"])
            if current_id is not None and decision_id != current_id:
                if decision_id in seen:
                    raise ValueError(f"Non-contiguous decision: {decision_id}")
                seen.add(str(current_id))
                yield current
                current = []
            current_id = decision_id
            current.append(row)
    if current:
        yield current


def _write_shard(
    groups: list[list[dict[str, object]]],
    player_index: int,
    path: Path,
) -> dict[str, object]:
    rows = [row for group in groups for row in group]
    matrix = feature_matrix(rows).tocsr().astype(np.float32)
    lengths = np.asarray([len(group) for group in groups], dtype=np.int32)
    starts = np.r_[0, np.cumsum(lengths[:-1])].astype(np.int64)
    chosen = []
    for group in groups:
        selected = [index for index, row in enumerate(group) if bool(row["chosen"])]
        if len(selected) != 1:
            raise ValueError("Each candidate decision must contain exactly one chosen move")
        chosen.append(selected[0])
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices.astype(np.int32),
        indptr=matrix.indptr.astype(np.int64),
        shape=np.asarray(matrix.shape, dtype=np.int64),
        base_logits=np.asarray([row["base_logit"] for row in rows], dtype=np.float32),
        starts=starts,
        lengths=lengths,
        chosen=np.asarray(chosen, dtype=np.int32),
        players=np.full(len(groups), player_index, dtype=np.int32),
    )
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "decisions": len(groups),
        "candidate_rows": len(rows),
    }


def prepare_candidate_shards(
    player_sources: dict[str, Path],
    output_dir: Path,
    *,
    max_decisions_per_player: int | None = None,
    decisions_per_shard: int = 512,
    evidence_status: str = "implementation_smoke_only",
) -> dict[str, object]:
    """Convert compact candidate Parquets into sparse, bounded NPZ shards."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    if len(player_sources) < 2:
        raise ValueError("Joint player training requires at least two players")
    if decisions_per_shard < 1:
        raise ValueError("decisions_per_shard must be positive")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    players = sorted(player.casefold() for player in player_sources)
    if len(set(players)) != len(players):
        shutil.rmtree(staging)
        raise ValueError("Duplicate case-insensitive player")
    player_to_index = {player: index for index, player in enumerate(players)}
    shards = []
    inputs = {}
    try:
        for player in players:
            source = next(path for name, path in player_sources.items() if name.casefold() == player)
            inputs[player] = {"path": str(source.resolve()), "sha256": digest(source)}
            pending = []
            used = 0
            for group in iter_decision_groups(source):
                if any(str(row["player_username"]).casefold() != player for row in group):
                    raise ValueError(f"Player mismatch in {source}")
                pending.append(group)
                used += 1
                if len(pending) == decisions_per_shard:
                    name = f"{player}-{len(shards):05d}.npz"
                    shard = _write_shard(pending, player_to_index[player], staging / name)
                    shards.append({**shard, "path": str((output_dir / name).resolve())})
                    pending = []
                if max_decisions_per_player is not None and used >= max_decisions_per_player:
                    break
            if pending:
                name = f"{player}-{len(shards):05d}.npz"
                shard = _write_shard(pending, player_to_index[player], staging / name)
                shards.append({**shard, "path": str((output_dir / name).resolve())})
            if used == 0:
                raise ValueError(f"No candidate decisions for {player}")
        manifest = {
            "schema_version": 1,
            "feature_version": VERSION,
            "feature_dimension": DIMENSION,
            "player_to_index": player_to_index,
            "evidence_status": evidence_status,
            "sequence_context_plies": 0,
            "inputs": inputs,
            "shards": shards,
            "decisions": sum(int(shard["decisions"]) for shard in shards),
            "candidate_rows": sum(int(shard["candidate_rows"]) for shard in shards),
        }
        _atomic_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_or_prepare_shards(
    sources: dict[str, Path], shard_dir: Path, evidence_status: str
) -> dict[str, object]:
    """Reuse sealed shards or preserve an old incomplete directory and rebuild."""

    manifest_path = shard_dir / "manifest.json"
    if shard_dir.exists() and not manifest_path.exists():
        quarantine = shard_dir.with_name(f"{shard_dir.name}.incomplete-{uuid4().hex}")
        shard_dir.rename(quarantine)
    if not shard_dir.exists():
        return prepare_candidate_shards(
            sources, shard_dir, decisions_per_shard=512, evidence_status=evidence_status
        )
    manifest = json.loads(manifest_path.read_text())
    players = sorted(player.casefold() for player in sources)
    if (manifest.get("schema_version") != 1
            or manifest.get("feature_version") != VERSION
            or manifest.get("feature_dimension") != DIMENSION
            or manifest.get("sequence_context_plies") != 0
            or manifest.get("evidence_status") != evidence_status
            or manifest.get("player_to_index") != {p: i for i, p in enumerate(players)}
            or set(manifest.get("inputs", {})) != set(players)):
        raise ValueError(f"Changed shard manifest schema or players: {manifest_path}")
    for player in players:
        source = next(path for name, path in sources.items() if name.casefold() == player)
        recorded = manifest["inputs"][player]
        if recorded != {"path": str(source.resolve()), "sha256": digest(source)}:
            raise ValueError(f"Changed inputs for {shard_dir} shards")
    shards = manifest.get("shards", [])
    if (not shards
            or manifest.get("decisions") != sum(int(item["decisions"]) for item in shards)
            or manifest.get("candidate_rows") != sum(int(item["candidate_rows"]) for item in shards)):
        raise ValueError(f"Invalid shard counts: {manifest_path}")
    if len({item["path"] for item in shards}) != len(shards):
        raise ValueError(f"Duplicate shard paths: {manifest_path}")
    for item in shards:
        path = Path(item["path"])
        if path.resolve().parent != shard_dir.resolve() or digest(path) != item["sha256"]:
            raise ValueError(f"Changed shard file: {path}")
    return manifest


def prepare_v1_cache_smoke(
    cohort_path: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    players_limit: int = 2,
    max_decisions_per_player: int = 256,
) -> dict[str, object]:
    """Prepare train/evaluation smoke shards, explicitly not Style v2 evidence."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    cohort = json.loads(cohort_path.read_text())
    ordered = cohort["development_players"] + cohort["evaluation_players"]
    selected = [
        player for player in ordered
        if (cache_dir / f"{player}_history.parquet").exists()
        and (cache_dir / f"{player}_validation.parquet").exists()
    ][:players_limit]
    if len(selected) < players_limit:
        raise FileNotFoundError("Not enough players have matching v1 history/validation caches")
    status = (
        "implementation_smoke_only; reuses mixed-time-control v1 data and must "
        "not be reported as Style Model v2 evidence"
    )
    training = prepare_candidate_shards(
        {player: cache_dir / f"{player}_history.parquet" for player in selected},
        output_dir / "train",
        max_decisions_per_player=max_decisions_per_player,
        decisions_per_shard=min(128, max_decisions_per_player),
        evidence_status=status,
    )
    evaluation_limit = max(64, max_decisions_per_player // 4)
    evaluation = prepare_candidate_shards(
        {player: cache_dir / f"{player}_validation.parquet" for player in selected},
        output_dir / "evaluation",
        max_decisions_per_player=evaluation_limit,
        decisions_per_shard=min(128, evaluation_limit),
        evidence_status=status,
    )
    bundle = {
        "schema_version": 1,
        "players": selected,
        "train_manifest": str((output_dir / "train/manifest.json").resolve()),
        "evaluation_manifest": str((output_dir / "evaluation/manifest.json").resolve()),
        "train_decisions": training["decisions"],
        "evaluation_decisions": evaluation["decisions"],
        "evidence_status": status,
    }
    (output_dir / "bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return bundle
