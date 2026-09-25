"""Predeclare a bounded strength diagnostic and evaluate sealed confirmation."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chess_clone.experiments.deep_style_cohort import load_cohort
from chess_clone.experiments.expanded_evaluation import seeded_key, verify_hashes
from chess_clone.experiments.move_quality import _write_json, run_move_quality
from chess_clone.experiments.profile_ablation import digest
from chess_clone.experiments.winning_chance import build_report, read_quality, paired_comparison


def declare(source, config):
    destination = source / "quality_declaration.json"
    if destination.exists() or (source / "residual/confirmation").exists():
        raise FileExistsError("Declare quality sampling before confirmation")
    settings = json.loads(config.read_text())
    _write_json(destination, {"config": settings, "input_sha256": {
        str(config): digest(config), settings["thresholds"]: digest(settings["thresholds"])}})


def run(source):
    declaration, cohort = load_cohort(source)
    quality = json.loads((source / "quality_declaration.json").read_text())
    verify_hashes(quality["input_sha256"])
    config = quality["config"]
    confirmation = source / "residual/confirmation"
    if json.loads((confirmation / "manifest.json").read_text())["status"] != "complete":
        raise ValueError("Confirmation must be complete")
    output = source / "quality_sample"
    if output.exists():
        raise FileExistsError(output)
    methods = {p.stem: pq.read_table(p).to_pylist() for p in sorted(confirmation.glob("test_predictions_*.parquet"))}
    if not methods:
        raise ValueError("No confirmation predictions")
    common = set.intersection(*({r["decision_id"] for r in rows} for rows in methods.values()))
    sources, selected = [], set()
    output.mkdir()
    for player, record in cohort.items():
        if record["role"] != "confirmation":
            continue
        positions = pq.read_table(record["paths"]["confirmation"]).to_pylist()
        games = sorted({r["game_id"] for r in positions}, key=lambda g: seeded_key(config["seed"], g))[:config["games_per_player"]]
        selected.update(f"{r['game_id']}:{r['ply']}" for r in positions if r["game_id"] in games)
        game_path = next(p for p in record["sha256"] if Path(p).name.startswith(f"games_{player}_"))
        sources.append({"username": player, "positions": record["paths"]["confirmation"], "games": str(Path(game_path).resolve())})
    selected &= common
    if not selected:
        raise ValueError("Empty common confirmation sample")
    for name, rows in methods.items():
        pq.write_table(pa.Table.from_pylist([r for r in rows if r["decision_id"] in selected]), output / f"{name}.parquet")
    _write_json(output / "cohort.json", {"schema_version": 1, "players": sources})
    _write_json(output / "sampling.json", {"decisions": len(selected), "declaration_sha256": digest(source / "quality_declaration.json"),
                                           "confirmation_prediction_sha256": {str(p): digest(p) for p in confirmation.glob("test_predictions_*.parquet")}})
    run_move_quality(output / "cohort.json", output, output / "engine")
    rows, _ = read_quality(output / "engine", expected_split="test")
    thresholds = json.loads(Path(config["thresholds"]).read_text())
    report = build_report(rows, thresholds)
    report["personal_comparisons"] = {name: paired_comparison(
        [r for r in rows if r["method"] == name], [r for r in rows if r["method"] == "personal"],
        thresholds, unit="player_username") for name in ("population", "shared", "wrong")
        if any(r["method"] == name for r in rows)}
    _write_json(output / "winning_chance.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("declare", "run"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/deep_style_quality_v1.json"))
    args = parser.parse_args()
    declare(args.source, args.config) if args.phase == "declare" else run(args.source)
