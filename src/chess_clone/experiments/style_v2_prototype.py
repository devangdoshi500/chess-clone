"""Frozen smallest-candidate comparison with an explicit early rejection gate.

The representation sees development history only. The expensive autonomous-play
gate is reached only if the necessary replay gate passes. No model is promoted
by this runner, even if the replay gate passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from chess_clone.experiments.deep_style import evaluate, macro, style_delta
from chess_clone.experiments.move_quality import _write_json
from chess_clone.experiments.style_v2_dataset import digest
from chess_clone.modeling import player_embedding, style_residual
from chess_clone.modeling.player_embedding import (
    MODEL_VERSION, PlayerConditionedPolicy, ShardDecisionDataset,
    collate_decisions, training_losses,
)
from chess_clone.modeling.style_residual import DIMENSION, VERSION, fit_residual


def verify_file(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"Changed sealed input: {path}")


def save_json_sealed(path: Path, value: object) -> None:
    _write_json(path, value)
    _write_json(path.with_suffix(path.suffix + ".seal"), {"sha256": digest(path)})


def load_json_sealed(path: Path) -> dict:
    seal = json.loads(path.with_suffix(path.suffix + ".seal").read_text())
    verify_file(path, seal["sha256"])
    return json.loads(path.read_text())


def select_role_manifest(manifest: dict, players: list[str]) -> dict:
    """Keep original integer identities while filtering whole player shards."""
    selected = set(players)
    if not selected <= set(manifest["player_to_index"]):
        raise ValueError("Unknown player in role manifest")
    shard_paths = []
    for shard in manifest["shards"]:
        with np.load(shard["path"]) as values:
            identities = set(int(value) for value in values["players"])
        allowed = {manifest["player_to_index"][player] for player in selected}
        if identities <= allowed:
            shard_paths.append(shard)
        elif identities & allowed:
            raise ValueError("Shard mixes development and evaluation players")
    return {**manifest, "shards": shard_paths,
            "decisions": sum(item["decisions"] for item in shard_paths)}


def remap_batch(batch: dict, indexes: dict[int, int]) -> dict:
    batch["players"] = torch.tensor([indexes[int(value)] for value in batch["players"]])
    return batch


def residual_from_embedding(model: PlayerConditionedPolicy, identity: int | None) -> dict:
    """Collapse the linear interaction for fast, numerically checked CPU inference."""
    with torch.no_grad():
        embedding = (model.player_embedding.weight.mean(dim=0) if identity is None
                     else model.player_embedding.weight[identity])
        beta = model.shared_adjustment.weight[0] + model.move_projection.weight.T @ embedding
    return {"version": VERSION, "coefficients": beta.detach().cpu().double().tolist(),
            "description": "Exact linear collapse of the frozen embedding scorer"}


def split_audit(selection: dict, development: list[str], evaluation: list[str]) -> dict:
    history = {game for item in selection.values() for game in item["game_ids"]["history"]}
    held_out = {game for item in selection.values() for split in ("validation", "evaluation")
                for game in item["game_ids"][split]}
    if history & held_out:
        raise ValueError(f"Shared games cross history and held-out splits: {sorted(history & held_out)}")
    reserved = {game for player in evaluation for split in ("validation", "evaluation")
                for game in selection[player]["game_ids"][split]}
    quarantine = {
        player: sorted(set(selection[player]["game_ids"]["validation"]) & reserved)
        for player in development
    }
    return {"development_validation_quarantine": quarantine,
            "unique_history_games": len(history),
            "history_held_out_overlap": 0,
            "zero_decision_games": {p: item["zero_decision_games"] for p, item in selection.items()
                                    if item["zero_decision_games"]}}


def replay_gate(records: dict) -> dict:
    metrics = ("style_tv", "negative_log_likelihood", "exact_move_accuracy", "top_3_accuracy",
               "top_5_accuracy", "mean_reciprocal_rank", "multiclass_brier_score",
               "top_1_expected_calibration_error")
    arms = ("population", "shared", "wrong", "personal", "v1_residual")
    panel = {arm: {key: macro(records, arm, key) for key in metrics} for arm in arms}
    personal = panel["personal"]
    checks = {
        f"style_beats_{arm}": personal["style_tv"] is not None and panel[arm]["style_tv"] is not None
        and personal["style_tv"] < panel[arm]["style_tv"]
        for arm in ("v1_residual", "shared", "wrong")
    }
    checks["nll_no_worse_than_shared"] = personal["negative_log_likelihood"] <= panel["shared"]["negative_log_likelihood"]
    checks["exact_within_two_points_of_shared"] = personal["exact_move_accuracy"] >= panel["shared"]["exact_move_accuracy"] - .02
    return {"passed": all(checks.values()), "checks": checks, "macro": panel,
            "style_differences": {f"personal_minus_{arm}": style_delta(records, "personal", arm)
                                  for arm in ("v1_residual", "shared", "wrong", "population")}}


def _train(
    manifest: dict, players: list[str], config: dict, output: Path,
    *, representation: PlayerConditionedPolicy | None = None,
) -> tuple[PlayerConditionedPolicy, list[dict]]:
    """Checkpoint each full epoch; resume optimizer and RNG exactly."""
    ordered = sorted(players)
    mapping = {manifest["player_to_index"][player]: index for index, player in enumerate(ordered)}
    manifest = select_role_manifest(manifest, ordered)
    torch.manual_seed(config["seed"])
    model = PlayerConditionedPolicy(len(players), DIMENSION, config["embedding_dimension"])
    adapting = representation is not None
    if adapting:
        model.move_projection.load_state_dict(representation.move_projection.state_dict())
        model.shared_adjustment.load_state_dict(representation.shared_adjustment.state_dict())
        for layer in (model.move_projection, model.shared_adjustment):
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
        with torch.no_grad():
            model.player_embedding.weight.copy_(representation.player_embedding.weight.mean(0))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    history = []
    if checkpoint_path.exists():
        verify_file(checkpoint_path, json.loads((output / "checkpoint.seal").read_text())["sha256"])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["players"] != ordered or state["config"] != config:
            raise ValueError("Training checkpoint configuration changed")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_state"])
        history = state["history"]
    epochs = config["adaptation_epochs"] if adapting else config["epochs"]
    for epoch in range(len(history), epochs):
        loader = DataLoader(
            ShardDecisionDataset(manifest, seed=config["seed"] + epoch, shuffle_buffer=128),
            batch_size=config["batch_decisions"], collate_fn=collate_decisions,
        )
        totals = {key: 0.0 for key in ("total", "imitation", "identity", "behavior", "population_kl")}
        decisions = 0
        started = time.perf_counter()
        model.train()
        for batch in loader:
            batch = remap_batch(batch, mapping)
            losses = training_losses(
                model, batch, identity_weight=config["identity_weight"],
                behavior_weight=config["behavior_weight"], population_kl_weight=config["population_kl_weight"],
            )
            if not all(torch.isfinite(value) for value in losses.values()):
                raise ValueError("Non-finite training objective")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            count = len(batch["chosen"])
            decisions += count
            for key in totals:
                totals[key] += float(losses[key].detach()) * count
        if decisions != manifest["decisions"]:
            raise ValueError("Training decision count differs from manifest")
        history.append({"epoch": epoch + 1, "decisions": decisions,
                        "seconds": time.perf_counter() - started,
                        **{key: value / decisions for key, value in totals.items()}})
        state = {"version": MODEL_VERSION, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "rng_state": torch.get_rng_state(),
                 "players": ordered, "config": config, "history": history}
        temporary = output / "checkpoint.partial"
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)
        _write_json(output / "checkpoint.seal", {"sha256": digest(checkpoint_path)})
        _write_json(output / "training.json", history)
        print(f"{'Embedding adaptation' if adapting else 'Joint training'} epoch {epoch + 1}/{epochs}: "
              f"NLL {history[-1]['imitation']:.5f}, {history[-1]['seconds']:.1f}s", flush=True)
    model.eval()
    return model, history


def _score_role(
    data: Path, output: Path, players: list[str], split: str,
    model: PlayerConditionedPolicy, shared: dict, quarantine: dict, config: dict,
) -> dict:
    ordered = sorted(players)
    records = {}
    output.mkdir(parents=True, exist_ok=True)
    for index, player in enumerate(ordered):
        result_path = output / f"{player}.json"
        if result_path.exists():
            records[player] = load_json_sealed(result_path)
            continue
        residual = fit_baseline(data, output, player, config["residual_penalty"])
        rows = pq.read_table(data / "candidate-cache" / split / f"{player}.parquet").to_pylist()
        excluded = set(quarantine.get(player, []))
        rows = [row for row in rows if row["game_id"] not in excluded]
        arms = {"population": None, "shared": shared,
                "personal": residual_from_embedding(model, index),
                "wrong": residual_from_embedding(model, (index + 1) % len(ordered)),
                "v1_residual": residual}
        records[player] = {"donor": ordered[(index + 1) % len(ordered)],
                           "quarantined_games": sorted(excluded),
                           "scored_games": len({row["game_id"] for row in rows})}
        for arm, residual_model in arms.items():
            records[player][arm], _ = evaluate(rows, residual_model)
        save_json_sealed(result_path, records[player])
        print(f"{split} replay: {len(records)}/{len(ordered)} players", flush=True)
    return records


def fit_baseline(data: Path, output: Path, player: str, penalty: float) -> dict:
    """Independently checkpoint history-only controls while preparing neural data."""
    history_path = data / "candidate-cache/history" / f"{player}.parquet"
    metadata = json.loads(history_path.with_suffix(".json").read_text())
    verify_file(history_path, metadata["sha256"])
    inputs = {"history_sha256": digest(history_path), "penalty": penalty,
              "residual_code_sha256": digest(Path(style_residual.__file__))}
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{player}-v1.json"
    if path.exists():
        saved = load_json_sealed(path)
        if saved.get("inputs") != inputs:
            raise ValueError(f"Baseline history/penalty changed: {player}")
        return saved["model"]
    rows = pq.read_table(history_path).to_pylist()
    model = fit_residual(rows, penalty=penalty)
    save_json_sealed(path, {"inputs": inputs, "model": model})
    return model


def run(config_path: Path, data: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text())
    if (config["schema_version"] != 1 or config["device"] != "cpu"
            or config["sequence_context_plies"] != 0 or config["minimum_style_opportunities"] != 20):
        raise ValueError("Unsupported smallest-candidate protocol")
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    bundle_path = data / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    verify_file(data / "selection.json", bundle["selection_sha256"])
    selection = json.loads((data / "selection.json").read_text())
    development, evaluation_players = bundle["development_players"], bundle["evaluation_players"]
    if (bundle["experiment_id"] != "style-model-v2-prototype"
            or len(development) != 14 or len(evaluation_players) != 6
            or len(set(development + evaluation_players)) != 20
            or set(selection) != set(development + evaluation_players)):
        raise ValueError("Expected the frozen 14-development/6-evaluation prototype cohort")
    for record in selection.values():
        if {name: len(ids) for name, ids in record["game_ids"].items()} != {
            "history": 400, "validation": 50, "evaluation": 50,
        }:
            raise ValueError("Expected the frozen 400/50/50 whole-game allocations")
    audit = split_audit(selection, development, evaluation_players)
    manifests = {}
    for split, item in bundle["splits"].items():
        verify_file(Path(item["path"]), item["sha256"])
        manifest = json.loads(Path(item["path"]).read_text())
        if manifest["feature_version"] != VERSION or manifest["feature_dimension"] != DIMENSION:
            raise ValueError("Incompatible feature schema")
        for entry in [*manifest["shards"], *manifest["inputs"].values()]:
            verify_file(Path(entry["path"]), entry["sha256"])
        manifests[split] = manifest
    if len({json.dumps(item["player_to_index"], sort_keys=True) for item in manifests.values()}) != 1:
        raise ValueError("Player indexes differ between splits")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {"protocol_sha256": digest(config_path), "bundle_sha256": digest(bundle_path),
              "runner_sha256": digest(Path(__file__)),
              "embedding_sha256": digest(Path(player_embedding.__file__)),
              "residual_sha256": digest(Path(style_residual.__file__))}
    declaration_path = output / "declaration.json"
    if declaration_path.exists():
        if json.loads(declaration_path.read_text())["inputs"] != inputs:
            raise ValueError("Prototype run inputs changed; resume refused")
    else:
        _write_json(declaration_path, {"inputs": inputs, "config": config, "split_audit": audit})
    report_path = output / "report.json"
    if report_path.exists():
        return load_json_sealed(report_path)
    model, history = _train(manifests["history"], development, config, output / "development-model")
    shared = residual_from_embedding(model, None)
    records = _score_role(data, output / "development-replay", development, "validation", model,
                          shared, audit["development_validation_quarantine"], config)
    gate = replay_gate(records)
    report = {"experiment_id": config["experiment_id"], "evidence_status": config["evidence_status"],
              "development": gate, "development_players": len(development),
              "development_decisions": sum(r["personal"]["ranking"]["decisions"] for r in records.values()),
              "development_assigned_validation_games": sum(selection[p]["games"]["validation"] for p in development),
              "development_scored_validation_games": sum(r["scored_games"] for r in records.values()),
              "split_audit": audit, "training": history,
              "development_checkpoint_sha256": digest(output / "development-model/checkpoint.pt"),
              "evaluation_player_outcomes_opened": False,
              "generated_identity_gate": "not_run", "engine_quality_gate": "not_run",
              "scale_authorized_by_evidence": False}
    if not gate["passed"]:
        report["status"] = "rejected_at_necessary_development_replay_gate"
        report["next_action"] = "Retain the v1 residual. Diagnose the failed smallest candidate using development data; preregister a new ablation before another run."
    else:
        # Evaluation identities cannot update the shared representation.
        adapted, _ = _train(manifests["history"], evaluation_players, config,
                            output / "evaluation-model", representation=model)
        for name in ("move_projection", "shared_adjustment"):
            for original, actual in zip(getattr(model, name).parameters(), getattr(adapted, name).parameters(), strict=True):
                if not torch.equal(original, actual):
                    raise ValueError("Representation changed during evaluation-player adaptation")
        _write_json(output / "adaptation-seal.json", {
            "development_model_sha256": digest(output / "development-model/checkpoint.pt"),
            "adapted_model_sha256": digest(output / "evaluation-model/checkpoint.pt"),
            "scope": "History-only embeddings; evaluation labels remain unopened pending generated selection."
        })
        report["status"] = "replay_passed_generated_and_quality_gates_required"
        report["next_action"] = "Complete the 25-game-per-player/arm generated identity and sampled engine-quality gates before opening reserved evaluation outcomes or funding scale."
    save_json_sealed(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/style_model_v2_prototype_run.json"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/benchmarks/style-v2-prototype-data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/style-v2-smallest-candidate-v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.data, args.output), indent=2))
