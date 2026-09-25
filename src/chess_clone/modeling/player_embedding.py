"""Small joint player-embedding policy and bounded sparse-shard trainer."""

from collections.abc import Iterator
import json
from pathlib import Path
import random

import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from chess_clone.experiments.profile_ablation import digest
from chess_clone.modeling.style_residual import BASE_FIELDS, DIMENSION, VERSION


MODEL_VERSION = "joint-player-embedding-v1"


class PlayerConditionedPolicy(nn.Module):
    """Low-rank player-by-move interaction over frozen population logits."""

    def __init__(self, players: int, feature_dimension: int, embedding_dimension: int = 32):
        super().__init__()
        if players < 2 or feature_dimension < 1 or embedding_dimension < 1:
            raise ValueError("Invalid player-conditioned model dimensions")
        self.players = players
        self.feature_dimension = feature_dimension
        self.embedding_dimension = embedding_dimension
        self.player_embedding = nn.Embedding(players, embedding_dimension)
        self.move_projection = nn.Linear(feature_dimension, embedding_dimension, bias=False)
        self.shared_adjustment = nn.Linear(feature_dimension, 1, bias=False)
        nn.init.normal_(self.player_embedding.weight, std=0.02)
        nn.init.normal_(self.move_projection.weight, std=0.02)
        nn.init.zeros_(self.shared_adjustment.weight)

    def forward(
        self,
        features: torch.Tensor,
        base_logits: torch.Tensor,
        player_indexes: torch.Tensor,
    ) -> torch.Tensor:
        move = self.move_projection(features)
        player = self.player_embedding(player_indexes).unsqueeze(1)
        personal = (move * player).sum(dim=-1)
        shared = self.shared_adjustment(features).squeeze(-1)
        return base_logits + shared + personal


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, -1e9), dim=1)


def training_losses(
    model: PlayerConditionedPolicy,
    batch: dict[str, torch.Tensor],
    *,
    identity_weight: float = 0.1,
    behavior_weight: float = 0.1,
    population_kl_weight: float = 0.02,
) -> dict[str, torch.Tensor]:
    features = batch["features"]
    base = batch["base_logits"]
    mask = batch["mask"]
    chosen = batch["chosen"]
    players = batch["players"]
    indexes = torch.arange(features.shape[0], device=features.device)
    logits = model(features, base, players)
    log_probabilities = masked_log_softmax(logits, mask)
    chosen_log_probability = log_probabilities[indexes, chosen]
    imitation = -chosen_log_probability.mean()

    if model.players > 1:
        wrong_players = (
            players
            + torch.randint(1, model.players, players.shape, device=players.device)
        ) % model.players
        wrong_log_probabilities = masked_log_softmax(
            model(features, base, wrong_players), mask
        )
        identity = F.softplus(
            0.1 - chosen_log_probability + wrong_log_probabilities[indexes, chosen]
        ).mean()
    else:
        identity = imitation.new_zeros(())

    probabilities = log_probabilities.exp() * mask
    style_width = len(BASE_FIELDS)
    expected = (probabilities.unsqueeze(-1) * features[:, :, :style_width]).sum(dim=1)
    observed = features[indexes, chosen, :style_width]
    behavior_terms = []
    for player in torch.unique(players):
        selected = players == player
        behavior_terms.append(
            (expected[selected].mean(dim=0) - observed[selected].mean(dim=0)).square().mean()
        )
    behavior = torch.stack(behavior_terms).mean()

    base_log_probabilities = masked_log_softmax(base, mask)
    population_kl = (
        probabilities * (log_probabilities - base_log_probabilities) * mask
    ).sum(dim=1).mean()
    total = (
        imitation
        + identity_weight * identity
        + behavior_weight * behavior
        + population_kl_weight * population_kl
    )
    return {
        "total": total,
        "imitation": imitation,
        "identity": identity,
        "behavior": behavior,
        "population_kl": population_kl,
    }


def _shard_decisions(path: Path) -> Iterator[dict[str, np.ndarray | int]]:
    with np.load(path) as values:
        shape = tuple(int(value) for value in values["shape"])
        matrix = csr_matrix(
            (values["data"], values["indices"], values["indptr"]), shape=shape
        )
        base = values["base_logits"]
        for start, length, chosen, player in zip(
            values["starts"], values["lengths"], values["chosen"], values["players"], strict=True
        ):
            start, length = int(start), int(length)
            yield {
                "features": matrix[start : start + length].toarray().astype(np.float32),
                "base_logits": base[start : start + length].astype(np.float32),
                "chosen": int(chosen),
                "player": int(player),
            }


class ShardDecisionDataset(IterableDataset):
    def __init__(self, manifest: dict[str, object], *, seed: int, shuffle_buffer: int = 512):
        super().__init__()
        self.manifest = manifest
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        randomizer = random.Random(self.seed)
        shards = list(self.manifest["shards"])
        randomizer.shuffle(shards)
        buffer = []
        for shard in shards:
            for decision in _shard_decisions(Path(shard["path"])):
                buffer.append(decision)
                if len(buffer) >= self.shuffle_buffer:
                    yield buffer.pop(randomizer.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(randomizer.randrange(len(buffer)))


def collate_decisions(items: list[dict[str, np.ndarray | int]]) -> dict[str, torch.Tensor]:
    maximum = max(item["features"].shape[0] for item in items)
    dimension = items[0]["features"].shape[1]
    features = torch.zeros((len(items), maximum, dimension), dtype=torch.float32)
    base = torch.full((len(items), maximum), -1e9, dtype=torch.float32)
    mask = torch.zeros((len(items), maximum), dtype=torch.bool)
    chosen = torch.empty(len(items), dtype=torch.long)
    players = torch.empty(len(items), dtype=torch.long)
    for index, item in enumerate(items):
        length = item["features"].shape[0]
        features[index, :length] = torch.from_numpy(item["features"])
        base[index, :length] = torch.from_numpy(item["base_logits"])
        mask[index, :length] = True
        chosen[index] = int(item["chosen"])
        players[index] = int(item["player"])
    return {"features": features, "base_logits": base, "mask": mask,
            "chosen": chosen, "players": players}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def smoke_diagnostic(
    model: PlayerConditionedPolicy,
    manifest: dict[str, object],
    *,
    batch_decisions: int,
    device: torch.device,
) -> dict[str, float | int]:
    """In-sample identity wiring check; never treat this as model evidence."""

    loader = DataLoader(
        ShardDecisionDataset(manifest, seed=0, shuffle_buffer=1),
        batch_size=batch_decisions,
        collate_fn=collate_decisions,
        num_workers=0,
    )
    totals = {"personal_nll": 0.0, "wrong_nll": 0.0, "population_nll": 0.0}
    preferred = decisions = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            indexes = torch.arange(batch["chosen"].shape[0], device=device)
            personal = masked_log_softmax(
                model(batch["features"], batch["base_logits"], batch["players"]),
                batch["mask"],
            )[indexes, batch["chosen"]]
            wrong = masked_log_softmax(
                model(
                    batch["features"], batch["base_logits"],
                    (batch["players"] + 1) % model.players,
                ),
                batch["mask"],
            )[indexes, batch["chosen"]]
            population = masked_log_softmax(
                batch["base_logits"], batch["mask"]
            )[indexes, batch["chosen"]]
            count = personal.shape[0]
            decisions += count
            totals["personal_nll"] += float(-personal.sum().cpu())
            totals["wrong_nll"] += float(-wrong.sum().cpu())
            totals["population_nll"] += float(-population.sum().cpu())
            preferred += int((personal > wrong).sum().cpu())
    return {
        "decisions": decisions,
        **{key: value / decisions for key, value in totals.items()},
        "personal_preferred_to_wrong_rate": preferred / decisions,
    }


def train_embedding(
    manifest_path: Path,
    output_dir: Path,
    *,
    evaluation_manifest_path: Path | None = None,
    embedding_dimension: int = 32,
    epochs: int = 3,
    batch_decisions: int = 64,
    learning_rate: float = 0.003,
    seed: int = 20260922,
) -> dict[str, object]:
    """Train a bounded smoke model; this function does not perform promotion evaluation."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = json.loads(manifest_path.read_text())
    if manifest["feature_version"] != VERSION or manifest["feature_dimension"] != DIMENSION:
        raise ValueError("Incompatible candidate feature manifest")
    if len(manifest["player_to_index"]) < 2:
        raise ValueError("At least two players are required")
    for shard in manifest["shards"]:
        if digest(shard["path"]) != shard["sha256"]:
            raise ValueError(f"Changed candidate shard: {shard['path']}")
    evaluation_manifest = None
    if evaluation_manifest_path is not None:
        evaluation_manifest = json.loads(evaluation_manifest_path.read_text())
        if (
            evaluation_manifest["feature_version"] != VERSION
            or evaluation_manifest["feature_dimension"] != DIMENSION
            or evaluation_manifest["player_to_index"] != manifest["player_to_index"]
        ):
            raise ValueError("Incompatible evaluation manifest")
        for shard in evaluation_manifest["shards"]:
            if digest(shard["path"]) != shard["sha256"]:
                raise ValueError(f"Changed evaluation shard: {shard['path']}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    model = PlayerConditionedPolicy(
        len(manifest["player_to_index"]), DIMENSION, embedding_dimension
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        loader = DataLoader(
            ShardDecisionDataset(manifest, seed=seed + epoch),
            batch_size=batch_decisions,
            collate_fn=collate_decisions,
            num_workers=0,
        )
        totals = {key: 0.0 for key in ("total", "imitation", "identity", "behavior", "population_kl")}
        decisions = 0
        model.train()
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses = training_losses(model, batch)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = batch["chosen"].shape[0]
            decisions += count
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * count
        history.append({"epoch": epoch + 1, "decisions": decisions,
                        **{key: value / decisions for key, value in totals.items()}})
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "model.pt"
    torch.save(
        {
            "version": MODEL_VERSION,
            "state_dict": model.state_dict(),
            "players": manifest["player_to_index"],
            "feature_dimension": DIMENSION,
            "embedding_dimension": embedding_dimension,
            "sequence_context_plies": 0,
            "seed": seed,
        },
        checkpoint,
    )
    training_diagnostic = smoke_diagnostic(
        model, manifest, batch_decisions=batch_decisions, device=device
    )
    evaluation_diagnostic = (
        smoke_diagnostic(
            model, evaluation_manifest, batch_decisions=batch_decisions, device=device
        )
        if evaluation_manifest is not None
        else None
    )
    report = {
        "status": "implementation_smoke_complete",
        "evidence_status": manifest["evidence_status"],
        "model_version": MODEL_VERSION,
        "device": str(device),
        "embedding_dimension": embedding_dimension,
        "sequence_context_plies": 0,
        "players": len(manifest["player_to_index"]),
        "decisions": manifest["decisions"],
        "history": history,
        "in_sample_identity_diagnostic": training_diagnostic,
        "held_out_smoke_diagnostic": evaluation_diagnostic,
        "manifest_sha256": digest(manifest_path),
        "evaluation_manifest_sha256": (
            digest(evaluation_manifest_path) if evaluation_manifest_path is not None else None
        ),
        "checkpoint_sha256": digest(checkpoint),
        "scope": "Engineering smoke test only; no held-out style or quality claim.",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
