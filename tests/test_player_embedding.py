from datetime import UTC, datetime, timedelta
import json

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from chess_clone.experiments.style_v2_dataset import prepare_candidate_shards
from chess_clone.modeling.legal_policy import build_all_legal_candidate_rows
from chess_clone.modeling.player_embedding import (
    PlayerConditionedPolicy,
    collate_decisions,
    train_embedding,
    training_losses,
)


def candidate_rows(player, chosen, count=8):
    positions = [
        {"game_id": f"{player}-{index}", "ply": 1, "move_number": 1,
         "fen": chess.STARTING_FEN, "player_username": player,
         "player_color": "white", "actual_move_uci": chosen}
        for index in range(count)
    ]
    dates = {
        row["game_id"]: datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)
        for index, row in enumerate(positions)
    }
    rows = build_all_legal_candidate_rows(
        positions, dates, {game: "history" for game in dates}
    )
    return [dict(row, position_key=" ".join(chess.STARTING_FEN.split()[:4]), base_logit=0.0)
            for row in rows]


def test_sparse_shards_and_training_smoke(tmp_path):
    sources = {}
    for player, chosen in (("alpha", "e2e4"), ("beta", "d2d4")):
        path = tmp_path / f"{player}.parquet"
        pq.write_table(pa.Table.from_pylist(candidate_rows(player, chosen)), path)
        sources[player] = path
    data = tmp_path / "data"
    manifest = prepare_candidate_shards(
        sources, data, max_decisions_per_player=8, decisions_per_shard=4
    )
    assert manifest["decisions"] == 16
    assert len(manifest["shards"]) == 4
    output = tmp_path / "model"
    report = train_embedding(
        data / "manifest.json", output, epochs=3, batch_decisions=8,
        learning_rate=0.02, seed=7,
    )
    assert report["embedding_dimension"] == 32
    assert report["sequence_context_plies"] == 0
    assert report["history"][-1]["total"] < report["history"][0]["total"]
    assert report["in_sample_identity_diagnostic"]["decisions"] == 16
    assert (output / "model.pt").exists()
    assert json.loads((output / "report.json").read_text())["scope"].startswith("Engineering")


def test_player_identity_interaction_can_learn_opposite_preferences():
    torch.manual_seed(1)
    items = []
    for player in (0, 1):
        for _ in range(32):
            features = np.zeros((2, 4), dtype=np.float32)
            features[0, 0] = 1
            features[1, 1] = 1
            items.append({"features": features, "base_logits": np.zeros(2, dtype=np.float32),
                          "chosen": player, "player": player})
    batch = collate_decisions(items)
    model = PlayerConditionedPolicy(2, 4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial = float(training_losses(model, batch)["total"].detach())
    for _ in range(80):
        loss = training_losses(model, batch)["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = float(training_losses(model, batch)["total"].detach())
    assert final < initial / 2
    with torch.no_grad():
        logits = model(batch["features"][:2], batch["base_logits"][:2], torch.tensor([0, 1]))
    assert logits[0, 0] > logits[0, 1]
    assert logits[1, 1] > logits[1, 0]
