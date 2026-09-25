"""Reconstruct the curated public showcase from sealed v1 confirmation artifacts.

Run from the repository root. Prints anonymized JSON; does not modify sealed data.
The nine examples are illustrative, not a representative evaluation sample.
"""

import json
from pathlib import Path

import chess
import pyarrow.parquet as pq

from chess_clone.modeling.style_residual import predict_residual


ROOT = Path("artifacts/benchmarks/deep-style-v1")
SELECTION = (
    ("opening_first", "opening", "seVyX6bY:1", 1),
    ("opening_second", "opening", "tm0Zvbue:5", 2),
    ("opening_outside", "opening", "seVyX6bY:5", 4),
    ("middlegame_first", "middlegame", "seVyX6bY:19", 1),
    ("middlegame_second", "middlegame", "6DhUFYmX:26", 2),
    ("middlegame_third", "middlegame", "FiGHsWDD:34", 3),
    ("endgame_first", "endgame", "hzRoaYNh:60", 1),
    ("endgame_second", "endgame", "hDhfMzW2:90", 2),
    ("endgame_outside", "endgame", "vALreqJp:104", 4),
)


def main() -> None:
    acquisition = json.loads((ROOT / "acquisition.json").read_text())["players"]
    residual = ROOT / "residual"
    penalty = json.loads((residual / "selection.json").read_text())["selected_penalty"]
    shared = json.loads((residual / "shared.json").read_text())[penalty]
    confirmation = residual / "confirmation"
    reports = {
        policy: {
            row["decision_id"]: row
            for row in pq.read_table(
                confirmation / f"test_predictions_{policy}.parquet",
                columns=["decision_id", "player_username", "actual_rank", "predicted_move_uci", "top_1_confidence"],
            ).to_pylist()
        }
        for policy in ("personal", "shared", "population")
    }
    examples = []
    for example_id, phase, decision_id, expected_rank in SELECTION:
        report = reports["personal"][decision_id]
        player = report["player_username"].lower()
        record = acquisition[player]
        assert record["role"] == "confirmation"
        position = next(
            row for row in pq.read_table(record["paths"]["confirmation"]).to_pylist()
            if f"{row['game_id']}:{row['ply']}" == decision_id
        )
        rows = pq.read_table(
            residual / "cache" / f"{player}_confirmation.parquet",
            filters=[("decision_id", "=", decision_id)],
        ).to_pylist()
        assert rows and rows[0]["game_phase"] == phase
        assert report["actual_rank"] == expected_rank
        board = chess.Board(position["fen"])
        if phase == "endgame":
            assert position["ply"] >= 60 and len(board.piece_map()) <= 16
        actual = chess.Move.from_uci(position["actual_move_uci"])
        assert actual in board.legal_moves
        policies = {}
        model = json.loads((residual / "models" / f"{player}.json").read_text())["models"][penalty]
        for policy, adapter in (("personal", model), ("shared", shared), ("population", None)):
            probabilities = predict_residual(rows, adapter)
            ranked = sorted(zip(rows, probabilities, strict=True), key=lambda pair: -pair[1])
            reference = reports[policy][decision_id]
            assert ranked[0][0]["candidate_move_uci"] == reference["predicted_move_uci"]
            assert abs(ranked[0][1] - reference["top_1_confidence"]) < 1e-8
            assert next(i for i, (row, _) in enumerate(ranked, 1) if row["candidate_move_uci"] == actual.uci()) == reference["actual_rank"]
            policies[policy] = [
                {
                    "uci": row["candidate_move_uci"],
                    "san": board.san(chess.Move.from_uci(row["candidate_move_uci"])),
                    "probability": round(float(probability), 6),
                }
                for row, probability in ranked[:3]
            ]
        examples.append({
            "id": example_id,
            "phase": phase,
            "fen": position["fen"],
            "actual": {"uci": actual.uci(), "san": board.san(actual)},
            "context": {
                "player_rating": position["player_rating"],
                "opponent_rating": position["opponent_rating"],
                "time_control": position["time_control"],
                "color": position["player_color"],
            },
            "policies": policies,
        })
    print(json.dumps({
        "schema_version": 1,
        "source": "Curated anonymized frozen deep-style v1 confirmation replay; not representative of aggregate performance",
        "examples": examples,
    }, indent=2))


if __name__ == "__main__":
    main()
