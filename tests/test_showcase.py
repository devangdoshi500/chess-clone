import json
from collections import Counter
from pathlib import Path

import chess


SHOWCASE = Path(__file__).parents[1] / "demo/public/showcase.json"


def test_showcase_fixture_is_valid_and_redistribution_safe():
    payload = json.loads(SHOWCASE.read_text())
    assert payload["schema_version"] == 1
    assert len(payload["examples"]) == 9
    assert Counter(row["phase"] for row in payload["examples"]) == {
        "opening": 3, "middlegame": 3, "endgame": 3
    }
    ranks = []
    assert not any(
        key in json.dumps(payload).casefold()
        for key in ("game_id", "player_username", "lichess")
    )
    for example in payload["examples"]:
        board = chess.Board(example["fen"])
        assert board.is_valid()
        actual = chess.Move.from_uci(example["actual"]["uci"])
        assert actual in board.legal_moves
        assert example["actual"]["san"] == board.san(actual)
        personal_top = [move["uci"] for move in example["policies"]["personal"]]
        ranks.append(personal_top.index(actual.uci()) + 1 if actual.uci() in personal_top else None)
        if example["phase"] == "endgame":
            assert len(board.piece_map()) <= 16
        for moves in example["policies"].values():
            assert len(moves) == 3
            assert [move["probability"] for move in moves] == sorted(
                (move["probability"] for move in moves), reverse=True
            )
            assert sum(move["probability"] for move in moves) <= 1
            assert len({move["uci"] for move in moves}) == 3
            for move in moves:
                parsed = chess.Move.from_uci(move["uci"])
                assert parsed in board.legal_moves
                assert move["san"] == board.san(parsed)
    assert Counter(ranks) == {1: 3, 2: 3, 3: 1, None: 2}
