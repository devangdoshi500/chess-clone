import os

import pytest

from chess_clone.providers import LichessProvider


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LICHESS_INTEGRATION") != "1",
    reason="set RUN_LICHESS_INTEGRATION=1 to call the live Lichess API",
)
def test_live_download_one_public_game() -> None:
    payload = LichessProvider().download_games("thibault", max_games=1)

    assert b"[Event " in payload
    assert b"[Variant \"Standard\"]" in payload
