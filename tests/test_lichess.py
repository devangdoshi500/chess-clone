from datetime import UTC, datetime

import httpx
import pytest

from chess_clone.providers import (
    InvalidUsernameError,
    LichessProvider,
    ProviderHTTPError,
    RateLimitError,
)


def test_download_requests_rated_standard_pgn_with_metadata() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["request"] = request
        return httpx.Response(200, content=b"untouched pgn\n", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = LichessProvider(client=client).download_games(
            "Test_User",
            max_games=100,
            since=datetime(2025, 1, 1, tzinfo=UTC),
            until="2025-01-31T23:59:59Z",
        )

    request = observed["request"]
    assert isinstance(request, httpx.Request)
    params = request.url.params
    assert result == b"untouched pgn\n"
    assert request.headers["accept"] == "application/x-chess-pgn"
    assert request.headers["user-agent"].startswith("chess-clone/")
    assert params["rated"] == "true"
    assert params["perfType"] == (
        "ultraBullet,bullet,blitz,rapid,classical,correspondence"
    )
    assert params["clocks"] == "true"
    assert params["opening"] == "true"
    assert params["max"] == "100"
    assert params["since"] == "1735689600000"
    assert params["until"] == "1738367999000"


@pytest.mark.parametrize("username", ["", "bad/name", "x" * 31])
def test_invalid_username_is_rejected_without_request(username: str) -> None:
    provider = LichessProvider()
    with pytest.raises(InvalidUsernameError, match="Invalid Lichess username"):
        provider.download_games(username, max_games=1)


def test_404_becomes_invalid_username_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404, request=request))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(InvalidUsernameError, match="was not found"):
            LichessProvider(client=client).download_games("missing-user", max_games=1)


def test_429_is_not_retried() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(429, headers={"Retry-After": "60"}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RateLimitError, match="Retry after 60"):
            LichessProvider(client=client).download_games("valid-user", max_games=1)

    assert request_count == 1


def test_other_http_error_is_cleanly_wrapped() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, text="maintenance", request=request)
    )
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ProviderHTTPError, match="HTTP 503: maintenance") as error:
            LichessProvider(client=client).download_games("valid-user", max_games=1)

    assert error.value.status_code == 503


def test_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="since must be earlier"):
        LichessProvider().download_games(
            "valid-user", max_games=1, since="2025-02-01", until="2025-01-01"
        )

