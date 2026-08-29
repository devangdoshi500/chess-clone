"""Lichess public user-games provider."""

from datetime import UTC, date, datetime
import re
from typing import Final

import httpx

from chess_clone.providers.base import (
    DateInput,
    GameProvider,
    InvalidUsernameError,
    ProviderError,
    ProviderHTTPError,
    RateLimitError,
)

_USERNAME_RE: Final = re.compile(r"^[A-Za-z0-9_-]{1,30}$")
_STANDARD_PERF_TYPES: Final = (
    "ultraBullet,bullet,blitz,rapid,classical,correspondence"
)


def to_epoch_milliseconds(value: DateInput, *, name: str) -> int | None:
    """Convert supported date inputs to the millisecond timestamps Lichess expects."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a timestamp or ISO-8601 date")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must not be negative")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must not be empty")
        if stripped.isdigit():
            return int(stripped)
        try:
            value = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"{name} must be epoch milliseconds or an ISO-8601 date/time"
            ) from exc
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    if not isinstance(value, datetime):
        raise TypeError(f"Unsupported {name} value: {type(value).__name__}")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


class LichessProvider(GameProvider):
    """Download public rated standard games from the Lichess API."""

    base_url = "https://lichess.org/api/games/user"

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._client = client
        self._timeout = timeout_seconds

    def download_games(
        self,
        username: str,
        *,
        max_games: int | None = None,
        since: DateInput = None,
        until: DateInput = None,
    ) -> bytes:
        username = username.strip()
        if not _USERNAME_RE.fullmatch(username):
            raise InvalidUsernameError(
                "Invalid Lichess username. Use 1-30 letters, numbers, underscores, or hyphens."
            )
        if max_games is not None and max_games < 1:
            raise ValueError("max_games must be at least 1")

        since_ms = to_epoch_milliseconds(since, name="since")
        until_ms = to_epoch_milliseconds(until, name="until")
        if since_ms is not None and until_ms is not None and since_ms > until_ms:
            raise ValueError("since must be earlier than or equal to until")

        params: dict[str, str | int | bool] = {
            "rated": True,
            "perfType": _STANDARD_PERF_TYPES,
            "moves": True,
            "tags": True,
            "clocks": True,
            "opening": True,
        }
        if max_games is not None:
            params["max"] = max_games
        if since_ms is not None:
            params["since"] = since_ms
        if until_ms is not None:
            params["until"] = until_ms

        headers = {
            "Accept": "application/x-chess-pgn",
            "User-Agent": "chess-clone/0.1.0 (personal chess research project)",
        }
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=self._timeout, follow_redirects=True)
        try:
            response = client.get(
                f"{self.base_url}/{username}", params=params, headers=headers
            )
        except httpx.RequestError as exc:
            raise ProviderError(f"Could not reach Lichess: {exc}") from exc
        finally:
            if owns_client:
                client.close()

        if response.status_code == 404:
            raise InvalidUsernameError(f"Lichess user '{username}' was not found.")
        if response.status_code == 429:
            # Deliberately do not retry: Lichess asks clients to back off after a 429.
            raise RateLimitError(response.headers.get("Retry-After"))
        if response.is_error:
            detail = response.text.strip()[:200]
            message = f"Lichess returned HTTP {response.status_code}"
            if detail:
                message += f": {detail}"
            raise ProviderHTTPError(response.status_code, message)
        return response.content
