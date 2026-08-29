"""Provider contracts and provider-facing exceptions."""

from abc import ABC, abstractmethod
from datetime import date, datetime

DateInput = int | date | datetime | str | None


class ProviderError(RuntimeError):
    """Base error raised while obtaining game data."""


class InvalidUsernameError(ProviderError):
    """The supplied username is malformed or does not exist."""


class ProviderHTTPError(ProviderError):
    """The provider returned an unsuccessful HTTP response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(ProviderHTTPError):
    """The provider rejected the request because of rate limiting."""

    def __init__(self, retry_after: str | None = None) -> None:
        self.retry_after = retry_after
        suffix = f" Retry after {retry_after}." if retry_after else " Try again later."
        super().__init__(429, f"Lichess rate limit reached.{suffix}")


class GameProvider(ABC):
    """Source of public PGN game data."""

    @abstractmethod
    def download_games(
        self,
        username: str,
        *,
        max_games: int | None = None,
        since: DateInput = None,
        until: DateInput = None,
    ) -> bytes:
        """Download games as untouched provider response bytes."""

