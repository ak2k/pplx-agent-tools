"""Typed error variants for pplx-agent-tools.

Each variant maps to a stable exit code so agents can implement retry without
parsing stderr. See the design plan's "Exit codes" table.
"""

from __future__ import annotations

from typing import Final

EXIT_OK: Final = 0
EXIT_GENERIC: Final = 1
EXIT_AUTH: Final = 2
EXIT_RATE_LIMIT: Final = 3
EXIT_NETWORK: Final = 4
EXIT_ANTI_BOT: Final = 5


class PplxError(Exception):
    """Base for all expected pplx-agent-tools failures.

    Unexpected exceptions (bugs) bubble up as generic and exit 1.
    """


class AuthError(PplxError):
    """Cookies missing, unreadable, expired, or rejected by /api/auth/session.

    Agent retry semantic: refresh cookies, then retry. Exit 2.
    """


class RateLimitError(PplxError):
    """Server returned 429. Agent retry semantic: exponential backoff. Exit 3."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class NetworkError(PplxError):
    """DNS / timeout / connection refused / TLS error. Agent retry: linear backoff. Exit 4."""


class AntiBotError(PplxError):
    """Cloudflare challenge or similar bot block. Agent retry: investigate, don't auto-retry. Exit 5."""


class SchemaError(PplxError):
    """Required field missing or unparseable response.

    Indicates Perplexity changed shape or our parser is wrong. Don't retry. Exit 1.
    The raw response may be persisted by the caller to $XDG_CACHE_HOME/pplx-tools/last-error.json
    for postmortem.
    """


def exit_code(err: BaseException) -> int:
    """Map an exception to the documented exit-code contract."""
    if isinstance(err, AuthError):
        return EXIT_AUTH
    if isinstance(err, RateLimitError):
        return EXIT_RATE_LIMIT
    if isinstance(err, NetworkError):
        return EXIT_NETWORK
    if isinstance(err, AntiBotError):
        return EXIT_ANTI_BOT
    return EXIT_GENERIC
