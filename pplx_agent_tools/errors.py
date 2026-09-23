"""Typed error variants for pplx-agent-tools.

Each variant maps to a stable exit code so agents can implement retry without
parsing stderr. See the design plan's "Exit codes" table.
"""

from __future__ import annotations

from typing import ClassVar, Final

EXIT_OK: Final = 0
EXIT_GENERIC: Final = 1
EXIT_AUTH: Final = 2
EXIT_RATE_LIMIT: Final = 3
EXIT_NETWORK: Final = 4
EXIT_ANTI_BOT: Final = 5
# Partial-success: stdout carries usable content, but the upstream stream did
# not signal COMPLETED (deadline or stall tripped, or server cut). Distinct from
# network (exit 4) because the retry semantic differs: bumping --timeout or
# accepting the partial is usually the right move, not a blind backoff retry.
EXIT_PARTIAL: Final = 6


class PplxError(Exception):
    """Base for all expected pplx-agent-tools failures.

    Unexpected exceptions (bugs) bubble up as generic and exit 1.
    """

    # Subclasses inherit their parent's code unless they set their own.
    exit_code: ClassVar[int] = EXIT_GENERIC


class AuthError(PplxError):
    """Cookies missing, unreadable, expired, or rejected by /api/auth/session.

    Agent retry semantic: refresh cookies, then retry. Exit 2.
    """

    exit_code = EXIT_AUTH


class RateLimitError(PplxError):
    """Server returned 429. Agent retry semantic: exponential backoff. Exit 3."""

    exit_code = EXIT_RATE_LIMIT

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class NetworkError(PplxError):
    """DNS / timeout / connection refused / TLS error. Agent retry: linear backoff. Exit 4."""

    exit_code = EXIT_NETWORK


class StreamDeadlineError(NetworkError):
    """SSE stream exceeded the caller-provided overall deadline.

    Subclasses NetworkError so the exit-code mapping treats it like any other
    timeout (exit 4). Distinguished from NetworkError so verbs that can salvage
    a partial result (e.g. `pplx fetch --prompt` accumulating chunks) can catch
    it specifically without swallowing real network failures.
    """


class StreamStallError(StreamDeadlineError):
    """SSE stream carried no new content for `seconds` (heartbeats and repeats don't count).

    A deadline subclass so every partial-salvage path treats it the same way;
    distinguished so the reported reason says the backend went quiet rather
    than that the caller's overall budget ran out.
    """

    def __init__(self, message: str, seconds: float) -> None:
        super().__init__(message)
        self.seconds = seconds


class AntiBotError(PplxError):
    """Cloudflare challenge or similar bot block. Agent retry: investigate, don't auto-retry. Exit 5."""

    exit_code = EXIT_ANTI_BOT


class SchemaError(PplxError):
    """Required field missing or unparseable response.

    Indicates Perplexity changed shape or our parser is wrong. Don't retry. Exit 1.
    The raw response may be persisted by the caller to $XDG_CACHE_HOME/pplx-tools/last-error.json
    for postmortem.
    """


class BlockedUrlError(PplxError):
    """A fetch URL refused before any request: not http(s), no or malformed
    host, or a host that resolves to a non-public address. Retrying cannot
    change the answer. Exit 1.
    """


class TargetHttpError(PplxError):
    """The fetched page answered 4xx (other than 408 and 429). Don't retry. Exit 1."""


def exit_code(err: BaseException) -> int:
    """Map an exception to the documented exit-code contract."""
    return err.exit_code if isinstance(err, PplxError) else EXIT_GENERIC
