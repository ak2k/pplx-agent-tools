"""Shared test-double base for `Client` subclasses.

Several tests subclass `pplx_agent_tools.wire.Client` to substitute the
network-touching methods (`post_json`, `sse_post`, `delete_thread`) with
canned responses. None of them want a real curl_cffi Session, but they
also can't simply skip `Client.__init__` — that trips CodeQL's
"missing super().__init__" rule on every new subclass.

`_TestClientBase` resolves both: it calls `super().__init__` with
throwaway cookies (CodeQL-clean) and centralises that setup so adding a
new attribute to `Client.__init__` only requires one update here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from pplx_agent_tools.errors import NetworkError
from pplx_agent_tools.wire import Client

# Thread ids for research doubles. The research driver takes only UUID-shaped
# ids from a frame, as Perplexity sends them, and ignores any other.
BU = "0b0b0b0b-1111-4222-8333-444444444444"
CTX = "0c0c0c0c-5555-4666-8777-888888888888"


class FakeTime:
    """A stand-in for `_research_stream.time` in tests with no wire clock:
    the driver's reconnect backoffs and timer waits pass at once."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _TestClientBase(Client):
    """Inherit from this instead of `Client` directly when writing a test
    double. Subclasses call `super().__init__()` (no args) to inherit the
    throwaway-cookie setup, or pass the cookies the double holds; CodeQL sees
    the chained super() and is happy.
    """

    def __init__(self, cookies: dict[str, str] | None = None) -> None:
        # Allocates a curl_cffi Session we never use — subclasses override
        # every method that would read `_session`, or replace it.
        super().__init__({"x": "y"} if cookies is None else cookies)
        self.terminated: list[tuple[str, str, str]] = []

    def terminate(self, entry_uuid: str, context_uuid: str, model_preference: str) -> bool:
        # Overridden here rather than per double: a double that forgets it
        # would send a real request to Perplexity from cleanup.
        self.terminated.append((entry_uuid, context_uuid, model_preference))
        return True

    def sse_reconnect(
        self,
        backend_uuid: str,
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        is_progress: Callable[[dict[str, Any]], bool] | None = None,
        stall_window: Callable[[], float | None] | None = None,
        silence_seconds: float | None = None,
        first_content_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        # For the same reason as `terminate`: a drop in a research double
        # would otherwise reconnect to Perplexity.
        raise NetworkError(f"no reconnect scripted for {backend_uuid}")
