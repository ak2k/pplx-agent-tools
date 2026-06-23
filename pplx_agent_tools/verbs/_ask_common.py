"""Shared orchestration for ask-family verbs (`ask`, `research`).

All three deep verbs hit /rest/sse/perplexity_ask and need the same plumbing:
429 retry honoring `retry-after`, an overall wall-clock deadline (soft-fail to a
partial result), a progress heartbeat, and capture of the thread identifiers +
completion/FAILED signals. Only the *accumulation* differs (copilot streams
`markdown_block` chunks; research streams full-snapshot `text`), so callers pass
an `on_event` callback and own their accumulator.

`fetch --prompt` predates this and keeps its own copy for now; it can migrate
here when it gains `--model`.
"""

from __future__ import annotations

import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import RateLimitError, StreamDeadlineError
from ..wire import Client

_RATE_LIMIT_MAX_ATTEMPTS = 3
_RATE_LIMIT_DEFAULT_BACKOFF = 5.0  # used when a 429 lacks a retry-after header
_RATE_LIMIT_BACKOFF_CAP = 60.0  # cap any single sleep so a hostile retry-after can't park us
_BACKOFF_JITTER_LOW = 0.85
_BACKOFF_JITTER_HIGH = 1.15  # ±15% jitter so parallel callers don't wake in lockstep
_PROGRESS_EVENT_STRIDE = 10


@dataclass
class AskStreamState:
    backend_uuid: str | None = None
    read_write_token: str | None = None
    saw_completed: bool = False
    failed: bool = False  # server emitted status=FAILED (e.g. model incompatible with mode)


def event_marks_completed(event: dict[str, Any]) -> bool:
    """True iff an SSE event signals the stream finished."""
    data = event.get("data")
    if not isinstance(data, dict):
        return False
    return data.get("status") == "COMPLETED" or bool(data.get("text_completed"))


def _event_marks_failed(event: dict[str, Any]) -> bool:
    data = event.get("data")
    return isinstance(data, dict) and data.get("status") == "FAILED"


def run_ask_stream(
    client: Client,
    endpoint: str,
    body: dict[str, Any],
    *,
    on_event: Callable[[dict[str, Any]], None],
    timeout: float | None,
    progress: bool,
    label: str,
) -> tuple[AskStreamState, bool]:
    """Drive the SSE call with retry/deadline; return (state, deadline_tripped).

    `on_event(event)` is invoked for every SSE event so the caller can accumulate
    (chunks or snapshot). The orchestrator captures `backend_uuid` /
    `read_write_token` and the completion / FAILED signals into the returned
    `AskStreamState`. Propagates a terminal `RateLimitError` (exit 3) when retries
    are exhausted; a tripped deadline returns with `deadline_tripped=True` so the
    caller can salvage whatever `on_event` accumulated.
    """
    state = AskStreamState()
    deadline_tripped = False
    overall_deadline = (time.monotonic() + timeout) if timeout else None

    def _remaining() -> float | None:
        if overall_deadline is None:
            return None
        return max(0.0, overall_deadline - time.monotonic())

    last_rate_limit: RateLimitError | None = None
    for attempt in range(1, _RATE_LIMIT_MAX_ATTEMPTS + 1):
        remaining = _remaining()
        if remaining == 0.0:
            if last_rate_limit is not None:
                raise last_rate_limit
            break
        try:
            _drive_one(
                client,
                endpoint,
                body,
                state,
                remaining_seconds=remaining,
                progress=progress,
                on_event=on_event,
            )
            break
        except StreamDeadlineError:
            deadline_tripped = True
            break
        except RateLimitError as e:
            last_rate_limit = e
            if attempt >= _RATE_LIMIT_MAX_ATTEMPTS:
                raise
            sleep_s = _rate_limit_backoff(e, _remaining())
            if sleep_s > 0:
                print(
                    f"pplx {label}: rate limited (attempt {attempt}/"
                    f"{_RATE_LIMIT_MAX_ATTEMPTS}); sleeping {sleep_s:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)
    return state, deadline_tripped


def _drive_one(
    client: Client,
    endpoint: str,
    body: dict[str, Any],
    state: AskStreamState,
    *,
    remaining_seconds: float | None,
    progress: bool,
    on_event: Callable[[dict[str, Any]], None],
) -> None:
    event_count = 0
    try:
        for event in client.sse_post(endpoint, body, max_total_seconds=remaining_seconds):
            event_count += 1
            if progress and event_count % _PROGRESS_EVENT_STRIDE == 0:
                print(".", end="", file=sys.stderr, flush=True)
            data = event.get("data")
            if isinstance(data, dict):
                if state.backend_uuid is None and isinstance(data.get("backend_uuid"), str):
                    state.backend_uuid = data["backend_uuid"]
                if state.read_write_token is None and isinstance(data.get("read_write_token"), str):
                    state.read_write_token = data["read_write_token"]
            on_event(event)
            # FAILED frames still carry text/blocks, so check before treating the
            # event as normal progress.
            if _event_marks_failed(event):
                state.failed = True
                return
            if event_marks_completed(event):
                state.saw_completed = True
                return
    finally:
        if progress and event_count >= _PROGRESS_EVENT_STRIDE:
            print("", file=sys.stderr, flush=True)


def _rate_limit_backoff(err: RateLimitError, remaining: float | None) -> float:
    base = err.retry_after if err.retry_after is not None else _RATE_LIMIT_DEFAULT_BACKOFF
    sleep_s = min(base, _RATE_LIMIT_BACKOFF_CAP) * random.uniform(
        _BACKOFF_JITTER_LOW, _BACKOFF_JITTER_HIGH
    )
    if remaining is not None:
        sleep_s = min(sleep_s, remaining)
    return max(0.0, sleep_s)
