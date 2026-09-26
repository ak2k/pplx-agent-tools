"""Shared layer for the ask-family verbs (`ask`, `research`, `fetch --prompt`).

All three hit /rest/sse/perplexity_ask and share: the request `params` block
(`base_ask_params`), the copilot chunk/source extractors + the `Source` type, and
the SSE orchestration (`run_ask_stream`: 429 retry honoring `retry-after`, an
overall wall-clock deadline, a no-new-content stall guard and a dropped
connection that all soft-fail to a partial result, first-content and silence
bounds, a progress heartbeat, and capture of the thread identifiers +
completion/FAILED signals; `release_on_exit` is the matching cleanup).
Only the *accumulation* differs (copilot streams `markdown_block` chunks +
`web_results` blocks; research streams full-snapshot `text`), so callers pass an
`on_event` callback and own their accumulator.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from ..errors import (
    NetworkError,
    PplxError,
    RateLimitError,
    SchemaError,
    StreamDeadlineError,
    StreamFirstContentError,
    StreamSilenceError,
    StreamStallError,
)
from ..wire import Client

_RATE_LIMIT_MAX_ATTEMPTS = 3
_RATE_LIMIT_DEFAULT_BACKOFF = 5.0  # used when a 429 lacks a retry-after header
_RATE_LIMIT_BACKOFF_CAP = 60.0  # cap any single sleep so a hostile retry-after can't park us
_BACKOFF_JITTER_LOW = 0.85
_BACKOFF_JITTER_HIGH = 1.15  # ±15% jitter so parallel callers don't wake in lockstep
_PROGRESS_EVENT_STRIDE = 10
# Default --stall-timeout for `research`: how long a stream may go without new
# content before it is cut.
DEFAULT_STALL_SECONDS = 240.0
# Default --stall-timeout for `ask` and `fetch --prompt`: a thinking model can
# go minutes without changing `blocks` before its answer arrives all at once,
# and a cut before that returns nothing.
COPILOT_STALL_SECONDS = 480.0
# How long ask waits for the COMPLETED frame once `text_completed` says the
# answer is whole. That frame arrived 0.2-0.34 s after `text_completed` in live
# captures; 15 s is ~50x that, and heartbeats (~15 s apart) are what trigger
# the check, so the worst-case wait is about 30 s instead of the stall window.
COPILOT_SETTLE_SECONDS = 15.0
# Total silence, heartbeats included, that ends an `ask` or `fetch --prompt`
# stream (curl's low-speed abort): 3x the largest gap on a healthy stream.
# Legacy streams send a heartbeat comment every 15.0 s, and on a quiet healthy
# stream that heartbeat alone sets the gap, so 3 x 15 s.
SILENCE_SECONDS = 45.0
# The same bound for `research`. A research run can wait out a clarifying
# question for its `auto_skip_seconds` (60 s in the captured run), and whether
# heartbeats keep flowing during that wait has not been measured. The window
# is fixed when the request starts, so it has to cover that wait: 60 s plus
# 30 s (two heartbeat intervals), or the window above if that is larger.
RESEARCH_SILENCE_SECONDS = max(SILENCE_SECONDS, 60.0 + 30.0)
# A stream whose first progress event has not arrived by then is cut. On the
# legacy body the first data frame, a query echo at 0.10-0.37 s, already counts
# as progress, so this fires only when heartbeats keep the connection alive
# but no data frame arrives.
FIRST_CONTENT_SECONDS = 90.0


@dataclass
class AskStreamState:
    backend_uuid: str | None = None
    read_write_token: str | None = None
    context_uuid: str | None = None
    # The last non-null one the frames carried: the model the server ran.
    display_model: str | None = None
    saw_completed: bool = False
    text_completed: bool = False
    failed: bool = False  # server emitted status=FAILED (e.g. model incompatible with mode)
    # The bound or dropped connection that cut the stream short, kept rather
    # than raised so the caller can salvage what it accumulated. Its type says
    # which one; `cutoff_warnings` and `no_content_error` report from it.
    cutoff: NetworkError | None = None


def run_may_be_live(state: AskStreamState, *, raised: bool, settles_after_text: bool) -> bool:
    """Whether the run may still be going on the server once pplx stops reading.

    Not after a COMPLETED or FAILED frame, nor after the completion the caller
    stops at. A caller that `settles_after_text` (ask) holds a whole answer
    once `text_completed` arrives, and that run completes on its own however
    the stream then ends, unless an exception stopped the read (`raised`, e.g.
    Ctrl-C): that interrupts a live stream.
    """
    if state.saw_completed or state.failed:
        return False
    return not (settles_after_text and state.text_completed and not raised)


@contextmanager
def release_on_exit(
    client: Client, state: AskStreamState, *, keep_thread: bool, settles_after_text: bool = False
) -> Iterator[None]:
    """Wrap `run_ask_stream`: on every exit, KeyboardInterrupt included,
    terminate the run if it may still be going, then delete its thread unless
    `keep_thread`.

    Terminate needs the context uuid and the display model as well as the
    backend uuid; delete needs the read_write_token. Both are best-effort and
    never raise, so cleanup cannot mask the exception already in flight.
    """
    raised = True
    try:
        yield
        raised = False
    finally:
        live = run_may_be_live(state, raised=raised, settles_after_text=settles_after_text)
        if live and state.backend_uuid and state.context_uuid and state.display_model:
            client.terminate(state.backend_uuid, state.context_uuid, state.display_model)
        if not keep_thread and state.backend_uuid and state.read_write_token:
            client.delete_thread(state.backend_uuid, state.read_write_token)


def cutoff_warnings(state: AskStreamState) -> list[str]:
    """The result warning naming which bound or drop cut the stream."""
    if state.cutoff is None:
        return []
    return [f"stream cut before COMPLETED, returning partial content: {state.cutoff}"]


def cutoff_silence(state: AskStreamState) -> float | None:
    """How long the stream carried no bytes at all when that cut it (a "stall"
    to `cutoff_cause`); None for any other end. A larger --stall-timeout
    cannot help this cut, so results name it apart from a progress stall."""
    return state.cutoff.seconds if isinstance(state.cutoff, StreamSilenceError) else None


def cutoff_cause(state: AskStreamState) -> Literal["stall", "deadline", "drop"] | None:
    """What cut the stream: "stall" (silence included), "deadline", "drop" (the
    connection died mid-stream), or None when none did (a completed stream, or
    one the server closed early).

    Results carry this on stdout because the retry differs by cause and agents
    commonly discard stderr: a larger --timeout cannot help a stall, and a
    retry may help a drop.
    """
    if isinstance(state.cutoff, StreamStallError):
        return "stall"
    if isinstance(state.cutoff, StreamDeadlineError):
        return "deadline"
    if state.cutoff is not None:
        return "drop"
    return None


def downgrade_verdict(state: AskStreamState, requested: str) -> tuple[bool | None, list[str]]:
    """(downgraded, warnings): whether the server ran a model other than the
    one requested, judged by the last `display_model` the frames carried.
    None when no frame named one."""
    served = state.display_model
    if served is None:
        return None, []
    if served == requested:
        return False, []
    return True, [f"model downgraded: requested {requested!r}, the server ran {served!r}"]


def base_ask_params(
    query: str, *, model_preference: str, is_incognito: bool = True
) -> dict[str, Any]:
    """The shared `params` block for /rest/sse/perplexity_ask (copilot mode), used
    by `ask`, `research`, and `fetch --prompt`. Callers add their own extras (e.g.
    `compare_model_preferences` for Model Council).

    `params.mode` stays "copilot" — the *model* is the real behavior selector (see
    verbs/research.py for the model-as-mode finding). `is_incognito` defaults True
    so created threads never enter history. `timezone` is hard-coded "UTC" rather
    than host-detected: detection leaks location, and `time.tzname` yields
    abbreviations ("EST") not the IANA names Perplexity expects.
    """
    frontend_uuid = str(uuid4())
    return {
        "query_source": "home",
        "prompt_source": "user",
        "source": "default",
        "version": "2.18",
        "language": "en-US",
        "timezone": "UTC",
        "search_focus": "internet",
        "sources": ["web"],
        "mode": "copilot",
        "model_preference": model_preference,
        "frontend_uuid": frontend_uuid,
        "frontend_context_uuid": str(uuid4()),
        "client_search_results_cache_key": frontend_uuid,
        "use_schematized_api": True,
        "send_back_text_in_streaming_api": True,
        "skip_search_enabled": True,
        "is_incognito": is_incognito,
        # Without these the server may hold the run for a confirmation or
        # approval only the web UI can give.
        "should_ask_for_mcp_tool_confirmation": False,
        "supports_tool_approval_modal": False,
        "attachments": [],
        "mentions": [],
        "client_coordinates": None,
        "dsl_query": query,
    }


@dataclass
class Source:
    """A cited source (url + optional title/snippet). Shared by ask + research."""

    url: str
    title: str | None = None
    snippet: str | None = None


def to_source(raw: Any) -> Source | None:
    """Pure: a raw web_result dict → Source, or None if it has no usable URL.
    Accepts both `name` (search/copilot) and `title` (some research blocks)."""
    if not isinstance(raw, dict):
        return None
    url = raw.get("url")
    if not isinstance(url, str) or not url:
        return None
    title = raw.get("name") or raw.get("title")
    return Source(
        url=url,
        title=title if isinstance(title, str) else None,
        snippet=raw.get("snippet") if isinstance(raw.get("snippet"), str) else None,
    )


def extract_web_results(event: dict[str, Any]) -> list[Any]:
    """Pure: pull the raw web_results list from a copilot `web_results` block
    (`blocks[].web_result_block.web_results`). Returns [] when absent; the caller
    converts via `to_source` and dedupes. Never raises."""
    data = event.get("data")
    if not isinstance(data, dict):
        return []
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        return []
    for block in blocks:
        if not isinstance(block, dict) or block.get("intended_usage") != "web_results":
            continue
        wrb = block.get("web_result_block")
        if isinstance(wrb, dict):
            wr = wrb.get("web_results")
            if isinstance(wr, list):
                return wr
    return []


def extract_chunks_from_event(event: dict[str, Any]) -> list[str]:
    """Pure: the streamed markdown chunks added by one copilot SSE event, in
    order, for a caller that stops at `text_completed` and appends. Never raises."""
    return [c for _, run in extract_chunk_patches(event) for c in run]


def extract_chunk_patches(event: dict[str, Any]) -> list[tuple[int | None, list[str]]]:
    """Pure: the `ask_text` chunks one copilot event carries, each run paired
    with its `chunk_starting_offset` (a chunk index, None when absent).

    The terminal COMPLETED frame repaints every chunk from offset 0, so a
    caller that reads past `text_completed` must place chunks by offset rather
    than append them. Only `intended_usage == "ask_text"` blocks are read, not
    the parallel `ask_text_0_markdown` blocks: they carry the same chunks and
    reading both double-counts. Never raises."""
    data = event.get("data")
    if not isinstance(data, dict):
        return []
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        return []
    out: list[tuple[int | None, list[str]]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("intended_usage") != "ask_text":
            continue
        mb = block.get("markdown_block")
        if not isinstance(mb, dict):
            continue
        chunks = mb.get("chunks")
        if not isinstance(chunks, list):
            continue
        offset = mb.get("chunk_starting_offset")
        valid = isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0
        out.append((offset if valid else None, [str(c) for c in chunks]))
    return out


def apply_chunk_patch(
    chunks: dict[int, str], offset: int | None, run: list[str], *, terminal: bool
) -> None:
    """Place one run from `extract_chunk_patches` into `chunks` (chunk index -> text).

    Keyed by index so a run that lands past a gap keeps its place. The terminal
    COMPLETED frame resends the answer from its offset to the end, so on that
    frame any chunk past the run is stale; mid-stream runs never truncate, since
    nothing says they carry the tail. A run with no offset appends, except on the
    terminal frame, which repaints from 0 in every captured stream."""
    if offset is None:
        offset = 0 if terminal else max(chunks, default=-1) + 1
    for i, text in enumerate(run):
        chunks[offset + i] = text
    if terminal and run:
        for stale in [i for i in chunks if i >= offset + len(run)]:
            del chunks[stale]


def status_completed(event: dict[str, Any]) -> bool:
    """Completion predicate for callers that need the terminal COMPLETED frame.

    The shared default also accepts `text_completed`, which Perplexity sets a
    few frames BEFORE that frame. Research needs it because it keeps whole
    snapshots; ask needs it because only that frame carries the web_results
    list in the order the answer's [n] citations index."""
    data = event.get("data")
    return isinstance(data, dict) and data.get("status") == "COMPLETED"


def event_marks_completed(event: dict[str, Any]) -> bool:
    """True iff an SSE event signals the stream finished."""
    data = event.get("data")
    if not isinstance(data, dict):
        return False
    return data.get("status") == "COMPLETED" or bool(data.get("text_completed"))


def blocks_changed() -> Callable[[dict[str, Any]], bool]:
    """A stall-guard progress predicate for copilot streams (`ask`, `fetch --prompt`)."""
    seen: set[int] = set()

    # Envelope, repeat and replayed frames flow while working or hung; only blocks
    # never seen before count.
    def is_progress(event: dict[str, Any]) -> bool:
        data = event.get("data")
        if not isinstance(data, dict) or "blocks" not in data:
            return False
        key = hash(json.dumps(data["blocks"], sort_keys=True, default=str))
        if key in seen:
            return False
        seen.add(key)
        return True

    return is_progress


def _event_marks_failed(event: dict[str, Any]) -> bool:
    data = event.get("data")
    return isinstance(data, dict) and data.get("status") == "FAILED"


def run_ask_stream(
    client: Client,
    endpoint: str,
    body: dict[str, Any],
    state: AskStreamState,
    *,
    on_event: Callable[[dict[str, Any]], None],
    timeout: float | None,
    stall_seconds: float | None,
    progress: bool,
    label: str,
    is_complete: Callable[[dict[str, Any]], bool] = event_marks_completed,
    is_progress: Callable[[dict[str, Any]], bool] | None = None,
    settle_seconds: float | None = None,
    silence_seconds: float = SILENCE_SECONDS,
) -> None:
    """Drive the SSE call with retry/deadline/stall guard, filling in `state`.

    `on_event(event)` is invoked for every SSE event so the caller can accumulate
    (chunks or snapshot). The orchestrator captures the thread ids, the display
    model and the completion / FAILED signals into `state`, which the caller
    owns so `release_on_exit` can clean up whatever ends the stream. Propagates
    a terminal `RateLimitError` (exit 3) when retries are exhausted. A tripped
    deadline, stall, silence or first-content bound, or a `NetworkError`,
    returns with `state.cutoff` set so the caller can salvage whatever
    `on_event` accumulated, or raise `no_content_error` when there is none.

    `is_complete` decides which event ends the stream. The default accepts the
    early `text_completed` flag, which is right for delta-accumulating callers
    (stopping there avoids double-counting the COMPLETED repaint). Snapshot
    callers need the repaint and override it — see verbs/research.py.

    `is_progress` decides which events reset the stall clock (see
    `Client.sse_post`); None counts any event carrying data.

    `settle_seconds`, for a caller that reads past `text_completed`, shrinks
    the stall window to that many seconds, counted from the first
    `text_completed` frame: the answer is whole by then and only the terminal
    frame is outstanding. The
    resulting stall cutoff lands in `state.cutoff` like any other.

    `silence_seconds` sizes the transport's abort on a stream that sends no
    bytes at all (see `Client.sse_post`); the first-content bound is
    `FIRST_CONTENT_SECONDS` for every verb.
    """
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
            # Budget already spent before this attempt — surface it as a tripped
            # deadline so a caller with no content raises StreamDeadlineError
            # (exit 4), not the generic "no content" SchemaError. Only 429s came
            # back, so no progress event arrived.
            state.cutoff = StreamDeadlineError(
                f"{label} stream on {endpoint} exceeded {timeout:.1f}s deadline"
            )
            break
        try:
            _drive_one(
                client,
                endpoint,
                body,
                state,
                remaining_seconds=remaining,
                stall_seconds=stall_seconds,
                progress=progress,
                on_event=on_event,
                is_complete=is_complete,
                is_progress=is_progress,
                settle_seconds=settle_seconds,
                silence_seconds=silence_seconds,
            )
            break
        except StreamStallError as e:
            state.cutoff = e
            break
        except StreamDeadlineError as e:
            # sse_post only sees the budget left after any 429 retries; name the
            # caller's bound instead.
            state.cutoff = (
                StreamDeadlineError(
                    f"{label} stream on {endpoint} exceeded {timeout:.1f}s deadline",
                    e.since_progress,
                )
                if timeout
                else e
            )
            break
        except NetworkError as e:
            # A connection that dies after content arrived still leaves a
            # partial worth returning; with no content the caller raises this.
            state.cutoff = e
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


def no_content_error(
    *, label: str, endpoint: str, timeout: float | None, cutoff: NetworkError | None
) -> PplxError:
    """The error for an ask-family stream that produced no usable content.

    The texts live here so the three verbs cannot drift apart on the one
    distinction an agent acts on: a bound that tripped, or a connection that
    dropped, before the first content is worth retrying (exit 4), whereas a
    stream the server closed empty is not (exit 1).
    """
    if isinstance(cutoff, StreamFirstContentError):
        return StreamFirstContentError(
            f"{label} stream on {endpoint} sent no first content within {cutoff.seconds:.1f}s",
            cutoff.seconds,
        )
    if isinstance(cutoff, StreamSilenceError):
        return StreamSilenceError(
            f"{label} stream on {endpoint} went silent: no bytes for {cutoff.seconds:.1f}s "
            f"before the first content arrived",
            cutoff.seconds,
        )
    if isinstance(cutoff, StreamStallError):
        return StreamStallError(
            f"{label} stream on {endpoint} stalled: no new content for {cutoff.seconds:.1f}s "
            f"before the first content arrived",
            cutoff.seconds,
        )
    if cutoff is not None and not isinstance(cutoff, StreamDeadlineError):
        return cutoff
    if cutoff is not None:
        # `timeout` is None only when the budget was spent by an earlier retry
        # rather than by a caller-supplied bound.
        budget = f"{timeout:.1f}s" if timeout is not None else "its"
        # Which retry fits depends on whether the run was still working when cut.
        since = cutoff.since_progress
        progress = (
            "no progress event arrived"
            if since is None
            else f"the last progress event came {since:.1f}s before the cut"
        )
        return StreamDeadlineError(
            f"{label} stream on {endpoint} exceeded {budget} deadline "
            f"before the first content arrived; {progress}",
            since,
        )
    return SchemaError(f"{label} stream on {endpoint} closed with no content")


def _drive_one(
    client: Client,
    endpoint: str,
    body: dict[str, Any],
    state: AskStreamState,
    *,
    remaining_seconds: float | None,
    stall_seconds: float | None,
    progress: bool,
    on_event: Callable[[dict[str, Any]], None],
    is_complete: Callable[[dict[str, Any]], bool],
    is_progress: Callable[[dict[str, Any]], bool] | None,
    settle_seconds: float | None,
    silence_seconds: float,
) -> None:
    event_count = 0
    window = stall_seconds
    settling = False
    # Passed only when used, so `sse_post` overrides without the parameter keep working.
    extra: dict[str, Any] = {}
    if settle_seconds is not None:
        extra["stall_window"] = lambda: window
        new_content = is_progress

        # The first `text_completed` frame counts as progress even when its
        # blocks repeat earlier ones, so the settle window runs from that frame
        # rather than from the last new block.
        def settle_or_progress(event: dict[str, Any]) -> bool:
            nonlocal window, settling
            progressed = new_content is None or new_content(event)
            if settling or not event_marks_completed(event):
                return progressed
            settling = True
            window = settle_seconds if window is None else min(window, settle_seconds)
            return True

        is_progress = settle_or_progress
    try:
        for event in client.sse_post(
            endpoint,
            body,
            max_total_seconds=remaining_seconds,
            stall_seconds=stall_seconds,
            is_progress=is_progress,
            silence_seconds=silence_seconds,
            first_content_seconds=FIRST_CONTENT_SECONDS,
            **extra,
        ):
            event_count += 1
            if progress and event_count % _PROGRESS_EVENT_STRIDE == 0:
                print(".", end="", file=sys.stderr, flush=True)
            data = event.get("data")
            if isinstance(data, dict):
                if state.backend_uuid is None and isinstance(data.get("backend_uuid"), str):
                    state.backend_uuid = data["backend_uuid"]
                if state.read_write_token is None and isinstance(data.get("read_write_token"), str):
                    state.read_write_token = data["read_write_token"]
                if state.context_uuid is None and isinstance(data.get("context_uuid"), str):
                    state.context_uuid = data["context_uuid"]
                if isinstance(data.get("display_model"), str):
                    state.display_model = data["display_model"]
                if data.get("text_completed"):
                    state.text_completed = True
            on_event(event)
            # FAILED frames still carry text/blocks, so check before treating the
            # event as normal progress.
            if _event_marks_failed(event):
                state.failed = True
                return
            if is_complete(event):
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
