"""pplx research verb: Perplexity deep research via /rest/sse/perplexity_ask.

The flagship differentiated capability — multi-step, cited research, far beyond
`search`'s ranked hits. Same endpoint as `fetch --prompt`, but the deep behaviour
is selected by `model_preference` (NOT `params.mode` — see `_MODE_MODEL` and
docs/wire/perplexity-ask-research.md), with two important differences:

  1. Session-creating. Research is an ask-family verb, so it creates a thread.
     We send `is_incognito: true` (the thread never enters the user's history;
     verified) and still issue a best-effort `delete_thread` as a secondary
     guard. See CLAUDE.md → "Endpoint selection principle".
  2. Schematized response. Unlike copilot mode's incremental `markdown_block`
     chunks, research streams full-snapshot frames whose `text` field is a JSON
     list of `{step_type, content, uuid}` blocks. The answer lives in the FINAL
     block's `content.answer`; sources accumulate across SEARCH_RESULTS blocks'
     `content.web_results`. We keep the latest snapshot and decode it once.

Deep research takes ~90-120s (multi-round, ~40+ sources); the verb supports the
same `--timeout` → partial-result (exit 6) contract as `fetch --prompt`, with
bounded 429 retry.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..errors import RateLimitError, SchemaError, StreamDeadlineError
from ..wire import Client
from .fetch import event_marks_completed

ENDPOINT = "/rest/sse/perplexity_ask"
DEFAULT_MODE = "research"

# Verified 2026-06-22: Perplexity selects deep-research behaviour by
# `model_preference`, NOT by `params.mode`. Sending params.mode="research" with
# model_preference="turbo" yields a plain copilot answer (1 search round); it's
# model_preference="pplx_alpha" that triggers real Deep Research (LOAD_SKILL +
# multiple SEARCH_WEB rounds + THOUGHT steps + far more sources). So we map the
# user-facing --mode to the model id that actually drives it, and keep
# params.mode coarse ("copilot"). Model ids here are the stable internal
# constants (ALPHA / AGENTIC_RESEARCH); the per-mode default_models can drift,
# but these identifiers have held across builds.
_MODE_MODEL = {
    "research": "pplx_alpha",  # Deep Research: multi-round + reasoning
    "agentic_research": "pplx_agentic_research",  # Model Council: multi-model
    "council": "pplx_agentic_research",  # friendly alias for agentic_research
}


def _model_for_mode(mode: str) -> str:
    """Map a user-facing --mode to its driving model_preference. An unknown value
    falls through as a literal model_preference so power users can pass a model id."""
    return _MODE_MODEL.get(mode, mode)


# 429 retry policy — mirrors fetch's tight bound (the agent contract documents
# exit-3 for callers wanting their own backoff). Research is rarer than fetch so
# we keep this conservative rather than aggressive.
_RATE_LIMIT_MAX_ATTEMPTS = 3
_RATE_LIMIT_DEFAULT_BACKOFF = 5.0
_RATE_LIMIT_BACKOFF_CAP = 60.0
_BACKOFF_JITTER_LOW = 0.85
_BACKOFF_JITTER_HIGH = 1.15
_PROGRESS_EVENT_STRIDE = 10


@dataclass
class ResearchSource:
    url: str
    title: str | None = None
    snippet: str | None = None


@dataclass
class ResearchResult:
    query: str
    answer: str
    sources: list[ResearchSource]
    mode: str
    # False iff the stream was cut before COMPLETED (deadline tripped / server cut).
    stream_complete: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class _StreamState:
    """Accumulator threaded across retry attempts. Research sends full snapshots,
    so we keep the *latest* `text` rather than concatenating deltas."""

    latest_text: str | None = None
    backend_uuid: str | None = None
    read_write_token: str | None = None
    saw_completed: bool = False
    failed: bool = False  # server emitted status=FAILED (e.g. model incompatible with mode)


def research(
    client: Client,
    query: str,
    *,
    mode: str = DEFAULT_MODE,
    keep_thread: bool = False,
    timeout: float | None = None,
    progress: bool = False,
) -> ResearchResult:
    """Run a deep-research query through the ask endpoint in `mode`.

    `timeout` bounds wall-clock; on deadline-with-partial we return the partial
    answer with `stream_complete=False` (the agent contract is "always something
    plus a flag", exit 6). `keep_thread` preserves the incognito thread instead
    of deleting it (default deletes).
    """
    body = _build_research_body(query, mode)
    state = _StreamState()
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
            _consume_stream(client, body, state, remaining_seconds=remaining, progress=progress)
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
                    f"pplx research: rate limited (attempt {attempt}/"
                    f"{_RATE_LIMIT_MAX_ATTEMPTS}); sleeping {sleep_s:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)

    if state.failed:
        raise SchemaError(
            f"research request on {ENDPOINT} returned status=FAILED; mode {mode!r} may "
            f"reject model_preference — check model↔mode compatibility via `pplx models`"
        )

    if state.latest_text is None:
        if deadline_tripped:
            raise StreamDeadlineError(
                f"research stream on {ENDPOINT} exceeded {timeout:.1f}s before any content"
            )
        raise SchemaError(f"no schematized text received from {ENDPOINT}")

    answer, sources = decode_research_text(state.latest_text)

    if not keep_thread and state.backend_uuid and state.read_write_token:
        client.delete_thread(state.backend_uuid, state.read_write_token)

    return ResearchResult(
        query=query,
        answer=answer,
        sources=sources,
        mode=mode,
        stream_complete=state.saw_completed,
    )


def decode_research_text(text: str) -> tuple[str, list[ResearchSource]]:
    """Pure decode of a research snapshot `text` (JSON block list) → (answer, sources).

    Total function over parse success: returns (answer, sources); raises
    SchemaError if `text` isn't a JSON list of blocks. Tolerant of unknown
    step_types and missing fields — extra block kinds are ignored, partial
    snapshots yield whatever FINAL/SEARCH_RESULTS content is present so far.

    The FINAL block's `content.answer` is itself a JSON string wrapping
    `{answer: <markdown>, web_results: [<cited sources>], ...}` — we unwrap it.
    Sources prefer the FINAL block's cited `web_results` (citation-aligned with
    the answer's [n] markers); we fall back to the intermediate SEARCH_RESULTS
    rounds when FINAL carries none.
    """
    try:
        blocks = json.loads(text)
    except (ValueError, TypeError) as e:
        raise SchemaError(f"research text is not JSON: {e}") from e
    if not isinstance(blocks, list):
        raise SchemaError(f"research text decoded to {type(blocks).__name__}, expected list")

    answer_parts: list[str] = []
    final_web: list[Any] | None = None
    search_web: list[Any] = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        content = blk.get("content")
        if not isinstance(content, dict):
            continue
        step = blk.get("step_type")
        if step == "FINAL":
            markdown, cited = _unwrap_final_answer(content.get("answer"))
            if markdown:
                answer_parts.append(markdown)
            if cited:
                final_web = cited
        elif step == "SEARCH_RESULTS":
            wr = content.get("web_results")
            if isinstance(wr, list):
                search_web.extend(wr)

    chosen = final_web if final_web else search_web
    sources: list[ResearchSource] = []
    seen: set[str] = set()
    for wr in chosen:
        src = _to_source(wr)
        if src is not None and src.url not in seen:
            seen.add(src.url)
            sources.append(src)
    return "\n\n".join(answer_parts).strip(), sources


def _unwrap_final_answer(raw: Any) -> tuple[str, list[Any]]:
    """FINAL `content.answer` → (markdown, cited web_results).

    Normally a JSON string `{"answer": <md>, "web_results": [...]}`; tolerate a
    plain-markdown string (returned as-is with no sources) so a server-side
    shape change degrades instead of crashing.
    """
    if not isinstance(raw, str) or not raw:
        return "", []
    try:
        inner = json.loads(raw)
    except (ValueError, TypeError):
        return raw, []  # already plain markdown
    if not isinstance(inner, dict):
        return raw, []
    md = inner.get("answer")
    web = inner.get("web_results")
    return (md if isinstance(md, str) else raw), (web if isinstance(web, list) else [])


def _to_source(raw: Any) -> ResearchSource | None:
    if not isinstance(raw, dict):
        return None
    url = raw.get("url")
    if not isinstance(url, str) or not url:
        return None
    title = raw.get("name") or raw.get("title")
    return ResearchSource(
        url=url,
        title=title if isinstance(title, str) else None,
        snippet=raw.get("snippet") if isinstance(raw.get("snippet"), str) else None,
    )


def _consume_stream(
    client: Client,
    body: dict[str, Any],
    state: _StreamState,
    *,
    remaining_seconds: float | None,
    progress: bool,
) -> None:
    """Drive one SSE call, keeping the latest full-snapshot `text` and ids.

    Propagates StreamDeadlineError / RateLimitError for the caller to handle.
    """
    event_count = 0
    try:
        for event in client.sse_post(ENDPOINT, body, max_total_seconds=remaining_seconds):
            event_count += 1
            if progress and event_count % _PROGRESS_EVENT_STRIDE == 0:
                print(".", end="", file=sys.stderr, flush=True)
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            if state.backend_uuid is None and isinstance(data.get("backend_uuid"), str):
                state.backend_uuid = data["backend_uuid"]
            if state.read_write_token is None and isinstance(data.get("read_write_token"), str):
                state.read_write_token = data["read_write_token"]
            if isinstance(data.get("text"), str):
                state.latest_text = data["text"]
            # A FAILED frame still carries a `text` field, so check it before the
            # completion/parse path — otherwise we'd treat a server-side failure
            # as an empty partial answer. Seen when model_preference is a model
            # that isn't valid for `mode` (server drops mode to CONCISE + FAILED).
            if data.get("status") == "FAILED":
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


def _build_research_body(query: str, mode: str) -> dict[str, Any]:
    """Ask-endpoint body for research. `model_preference` (derived from `mode`)
    is what selects Deep Research vs Model Council — see `_MODE_MODEL`. `params.mode`
    stays "copilot" (coarse; the server derives the real mode from the model).
    `is_incognito` is True so the created thread stays out of history."""
    frontend_uuid = str(uuid4())
    return {
        "query_str": query,
        "params": {
            "query_source": "home",
            "prompt_source": "user",
            "source": "default",
            "version": "2.18",
            "language": "en-US",
            "timezone": "UTC",
            "search_focus": "internet",
            "sources": ["web"],
            "mode": "copilot",
            "model_preference": _model_for_mode(mode),
            "frontend_uuid": frontend_uuid,
            "frontend_context_uuid": str(uuid4()),
            "client_search_results_cache_key": frontend_uuid,
            "use_schematized_api": True,
            "send_back_text_in_streaming_api": True,
            "skip_search_enabled": True,
            "is_incognito": True,
            "attachments": [],
            "mentions": [],
            "client_coordinates": None,
            "dsl_query": query,
        },
    }
