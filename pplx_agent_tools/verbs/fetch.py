"""pplx fetch verb: URL → cleaned content (optional LLM extraction via --prompt).

Hybrid implementation:
  - Plain mode (no --prompt): fetch the URL ourselves via curl_cffi (chrome-
    impersonate, same Cloudflare-handling as Perplexity calls), extract main
    content with trafilatura.
  - --prompt mode: route the URL + prompt through /rest/sse/perplexity_ask
    (the LLM has URL-fetching as a tool), parse out the answer.

Why the hybrid: Perplexity's web-session API surface has no URL→content
fetch endpoint we can reach (RE'd 2026-05-12; see plan's "Open questions").
Their internal `pplx content fetch` CLI must use Sonar-API or internal-only
auth. Implementing fetch ourselves loses the `is_paywall` / `is_cached`
signals but keeps the agent-shape single-command primitive.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from curl_cffi import requests as cf_requests

from ..errors import NetworkError, RateLimitError, SchemaError, StreamDeadlineError
from ..wire import Client

_PROMPT_ENDPOINT = "/rest/sse/perplexity_ask"

# Schemes accepted for outbound fetch. Anything else (file://, ftp://,
# gopher://, custom) is rejected up front — we never want curl_cffi to
# touch the local filesystem or non-HTTP backends from a user-supplied URL.
_ALLOWED_FETCH_SCHEMES = frozenset({"http", "https"})

# Auto-retry policy for 429s from /rest/sse/perplexity_ask. Tight bound — we
# don't want a thundering-herd retry loop, and the agent contract documents
# exit-code 3 for callers that want their own backoff. Three attempts total
# means a server with a 1-minute retry-after still resolves within 2 minutes.
_RATE_LIMIT_MAX_ATTEMPTS = 3
_RATE_LIMIT_DEFAULT_BACKOFF = 5.0  # used when 429 lacks a retry-after header
# Cap any single sleep so a hostile/buggy retry-after can't park us for an hour.
_RATE_LIMIT_BACKOFF_CAP = 60.0

# Heartbeat cadence: emit one stderr char per N SSE events when progress is on.
# Tuned for the observed 3-4 events/sec rate from Perplexity - a dot every
# 10 events is roughly every 2-3 s, frequent enough to see liveness without
# flooding stderr on a long stream.
_PROGRESS_EVENT_STRIDE = 10

# ±15% multiplicative jitter on rate-limit backoff. Without it, N parallel
# `pplx fetch --prompt` processes honoring the same `retry-after` value all
# wake up at the same instant and recreate the herd. Range is conservative:
# enough to disperse the herd across ~1.5s, not so much that we exceed the
# server's retry-after by a meaningful margin.
_BACKOFF_JITTER_LOW = 0.85
_BACKOFF_JITTER_HIGH = 1.15


def _require_http_url(url: str) -> None:
    """Reject non-HTTP(S) URLs and URLs missing a host. Raises NetworkError.

    Prevents SSRF via file:// and custom schemes, and rejects obviously
    malformed inputs (e.g. `localhost:8080` parsed without a scheme).
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_FETCH_SCHEMES:
        raise NetworkError(
            f"fetch {url}: unsupported URL scheme {parsed.scheme!r} (only http/https allowed)"
        )
    if not parsed.netloc:
        raise NetworkError(f"fetch {url}: URL has no host")


@dataclass
class FetchResult:
    url: str
    title: str | None
    domain: str
    content: str
    is_extracted: bool  # True iff --prompt was used (content is LLM-generated)
    published_date: str | None = None
    truncated: bool = False
    # False iff the server stream was cut before a COMPLETED signal arrived
    # (only meaningful for --prompt mode; plain mode is always True).
    stream_complete: bool = True


@dataclass
class _StreamState:
    """Mutable accumulator threaded through `_consume_one_stream`.

    Lives across retry attempts so partial progress (chunks already received,
    thread identifiers already captured) is not lost when the SSE call raises
    a recoverable error like RateLimitError on a subsequent attempt.
    """

    chunks: list[str]
    backend_uuid: str | None = None
    read_write_token: str | None = None
    saw_completed: bool = False


def extract_chunks_from_event(event: dict[str, Any]) -> list[str]:
    """Pure: pull the streamed markdown chunks added by one SSE event.

    Returns the list of text fragments to append to the accumulating answer.
    Total function: never raises, returns `[]` for any event without the
    expected `ask_text` markdown_block structure. Independently fuzzable.

    Decision filter: we only consume `intended_usage == "ask_text"` blocks,
    not the parallel `ask_text_0_markdown` blocks the server emits — they
    carry the same chunks and reading both would double-count.
    """
    data = event.get("data")
    if not isinstance(data, dict):
        return []
    out: list[str] = []
    for block in data.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        if block.get("intended_usage") != "ask_text":
            continue
        mb = block.get("markdown_block")
        if not isinstance(mb, dict):
            continue
        chunks = mb.get("chunks") or []
        if isinstance(chunks, list):
            out.extend(str(c) for c in chunks)
    return out


def event_marks_completed(event: dict[str, Any]) -> bool:
    """Pure: True iff the SSE event signals the stream has finished."""
    data = event.get("data")
    if not isinstance(data, dict):
        return False
    return data.get("status") == "COMPLETED" or bool(data.get("text_completed"))


def _consume_one_stream(
    client: Client,
    body: dict[str, Any],
    state: _StreamState,
    *,
    remaining_seconds: float | None,
    progress: bool,
) -> None:
    """Drive one SSE call, mutating `state` with whatever it captures.

    Returns normally when the stream ends (COMPLETED or natural close).
    Propagates StreamDeadlineError and RateLimitError to the caller; the
    caller decides whether to retry or salvage.

    Per-event logic is split into pure helpers (`extract_chunks_from_event`,
    `event_marks_completed`) so the parsing rules can be fuzzed in isolation
    from the wire orchestration.
    """
    event_count = 0
    try:
        for event in client.sse_post(
            _PROMPT_ENDPOINT,
            body,
            max_total_seconds=remaining_seconds,
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
            state.chunks.extend(extract_chunks_from_event(event))
            if event_marks_completed(event):
                state.saw_completed = True
                return
    finally:
        # Always close out the heartbeat line on exit (normal, COMPLETED, or
        # exception) so the next stderr writer starts on a fresh line.
        if progress and event_count >= _PROGRESS_EVENT_STRIDE:
            print("", file=sys.stderr, flush=True)


def _rate_limit_backoff(err: RateLimitError, remaining: float | None) -> float:
    """How long to sleep after a 429, bounded by the overall deadline.

    Multiplicative jitter (±15%) is applied AFTER the cap and BEFORE the
    deadline clip — so the cap still bounds the worst case but parallel
    callers honoring the same `retry-after` won't wake in lockstep.
    """
    base = err.retry_after if err.retry_after is not None else _RATE_LIMIT_DEFAULT_BACKOFF
    sleep_s = min(base, _RATE_LIMIT_BACKOFF_CAP)
    sleep_s *= random.uniform(_BACKOFF_JITTER_LOW, _BACKOFF_JITTER_HIGH)
    if remaining is not None:
        sleep_s = min(sleep_s, remaining)
    return max(0.0, sleep_s)


def fetch(
    client: Client,
    url: str,
    *,
    prompt: str | None = None,
    max_chars: int | None = None,
    keep_thread: bool = False,
    timeout: float | None = None,
    progress: bool = False,
) -> FetchResult:
    """Fetch a URL, optionally route through Perplexity's LLM for extraction.

    `max_chars` caps the returned content; the result's `truncated` flag
    indicates whether truncation occurred.

    `keep_thread` controls whether the chat-endpoint thread created by
    `--prompt` mode is preserved in the user's Perplexity UI. Default
    (False) deletes it post-call.

    `timeout` bounds the wall-clock duration of `--prompt` mode (the SSE
    chat call). When the deadline trips with any accumulated content, the
    partial answer is returned with `stream_complete=False`. Plain mode
    uses curl's own connect/read timeouts and ignores this parameter.

    `progress`, when True, emits a single stderr char every N SSE events
    in `--prompt` mode so concurrent backgrounded calls show liveness.
    """
    domain = urlparse(url).netloc or "(unknown)"
    if prompt is None:
        return fetch_page(url, domain, max_chars=max_chars)
    return _fetch_with_prompt(
        client,
        url,
        prompt,
        domain,
        max_chars=max_chars,
        keep_thread=keep_thread,
        timeout=timeout,
        progress=progress,
    )


def fetch_page(
    url: str,
    domain: str,
    *,
    max_chars: int | None,
    session: cf_requests.Session | None = None,
) -> FetchResult:
    """Public: fetch a URL via curl_cffi and extract content with trafilatura.

    No auth: uses a curl_cffi session without perplexity.ai cookies so they
    are not leaked to third-party hosts. Used by `fetch()` (no-prompt mode)
    and by `verbs/snippets._fetch_all` for the concurrent-fetch path.

    `session` (optional): pass a pre-existing curl_cffi Session to reuse the
    TCP connection across calls. The snippets verb uses this to share one
    Session per host group — TCP reuse plus HTTP/2 multiplexing means 6
    same-host URLs cost 1 handshake instead of 6, and one connection per
    host is markedly less Cloudflare-antagonizing than rapid TCP setups.
    When None (default), a fresh session is created and torn down per call.
    """
    _require_http_url(url)
    try:
        if session is None:
            # Standalone path: fresh session, torn down on exit. curl_cffi
            # keeps the chrome TLS fingerprint which handles Cloudflare-
            # protected sources transparently.
            with cf_requests.Session(impersonate="chrome") as sess:
                resp = sess.get(url, timeout=30, allow_redirects=True)
        else:
            # Caller owns the session lifecycle (typically one per host group).
            resp = session.get(url, timeout=30, allow_redirects=True)
    except NetworkError:
        raise
    except Exception as e:
        raise NetworkError(f"fetch {url}: {e!s}") from e

    if resp.status_code >= 400:
        raise NetworkError(f"fetch {url}: HTTP {resp.status_code}")

    html = resp.text or ""
    try:
        import trafilatura
    except ImportError as e:
        raise SchemaError(f"trafilatura is required for local fetch: {e}") from e

    content = (
        trafilatura.extract(
            html,
            output_format="markdown",
            include_links=False,
            include_comments=False,
            favor_recall=True,
        )
        or ""
    )

    # Also pull metadata where we can — trafilatura returns a metadata
    # object with title / date if available.
    md = trafilatura.extract_metadata(html)
    title = getattr(md, "title", None) if md else None
    published = getattr(md, "date", None) if md else None

    truncated = False
    if max_chars and len(content) > max_chars:
        content = content[:max_chars]
        truncated = True

    return FetchResult(
        url=url,
        title=title,
        domain=domain,
        content=content,
        is_extracted=False,
        published_date=published,
        truncated=truncated,
    )


def _fetch_with_prompt(
    client: Client,
    url: str,
    prompt: str,
    domain: str,
    *,
    max_chars: int | None,
    keep_thread: bool = False,
    timeout: float | None = None,
    progress: bool = False,
) -> FetchResult:
    """Submit url+prompt to /rest/sse/perplexity_ask; Perplexity's LLM has
    URL-fetching as a tool and will fetch+extract+answer in one round trip.

    We accumulate the markdown_block text across events as the answer
    streams in. Unless `keep_thread` is True, we also delete the thread
    Perplexity creates in the UI post-call (default behavior is to clean
    up so agent calls don't pollute the user's thread history).

    `timeout` is the overall wall-clock deadline (None = no deadline). When
    the deadline trips after partial content has accumulated, we return the
    partial answer with `stream_complete=False` so callers can decide
    whether to retry — the agent contract is "you always get *something*
    plus a flag", not "deadline → exception".

    Auto-retry on `RateLimitError` follows `retry_after` and is bounded by
    the overall deadline so a stubborn 429 can't push past `timeout`.
    """
    body = _build_chat_body(f"{prompt}\n\nFor URL: {url}")
    # The chat endpoint streams the answer one chunk per event. Each event
    # may have parallel blocks (`ask_text` for the incremental stream and
    # `ask_text_0_markdown` for the markdown-rendered variant) carrying the
    # same chunk - we read only `ask_text` to avoid double-counting.
    state = _StreamState(chunks=[])
    deadline_tripped = False

    # Wall-clock budget for the whole verb call. Rate-limit retries share the
    # same deadline as the stream itself so a 429 burst can't overrun it.
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
            _consume_one_stream(client, body, state, remaining_seconds=remaining, progress=progress)
            break  # stream ran to a normal end (COMPLETED or natural close)
        except StreamDeadlineError:
            # Soft-fail: keep accumulated chunks; retrying would just consume
            # the (already-zero) remaining budget for no benefit.
            deadline_tripped = True
            break
        except RateLimitError as e:
            last_rate_limit = e
            if attempt >= _RATE_LIMIT_MAX_ATTEMPTS:
                raise
            sleep_s = _rate_limit_backoff(e, _remaining())
            if sleep_s > 0:
                print(
                    f"pplx fetch: rate limited (attempt {attempt}/"
                    f"{_RATE_LIMIT_MAX_ATTEMPTS}); sleeping {sleep_s:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)

    content = "".join(state.chunks).strip()
    if not content and not state.saw_completed:
        if deadline_tripped:
            raise StreamDeadlineError(
                f"SSE stream on {_PROMPT_ENDPOINT} exceeded {timeout:.1f}s "
                f"deadline before any content arrived"
            )
        raise SchemaError(f"no markdown_block content received from {_PROMPT_ENDPOINT}")

    # Best-effort thread cleanup. client.delete_thread is documented + actually
    # implemented as best-effort: any failure prints to stderr and returns
    # False, so the user's call survives an orphaned thread on Perplexity's side.
    if not keep_thread and state.backend_uuid and state.read_write_token:
        client.delete_thread(state.backend_uuid, state.read_write_token)

    truncated = False
    if max_chars and len(content) > max_chars:
        content = content[:max_chars]
        truncated = True

    return FetchResult(
        url=url,
        title=None,  # not available from the chat response (no header equivalent)
        domain=domain,
        content=content,
        is_extracted=True,
        published_date=None,
        truncated=truncated,
        stream_complete=state.saw_completed,
    )


def _build_chat_body(query: str) -> dict[str, Any]:
    """Minimum-viable body for /rest/sse/perplexity_ask. See docs/wire/search-web.md
    for the full captured shape; we strip UI-specific fields here.

    `timezone` is set to "UTC" rather than detected from the host: detection
    actively leaks the user's location, and `time.tzname` returns
    abbreviations ("EST") rather than the IANA names ("America/New_York")
    Perplexity expects. UTC is deterministic and accepted everywhere.
    """
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
            "model_preference": "turbo",
            "frontend_uuid": frontend_uuid,
            "frontend_context_uuid": str(uuid4()),
            "client_search_results_cache_key": frontend_uuid,
            "use_schematized_api": True,
            "send_back_text_in_streaming_api": True,
            "skip_search_enabled": True,
            "is_incognito": False,
            "attachments": [],
            "mentions": [],
            "client_coordinates": None,
            "dsl_query": query,
        },
    }
