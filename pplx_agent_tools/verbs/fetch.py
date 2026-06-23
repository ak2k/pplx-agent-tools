"""pplx fetch verb: URL → cleaned content (optional LLM extraction via --prompt).

Hybrid implementation:
  - Plain mode (no --prompt): fetch the URL ourselves via curl_cffi (chrome-
    impersonate, same Cloudflare-handling as Perplexity calls), extract main
    content with trafilatura.
  - --prompt mode: route the URL + prompt through /rest/sse/perplexity_ask
    (the LLM has URL-fetching as a tool), parse out the answer — sharing the
    ask-family SSE orchestration in `_ask_common` (retry/deadline/heartbeat/
    cleanup) with `ask` and `research`.

Why the hybrid: Perplexity's web-session API surface has no URL→content
fetch endpoint we can reach (RE'd 2026-05-12; see plan's "Open questions").
Their internal `pplx content fetch` CLI must use Sonar-API or internal-only
auth. Implementing fetch ourselves loses the `is_paywall` / `is_cached`
signals but keeps the agent-shape single-command primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from curl_cffi import requests as cf_requests

from ..errors import NetworkError, SchemaError, StreamDeadlineError
from ..wire import Client
from ._ask_common import base_ask_params, extract_chunks_from_event, run_ask_stream

_PROMPT_ENDPOINT = "/rest/sse/perplexity_ask"

# Schemes accepted for outbound fetch. Anything else (file://, ftp://,
# gopher://, custom) is rejected up front — we never want curl_cffi to
# touch the local filesystem or non-HTTP backends from a user-supplied URL.
_ALLOWED_FETCH_SCHEMES = frozenset({"http", "https"})


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


def fetch(
    client: Client,
    url: str,
    *,
    prompt: str | None = None,
    max_chars: int | None = None,
    keep_thread: bool = False,
    timeout: float | None = None,
    progress: bool = False,
    model: str = "turbo",
) -> FetchResult:
    """Fetch a URL, optionally route through Perplexity's LLM for extraction.

    `max_chars` caps the returned content; the result's `truncated` flag
    indicates whether truncation occurred.

    `keep_thread` controls whether the chat-endpoint thread created by
    `--prompt` mode is preserved in the user's Perplexity UI. Default
    (False) deletes it post-call. `--prompt` runs incognito so the thread
    never enters history regardless.

    `model` is the `model_preference` for `--prompt` mode (default `turbo`).

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
        model=model,
    )


def fetch_page(
    url: str,
    domain: str,
    *,
    max_chars: int | None,
    session: cf_requests.Session[cf_requests.Response] | None = None,
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
    model: str = "turbo",
) -> FetchResult:
    """Submit url+prompt to /rest/sse/perplexity_ask; Perplexity's LLM has
    URL-fetching as a tool and answers in one round-trip.

    Shares the ask-family SSE orchestration (429 retry + wall-clock deadline +
    heartbeat + thread-id/completion/FAILED capture) via
    `_ask_common.run_ask_stream`; we accumulate the `markdown_block` chunks. The
    created thread runs incognito and is best-effort deleted (unless
    `keep_thread`) on every exit path. On a tripped deadline with partial content
    we return it with `stream_complete=False` (the agent contract is "you always
    get *something* plus a flag").
    """
    body = _build_chat_body(f"{prompt}\n\nFor URL: {url}", model_preference=model)
    chunks: list[str] = []

    def on_event(event: dict[str, Any]) -> None:
        chunks.extend(extract_chunks_from_event(event))

    state, deadline_tripped = run_ask_stream(
        client,
        _PROMPT_ENDPOINT,
        body,
        on_event=on_event,
        timeout=timeout,
        progress=progress,
        label="fetch",
    )

    # Cleanup runs on EVERY exit path (success, FAILED, no-content) — delete_thread
    # never raises, so doing it before the error checks stops a leak.
    if not keep_thread and state.backend_uuid and state.read_write_token:
        client.delete_thread(state.backend_uuid, state.read_write_token)

    if state.failed:
        raise SchemaError(
            f"fetch --prompt on {_PROMPT_ENDPOINT} returned status=FAILED; model "
            f"{model!r} may be invalid — check `pplx models`"
        )

    content = "".join(chunks).strip()
    if not content and not state.saw_completed:
        if deadline_tripped:
            raise StreamDeadlineError(
                f"SSE stream on {_PROMPT_ENDPOINT} exceeded {timeout:.1f}s "
                f"deadline before any content arrived"
            )
        raise SchemaError(f"no markdown_block content received from {_PROMPT_ENDPOINT}")

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


def _build_chat_body(
    query: str, *, model_preference: str = "turbo", is_incognito: bool = True
) -> dict[str, Any]:
    """Copilot ask body for fetch --prompt. Delegates the shared field set to
    `_ask_common.base_ask_params`; `is_incognito` defaults True so `--prompt`
    threads never enter history (delete_thread cleanup is then belt-and-
    suspenders, not load-bearing)."""
    return {
        "query_str": query,
        "params": base_ask_params(
            query, model_preference=model_preference, is_incognito=is_incognito
        ),
    }
