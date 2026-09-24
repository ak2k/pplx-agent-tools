"""pplx fetch verb: URL → cleaned content (optional LLM extraction via --prompt).

Hybrid implementation:
  - Plain mode (no --prompt): fetch the URL ourselves via curl_cffi (chrome-
    impersonate, same Cloudflare-handling as Perplexity calls), extract main
    content with trafilatura.
  - --prompt mode: route the URL + prompt through /rest/sse/perplexity_ask
    (the LLM has URL-fetching as a tool), parse out the answer — sharing the
    ask-family SSE orchestration in `_ask_common` (retry/deadline/stall/
    heartbeat/cleanup) with `ask` and `research`.

Why the hybrid: Perplexity's web-session API surface has no URL→content
fetch endpoint we can reach (RE'd 2026-05-12; see plan's "Open questions").
Their internal `pplx content fetch` CLI must use Sonar-API or internal-only
auth. Implementing fetch ourselves loses the `is_paywall` / `is_cached`
signals but keeps the agent-shape single-command primitive.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from curl_cffi import requests as cf_requests

from ..errors import (
    NetworkError,
    PplxError,
    RateLimitError,
    SchemaError,
    TargetHttpError,
)
from ..netguard import (
    PublicUrl,
    check_authority,
    check_url,
    join_location,
    redact,
    strip_userinfo,
)
from ..wire import Client
from ._ask_common import (
    AskStreamState,
    base_ask_params,
    blocks_changed,
    cutoff_cause,
    cutoff_warnings,
    extract_chunks_from_event,
    no_content_error,
    release_thread,
    run_ask_stream,
)

_PROMPT_ENDPOINT = "/rest/sse/perplexity_ask"

# Manual redirect handling: curl_cffi's allow_redirects would follow a 3xx into
# an internal host without re-running the SSRF guard, so we cap the hops and
# check each Location ourselves (see _get_guarded).
_MAX_REDIRECTS = 5
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


def _fetch_hop(
    session: cf_requests.Session[cf_requests.Response],
    target: PublicUrl,
    *,
    timeout: float,
) -> cf_requests.Response:
    """The one place plain fetch sends a request; only a checked URL gets here."""
    return session.get(target.url, timeout=timeout, allow_redirects=False, auth=target.auth)


def _get_guarded(
    session: cf_requests.Session[cf_requests.Response],
    url: str,
    *,
    timeout: float = 30.0,
) -> cf_requests.Response:
    """GET `url`, following redirects manually so every hop passes `check_url`."""
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        resp = _fetch_hop(session, check_url(current), timeout=timeout)
        location = resp.headers.get("location")
        if resp.status_code in _REDIRECT_CODES and location:
            current = join_location(current, location)
            continue
        return resp
    raise TargetHttpError(f"fetch {redact(url)}: exceeded {_MAX_REDIRECTS} redirects")


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
    warnings: list[str] = field(default_factory=list)
    # "stall" | "deadline" when that bound cut the --prompt stream; None otherwise.
    cut_by: str | None = None


def fetch(
    client: Client,
    url: str,
    *,
    prompt: str | None = None,
    max_chars: int | None = None,
    keep_thread: bool = False,
    timeout: float | None = None,
    stall_seconds: float | None = None,
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
    chat call) and `stall_seconds` its time without new content. When
    either trips with any accumulated content, the partial answer is returned
    with `stream_complete=False` and a warning naming which one. Plain mode
    uses curl's own connect/read timeouts and ignores both.

    `progress`, when True, emits a single stderr char every N SSE events
    in `--prompt` mode so concurrent backgrounded calls show liveness.
    """
    if prompt is None:
        return fetch_plain(url, max_chars=max_chars)
    check_authority(url)
    shown = strip_userinfo(url)
    if shown != url:
        print("pplx fetch: removed credentials from the URL sent to Perplexity", file=sys.stderr)
    return _fetch_with_prompt(
        client,
        shown,
        prompt,
        _domain(shown),
        max_chars=max_chars,
        keep_thread=keep_thread,
        timeout=timeout,
        stall_seconds=stall_seconds,
        progress=progress,
        model=model,
    )


def _domain(url: str) -> str:
    """`host[:port]` of `url`; never the userinfo part of the netloc."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return "(unknown)"
    if not host:
        return "(unknown)"
    if ":" in host:
        host = f"[{host}]"
    return host if port is None else f"{host}:{port}"


def fetch_plain(url: str, *, max_chars: int | None = None) -> FetchResult:
    """Plain-mode fetch. Takes no Client: it never sends Perplexity cookies."""
    return fetch_page(url, _domain(url), max_chars=max_chars)


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
    shown = redact(url)
    try:
        if session is None:
            # Standalone path: fresh session, torn down on exit. curl_cffi
            # keeps the chrome TLS fingerprint which handles Cloudflare-
            # protected sources transparently.
            with cf_requests.Session(impersonate="chrome") as sess:
                resp = _get_guarded(sess, url)
        else:
            # Caller owns the session lifecycle (typically one per host group).
            resp = _get_guarded(session, url)
    except PplxError:
        raise
    except Exception as e:
        raise NetworkError(f"fetch {shown}: {e!s}") from e

    status = resp.status_code
    if status == 429:
        raise RateLimitError(f"fetch {shown}: HTTP 429")
    if status == 408 or status >= 500:
        raise NetworkError(f"fetch {shown}: HTTP {status}")
    if status >= 400:
        raise TargetHttpError(f"fetch {shown}: HTTP {status}")

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
        url=strip_userinfo(url),
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
    stall_seconds: float | None = None,
    progress: bool = False,
    model: str = "turbo",
) -> FetchResult:
    """Submit url+prompt to /rest/sse/perplexity_ask; Perplexity's LLM has
    URL-fetching as a tool and answers in one round-trip.

    Shares the ask-family SSE orchestration (429 retry + wall-clock deadline +
    stall guard + heartbeat + thread-id/completion/FAILED capture) via
    `_ask_common.run_ask_stream`; we accumulate the `markdown_block` chunks. The
    created thread runs incognito and is best-effort deleted (unless
    `keep_thread`) on every exit path. On a tripped deadline or stall with
    partial content we return it with `stream_complete=False` (the agent contract is "you always
    get *something* plus a flag").
    """
    body = _build_chat_body(f"{prompt}\n\nFor URL: {url}", model_preference=model)
    chunks: list[str] = []

    def on_event(event: dict[str, Any]) -> None:
        chunks.extend(extract_chunks_from_event(event))

    state = AskStreamState()
    try:
        run_ask_stream(
            client,
            _PROMPT_ENDPOINT,
            body,
            state,
            on_event=on_event,
            timeout=timeout,
            stall_seconds=stall_seconds,
            progress=progress,
            label="fetch",
            is_progress=blocks_changed(),
        )
    finally:
        release_thread(client, state, keep_thread=keep_thread)

    if state.failed:
        raise SchemaError(
            f"fetch --prompt on {_PROMPT_ENDPOINT} returned status=FAILED; model "
            f"{model!r} may be invalid — check `pplx models`"
        )

    content = "".join(chunks).strip()
    if not content and not state.saw_completed:
        raise no_content_error(
            label="fetch --prompt",
            endpoint=_PROMPT_ENDPOINT,
            timeout=timeout,
            cutoff=state.cutoff,
        )

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
        cut_by=cutoff_cause(state),
        warnings=cutoff_warnings(state),
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
