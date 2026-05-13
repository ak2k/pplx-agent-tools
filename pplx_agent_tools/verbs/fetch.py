"""pplx-fetch verb: URL → cleaned content (optional LLM extraction via --prompt).

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

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from ..errors import NetworkError, SchemaError
from ..wire import Client

_PROMPT_ENDPOINT = "/rest/sse/perplexity_ask"


@dataclass
class FetchResult:
    url: str
    title: str | None
    domain: str
    content: str
    is_extracted: bool  # True iff --prompt was used (content is LLM-generated)
    published_date: str | None = None
    truncated: bool = False


def fetch(
    client: Client,
    url: str,
    *,
    prompt: str | None = None,
    max_chars: int | None = None,
) -> FetchResult:
    """Fetch a URL, optionally route through Perplexity's LLM for extraction.

    `max_chars` caps the returned content; the result's `truncated` flag
    indicates whether truncation occurred.
    """
    domain = urlparse(url).netloc or "(unknown)"
    if prompt is None:
        return _fetch_local(url, domain, max_chars=max_chars)
    return _fetch_with_prompt(client, url, prompt, domain, max_chars=max_chars)


def _fetch_local(url: str, domain: str, *, max_chars: int | None) -> FetchResult:
    """Fetch the URL via curl_cffi and extract content with trafilatura."""
    try:
        # Use a fresh session so we don't send perplexity.ai cookies to a
        # random third-party host. curl_cffi keeps the chrome TLS fingerprint
        # which is what we want for Cloudflare-protected sources too.
        from curl_cffi import requests as cf_requests

        with cf_requests.Session(impersonate="chrome") as sess:
            resp = sess.get(url, timeout=30, allow_redirects=True)
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
) -> FetchResult:
    """Submit url+prompt to /rest/sse/perplexity_ask; Perplexity's LLM has
    URL-fetching as a tool and will fetch+extract+answer in one round trip.

    We accumulate the markdown_block text across events as the answer
    streams in.
    """
    query = f"{prompt}\n\nFor URL: {url}"
    body = _build_chat_body(query)

    # Title is not available from the chat response (no header equivalent).
    title: str | None = None
    # The chat endpoint streams the answer one chunk per event. Each event
    # may have parallel blocks (`ask_text` for the incremental stream and
    # `ask_text_0_markdown` for the markdown-rendered variant) carrying the
    # same chunk — accumulate from only one to avoid duplication.
    chunks_acc: list[str] = []
    saw_completed = False

    for event in client.sse_post(_PROMPT_ENDPOINT, body):
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        for block in data.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            if block.get("intended_usage") != "ask_text":
                continue
            mb = block.get("markdown_block")
            if isinstance(mb, dict):
                chunks = mb.get("chunks") or []
                if isinstance(chunks, list):
                    chunks_acc.extend(str(c) for c in chunks)
        if data.get("status") == "COMPLETED" or data.get("text_completed"):
            saw_completed = True
            break

    content = "".join(chunks_acc).strip()
    if not content and not saw_completed:
        raise SchemaError(f"no markdown_block content received from {_PROMPT_ENDPOINT}")

    truncated = False
    if max_chars and len(content) > max_chars:
        content = content[:max_chars]
        truncated = True

    return FetchResult(
        url=url,
        title=title,
        domain=domain,
        content=content,
        is_extracted=True,
        published_date=None,
        truncated=truncated,
    )


def _build_chat_body(query: str) -> dict[str, Any]:
    """Minimum-viable body for /rest/sse/perplexity_ask. See docs/wire/search-web.md
    for the full captured shape; we strip UI-specific fields here.
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
            "timezone": "America/New_York",
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
