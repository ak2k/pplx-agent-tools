"""pplx-search verb: ranked hits via /rest/sse/perplexity_ask.

See docs/wire/search-web.md for the wire format. Strategy:
  1. POST a query body to the SSE chat endpoint
  2. Stream events; the first event with a populated
     blocks[].web_result_block.web_results[] carries the full ranked list
  3. Break out of the stream once we have hits (saves ~95% of bytes and
     skips the LLM synthesis cost we don't use)
  4. Filter out is_navigational / is_widget / is_knowledge_card / is_image /
     is_memory / is_attachment hits — those aren't "web search results" in
     the conventional sense the caller wants
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..errors import SchemaError
from ..wire import Client

ENDPOINT = "/rest/sse/perplexity_ask"
SEARCH_TYPES = ("web", "academic", "images", "videos", "shopping")

# Hits with any of these flags set are dropped from the web result list.
# (-t images / -t videos / etc. would reset these filters when implemented.)
_DROP_FLAGS_WEB: tuple[str, ...] = (
    "is_navigational",
    "is_widget",
    "is_knowledge_card",
    "is_image",
    "is_memory",
    "is_conversation_history",
    "is_conversation_summary",
    "is_attachment",
    "is_client_context",
)


@dataclass
class Hit:
    url: str
    title: str
    domain: str | None
    snippet: str | None
    published_date: str | None = None
    images: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    query: str
    type: str
    hits: list[Hit]
    total: int
    warnings: list[str] = field(default_factory=list)


def search(
    client: Client,
    query: str,
    *,
    search_type: str = "web",
    limit: int = 10,
    country: str = "US",
    domains: list[str] | None = None,
    excluded_domains: list[str] | None = None,
) -> SearchResult:
    """Run one search query, return parsed SearchResult.

    For now `search_type` only supports "web"; the other types need separate RE
    (see plan Step 9).
    """
    if search_type != "web":
        raise NotImplementedError(
            f"search_type={search_type!r} not yet implemented (Step 9 in the plan)"
        )

    body = _build_body(
        query,
        domains=domains,
        excluded_domains=excluded_domains,
        country=country,
    )

    raw_hits: list[dict[str, Any]] = []
    warnings: list[str] = []
    saw_classifier_skip = False

    for event in client.sse_post(ENDPOINT, body):
        data = event.get("data")
        if not isinstance(data, dict):
            continue

        # Soft signal: if the server-side classifier elected to skip search,
        # the response will have empty web_results and the LLM will answer
        # without sources. Surface as a warning.
        cls = data.get("classifier_results")
        if isinstance(cls, dict) and cls.get("skip_search") and not saw_classifier_skip:
            saw_classifier_skip = True
            warnings.append("perplexity classifier elected to skip web search for this query")

        for hit_list in _iter_web_results(data):
            raw_hits = hit_list
            break  # outer loop continues but we've captured what we wanted
        if raw_hits:
            break  # early termination — close the SSE connection

    hits = [_to_hit(h) for h in raw_hits if _keep(h)]
    hits = hits[:limit]
    return SearchResult(
        query=query,
        type=search_type,
        hits=hits,
        total=len(hits),
        warnings=warnings,
    )


def _iter_web_results(data: dict[str, Any]):
    """Yield each non-empty web_results list found in this event's blocks."""
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        wrb = block.get("web_result_block")
        if not isinstance(wrb, dict):
            continue
        results = wrb.get("web_results")
        if isinstance(results, list) and results:
            yield results


def _keep(hit: dict[str, Any]) -> bool:
    if not isinstance(hit, dict):
        return False
    return not any(hit.get(flag) for flag in _DROP_FLAGS_WEB)


def _to_hit(raw: dict[str, Any]) -> Hit:
    url = raw.get("url")
    title = raw.get("name")
    if not isinstance(url, str) or not isinstance(title, str):
        raise SchemaError(
            f"web_result missing url/name: keys={sorted(raw.keys())[:10]}"
        )
    meta = raw.get("meta_data") or {}
    images_raw = meta.get("images") if isinstance(meta, dict) else None
    images = [str(u) for u in images_raw] if isinstance(images_raw, list) else []
    return Hit(
        url=url,
        title=title,
        domain=(meta.get("citation_domain_name") if isinstance(meta, dict) else None),
        snippet=raw.get("snippet") or None,
        published_date=raw.get("timestamp") or None,
        images=images,
    )


def _build_body(
    query: str,
    *,
    domains: list[str] | None,
    excluded_domains: list[str] | None,
    country: str,
) -> dict[str, Any]:
    """Minimal viable body — see docs/wire/search-web.md for the full captured shape."""
    frontend_uuid = str(uuid4())
    params: dict[str, Any] = {
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
        "is_nav_suggestions_disabled": False,
        "is_incognito": False,
        "is_related_query": False,
        "is_sponsored": False,
        "always_search_override": False,
        "override_no_search": False,
        "local_search_enabled": False,
        "extended_context": False,
        "attachments": [],
        "mentions": [],
        "client_coordinates": None,
        "dsl_query": query,
    }
    # `country` is captured as part of `params` in the upstream `pplx --help`;
    # the web frontend doesn't send it explicitly. Forwarded only if non-default
    # so the request remains close to what the web UI sends.
    if country and country.upper() != "US":
        params["country"] = country.upper()
    if domains:
        params["domain_filter"] = list(domains)
    if excluded_domains:
        params["excluded_domains"] = list(excluded_domains)
    return {"query_str": query, "params": params}
