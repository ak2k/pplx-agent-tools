"""pplx search verb: ranked hits via /rest/realtime/search-web.

The web SPA also offers SSE-chat-based search (/rest/sse/perplexity_ask), but
the realtime endpoint is what Perplexity uses internally for fast ranked-hit
retrieval — JSON in, JSON out, no LLM, native multi-query.

See docs/wire/search-web.md for the wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..errors import SchemaError
from ..wire import Client

ENDPOINT = "/rest/realtime/search-web"
SEARCH_TYPES = ("web", "academic", "images", "videos", "shopping")

# Drop hits flagged as anything other than a real web result. Same filter logic
# as the SSE response — Perplexity uses the same is_* booleans across both.
_DROP_FLAGS_WEB: tuple[str, ...] = (
    "is_navigational",
    "is_widget",
    "is_knowledge_card",
    "is_image",
    "is_video",
    "is_audio",
    "is_map",
    "is_memory",
    "is_conversation_history",
    "is_conversation_summary",
    "is_attachment",
    "is_extra_info",
    "is_pro_search_table",
)


@dataclass
class Hit:
    url: str
    title: str
    domain: str | None
    snippet: str | None
    summary: str | None = None
    published_date: str | None = None
    images: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    query: str
    type: str
    hits: list[Hit]
    # Count of hits actually returned (post-dedup, post-limit). The
    # `/rest/realtime/search-web` endpoint does not return a server-side
    # total, so this is NOT the count of available matches on Perplexity's
    # side — just `len(hits)`, surfaced as a field for caller ergonomics.
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
    """Run a single search query against /rest/realtime/search-web.

    `search_type` is currently only `"web"`. The other types (academic,
    images, videos, shopping) don't have dedicated cookie-auth endpoints
    (RE-confirmed). They're Phase 2 territory: would route through
    /rest/sse/perplexity_ask with sources=["scholar"/"videos"/etc.] and
    parse type-specific block shapes.
    """
    if search_type != "web":
        raise NotImplementedError(
            f"search_type={search_type!r} is Phase 2 — no dedicated endpoint "
            f"for cookie auth. Track via plan-doc Step 9."
        )

    return search_many(
        client,
        [query],
        search_type=search_type,
        limit=limit,
        country=country,
        domains=domains,
        excluded_domains=excluded_domains,
    )


def search_many(
    client: Client,
    queries: list[str],
    *,
    search_type: str = "web",
    limit: int = 10,
    country: str = "US",
    domains: list[str] | None = None,
    excluded_domains: list[str] | None = None,
) -> SearchResult:
    """Run multiple queries in one round-trip. The endpoint takes queries[]
    natively and merges/dedupes server-side.
    """
    if not queries:
        return SearchResult(query="", type=search_type, hits=[], total=0)

    body = _build_body(
        queries,
        domains=domains,
        excluded_domains=excluded_domains,
        country=country,
    )
    raw = client.post_json(ENDPOINT, body)
    if not isinstance(raw, dict):
        raise SchemaError(f"unexpected response type from {ENDPOINT}: {type(raw).__name__}")

    raw_hits = raw.get("web_results") or []
    if not isinstance(raw_hits, list):
        raise SchemaError(f"{ENDPOINT} returned non-list web_results: {type(raw_hits).__name__}")

    hits = [_to_hit(h) for h in raw_hits if _keep(h)]
    # Stable de-dupe by URL (server may already do this for queries[]; we
    # belt-and-suspender for safety).
    seen: set[str] = set()
    deduped: list[Hit] = []
    for h in hits:
        if h.url in seen:
            continue
        seen.add(h.url)
        deduped.append(h)

    # `total` mirrors `len(hits)` deliberately — the endpoint doesn't ship
    # a server-side count, so a "true total" would just be misleading.
    hits = deduped[:limit]
    return SearchResult(
        query=" | ".join(queries),
        type=search_type,
        hits=hits,
        total=len(hits),
    )


def _keep(hit: dict[str, Any]) -> bool:
    if not isinstance(hit, dict):
        return False
    return not any(hit.get(flag) for flag in _DROP_FLAGS_WEB)


def _to_hit(raw: dict[str, Any]) -> Hit:
    url = raw.get("url")
    title = raw.get("name")
    if not isinstance(url, str) or not isinstance(title, str):
        raise SchemaError(f"web_result missing url/name: keys={sorted(raw.keys())[:10]}")
    images_raw = []
    meta = raw.get("meta_data")
    if isinstance(meta, dict):
        images_raw = meta.get("images") or []
    return Hit(
        url=url,
        title=title,
        domain=raw.get("domain") or None,
        snippet=raw.get("snippet") or None,
        summary=raw.get("summary") or None,
        published_date=raw.get("timestamp") or None,
        images=[str(u) for u in images_raw] if isinstance(images_raw, list) else [],
    )


def _build_body(
    queries: list[str],
    *,
    domains: list[str] | None,
    excluded_domains: list[str] | None,
    country: str,
) -> dict[str, Any]:
    """Minimal body — see docs/wire/search-web.md. session_id is just a
    per-call tracking UUID; the endpoint doesn't reuse state across calls.
    """
    body: dict[str, Any] = {
        "session_id": str(uuid4()),
        "queries": list(queries),
    }
    # These params haven't been validated against this endpoint yet; the
    # endpoint may silently ignore them or 422. Document if/when encountered.
    if country and country.upper() != "US":
        body["country"] = country.upper()
    if domains:
        body["domain_filter"] = list(domains)
    if excluded_domains:
        body["excluded_domains"] = list(excluded_domains)
    return body
