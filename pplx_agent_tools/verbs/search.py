"""pplx search verb: ranked hits via /rest/realtime/search-web.

The web SPA also offers SSE-chat-based search (/rest/sse/perplexity_ask), but
the realtime endpoint is what Perplexity uses internally for fast ranked-hit
retrieval — JSON in, JSON out, no LLM, native multi-query.

See docs/wire/search-web.md for the wire format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4

from ..errors import SchemaError
from ..jsonval import as_array, as_object, str_or_none
from ..wire import Client

ENDPOINT = "/rest/realtime/search-web"

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
    hits: list[Hit]
    # Count of hits actually returned (post-dedup, post-limit). The
    # `/rest/realtime/search-web` endpoint does not return a server-side
    # total, so this is NOT the count of available matches on Perplexity's
    # side — just `len(hits)`, surfaced as a field for caller ergonomics.
    total: int
    warnings: list[str] = field(default_factory=list)


def search_many(
    client: Client,
    queries: list[str],
    *,
    limit: int = 10,
) -> SearchResult:
    """Run multiple queries in one round-trip. The endpoint takes queries[]
    natively and merges/dedupes server-side.

    Thin orchestrator: builds the body, calls the wire, hands the raw
    response to `decode_search_response()`. The decode logic is pure and
    independently fuzzable (see tests/test_fuzz_robustness.py).
    """
    if not queries:
        return SearchResult(query="", hits=[], total=0)

    raw = client.post_json(ENDPOINT, _build_body(queries))
    return decode_search_response(raw, query=" | ".join(queries), limit=limit)


def decode_search_response(raw: object, *, query: str, limit: int) -> SearchResult:
    """Pure decode: raw `/rest/realtime/search-web` response → SearchResult.

    No I/O. Total function: returns a SearchResult or raises SchemaError on
    any structural mismatch. Used by `search_many` (with real wire response)
    and by fixture-replay + fuzz tests (with canned / adversarial dicts).
    """
    body = as_object(raw)
    if body is None:
        raise SchemaError(f"unexpected response type from {ENDPOINT}: {type(raw).__name__}")

    web_results: object = body.get("web_results") or []
    raw_hits = as_array(web_results)
    if raw_hits is None:
        raise SchemaError(f"{ENDPOINT} returned non-list web_results: {type(web_results).__name__}")

    hits = [_to_hit(h) for h in map(as_object, raw_hits) if h is not None and _keep(h)]
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
    return SearchResult(query=query, hits=hits, total=len(hits))


def _keep(hit: object) -> bool:
    """Filter a candidate web_result, dropping non-objects and anything
    flagged as widget/image/nav/etc."""
    obj = as_object(hit)
    if obj is None:
        return False
    return not any(obj.get(flag) for flag in _DROP_FLAGS_WEB)


def _to_hit(raw: Mapping[str, object]) -> Hit:
    url = raw.get("url")
    title = raw.get("name")
    if not isinstance(url, str) or not isinstance(title, str):
        raise SchemaError(f"web_result missing url/name: keys={sorted(raw.keys())[:10]}")
    meta = as_object(raw.get("meta_data")) or {}
    images = as_array(meta.get("images")) or []
    return Hit(
        url=url,
        title=title,
        domain=str_or_none(raw.get("domain")),
        snippet=str_or_none(raw.get("snippet")),
        summary=str_or_none(raw.get("summary")),
        published_date=str_or_none(raw.get("timestamp")),
        images=[u for u in images if isinstance(u, str)],
    )


def _build_body(queries: list[str]) -> dict[str, object]:
    """Minimal body — see docs/wire/search-web.md. session_id is just a
    per-call tracking UUID; the endpoint doesn't reuse state across calls.
    """
    return {
        "session_id": str(uuid4()),
        "queries": list(queries),
    }
