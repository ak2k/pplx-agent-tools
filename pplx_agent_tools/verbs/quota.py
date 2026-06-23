"""pplx quota verb: subscription rate-limit / availability via /rest/rate-limit/status.

Stateless GET (creates no thread). Lets an agent loop check whether an expensive
mode (research, agentic_research) is still available before firing, and lets the
user see where they stand on metered sources.

Response shape (observed 2026-06-22):

    {
      "free_queries": {"available": bool, "remaining_detail": {...}},
      "modes":   {"<mode>":   {"available": bool, "remaining_detail": {...}}, ...},
      "sources": {"<source>": {"available": bool, "remaining_detail": {...}}, ...}
    }

`remaining_detail.kind` is "not_provided" (Pro: effectively unmetered) or "exact"
with a `remaining` integer (metered connectors, e.g. an exhausted source at 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import SchemaError
from ..wire import Client

ENDPOINT = "/rest/rate-limit/status"


@dataclass
class QuotaItem:
    name: str
    available: bool
    remaining: int | None  # exact count when known; None when "not_provided"


@dataclass
class QuotaResult:
    free_queries: QuotaItem | None
    modes: list[QuotaItem]
    sources: list[QuotaItem]
    warnings: list[str] = field(default_factory=list)


def quota(client: Client) -> QuotaResult:
    """Fetch current rate-limit / availability status. Stateless GET."""
    raw = client.get_json(ENDPOINT)
    return decode_quota(raw)


def decode_quota(raw: Any) -> QuotaResult:
    """Pure decode: raw /rest/rate-limit/status response → QuotaResult.

    Tolerant of missing groups (returns empty lists) but raises SchemaError if
    the top-level shape isn't an object — that signals Perplexity changed the
    contract rather than just dropping an optional field.
    """
    if not isinstance(raw, dict):
        raise SchemaError(f"unexpected response type from {ENDPOINT}: {type(raw).__name__}")
    fq_raw = raw.get("free_queries")
    free_queries = _item("free_queries", fq_raw) if isinstance(fq_raw, dict) else None
    return QuotaResult(
        free_queries=free_queries,
        modes=_group(raw.get("modes")),
        sources=_group(raw.get("sources")),
    )


def _group(raw: Any) -> list[QuotaItem]:
    if not isinstance(raw, dict):
        return []
    return [_item(name, v) for name, v in sorted(raw.items()) if isinstance(v, dict)]


def _item(name: str, raw: dict[str, Any]) -> QuotaItem:
    detail = raw.get("remaining_detail")
    remaining: int | None = None
    if isinstance(detail, dict) and detail.get("kind") == "exact":
        rv = detail.get("remaining")
        if isinstance(rv, int):
            remaining = rv
    return QuotaItem(name=name, available=bool(raw.get("available")), remaining=remaining)
