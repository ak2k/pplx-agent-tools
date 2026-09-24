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

Anonymous / expired-session shape (observed 2026-09-09): the endpoint still
answers 200, with `free_queries` available and every mode and source
`{"available": false, "remaining_detail": {"kind": "exact", "remaining": 0}}`.
Nothing in the payload marks it as anonymous, so it decodes as a legitimately
exhausted account. See tests/fixtures/rate-limit-status/anonymous.json.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..errors import SchemaError
from ..jsonval import as_object, int_or_none
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
    """Fetch current rate-limit / availability status. Creates no thread.

    Pre-flights `/api/auth/session` because the anonymous payload (see module
    docstring) is indistinguishable from an exhausted account: without it an
    expired cookie renders as "you have used everything up" and exits 0.
    """
    client.auth_session()
    raw = client.get_json(ENDPOINT)
    return decode_quota(raw)


def decode_quota(raw: object) -> QuotaResult:
    """Pure decode: raw /rest/rate-limit/status response → QuotaResult.

    Tolerant of missing groups (returns empty lists) but raises SchemaError if
    the top-level shape isn't an object — that signals Perplexity changed the
    contract rather than just dropping an optional field.
    """
    body = as_object(raw)
    if body is None:
        raise SchemaError(f"unexpected response type from {ENDPOINT}: {type(raw).__name__}")
    fq_raw = as_object(body.get("free_queries"))
    return QuotaResult(
        free_queries=_item("free_queries", fq_raw) if fq_raw is not None else None,
        modes=_group(body.get("modes")),
        sources=_group(body.get("sources")),
    )


def _group(raw: object) -> list[QuotaItem]:
    group = as_object(raw) or {}
    items = ((name, as_object(group[name])) for name in sorted(group))
    return [_item(name, v) for name, v in items if v is not None]


def _item(name: str, raw: Mapping[str, object]) -> QuotaItem:
    detail = as_object(raw.get("remaining_detail")) or {}
    remaining = int_or_none(detail.get("remaining")) if detail.get("kind") == "exact" else None
    # Only a literal `true` counts: a truthy string or number is not a
    # promise the mode will accept a request.
    return QuotaItem(name=name, available=raw.get("available") is True, remaining=remaining)
