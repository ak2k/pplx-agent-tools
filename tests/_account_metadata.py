"""Leak checks for sanitized fixtures: find account metadata and identity values
in an event, including copies nested inside JSON-encoded strings (research
`text`, FINAL `content.answer`, any other value), which a top-level check never
sees; plus the planted values the sanitizer tests feed in."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

ACCOUNT_PARENTS = ("_extras", "telemetry_data")
IDENTITY_KEYS = frozenset(
    {
        "backend_uuid",
        "context_uuid",
        "uuid",
        "frontend_uuid",
        "frontend_context_uuid",
        "cursor",
        "read_write_token",
        "thread_url_slug",
        "backend_uuid_slug",
    }
)
_IDENTITY_PREFIXES = ("author_", "user_")
_NOT_IDENTITY = frozenset({"user_selected_model"})


def _decoded(node: str) -> Any:
    # json.loads accepts leading whitespace, so the shape test must too.
    if node.lstrip()[:1] not in ("{", "["):
        return None
    try:
        return json.loads(node)
    except json.JSONDecodeError:
        return None


def account_values(
    node: Any, keys: Iterable[str], parent: str | None = None
) -> Iterator[tuple[str, str, Any]]:
    """Yield (parent, key, value) for every account key under an account parent."""
    keys = tuple(keys)
    if isinstance(node, dict):
        for key, value in node.items():
            if parent in ACCOUNT_PARENTS and key in keys:
                yield parent, key, value
            yield from account_values(value, keys, key)
    elif isinstance(node, list):
        for item in node:
            yield from account_values(item, keys)
    elif isinstance(node, str):
        decoded = _decoded(node)
        if decoded is not None:
            yield from account_values(decoded, keys)


def is_identity_key(key: str) -> bool:
    return key in IDENTITY_KEYS or (key.startswith(_IDENTITY_PREFIXES) and key not in _NOT_IDENTITY)


def identity_values(node: Any) -> Iterator[tuple[str, Any]]:
    """Yield (key, value) for every identity-keyed value at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if is_identity_key(key):
                yield key, value
            else:
                yield from identity_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from identity_values(item)
    elif isinstance(node, str):
        decoded = _decoded(node)
        if decoded is not None:
            yield from identity_values(decoded)


_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
# Search-result thumbnails are content-addressed public CDN paths, not account ids.
_THUMBNAIL_RE = re.compile(rf"cloudfront\.net/thumbnails/{_UUID_RE.pattern}/{_UUID_RE.pattern}")


def stray_uuids(text: str, allowed: Iterable[str]) -> set[str]:
    """Every uuid in a fixture's raw text that is neither allowed nor a thumbnail."""
    return set(_UUID_RE.findall(_THUMBNAIL_RE.sub("", text))) - set(allowed)


# Values a real capture could carry, planted by the sanitizer tests; distinctive
# so their absence from the scrubbed output can be checked as a substring.
PLANTED_ACCOUNT = {
    "subscription_tier": "planted-tier",
    "payment_tier": "planted-payment",
    "country": "planted-country",
}
PLANTED_DOC = json.dumps(
    {
        "_extras": PLANTED_ACCOUNT,
        "backend_uuid": "11111111-2222-4333-8444-555555555555",
        "user_id": "planted-user-42",
    }
)
PLANTED_VALUES = (
    *PLANTED_ACCOUNT.values(),
    "11111111-2222-4333-8444-555555555555",
    "planted-user-42",
)


def sliced(text: str) -> list[str]:
    """`text` as a `chunks` list whose every slice is undecodable on its own."""
    return [text[:20], text[20:50], text[50:]]
