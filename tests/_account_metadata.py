"""Find account metadata in a fixture event, including copies nested inside
JSON-encoded strings (research `text`, FINAL `content.answer`), which a
top-level check never sees."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

ACCOUNT_PARENTS = ("_extras", "telemetry_data")


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
    elif isinstance(node, str) and node[:1] in ("{", "["):
        try:
            decoded = json.loads(node)
        except json.JSONDecodeError:
            return
        yield from account_values(decoded, keys)
