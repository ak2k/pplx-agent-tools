"""Account-metadata policy shared by both fixture sanitizers and the tests' leak
walk, so the key sets and the container decode cannot drift between them.

Account metadata (ACCOUNT_KEYS under an ACCOUNT_PARENTS key) becomes a fixed
placeholder: no test depends on the real value, and it describes the capturing
account, not the wire shape.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

ACCOUNT_KEYS = ("subscription_tier", "payment_tier", "country")
ACCOUNT_PARENTS = frozenset({"_extras", "telemetry_data"})
ACCOUNT_PLACEHOLDER = "REDACTED"


def account_container(value: Any) -> dict[str, Any] | None:
    """An account parent's value as a dict, decoding one serialized into a JSON
    string; None when it is neither."""
    if isinstance(value, str) and value.lstrip()[:1] == "{":
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def scrub_account_parent(value: Any, scrub: Callable[[Any], Any]) -> Any | None:
    """An account parent's value → `scrub`bed with its account keys replaced, in
    the value's own form (dict or JSON string); None if it holds no dict."""
    meta = account_container(value)
    if meta is None:
        return None
    out = scrub(meta)
    for key in ACCOUNT_KEYS:
        if out.get(key) is not None:
            out[key] = ACCOUNT_PLACEHOLDER
    if isinstance(value, dict):
        return out
    # Re-serializing a clean container would rewrite the server's own spacing.
    return value if out == meta else json.dumps(out, separators=(",", ":"))
