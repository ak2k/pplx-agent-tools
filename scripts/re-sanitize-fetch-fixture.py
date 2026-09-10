#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.10"
# ///
"""Sanitize a raw chat-fetch-prompt SSE capture into a checked-in fixture.

Takes a `.events.jsonl` from re-fixtures/fetch-url/ (one event payload per
line, as captured by scripts/re-capture-fetch-prompt.py) and emits a sanitized
JSONL safe to commit under tests/fixtures/fetch-url/.

Scrubbing is recursive: identifiers nest inside `blocks[*]`, inside `plan`,
and inside the `text` field, which is itself a JSON document serialized into
a string. Anything reachable is reachable by a leak, so the walk descends
into all of it.

What gets replaced (deterministically, so reruns are diff-free):
  - account-bound UUIDs (backend_uuid, context_uuid, frontend_uuid,
    frontend_context_uuid, uuid, cursor) at any depth
  - read_write_token (session-bound thread token) at any depth
  - any `author_*` / `user_*` key at any depth, except the entries in
    `_PRESERVED_PREFIXED_KEYS` that name a setting rather than a person.
    The key decides, not the value's JSON type: a nested object, a list or a
    bare number under such a key is replaced wholesale, since an identity
    hides just as well in `{"author_profile": {"account_id": ...}}`
  - thread_url_slug (often the backend_uuid again)
  - any email-shaped substring in any string value, including one that only
    exists across a `chunks` slice boundary (see `_scrub_chunks`)

What is preserved verbatim:
  - blocks[*] (incl. chunk_starting_offset / progress; markdown_block chunks
    keep their original slicing unless something in the joined text matched)
  - status, text_completed, final_sse_message
  - thread_title (it's the user's prompt — fixture's whole point)
  - empty strings: the server emits `"uuid":""` for un-started plan steps, and
    replacing that would both churn the fixture and hide the real shape

Rerunning it over its own output is a no-op, so re-sanitizing a committed
fixture is a safe way to check nothing new leaked in.

Usage:
  uv run scripts/re-sanitize-fetch-fixture.py \\
    re-fixtures/fetch-url/multi-block-prompt.events.jsonl \\
    tests/fixtures/fetch-url/multi-block-prompt.events.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, cast

# Fixed sentinel values — any test asserting on these can hard-code them.
SENTINELS = {
    "backend_uuid": "00000000-0000-4000-8000-000000000001",
    "context_uuid": "00000000-0000-4000-8000-000000000002",
    "uuid": "00000000-0000-4000-8000-000000000003",
    "frontend_uuid": "00000000-0000-4000-8000-000000000004",
    "frontend_context_uuid": "00000000-0000-4000-8000-000000000005",
    "cursor": "00000000-0000-4000-8000-000000000006",
    "read_write_token": "TEST_RW_TOKEN",
    "author_id": "00000000-0000-4000-8000-00000000000a",
    "author_username": "test_user",
    "thread_url_slug": "00000000-0000-4000-8000-000000000001",
}


# Applied to any key matching `_REDACTED_KEY_PREFIXES` that has no entry of
# its own in SENTINELS — a name we have not seen before is assumed identifying.
SENTINEL_REDACTED = "REDACTED"
# Applied to email-shaped substrings wherever they appear, including in answer
# prose scraped off the fetched page.
SENTINEL_EMAIL = "redacted@example.invalid"

# Prefixes whose keys are person-bound by default.
_REDACTED_KEY_PREFIXES = ("author_", "user_")
# Exceptions to the prefix rule: these name a request setting, not a person,
# and redacting them would destroy behavior the fixture exists to pin.
_PRESERVED_PREFIXED_KEYS = frozenset({"user_selected_model"})
# Keys whose string value is itself a JSON document.
_EMBEDDED_JSON_KEYS = frozenset({"text"})

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _is_identity_key(key: str | None) -> bool:
    if key is None or key in _PRESERVED_PREFIXED_KEYS:
        return False
    return key in SENTINELS or key.startswith(_REDACTED_KEY_PREFIXES)


def _redact_identity(key: str, value: Any) -> Any:
    """Replace an identity-keyed value outright, whatever its JSON type.

    Empty containers, empty strings and null carry shape but no secret (the
    server emits `"uuid":""` for un-started plan steps), so they ride out
    untouched; substituting them would churn the fixture and hide the shape.
    """
    if value is None or (isinstance(value, (str, list, dict)) and not value):
        return value
    return SENTINELS.get(key, SENTINEL_REDACTED)


def _scrub_str(key: str | None, value: str) -> str:
    # Only non-empty strings are rewritten: `""` carries shape information
    # (an un-started plan step) and no secret.
    if not value:
        return value
    if key is not None and key in _EMBEDDED_JSON_KEYS:
        return _scrub_embedded_json(value)
    return _EMAIL_RE.sub(SENTINEL_EMAIL, value)


def _scrub_embedded_json(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return _EMAIL_RE.sub(SENTINEL_EMAIL, value)
    if not isinstance(parsed, (dict, list)):
        return _EMAIL_RE.sub(SENTINEL_EMAIL, value)
    scrubbed = _scrub_node(parsed)
    if scrubbed == parsed:
        # Re-serializing an already-clean payload would rewrite the server's
        # own spacing and escaping, making every rerun a diff.
        return value
    return json.dumps(scrubbed, separators=(",", ":"))


def _scrub_chunks(chunks: list[str]) -> list[str]:
    """A `chunks` list → scrubbed, scrubbing the JOINED text.

    Perplexity ships the answer twice: whole, and sliced into ~23-char `chunks`.
    Per-string scrubbing therefore redacts an email in the whole string and
    keeps it verbatim in `chunks` whenever it straddles a slice boundary.
    Scrubbing the join closes that. A clean capture keeps its original slicing
    (nothing matched), so re-running the sanitizer over an already-clean input
    is byte-for-byte stable.
    """
    joined = "".join(chunks)
    scrubbed = _EMAIL_RE.sub(SENTINEL_EMAIL, joined)
    if scrubbed == joined:
        return list(chunks)
    return [scrubbed]


def _scrub_node(node: Any, key: str | None = None) -> Any:
    # The key is consulted BEFORE the type: dispatching on type first let an
    # identity ride out under any non-string value.
    if key is not None and _is_identity_key(key):
        return _redact_identity(key, node)
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "chunks" and isinstance(v, list) and all(isinstance(i, str) for i in v):
                out[k] = _scrub_chunks(v)
            else:
                out[k] = _scrub_node(v, k)
        return out
    if isinstance(node, list):
        # The parent key rides along so `{"author_ids": [...]}` scrubs its items.
        return [_scrub_node(v, key) for v in node]
    if isinstance(node, str):
        return _scrub_str(key, node)
    # Numbers, bools and null carry no identifier we can recognize.
    return node


def _scrub(payload: dict[str, Any]) -> dict[str, Any]:
    # Routed through `_scrub_node` so the top level gets the same key dispatch
    # as every nested dict rather than a second copy of it.
    return cast(dict[str, Any], _scrub_node(payload))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path, help="raw .events.jsonl")
    p.add_argument("output", type=Path, help="sanitized .events.jsonl path")
    args = p.parse_args(argv)

    raw = args.input.read_text().splitlines()
    if not raw:
        print(f"error: empty input: {args.input}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w") as f:
        for lineno, line in enumerate(raw, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as e:
                print(f"error: invalid JSON on line {lineno} of {args.input}: {e}", file=sys.stderr)
                return 1
            f.write(json.dumps(_scrub(event), separators=(",", ":")))
            f.write("\n")
            written += 1
    print(f"wrote {args.output} ({written} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
