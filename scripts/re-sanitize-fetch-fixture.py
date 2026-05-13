#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.10"
# ///
"""Sanitize a raw chat-fetch-prompt SSE capture into a checked-in fixture.

Takes a `.events.jsonl` from re-fixtures/fetch-url/ (one event payload per
line, as captured by scripts/re-capture-paste.py) and emits a sanitized
JSONL safe to commit under tests/fixtures/fetch-url/.

What gets replaced (deterministically, so reruns are diff-free):
  - account-bound UUIDs (backend_uuid, context_uuid, frontend_uuid,
    frontend_context_uuid, uuid, cursor)
  - read_write_token (session-bound thread token)
  - author_id, author_username
  - thread_url_slug (often the backend_uuid again)

What is preserved verbatim:
  - blocks[*] (incl. markdown_block chunks / chunk_starting_offset / progress)
  - status, text_completed, final_sse_message
  - thread_title (it's the user's prompt — fixture's whole point)

Usage:
  uv run scripts/re-sanitize-fetch-fixture.py \\
    re-fixtures/fetch-url/chat-fetch-prompt-2026-05-13T00-20-56Z.events.jsonl \\
    tests/fixtures/fetch-url/example-com-prompt.events.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def _scrub(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for key, sentinel in SENTINELS.items():
        if key in out and out[key] is not None:
            out[key] = sentinel
    return out


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
