#!/usr/bin/env -S uv run --python 3.12 python
"""Capture one live `pplx fetch --prompt` SSE stream into re-fixtures/fetch-url/.

Drives the same code path the verb does — `_build_chat_body` (incognito) posted
to /rest/sse/perplexity_ask via `Client.sse_post` — so the capture is the real
wire shape rather than a hand-written approximation. The thread is deleted by
backend_uuid afterwards; incognito already keeps it out of history, so deletion
is belt-and-braces.

The whole stream is consumed, not just the prefix the verb reads: the verb stops
at `text_completed`, but the COMPLETED event that follows carries Perplexity's
authoritative join of the delta chunks, which is what replay tests assert
against.

Writes two files per capture:
  <label>.events.jsonl  one bare `data` payload per line — the sanitizer's input
  <label>.raw.sse       the event stream re-framed as `event:`/`data:` lines,
                        for eyeballing framing and event ordering

Output lands in re-fixtures/fetch-url/, which is gitignored: the payloads still
carry the session's thread token and account identifiers. Run
scripts/re-sanitize-fetch-fixture.py over the .events.jsonl before committing
anything under tests/fixtures/.

Unlike the stdlib-only scripts here this one is not a PEP 723 `--script`: it
imports the package, so it needs the project environment rather than an
isolated one.

Usage (from the repo root):
  uv run --python 3.12 python scripts/re-capture-fetch-prompt.py \\
    <label> <url> <prompt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pplx_agent_tools.errors import PplxError, exit_code
from pplx_agent_tools.verbs.fetch import _build_chat_body
from pplx_agent_tools.wire import Client

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "re-fixtures/fetch-url"
ENDPOINT = "/rest/sse/perplexity_ask"
# Long enough for a multi-section answer on a content-rich page; a capture that
# trips this is a bad fixture anyway, not something to salvage.
CAPTURE_TIMEOUT_SECONDS = 300.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("label", help="fixture basename, e.g. paywalled-article-prompt")
    p.add_argument("url", help="URL to hand to the fetch prompt")
    p.add_argument("prompt", help="the prompt text")
    args = p.parse_args(argv)

    try:
        client = Client.from_default_cookies(profile=None)
    except PplxError as e:
        print(f"re-capture-fetch-prompt: {e}", file=sys.stderr)
        return exit_code(e)

    # Mirrors verbs/fetch._fetch_with_prompt: same composed query, same body.
    body = _build_chat_body(f"{args.prompt}\n\nFor URL: {args.url}")

    events: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    backend_uuid: str | None = None
    read_write_token: str | None = None
    try:
        for event in client.sse_post(ENDPOINT, body, max_total_seconds=CAPTURE_TIMEOUT_SECONDS):
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            events.append(data)
            name = event.get("event")
            if name:
                raw_lines.append(f"event: {name}")
            raw_lines.append(f"data: {json.dumps(data, separators=(',', ':'))}")
            raw_lines.append("")
            if backend_uuid is None and isinstance(data.get("backend_uuid"), str):
                backend_uuid = data["backend_uuid"]
            if read_write_token is None and isinstance(data.get("read_write_token"), str):
                read_write_token = data["read_write_token"]
            print(f"  event {len(events):3d}  status={data.get('status')}", file=sys.stderr)
    except PplxError as e:
        print(f"re-capture-fetch-prompt: {e}", file=sys.stderr)
        return exit_code(e)
    finally:
        if backend_uuid and read_write_token:
            client.delete_thread(backend_uuid, read_write_token)

    if not events:
        print("re-capture-fetch-prompt: stream produced no events", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUT_DIR / f"{args.label}.events.jsonl"
    jsonl.write_text(
        "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events),
    )
    raw = OUT_DIR / f"{args.label}.raw.sse"
    raw.write_text("\n".join(raw_lines) + "\n")
    print(f"wrote {jsonl} ({len(events)} events)")
    print(f"wrote {raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
