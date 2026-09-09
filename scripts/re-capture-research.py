#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.12"
# ///
"""Capture a raw `pplx research` SSE stream to re-fixtures/research/.

Drives /rest/sse/perplexity_ask the way `verbs/research.py` does (same body via
`_build_research_body`, same `client.sse_post` loop) but WITHOUT the early-exit
completion predicate, so every frame the server sends is recorded — including
the ones the verb stops before. That is the point: this script exists to show
what the verb is missing.

One JSON line per SSE event = that event's bare `data` payload (the shape the
`tests/fixtures/**/*.events.jsonl` replay clients expect).

Output is scratch and may carry account-bound identifiers: re-fixtures/ is
gitignored — never `git add` it. Sanitize with
scripts/re-sanitize-research-fixture.py before committing anything under
tests/fixtures/research/.

Research is session-creating: the body is incognito so the thread never enters
history, and we still DELETE it by backend_uuid afterwards as the secondary
guard (CLAUDE.md -> "Endpoint selection principle").

Usage:
  uv run --python 3.12 scripts/re-capture-research.py "your query" --timeout 600
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pplx_agent_tools.verbs.research import (  # noqa: E402
    _COUNCIL_MODEL,
    _DEFAULT_COUNCIL_MODELS,
    ENDPOINT,
    _build_research_body,
    _model_for_mode,
)
from pplx_agent_tools.wire import Client  # noqa: E402

OUT_DIR = REPO / "re-fixtures" / "research"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query")
    ap.add_argument("--mode", default="research")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default=None, help="output basename (default: derived from query)")
    args = ap.parse_args(argv)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    slug = re.sub(r"[^a-z0-9]+", "-", args.query.lower())[:40].strip("-")
    label = args.label or f"{slug}-{ts}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{label}.events.jsonl"

    client = Client.from_default_cookies(profile=None)
    model = _model_for_mode(args.mode)
    # Model Council stalls forever without compare_model_preferences; the verb
    # defaults the trio in, so a capture that omits it never returns a frame.
    council = list(_DEFAULT_COUNCIL_MODELS) if model == _COUNCIL_MODEL else None
    body = _build_research_body(args.query, model, council_models=council)

    backend_uuid: str | None = None
    read_write_token: str | None = None
    count = 0
    try:
        with out_path.open("w") as f:
            for event in client.sse_post(ENDPOINT, body, max_total_seconds=args.timeout):
                data: Any = event.get("data")
                f.write(json.dumps(data, separators=(",", ":")))
                f.write("\n")
                f.flush()
                count += 1
                if isinstance(data, dict):
                    if backend_uuid is None and isinstance(data.get("backend_uuid"), str):
                        backend_uuid = data["backend_uuid"]
                    if read_write_token is None and isinstance(data.get("read_write_token"), str):
                        read_write_token = data["read_write_token"]
                    if data.get("status") in ("COMPLETED", "FAILED"):
                        print(f"  frame {count}: status={data.get('status')}", file=sys.stderr)
                if count % 10 == 0:
                    print(f"  {count} frames...", file=sys.stderr)
    finally:
        # A mid-stream failure still created the thread, and the partial capture
        # on disk still holds a live read_write_token at the default umask.
        if out_path.exists():
            out_path.chmod(0o600)
        print(f"wrote {out_path} ({count} events)", file=sys.stderr)
        if backend_uuid and read_write_token:
            ok = client.delete_thread(backend_uuid, read_write_token)
            print(f"thread cleanup: {'deleted' if ok else 'FAILED'}", file=sys.stderr)
        else:
            print("warning: no backend_uuid/read_write_token seen; no cleanup", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
