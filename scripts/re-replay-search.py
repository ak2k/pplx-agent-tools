#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.12"
# dependencies = ["curl_cffi>=0.7"]
# ///
"""Replay a /rest/sse/perplexity_ask request via curl_cffi to capture the SSE
response (which Playwright's HAR can't record).

Uses the request-body shape captured by re-capture.py but with fresh UUIDs,
streams the SSE response, and dumps each event as a JSONL line for inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pplx_agent_tools.wire import Client  # noqa: E402

ENDPOINT = "/rest/sse/perplexity_ask"


def build_body(query: str, *, send_text: bool = True) -> dict:
    """Minimal-ish body, modelled on the captured frontend request.

    Fields stripped: UI-specific (supported_block_use_cases, client_coordinates,
    mentions, etc.) — agents don't render widgets. Kept: the ones plausibly
    needed for the search to fire and return hits.
    """
    frontend_uuid = str(uuid4())
    return {
        "query_str": query,
        "params": {
            "attachments": [],
            "language": "en-US",
            "timezone": "America/New_York",
            "search_focus": "internet",
            "sources": ["web"],
            "frontend_uuid": frontend_uuid,
            "mode": "copilot",
            "model_preference": "turbo",
            "is_related_query": False,
            "is_sponsored": False,
            "frontend_context_uuid": str(uuid4()),
            "prompt_source": "user",
            "query_source": "home",
            "is_incognito": False,
            "local_search_enabled": False,
            "use_schematized_api": True,
            "send_back_text_in_streaming_api": send_text,
            "client_coordinates": None,
            "mentions": [],
            "dsl_query": query,
            "skip_search_enabled": True,
            "is_nav_suggestions_disabled": False,
            "source": "default",
            "always_search_override": False,
            "override_no_search": False,
            "client_search_results_cache_key": frontend_uuid,
            "extended_context": False,
            "version": "2.18",
        },
    }


def parse_sse_event(buffer: str) -> tuple[dict | None, str]:
    """Pull one SSE event off the buffer if complete. Return (event, remainder).

    SSE protocol uses \\r\\n line endings (RFC). We accept either, normalizing
    on read. Events end at a double-newline boundary.
    """
    # Normalize CRLF → LF so the rest of this parser only has to handle LF.
    buffer = buffer.replace("\r\n", "\n")
    sep = "\n\n"
    if sep not in buffer:
        return None, buffer
    raw, rest = buffer.split(sep, 1)
    event_type: str | None = None
    data_lines: list[str] = []
    for line in raw.split("\n"):
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        return {"event": event_type, "data": None, "raw": raw}, rest
    data_str = "\n".join(data_lines)
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        data = data_str
    return {"event": event_type, "data": data}, rest


def main(query: str, send_text: bool, out_dir: Path) -> int:
    client = Client.from_default_cookies()
    body = build_body(query, send_text=send_text)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    label = f"{query.replace(' ', '_')[:30]}-text_{send_text}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"{label}.events.jsonl"
    raw_path = out_dir / f"{label}.raw.sse"

    print(f"→ POST {ENDPOINT}", file=sys.stderr)
    print(f"  query={query!r} send_text={send_text}", file=sys.stderr)

    resp = client._session.post(
        client._base_url + ENDPOINT,
        cookies=client._cookies,
        json=body,
        headers={"accept": "text/event-stream"},
        stream=True,
        timeout=60,
    )
    print(f"  status={resp.status_code}", file=sys.stderr)
    if resp.status_code != 200:
        try:
            print(resp.text[:500], file=sys.stderr)
        except Exception:
            pass
        return 1

    buffer = ""
    n_events = 0
    event_types: dict[str, int] = {}
    with (
        raw_path.open("w", encoding="utf-8") as raw_fh,
        events_path.open("w", encoding="utf-8") as ev_fh,
    ):
        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            text = chunk.decode("utf-8", errors="replace")
            raw_fh.write(text)
            buffer += text
            while True:
                event, buffer = parse_sse_event(buffer)
                if event is None:
                    break
                n_events += 1
                et = event.get("event") or "(no event type)"
                event_types[et] = event_types.get(et, 0) + 1
                ev_fh.write(json.dumps(event) + "\n")

    print(f"\n→ captured {n_events} events", file=sys.stderr)
    print(f"  raw:    {raw_path}", file=sys.stderr)
    print(f"  events: {events_path}", file=sys.stderr)
    print("\nevent type counts:", file=sys.stderr)
    for et, n in sorted(event_types.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {et}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query")
    ap.add_argument(
        "--no-text", action="store_true", help="set send_back_text_in_streaming_api=false"
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=REPO / "re-fixtures/search-web",
    )
    args = ap.parse_args()
    sys.exit(main(args.query, send_text=not args.no_text, out_dir=args.outdir))
