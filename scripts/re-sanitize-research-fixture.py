#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.12"
# ///
"""Sanitize a raw deep-research SSE capture into a checked-in fixture.

Input: a `.events.jsonl` from re-fixtures/research/ as written by
scripts/re-capture-research.py (one bare SSE `data` payload per line). Output:
a sanitized, bounded JSONL safe to commit under tests/fixtures/research/.

Research frames are full-snapshot repaints: a single capture is ~800 frames and
~70 MB, so the fixture is trimmed as well as scrubbed.

Trimming (what is KEPT):
  - frame 0 (carries the initial shape), plus two mid-stream frames at a third
    and two thirds of the pre-completion span (search + report-streaming phases)
  - EVERY frame from the first one carrying `text_completed` through the end of
    the stream — the frames the completion bug lives in, so the replay can drive
    both the buggy and the fixed exit points.
  - nested `web_results` lists are capped (`MAX_WEB_RESULTS`): they are ~60% of a
    snapshot's bytes and identical in shape, so a cap bounds the fixture without
    changing the code paths under test.

Scrubbing — deterministic (reruns are diff-free), applied recursively to the
payload AND inside `data.text` (a JSON *string* holding the research block list)
AND inside the FINAL block's `content.answer` (a JSON string again):
  - account/thread-bound ids (see SENTINELS): backend_uuid, read_write_token,
    context/frontend uuids, per-block `uuid`, cursor, slugs, author_*/user_*
  - the RESEARCH_ANSWER report URL: a *signed* CloudFront/S3 link (Policy +
    Signature query params), i.e. a time-limited credential
  - any email-shaped string anywhere

Preserved verbatim: `status`, `text_completed`, `step_type`s, the report body
(`assets[].research_report.source_content`), FINAL `answer`/`chunks` — the
fields the verb decodes and the tests assert on.

Usage:
  uv run --python 3.12 scripts/re-sanitize-research-fixture.py \\
    re-fixtures/research/weather-nowcasting-apis.events.jsonl \\
    tests/fixtures/research/weather-nowcasting-apis.events.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Fixed sentinel values — tests asserting on these hard-code them; drift between
# this dict and the test constants is caught by a test in
# tests/test_fixture_replay_research.py.
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
    "backend_uuid_slug": "00000000-0000-4000-8000-000000000001",
}
SENTINEL_REPORT_URL = "https://fixture.invalid/research-report.md"
SENTINEL_EMAIL = "user@example.invalid"

# `user_selected_model` is a model id (part of the wire shape under test), not an
# identity field — exempt it from the user_*/author_* prefix rule.
PREFIX_EXEMPT = {"user_selected_model"}
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# The generated report is served from a pre-signed CDN link; the same link is
# repeated under several asset keys (`url`, `download_info[].url`,
# `fallback_info.webview_url`), so match the credential itself, not the key name.
# `/web/direct-files/<account hash>/...` is the unsigned twin of the same asset.
_SIGNED_URL_RE = re.compile(r"https?://\S*(?:[?&]Policy=|/web/direct-files/)\S*")

MAX_WEB_RESULTS = 10


def _scrub_string(value: str) -> str:
    return _SIGNED_URL_RE.sub(SENTINEL_REPORT_URL, _EMAIL_RE.sub(SENTINEL_EMAIL, value))


def _is_identity_key(key: str) -> bool:
    return key not in PREFIX_EXEMPT and key.startswith(("author_", "user_"))


def _scrub(node: Any) -> Any:
    """Recursively replace identity fields, cap web_results, scrub emails."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in SENTINELS and value is not None:
                out[key] = SENTINELS[key]
            elif _is_identity_key(key) and isinstance(value, str):
                out[key] = SENTINELS.get(key, "REDACTED")
            elif key == "web_results" and isinstance(value, list):
                out[key] = [_scrub(v) for v in value[:MAX_WEB_RESULTS]]
            elif key == "research_report" and isinstance(value, dict):
                rr = _scrub(value)
                if isinstance(rr, dict) and rr.get("url") is not None:
                    rr["url"] = SENTINEL_REPORT_URL
                out[key] = rr
            else:
                out[key] = _scrub(value)
        return out
    if isinstance(node, list):
        return [_scrub(v) for v in node]
    if isinstance(node, str):
        return _scrub_string(node)
    return node


def _scrub_research_text(text: str) -> str:
    """`data.text` is a JSON string of research blocks; scrub inside it, plus the
    FINAL block's `content.answer`, which is a JSON string one level deeper."""
    try:
        blocks = json.loads(text)
    except json.JSONDecodeError:
        return _scrub_string(text)
    blocks = _scrub(blocks)
    if isinstance(blocks, list):
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            content = blk.get("content")
            if not isinstance(content, dict):
                continue
            if blk.get("step_type") == "RESEARCH_ANSWER" and content.get("url") is not None:
                content["url"] = SENTINEL_REPORT_URL
            if blk.get("step_type") == "FINAL" and isinstance(content.get("answer"), str):
                try:
                    inner = json.loads(content["answer"])
                except json.JSONDecodeError:
                    continue
                content["answer"] = json.dumps(_scrub(inner), separators=(",", ":"))
    return json.dumps(blocks, separators=(",", ":"))


def _scrub_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return _scrub(payload)
    out = _scrub(payload)
    if isinstance(payload.get("text"), str):
        out["text"] = _scrub_research_text(payload["text"])
    return out


def _marks_text_completed(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload.get("text_completed"))


def _select(payloads: list[Any]) -> list[int]:
    tail = next((i for i, p in enumerate(payloads) if _marks_text_completed(p)), None)
    if tail is None:
        raise SystemExit("no frame carries text_completed — capture looks truncated")
    mids = sorted({max(1, tail // 3), max(2, (tail * 2) // 3)})
    return sorted({0, *mids, *range(tail, len(payloads))})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="raw .events.jsonl from re-capture-research.py")
    ap.add_argument("output", type=Path, help="sanitized .events.jsonl path")
    args = ap.parse_args(argv)

    raw = [line for line in args.input.read_text().splitlines() if line.strip()]
    if not raw:
        print(f"error: empty input: {args.input}", file=sys.stderr)
        return 1
    payloads = [json.loads(line) for line in raw]

    keep = _select(payloads)
    first = payloads[keep[0]]
    if not (
        isinstance(first, dict) and first.get("backend_uuid") and first.get("read_write_token")
    ):
        # The replay asserts thread cleanup, which needs both ids on a kept
        # frame; the verb only captures them from the FIRST frame that has them.
        keep = sorted(
            set(keep)
            | {
                next(
                    i
                    for i, p in enumerate(payloads)
                    if isinstance(p, dict) and p.get("backend_uuid") and p.get("read_write_token")
                )
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for i in keep:
            f.write(json.dumps(_scrub_payload(payloads[i]), separators=(",", ":")))
            f.write("\n")
    size = args.output.stat().st_size
    print(
        f"wrote {args.output} ({len(keep)} of {len(payloads)} frames, "
        f"{size / 1024:.0f} KiB; kept indices {keep})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
