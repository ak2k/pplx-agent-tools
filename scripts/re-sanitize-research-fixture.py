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
  - the frames with the longest report body and the longest decoded answer: the
    verb flags a shortfall against those high-water marks, so a replay without
    them cannot reproduce one seen in the raw stream.
  - nested `web_results` lists are capped (`MAX_WEB_RESULTS`): they are ~60% of a
    snapshot's bytes and identical in shape, so a cap bounds the fixture without
    changing the code paths under test. FINAL's cited list is the exception: the
    answer's [n] markers index it by position, so it keeps the prefix up to the
    highest marker in the snapshot, the shortest list every citation resolves in.
  - step lists (research blocks, `plan_block.steps`) keep the first
    `MAX_STEP_BLOCKS` of each step type the verb does not return as answer text:
    a broad run repeats ~200 THOUGHT/SEARCH_* steps in every snapshot.

Scrubbing — deterministic (reruns are diff-free), applied recursively to the
payload AND inside every string that is itself a JSON document: `data.text` (the
research block list), the FINAL block's `content.answer`, a joined `chunks`
list, or any other value, whatever its key:
  - account/thread-bound ids (see SENTINELS): backend_uuid, read_write_token,
    context/frontend uuids, per-block `uuid`, cursor, slugs, author_*/user_*
  - the RESEARCH_ANSWER report URL: a *signed* CloudFront/S3 link (custom- or
    canned-policy query params), i.e. a time-limited credential
  - any email-shaped string anywhere
  - account metadata (ACCOUNT_KEYS under `_extras` / `telemetry_data`) becomes a
    fixed placeholder: no test depends on the real value, and it describes the
    capturing account, not the wire shape.

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
from collections import Counter
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
# CloudFront signs two ways and only the custom-policy form carries `Policy=`; the
# canned-policy form carries just `Signature=` + `Key-Pair-Id=`, so matching
# `Policy=` alone would let that second form ride out intact.
# `/web/direct-files/<account hash>/...` is the unsigned twin of the same asset.
# `\S` ran past the closing quote of a URL embedded in compact JSON and ate the
# rest of the document, so the run is bounded to characters a URL may contain.
# S3 SigV4 and GCS prefix the param (`X-Amz-Signature=`, `X-Goog-Signature=`),
# which a bare `[?&]Signature=` alternation misses.
_URL_CHAR = r"""[^\s"'<>\\]"""
_CREDENTIAL_PARAM = r"[?&][\w-]*(?:Policy|Signature|Key-Pair-Id|Credential|Security-Token)="
_SIGNED_URL_RE = re.compile(
    rf"https?://{_URL_CHAR}*(?:{_CREDENTIAL_PARAM}|/web/direct-files/){_URL_CHAR}*"
)

MAX_WEB_RESULTS = 10
MAX_STEP_BLOCKS = 4
# Step types whose blocks become the verb's answer text; never capped.
ANSWER_STEPS = {"FINAL", "RESEARCH_ANSWER"}
ACCOUNT_KEYS = ("subscription_tier", "payment_tier", "country")
SENTINEL_ACCOUNT_VALUE = "REDACTED"
_ACCOUNT_METADATA_PARENTS = {"_extras", "telemetry_data"}
# `[3]`, `[^3]`, `[web:3]`, grouped `[3, 5]`, ranged `[3-5]` (hyphen or en dash), and
# `【3】` / `【3†source】`, whose tail after the index is not a citation.
_CITATION_RE = re.compile(
    r"\[(?:\^|web:)?(\d+(?:\s*[,\u2013-]\s*\d+)*)\]|\u3010(\d+)[^\u3011\n]{0,40}\u3011"
)


def _scrub_string(value: str) -> str:
    return _SIGNED_URL_RE.sub(SENTINEL_REPORT_URL, _EMAIL_RE.sub(SENTINEL_EMAIL, value))


def _scrub_text(value: str) -> str:
    """A string value → scrubbed, descending into it when it is a JSON document:
    any key can carry one, and the identity and account rules are key-based."""
    if value.lstrip()[:1] in ("{", "["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            cleaned = _scrub(parsed)
            # Re-serializing a clean document would rewrite the server's own
            # spacing and escaping, making every rerun a diff.
            if cleaned != parsed:
                value = json.dumps(cleaned, separators=(",", ":"))
    # Still run over the raw text: an address in a JSON key is not a value.
    return _scrub_string(value)


def _scrub_chunks(chunks: list[str]) -> list[str]:
    """A `chunks` list → scrubbed, scrubbing the JOINED text.

    Perplexity ships the answer twice: whole, and sliced into ~23-char `chunks`.
    Per-string scrubbing therefore redacts a credential in `answer` and keeps it
    verbatim in `chunks` whenever it straddles a slice boundary. Scrubbing the
    join closes that. A clean capture keeps its original slicing (nothing
    matched), so re-running the sanitizer over an already-clean input is
    byte-for-byte stable.
    """
    joined = "".join(chunks)
    scrubbed = _scrub_text(joined)
    if scrubbed == joined:
        return list(chunks)
    return [scrubbed]


def _is_identity_key(key: str) -> bool:
    return key not in PREFIX_EXEMPT and key.startswith(("author_", "user_"))


def _redact_identity(key: str, value: Any) -> Any:
    """Replace an `author_*`/`user_*` value outright, whatever its JSON type.

    A non-empty value under an identity key IS the identity, so nothing under it
    is inspected: recursing into a dict "keeping the shape" left a profile object
    whose own keys match neither SENTINELS nor the prefix rule fully intact. A
    list is redacted element-wise, which applies this same rule to each element.

    Empty containers, empty strings and null carry shape but no secret, so they
    ride out untouched; substituting them would churn the fixture and invent
    fields the wire never sent.
    """
    if value is None or (isinstance(value, (str, list, dict)) and not value):
        return value
    if isinstance(value, list):
        return [_redact_identity(key, item) for item in value]
    return SENTINELS.get(key, "REDACTED")


def _cap_steps(steps: list[Any]) -> list[Any]:
    seen: Counter[Any] = Counter()
    kept: list[Any] = []
    for step in steps:
        kind = step.get("step_type") if isinstance(step, dict) else None
        # A malformed (e.g. list) step_type is unhashable; bucket it with the rest.
        kind = kind if isinstance(kind, str) else None
        if kind not in ANSWER_STEPS:
            seen[kind] += 1
            if seen[kind] > MAX_STEP_BLOCKS:
                continue
        kept.append(step)
    return kept


def _is_step_list(node: list[Any]) -> bool:
    return any(isinstance(v, dict) and "step_type" in v for v in node)


def _scrub(node: Any) -> Any:
    """Recursively replace identity fields, cap web_results and step lists,
    redact account metadata, scrub emails."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in SENTINELS and value is not None:
                out[key] = SENTINELS[key]
            elif _is_identity_key(key):
                out[key] = _redact_identity(key, value)
            elif (
                key == "chunks"
                and isinstance(value, list)
                and all(isinstance(v, str) for v in value)
            ):
                out[key] = _scrub_chunks(value)
            elif key == "web_results" and isinstance(value, list):
                out[key] = [_scrub(v) for v in value[:MAX_WEB_RESULTS]]
            elif key in _ACCOUNT_METADATA_PARENTS and isinstance(value, dict):
                extras = _scrub(value)
                for account_key in ACCOUNT_KEYS:
                    if extras.get(account_key) is not None:
                        extras[account_key] = SENTINEL_ACCOUNT_VALUE
                out[key] = extras
            elif key == "research_report" and isinstance(value, dict):
                rr = _scrub(value)
                if isinstance(rr, dict) and rr.get("url") is not None:
                    rr["url"] = SENTINEL_REPORT_URL
                out[key] = rr
            else:
                out[key] = _scrub(value)
        return out
    if isinstance(node, list):
        return [_scrub(v) for v in (_cap_steps(node) if _is_step_list(node) else node)]
    if isinstance(node, str):
        return _scrub_text(node)
    return node


def _scrub_research_text(text: str) -> str:
    """`data.text` is a JSON string of research blocks; scrub inside it, plus the
    FINAL block's `content.answer`, which is a JSON string one level deeper."""
    try:
        blocks = json.loads(text)
    except json.JSONDecodeError:
        return _scrub_string(text)
    # The generic walk below already decodes FINAL answers and caps their
    # web_results flat; the citation-aligned cap needs the uncapped originals.
    # FINAL steps are never dropped by `_cap_steps`, so order pairs them up.
    raw_finals = iter(
        blk["content"]["answer"]
        for blk in (blocks if isinstance(blocks, list) else [])
        if _is_final_answer(blk)
    )
    blocks = _scrub(blocks)
    if isinstance(blocks, list):
        cited = _max_citation(blocks)
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            content = blk.get("content")
            if not isinstance(content, dict):
                continue
            if blk.get("step_type") == "RESEARCH_ANSWER" and content.get("url") is not None:
                content["url"] = SENTINEL_REPORT_URL
            if _is_final_answer(blk):
                try:
                    inner = json.loads(next(raw_finals))
                except json.JSONDecodeError:
                    continue
                cleaned = _scrub(inner)
                if isinstance(inner, dict) and isinstance(inner.get("web_results"), list):
                    cap = max(cited, MAX_WEB_RESULTS)
                    cleaned["web_results"] = [_scrub(v) for v in inner["web_results"][:cap]]
                content["answer"] = _scrub_string(json.dumps(cleaned, separators=(",", ":")))
    return json.dumps(blocks, separators=(",", ":"))


def _is_final_answer(blk: Any) -> bool:
    content = blk.get("content") if isinstance(blk, dict) else None
    return (
        isinstance(blk, dict)
        and blk.get("step_type") == "FINAL"
        and isinstance(content, dict)
        and isinstance(content.get("answer"), str)
    )


def _answer_parts(blocks: list[Any]) -> tuple[list[str], list[str]]:
    """Research blocks → (cover note parts, report body parts), decoded as
    `verbs/research.py::_decode_parts` does; tests pin the two together."""
    cover: list[str] = []
    bodies: list[str] = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        content = blk.get("content")
        if blk.get("step_type") == "RESEARCH_ANSWER":
            found = []
            assets = blk.get("assets")
            for asset in assets if isinstance(assets, list) else []:
                report = asset.get("research_report") if isinstance(asset, dict) else None
                body = report.get("source_content") if isinstance(report, dict) else None
                if isinstance(body, str) and body.strip():
                    found.append(body.strip())
            inline = content.get("answer") if isinstance(content, dict) else None
            if not found and isinstance(inline, str) and inline.strip():
                found.append(inline.strip())
            bodies.extend(found)
        elif blk.get("step_type") == "FINAL" and isinstance(content, dict):
            raw = content.get("answer")
            if not isinstance(raw, str) or not raw:
                continue
            try:
                inner = json.loads(raw)
            except json.JSONDecodeError:
                inner = raw
            md = inner.get("answer") if isinstance(inner, dict) else None
            text = md if isinstance(md, str) else raw
            if text:
                cover.append(text)
    return cover, bodies


def _max_citation(blocks: list[Any]) -> int:
    cover, bodies = _answer_parts(blocks)
    indices = [
        int(n)
        for bracket, lenticular in _CITATION_RE.findall("\n".join(cover + bodies))
        for n in re.findall(r"\d+", bracket or lenticular)
    ]
    return max(indices, default=0)


def _measure(payload: Any) -> tuple[int, int]:
    """A frame → (report body chars, decoded answer chars), the two high-water
    marks the verb's shortfall check compares against; (0, 0) if undecodable."""
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        return 0, 0
    try:
        blocks = json.loads(text)
    except json.JSONDecodeError:
        return 0, 0
    if not isinstance(blocks, list):
        return 0, 0
    cover, bodies = _answer_parts(blocks)
    joined_cover = "\n\n".join(cover).strip()
    answer = "\n\n".join(cover + [b for b in bodies if b not in joined_cover]).strip()
    return len("\n\n".join(bodies).strip()), len(answer)


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
    mids = {max(1, tail // 3), max(2, (tail * 2) // 3)}
    sizes = [_measure(p) for p in payloads]
    peaks: set[int] = set()
    for axis in (0, 1):
        column = [size[axis] for size in sizes]
        if max(column):
            peaks.add(column.index(max(column)))
    # The mid-stream floors (1, 2) can exceed a very short capture's length.
    return sorted(i for i in {0, *mids, *peaks, *range(tail, len(payloads))} if i < len(payloads))


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
