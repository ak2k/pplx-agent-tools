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

Scrubbing is per-event, but the answer is streamed as one chunk per event, so
an address split across two events is invisible to every individual scrub and
reassembles when a client replays the fixture. `residual_chunk_emails` is the
gate for that: once every event is scrubbed, each block's chunks are joined in
stream order across all of them — exactly as a replaying client accumulates
them, with no reconstruction from `chunk_starting_offset` — and any match other
than the sentinel refuses. A `chunks` list holding anything but strings is a
finding too, not a skip: the join cannot see through it, so the one check that
covers cross-event addresses would silently not apply. The sanitized events are
staged under a unique name in the destination's directory and moved into place
only after the gate passes, so a refusal (or a malformed input) leaves an
existing fixture untouched instead of destroying it, and two runs writing
different fixtures in the same directory cannot land on each other's staging
file.

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
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple, cast

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


class ChunkLeak(NamedTuple):
    """One email-shaped match in a block's cross-event chunk join."""

    block: str
    first_event: int
    last_event: int
    match: str

    def describe(self) -> str:
        return (
            f"block {self.block!r} chunks join to {self.match!r} "
            f"across events {self.first_event}-{self.last_event}"
        )


class ChunkShape(NamedTuple):
    """A `chunks` list carrying something other than strings.

    Reported rather than skipped: the cross-event join is the only check that
    sees an address split across events, and a list it cannot join is a place
    that check does not reach. Whether the odd element is itself a secret is
    beside the point — an unreviewed shape refuses.
    """

    block: str
    event: int
    kind: str

    def describe(self) -> str:
        return (
            f"block {self.block!r} has a {self.kind} in its chunks at event "
            f"{self.event}; the cross-event email join cannot cover it"
        )


def _block_key(index: int, block: dict[str, Any]) -> str:
    usage = block.get("intended_usage")
    return usage if isinstance(usage, str) and usage else f"blocks[{index}]"


def _chunk_stream(
    events: list[dict[str, Any]],
) -> tuple[dict[str, list[tuple[str, int]]], list[ChunkShape]]:
    """Events → (block key → its chunks with the event index, in stream order),
    plus every `chunks` list this walk could not read as text.

    The streams are exactly what a replaying client appends
    (`extract_chunks_from_event` over the events in order) — nothing is
    reconstructed from `chunk_starting_offset`, because interpreting the offsets
    lets the COMPLETED repaint mask a delta that no longer sits at the slot it
    repaints.
    """
    streams: dict[str, list[tuple[str, int]]] = {}
    shapes: list[ChunkShape] = []
    for event_index, event in enumerate(events):
        blocks = event.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            markdown = block.get("markdown_block")
            if not isinstance(markdown, dict):
                continue
            chunks = markdown.get("chunks")
            if not isinstance(chunks, list):
                continue
            key = _block_key(block_index, block)
            odd = next((c for c in chunks if not isinstance(c, str)), None)
            if odd is not None:
                shapes.append(ChunkShape(key, event_index, type(odd).__name__))
                continue
            stream = streams.setdefault(key, [])
            stream.extend((chunk, event_index) for chunk in chunks)
    return streams, shapes


def residual_chunk_emails(events: list[dict[str, Any]]) -> list[ChunkLeak | ChunkShape]:
    """Already-scrubbed events → everything that keeps this fixture from being
    written: emails surviving a cross-event chunk join, and chunk lists the join
    could not read.

    The sentinel is itself email-shaped, so a match equal to it is what a
    successful scrub looks like, not a finding. Every other match refuses,
    including one formed only at the seam between the last delta and the
    COMPLETED repaint: refusing is the safe direction, and a seam hit is for a
    human to look at rather than for this check to reason away.
    """
    streams, shapes = _chunk_stream(events)
    leaks: list[ChunkLeak | ChunkShape] = list(shapes)
    for key, ordered in streams.items():
        joined = "".join(text for text, _ in ordered)
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        for text, event_index in ordered:
            spans.append((cursor, cursor + len(text), event_index))
            cursor += len(text)
        for m in _EMAIL_RE.finditer(joined):
            if m.group(0) == SENTINEL_EMAIL:
                continue
            touched = [e for start, end, e in spans if start < m.end() and end > m.start()]
            leaks.append(ChunkLeak(key, min(touched), max(touched), m.group(0)))
    return leaks


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
    # Staged in the destination's own directory (so the move is atomic) under a
    # name no other run can pick, and moved into place only once the leak check
    # passes: the usual invocation re-sanitizes a committed fixture over itself,
    # where writing first would let a refusal destroy the good file it was meant
    # to protect, and a fixed name lets two concurrent runs write each other's
    # staging file.
    fd, staged_name = tempfile.mkstemp(
        dir=args.output.parent, prefix=args.output.name + ".", suffix=".tmp"
    )
    staged = Path(staged_name)
    scrubbed: list[dict[str, Any]] = []
    try:
        with os.fdopen(fd, "w") as f:
            # mkstemp creates 0600; the fixture it replaces is a committed,
            # world-readable file, so the mode the umask would have given it is
            # restored before the move.
            umask = os.umask(0)
            os.umask(umask)
            os.fchmod(f.fileno(), 0o666 & ~umask)
            for lineno, line in enumerate(raw, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as e:
                    print(
                        f"error: invalid JSON on line {lineno} of {args.input}: {e}",
                        file=sys.stderr,
                    )
                    return 1
                event = _scrub(event)
                scrubbed.append(event)
                f.write(json.dumps(event, separators=(",", ":")))
                f.write("\n")

        leaks = residual_chunk_emails(scrubbed)
        if leaks:
            for leak in leaks:
                print(f"error: {args.output} withheld: {leak.describe()}", file=sys.stderr)
            return 1
        staged.replace(args.output)
    finally:
        staged.unlink(missing_ok=True)
    print(f"wrote {args.output} ({len(scrubbed)} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
