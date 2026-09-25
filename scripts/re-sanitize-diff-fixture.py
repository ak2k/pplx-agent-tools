"""Sanitize raw diff-mode ask captures into checked-in fixtures.

Input: capture JSONL files, one record per line; a record with `"k": "frame"`
and a dict `data` is one SSE payload, every other record is skipped. A line
that is already a bare payload (a committed fixture) is read as one too.
Output: `<out_dir>/<input stem>.events.jsonl`, one bare payload per line.

The files of one invocation share one UUID map, so an initial stream and its
reconnect streams keep matching ids. Pass a run's files together, in order.

Diff frames are RFC 6902 patches against earlier frames, so this keeps every
list and every op the replay needs: nothing is capped or collapsed, since that
would move the indices later ops address. Scrubbing, applied to every string
at any depth, including inside strings that hold a JSON document:
  - identity keys (SENTINELS, `author_*`/`user_*`) to fixed sentinels, both as
    object keys and as the last segment of a patch op's `path`
  - every other UUID to `00000000-0000-4000-9000-<n>`, numbered in order of
    first appearance; thumbnail CDN paths are public content hashes and kept
  - signed or account-scoped report URLs to SENTINEL_REPORT_URL
  - email-shaped strings to SENTINEL_EMAIL
  - account metadata per `_fixture_account.py`; `telemetry_data` and
    `classifier_results` to `{}`

Thinning, for size: a whole-string `replace` of a report `source_content` whose
next op in the same field is a `replace` of the same path is overwritten
before anything reads it, so only every `--keep-every`th of those is kept.
A frame left with no blocks, no `text` and an unchanged status is dropped.

After writing, the files are replayed through the block store and the
projections (answer, sources, report body) must hold no email other than the
sentinel and no UUID the map did not issue: a value split across `chunks`
escapes the per-string scrub, and only the joined text shows it.

Usage (the replay needs the package):
  uv run --extra dev python scripts/re-sanitize-diff-fixture.py \\
    tests/fixtures/ask-diff CAPTURES/p3-ask-initial.jsonl CAPTURES/p3-ask-reconnect1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from _fixture_account import ACCOUNT_PARENTS, scrub_account_parent

from pplx_agent_tools.askstream import projections as P
from pplx_agent_tools.askstream.blocks import BlockStore
from pplx_agent_tools.askstream.frames import AskFrame, decode_frame
from pplx_agent_tools.askstream.patch import Limits

# The research sanitizer's values, so both fixture sets share one allow-list;
# a test pins the two dicts equal.
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
MAPPED_PREFIX = "00000000-0000-4000-9000-"
SANITIZED_UUID = re.compile(r"00000000-0000-4000-[89]000-[0-9a-f]{12}")
PREFIX_EXEMPT = frozenset({"user_selected_model"})
# Per-run telemetry and per-query classifier scores: no reader needs them, and
# the classifier block alone is about 2 KB on every frame.
TELEMETRY = frozenset({"telemetry_data", "classifier_results"})

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_THUMBNAIL_RE = re.compile(rf"cloudfront\.net/thumbnails/{UUID_RE.pattern}/{UUID_RE.pattern}", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_CHAR = r"""[^\s"'<>\\]"""
_CREDENTIAL_PARAM = r"[?&][\w-]*(?:Policy|Signature|Key-Pair-Id|Credential|Security-Token)="
SIGNED_URL_RE = re.compile(
    rf"https?://{_URL_CHAR}*(?:{_CREDENTIAL_PARAM}|/web/direct-files/){_URL_CHAR}*"
)

REPORT_BODY = "source_content"


def is_identity_key(key: str) -> bool:
    return key in SENTINELS or (key.startswith(("author_", "user_")) and key not in PREFIX_EXEMPT)


def _identity_value(key: str, value: Any) -> Any:
    # Empty values carry shape but no secret; a list is redacted per element.
    if value is None or (isinstance(value, (str, list, dict)) and not value):
        return value
    if isinstance(value, list):
        return [_identity_value(key, v) for v in value]
    return SENTINELS.get(key, "REDACTED")


def _last_segment(path: Any) -> str:
    return path.rsplit("/", 1)[-1] if isinstance(path, str) else ""


class Scrubber:
    def __init__(self) -> None:
        self.uuids: dict[str, str] = {}

    def seed(self, node: Any, key: str | None = None) -> None:
        """Map every UUID held under an identity key to that key's sentinel,
        so a copy of it elsewhere (a URL, a slug) becomes the same sentinel."""
        if isinstance(node, dict):
            if isinstance(node.get("op"), str) and "path" in node:
                self.seed(node.get("value"), _last_segment(node["path"]))
            for k, v in node.items():
                self.seed(v, k)
        elif isinstance(node, list):
            for v in node:
                self.seed(v, key)
        elif isinstance(node, str) and key in SENTINELS:
            for u in UUID_RE.findall(node):
                self.uuids.setdefault(u.lower(), SENTINELS[key])

    def _uuid(self, m: re.Match[str]) -> str:
        u = m.group(0).lower()
        if SANITIZED_UUID.fullmatch(u):
            return u
        if u not in self.uuids:
            self.uuids[u] = f"{MAPPED_PREFIX}{len(self.uuids):012x}"
        return self.uuids[u]

    def _uuids_outside_thumbnails(self, s: str) -> str:
        out: list[str] = []
        pos = 0
        for t in _THUMBNAIL_RE.finditer(s):
            out.append(UUID_RE.sub(self._uuid, s[pos : t.start()]))
            out.append(t.group(0))
            pos = t.end()
        out.append(UUID_RE.sub(self._uuid, s[pos:]))
        return "".join(out)

    def string(self, s: str) -> str:
        if s.lstrip()[:1] in ("{", "["):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                cleaned = self.node(parsed)
                # Re-serializing a clean document would rewrite its spacing.
                if cleaned != parsed:
                    s = json.dumps(cleaned, separators=(",", ":"))
        s = SIGNED_URL_RE.sub(SENTINEL_REPORT_URL, s)
        s = EMAIL_RE.sub(SENTINEL_EMAIL, s)
        return self._uuids_outside_thumbnails(s)

    def node(self, node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if is_identity_key(key):
                    out[key] = _identity_value(key, value)
                elif key in TELEMETRY:
                    out[key] = None if value is None else {}
                elif (
                    key in ACCOUNT_PARENTS
                    and (meta := scrub_account_parent(value, self.node)) is not None
                ):
                    out[key] = meta
                else:
                    out[key] = self.node(value)
            if isinstance(out.get("op"), str) and is_identity_key(_last_segment(out.get("path"))):
                out["value"] = _identity_value(_last_segment(out["path"]), node.get("value"))
            return out
        if isinstance(node, list):
            return [self.node(v) for v in node]
        if isinstance(node, str):
            return self.string(node)
        return node


def read_payloads(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        if "k" in record:
            if record["k"] == "frame" and isinstance(record.get("data"), dict):
                out.append(record["data"])
        else:
            out.append(record)
    return out


def _diffs(payload: dict[str, Any]) -> Iterator[tuple[tuple[Any, Any], dict[str, Any] | None]]:
    """(field key, diff_block or None for a snapshot) for each block."""
    blocks = payload.get("blocks")
    for b in blocks if isinstance(blocks, list) else []:
        if not isinstance(b, dict):
            continue
        db = b.get("diff_block")
        if isinstance(db, dict):
            yield (b.get("intended_usage"), db.get("field")), db
        else:
            fields = [k for k in b if k != "intended_usage"]
            yield (b.get("intended_usage"), fields[0] if fields else None), None


def _overwritten(payloads: list[dict[str, Any]]) -> set[int]:
    """ids of report-body replace ops the next op in their field overwrites."""
    last: dict[tuple[Any, Any], dict[str, Any] | None] = {}
    out: set[int] = set()
    for p in payloads:
        for key, db in _diffs(p):
            if db is None:
                last[key] = None
                continue
            ops = db.get("patches")
            for op in ops if isinstance(ops, list) else []:
                prev = last.get(key)
                if (
                    prev is not None
                    and isinstance(op, dict)
                    and op.get("op") == "replace"
                    and op.get("path") == prev.get("path")
                ):
                    out.add(id(prev))
                last[key] = op if _is_body_replace(op) else None
    return out


def _is_body_replace(op: Any) -> bool:
    return (
        isinstance(op, dict)
        and op.get("op") == "replace"
        and _last_segment(op.get("path")) == REPORT_BODY
        and isinstance(op.get("value"), str)
    )


def _stage(p: dict[str, Any]) -> tuple[Any, Any]:
    return p.get("status"), p.get("text_completed")


def thin(payloads: list[dict[str, Any]], keep_every: int) -> list[dict[str, Any]]:
    droppable = _overwritten(payloads)
    runs: dict[Any, int] = {}
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(payloads):
        p = raw
        if isinstance(raw.get("blocks"), list):
            p = {**raw, "blocks": _thin_blocks(raw["blocks"], droppable, runs, keep_every)}
        empty = not p.get("blocks") and p.get("text") is None
        last = i == len(payloads) - 1
        if out and empty and not last and _stage(p) == _stage(out[-1]):
            continue
        out.append(p)
    return out


def _thin_blocks(
    blocks: list[Any], droppable: set[int], runs: dict[Any, int], keep_every: int
) -> list[Any]:
    kept_blocks: list[Any] = []
    for b in blocks:
        db = b.get("diff_block") if isinstance(b, dict) else None
        ops = db.get("patches") if isinstance(db, dict) else None
        if not isinstance(ops, list):
            kept_blocks.append(b)
            continue
        kept_ops: list[Any] = []
        for op in ops:
            if id(op) in droppable:
                n = runs.get(op["path"], 0)
                runs[op["path"]] = n + 1
                if n % keep_every:
                    continue
            else:
                runs.pop(op.get("path") if isinstance(op, dict) else None, None)
            kept_ops.append(op)
        if kept_ops or not ops:
            kept_blocks.append({**b, "diff_block": {**db, "patches": kept_ops}})
    return kept_blocks


def sanitize(runs: Iterable[list[dict[str, Any]]], keep_every: int) -> list[list[dict[str, Any]]]:
    runs = list(runs)
    scrubber = Scrubber()
    for payloads in runs:
        for p in payloads:
            scrubber.seed(p)
    return [[scrubber.node(p) for p in thin(ps, keep_every)] for ps in runs]


def leaks(text: str) -> list[str]:
    """Emails and UUIDs in `text` that the scrub did not issue."""
    found = [e for e in EMAIL_RE.findall(text) if e != SENTINEL_EMAIL]
    kept = _THUMBNAIL_RE.sub("", text)
    found += [u for u in UUID_RE.findall(kept) if not SANITIZED_UUID.fullmatch(u.lower())]
    return found


def projected_leaks(runs: list[list[dict[str, Any]]]) -> list[str]:
    """Leaks in the answer, sources and report body after replaying `runs` as
    one stream with a reconnect between files."""
    store = BlockStore("ask_text_or_workflow", Limits(), track_all=True)
    found: list[str] = []
    for i, payloads in enumerate(runs):
        if i:
            store.begin_reconnect()
        for p in payloads:
            frame, _ = decode_frame(json.dumps(p))
            if isinstance(frame, AskFrame):
                store.apply_frame(frame)
            ans = [P.answer(store, paths)[0] for paths in ("ask_text_or_workflow", "ask_text_only")]
            texts = [a.streamed for a in ans] + [a.final or "" for a in ans]
            texts += [f"{s.url} {s.title} {s.snippet}" for s in P.sources(store)]
            texts.append(P.report_body(store))
            found += leaks("\n".join(texts))
    return sorted(set(found))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("inputs", type=Path, nargs="+", help="one run's capture files, in order")
    ap.add_argument("--keep-every", type=int, default=64)
    args = ap.parse_args(argv)

    raw = [read_payloads(p) for p in args.inputs]
    runs = sanitize(raw, args.keep_every)
    if bad := projected_leaks(runs):
        print(f"refusing: projections still hold {len(bad)} value(s) to scrub", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for src, before, payloads in zip(args.inputs, raw, runs, strict=True):
        out = (
            args.out_dir / f"{src.name.removesuffix('.jsonl').removesuffix('.events')}.events.jsonl"
        )
        text = "".join(json.dumps(p, separators=(",", ":")) + "\n" for p in payloads)
        if bad := leaks(text):
            print(f"refusing {out.name}: {len(bad)} value(s) to scrub", file=sys.stderr)
            return 1
        out.write_text(text)
        print(f"wrote {out} ({len(payloads)} of {len(before)} frames, {len(text) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
