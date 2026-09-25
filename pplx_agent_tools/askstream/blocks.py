"""The accumulated block documents of one run, and what each frame changed.

`BlockStore` has one owner. It applies each frame's snapshots and patches
under one `Budget`: after any `apply_frame`, `budget.total_weight` equals the
weight of the documents the store holds plus the retained sources list
(`run_sources`), and no cap is passed, even for a
moment, because every charge is checked before the work it pays for.
A cap hit ends the store: it returns the same `CapExceeded` from then on.

Fields a projection reads, and `content` and `report_asset` fields, are
tracked. Other fields keep no document, only a byte count.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, final

from pplx_agent_tools.askstream import patch
from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import (
    AskFrame,
    BlockDiff,
    BlockMalformed,
    BlockSnapshot,
    FieldKey,
)
from pplx_agent_tools.askstream.jsonval import JsonValue, measure, weight
from pplx_agent_tools.askstream.patch import Budget, CapName, Limits, Probe, node_units
from pplx_agent_tools.askstream.projections import (
    READ_FIELDS,
    READS,
    Answer,
    AnswerPaths,
    AskTextPath,
    ReadKey,
    WebSource,
    answer,
    citations_renumbered,
    latest_sources,
    projection_missing,
    report_body,
)

# Whether the frame changed content the progress rule counts.
Change = Literal["idle", "progress"]
FieldClass = Literal["content", "report_asset", "chrome", "unknown"]
Verdict = Literal["equal", "citations_renumbered", "at_or_ahead", "mismatch", "not_compared"]

FIELD_CLASS: Mapping[str, FieldClass] = {
    "markdown_block": "content",
    "plan_block": "content",
    "workflow_block": "content",
    "web_result_block": "content",
    "reasoning_plan_block": "content",
    "unified_assets_block": "report_asset",
    "sources_mode_block": "chrome",
    "answer_tabs_block": "chrome",
    "assets_mode_block": "chrome",
    "canvas_block": "chrome",
    "in_context_suggestions_block": "chrome",
    "pending_followups_block": "chrome",
    "answer_modes_block": "chrome",
    "inline_entity_block": "chrome",
}

# About 4 MB of hashes per field; an evicted value counts as new again only
# after this many newer distinct values.
SEEN_CAP = 65_536

# The weight of `{}`, the document a diff starts from on a field it has no
# document for.
EMPTY_OBJECT_WEIGHT = 2


def classify_field(field: str) -> FieldClass:
    return FIELD_CLASS.get(field, "unknown")


@final
@dataclass(frozen=True, slots=True)
class Synced:
    doc: JsonValue
    weight: int


@final
@dataclass(frozen=True, slots=True)
class Desynced:
    reason: patch.RejectReason
    dropped_ops: int


@final
@dataclass(frozen=True, slots=True)
class Untracked:
    bytes_in: int


FieldState: TypeAlias = Synced | Desynced | Untracked


@final
@dataclass(frozen=True, slots=True)
class Parity:
    """How the projections after a terminal or reconnect frame compare with
    those before it. `sources` is compared only at the terminal frame."""

    kind: Literal["terminal", "reconnect"]
    answer: Verdict
    sources: Verdict
    report_body: Verdict


@final
@dataclass(frozen=True, slots=True)
class FrameApplied:
    change: Change
    drift: tuple[Drift, ...]
    parity: Parity | None


@final
@dataclass(frozen=True, slots=True)
class CapExceeded:
    cap: CapName
    limit: int
    observed: int
    # None for a cap on the whole frame.
    field: FieldKey | None


@final
@dataclass(frozen=True, slots=True)
class _Projected:
    answer: Answer
    urls: tuple[str, ...]
    report_body: str


class _Seen:
    """A FIFO-bounded set of value hashes."""

    __slots__ = ("_d",)

    def __init__(self) -> None:
        self._d: dict[int, None] = {}

    def add(self, h: int) -> bool:
        """True when `h` was not held."""
        if h in self._d:
            return False
        if len(self._d) >= SEEN_CAP:
            del self._d[next(iter(self._d))]
        self._d[h] = None
        return True

    def __len__(self) -> int:
        return len(self._d)


def _field_name(key: FieldKey) -> str:
    return key[1][0] if key[1] else ""


def _hash(v: JsonValue) -> int:
    return hash(json.dumps(v, sort_keys=True))


class BlockStore:
    __slots__ = (
        "_answer_paths",
        "_dead",
        "_diff_mode",
        "_drift_once",
        "_expect_text",
        "_fields",
        "_probe",
        "_reconnect_pending",
        "_report_high",
        "_seen",
        "_seen_text",
        "_sources",
        "_sources_weight",
        "_track_all",
        "budget",
    )

    def __init__(
        self,
        answer_paths: AnswerPaths,
        limits: Limits,
        *,
        track_all: bool = False,
        expect_text: bool = False,
        probe: Probe | None = None,
    ) -> None:
        self._answer_paths: AnswerPaths = answer_paths
        self.budget = Budget(limits)
        self._track_all = track_all
        self._expect_text = expect_text
        self._probe = probe
        self._fields: dict[FieldKey, FieldState] = {}
        self._seen: dict[FieldKey, _Seen] = {}
        self._seen_text = _Seen()
        self._report_high = 0
        self._diff_mode = False
        self._reconnect_pending = False
        self._dead: CapExceeded | None = None
        self._drift_once: set[Drift] = set()
        # Charged like a tracked field: it outlives the `web_results` document
        # it came from, which an empty list or a desync can drop.
        self._sources: tuple[WebSource, ...] = ()
        self._sources_weight = 0

    # --- read side ------------------------------------------------------------

    def get(self, key: ReadKey) -> JsonValue | None:
        state = self._fields.get(READS[key])
        return state.doc if isinstance(state, Synced) else None

    def state(self, key: FieldKey) -> FieldState | None:
        return self._fields.get(key)

    @property
    def fields(self) -> Mapping[FieldKey, FieldState]:
        return dict(self._fields)

    @property
    def run_sources(self) -> tuple[WebSource, ...]:
        """The run's sources: the latest non-empty `web_results` list."""
        return self._sources

    def seen_sizes(self) -> tuple[int, ...]:
        return (len(self._seen_text), *(len(s) for s in self._seen.values()))

    # --- write side -----------------------------------------------------------

    def begin_reconnect(self) -> None:
        """The next frame is a reconnect snapshot: markdown repaints from
        chunk 0, and parity is "at or ahead" rather than equal."""
        self._reconnect_pending = True

    def apply_frame(self, frame: AskFrame) -> FrameApplied | CapExceeded:
        if self._dead is not None:
            return self._dead
        b = self.budget
        b.start_frame(frame.size)
        ops = sum(
            len(u.ops) if isinstance(u, BlockDiff) else 1
            for u in frame.blocks
            if not isinstance(u, BlockMalformed)
        )
        if ops > b.limits.ops_per_frame:
            return self._die(CapExceeded("ops_per_frame", b.limits.ops_per_frame, ops, None))

        reconnect, self._reconnect_pending = self._reconnect_pending, False
        terminal = frame.stage == "completed"
        self._diff_mode = self._diff_mode or any(isinstance(u, BlockDiff) for u in frame.blocks)
        # Before any diff the store holds whole snapshots, so there is no
        # accumulated state for parity to check.
        kind: Literal["terminal", "reconnect"] | None = (
            "reconnect" if reconnect else "terminal" if terminal and self._diff_mode else None
        )
        before = self._project() if kind is not None else None

        drift: list[Drift] = []
        progress = frame.text is not None and self._seen_text.add(hash(frame.text))
        report_touched = False
        for u in frame.blocks:
            if isinstance(u, BlockMalformed):
                continue
            r = self._update(u, terminal or reconnect, drift)
            if isinstance(r, CapExceeded):
                return self._die(r)
            # Per block, not per frame: a later empty block in the same frame
            # must not hide an earlier non-empty one.
            if u.key == READS["web_results"]:
                cap = self._retain_sources(latest_sources(self._sources, self))
                if cap is not None:
                    return self._die(cap)
            cls = classify_field(_field_name(u.key))
            report_touched = report_touched or cls == "report_asset"
            progress = progress or (r and cls == "content")
        if report_touched:
            n = len(report_body(self))
            if n > self._report_high:
                self._report_high = n
                progress = True

        parity = None
        if before is not None and kind is not None:
            parity = self._parity(kind, before, self._project(), drift)
        if terminal:
            _, ambiguous = answer(self, self._answer_paths)
            drift += self._once(ambiguous + projection_missing(self))
            if self._expect_text and frame.text is None:
                drift += self._once((Drift("projection_missing", name_of("text")),))
        return FrameApplied("progress" if progress else "idle", tuple(drift), parity)

    # --- internals ------------------------------------------------------------

    def _die(self, cap: CapExceeded) -> CapExceeded:
        self._dead = cap
        return cap

    def _once(self, items: tuple[Drift, ...]) -> list[Drift]:
        out = [d for d in items if d not in self._drift_once]
        self._drift_once.update(out)
        return out

    def _tracked(self, key: FieldKey) -> bool:
        if self._track_all or key in READ_FIELDS:
            return True
        return classify_field(_field_name(key)) in ("content", "report_asset")

    def _update(
        self, u: BlockSnapshot | BlockDiff, repaint: bool, drift: list[Drift]
    ) -> bool | CapExceeded:
        """Apply one block update; True when it is new content."""
        if classify_field(_field_name(u.key)) == "unknown":
            drift += self._once((Drift("unknown_block_field", name_of(_field_name(u.key))),))
        if not self._tracked(u.key):
            state = self._fields.get(u.key)
            prior = state.bytes_in if isinstance(state, Untracked) else 0
            self._fields[u.key] = Untracked(prior + _bytes_in(u))
            return False
        if isinstance(u, BlockSnapshot):
            return self._snapshot(u, repaint, drift)
        return self._diff(u, drift)

    def _spend(self, units: int, key: FieldKey) -> CapExceeded | None:
        b = self.budget
        if b.frame_work + units > b.limits.work_per_frame:
            return CapExceeded("work_per_frame", b.limits.work_per_frame, b.frame_work + units, key)
        if b.run_work + units > b.run_work_limit:
            return CapExceeded("work_per_run", b.run_work_limit, b.run_work + units, key)
        b.frame_work += units
        b.run_work += units
        return None

    def _snapshot(self, u: BlockSnapshot, repaint: bool, drift: list[Drift]) -> bool | CapExceeded:
        state = self._fields.get(u.key)
        doc, w = (state.doc, state.weight) if isinstance(state, Synced) else (None, 0)
        value: dict[str, JsonValue] = u.value
        if _field_name(u.key) == "markdown_block" and isinstance(u.value.get("chunks"), list):
            old = doc.get("chunks") if isinstance(doc, dict) else None
            old_chunks = old if isinstance(old, list) else []
            # The merge copies and re-measures every held chunk.
            cap = self._spend(sum(node_units(None, c) for c in old_chunks) + 1, u.key)
            if cap is not None:
                self._drop(u.key, w)
                return cap
            value = _merge_chunks(u.value, old_chunks, repaint)
        m = measure(value)
        if m is None:
            # Unreachable for a decoded frame, whose depth is already bounded.
            self._store(u.key, patch.Rejected(0, "too_deep", w), drift)
            return False
        op = patch.Add((), m.value, m.weight, m.nodes, m.depth)
        r = patch.apply_ops(doc, w, (op,), self.budget, self._probe)
        cap = self._store(u.key, r, drift)
        if cap is not None:
            return cap
        return isinstance(r, patch.Applied) and self._seen.setdefault(u.key, _Seen()).add(
            _hash(u.value)
        )

    def _diff(self, u: BlockDiff, drift: list[Drift]) -> bool | CapExceeded:
        state = self._fields.get(u.key)
        if isinstance(state, Desynced):
            self._fields[u.key] = Desynced(state.reason, state.dropped_ops + len(u.ops))
            return False
        if isinstance(state, Synced):
            doc, w = state.doc, state.weight
        else:
            lim = self.budget.limits
            total = self.budget.total_weight + EMPTY_OBJECT_WEIGHT
            if total > lim.total_weight:
                return CapExceeded("total_weight", lim.total_weight, total, u.key)
            doc, w = {}, EMPTY_OBJECT_WEIGHT
            self.budget.total_weight = total
        r = patch.apply_ops(doc, w, u.ops, self.budget, self._probe)
        cap = self._store(u.key, r, drift)
        if cap is not None:
            return cap
        return isinstance(r, patch.Applied) and r.changed == "changed"

    def _store(
        self,
        key: FieldKey,
        r: patch.Applied | patch.Rejected | patch.CapExceeded,
        drift: list[Drift],
    ) -> CapExceeded | None:
        """Keep `r.doc` at `r.weight`, or drop the field and its weight."""
        match r:
            case patch.Applied():
                self._fields[key] = Synced(r.doc, r.weight)
                return None
            case patch.Rejected():
                self._drop(key, r.weight)
                self._fields[key] = Desynced(r.reason, 0)
                drift.append(
                    Drift("patch_rejected", name_of(f"{key[0]}/{_field_name(key)}:{r.reason}"))
                )
                return None
            case patch.CapExceeded():
                self._drop(key, r.weight)
                return CapExceeded(r.cap, r.limit, r.observed, key)

    def _retain_sources(self, held: tuple[WebSource, ...]) -> CapExceeded | None:
        if held is self._sources:
            return None
        w = _sources_rows_weight(held)
        lim = self.budget.limits
        key = READS["web_results"]
        if w > lim.field_weight:
            return CapExceeded("field_weight", lim.field_weight, w, key)
        total = self.budget.total_weight - self._sources_weight + w
        if total > lim.total_weight:
            return CapExceeded("total_weight", lim.total_weight, total, key)
        self.budget.total_weight = total
        self._sources, self._sources_weight = held, w
        return None

    def _drop(self, key: FieldKey, weight: int) -> None:
        self._fields.pop(key, None)
        self.budget.total_weight -= weight

    def _project(self) -> _Projected:
        ans, _ = answer(self, self._answer_paths)
        return _Projected(ans, tuple(s.url for s in self._sources), report_body(self))

    def _parity(
        self,
        kind: Literal["terminal", "reconnect"],
        before: _Projected,
        after: _Projected,
        drift: list[Drift],
    ) -> Parity:
        a, b = before.answer, after.answer
        same_source = type(a.source) is type(b.source)
        if kind == "reconnect":
            # A snapshot may be ahead of what the dropped stream delivered.
            ans = _prefix(a.streamed, b.streamed) if same_source or not a.streamed else "mismatch"
            p = Parity(kind, ans, "not_compared", _prefix(before.report_body, after.report_body))
        else:
            p = Parity(
                kind,
                _terminal_answer(a, b) if same_source else "mismatch",
                "equal" if before.urls == after.urls else "mismatch",
                "equal" if before.report_body == after.report_body else "mismatch",
            )
        verdicts = (("answer", p.answer), ("sources", p.sources), ("report_body", p.report_body))
        for name, verdict in verdicts:
            if verdict == "mismatch":
                drift += self._once((Drift("projection_mismatch", name_of(name)),))
        return p


def _prefix(before: str, after: str) -> Verdict:
    return "at_or_ahead" if after.startswith(before) else "mismatch"


def _terminal_answer(before: Answer, after: Answer) -> Verdict:
    """The accumulated chunk join against the terminal frame's, and against
    its final text. Only `ask_text`'s final may renumber citations."""
    if before.streamed != after.streamed:
        return "mismatch"
    if after.final is None or after.final == before.streamed:
        return "equal"
    if isinstance(after.source, AskTextPath) and citations_renumbered(before.streamed, after.final):
        return "citations_renumbered"
    return "mismatch"


def _merge_chunks(
    value: dict[str, JsonValue], old: list[JsonValue], repaint: bool
) -> dict[str, JsonValue]:
    """A markdown snapshot placed over the held chunks, as a new document.

    The run lands at `chunk_starting_offset`, padding a gap with null so
    later diffs index the server's list. With no offset it appends, except on
    a repaint (terminal or reconnect frame), which starts at 0 and drops any
    chunk past a non-empty run."""
    run = value["chunks"]
    assert isinstance(run, list)
    raw = value.get("chunk_starting_offset")
    offset = raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None
    if offset is None:
        offset = 0 if repaint else len(old)
    chunks: list[JsonValue] = list(old)
    if len(chunks) < offset:
        chunks.extend([None] * (offset - len(chunks)))
    end = offset + len(run)
    chunks[offset:end] = run
    if repaint and run:
        del chunks[end:]
    return {**value, "chunks": chunks}


def _sources_rows_weight(srcs: tuple[WebSource, ...]) -> int:
    """The weight of a sources list as JSON rows of (url, title, snippet);
    no list retained weighs nothing."""
    return weight([[s.url, s.title, s.snippet] for s in srcs]) if srcs else 0


def _bytes_in(u: BlockSnapshot | BlockDiff) -> int:
    if isinstance(u, BlockSnapshot):
        m = measure(u.value)
        return 0 if m is None else m.weight
    return sum(op.value_weight for op in u.ops if isinstance(op, (patch.Add, patch.Replace)))
