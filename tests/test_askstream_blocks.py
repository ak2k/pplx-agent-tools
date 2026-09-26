"""BlockStore (plan §2.4, §2.4.1, §2.4.2): repaint rows, tracking, the
weight rule across fields, caps, progress, and terminal and reconnect parity
(U3 oracles 3, 5 and 10; I20, I21)."""

from __future__ import annotations

import json
import random
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from pplx_agent_tools.askstream import blocks
from pplx_agent_tools.askstream.blocks import (
    BlockStore,
    CapExceeded,
    Desynced,
    FrameApplied,
    Parity,
    Synced,
    Untracked,
)
from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import AskFrame, MalformedReason, decode_frame
from pplx_agent_tools.askstream.jsonval import JsonValue, weight
from pplx_agent_tools.askstream.patch import Limits
from pplx_agent_tools.askstream.projections import AnswerPaths, answer, answer_text, sources
from pplx_agent_tools.verbs._ask_common import extract_web_results, to_source
from tests._askframes import (
    add,
    diff,
    frame,
    md,
    remove,
    replace,
    report,
    snap,
    terminal,
    web,
    workflow_text,
)
from tests._patch_strategies import DOCS, op_for

FIXTURES = Path(__file__).parent / "fixtures"
MD_KEY = ("ask_text", ("markdown_block",))
EVENT_FIXTURES = sorted(FIXTURES.rglob("*.events.jsonl"))

SMALL = Limits(
    field_weight=512,
    total_weight=768,
    ops_per_frame=8,
    work_per_frame=64,
    # A run cap independent of bytes, so a short run can reach it.
    run_work_factor=0,
    run_work_base=300,
)
# Only the frame work cap tight, so the markdown merge charge is what hits it.
MERGE_WORK = Limits(
    field_weight=1 << 20,
    total_weight=1 << 20,
    ops_per_frame=64,
    work_per_frame=24,
    run_work_factor=0,
    run_work_base=1 << 20,
)
# Weight caps loose, work caps tight: reaches the work caps SMALL rarely does.
SMALL_WORK = Limits(
    field_weight=1 << 20,
    total_weight=1 << 20,
    ops_per_frame=64,
    work_per_frame=40,
    run_work_factor=0,
    run_work_base=200,
)
VALUES = DOCS | DOCS | DOCS | st.integers(0, 300).map(lambda n: "x" * n)
# Rows that `sources` keeps, so the retained list is non-empty, grows and can
# reach the weight caps on its own.
SOURCE_ROWS = st.lists(
    st.fixed_dictionaries(
        {
            "url": st.text("uv", min_size=1, max_size=2)
            | st.integers(1, 300).map(lambda n: "u" * n),
            "name": st.text("n", max_size=2),
            "snippet": st.text("s", max_size=2) | st.integers(0, 300).map(lambda n: "s" * n),
        }
    ),
    max_size=4,
)


def store(paths: AnswerPaths = "ask_text_or_workflow", **kw: Any) -> BlockStore:
    return BlockStore(paths, Limits(), **kw)


def ok(r: FrameApplied | CapExceeded) -> FrameApplied:
    assert isinstance(r, FrameApplied), r
    return r


def feed(s: BlockStore, *frames: AskFrame) -> list[FrameApplied]:
    return [ok(s.apply_frame(f)) for f in frames]


def text_of(s: BlockStore, completed: bool = True) -> str:
    return answer_text(answer(s, "ask_text_or_workflow")[0], completed).text


def held_weight(s: BlockStore) -> int:
    """The held documents plus the retained sources list, as JSON rows."""
    docs = sum(st.weight for st in s.fields.values() if isinstance(st, Synced))
    rows: list[JsonValue] = [[x.url, x.title, x.snippet] for x in s.run_sources]
    return docs + (weight(rows) if rows else 0)


def mismatches(results: list[FrameApplied]) -> list[Drift]:
    return [d for r in results for d in r.drift if d.kind == "projection_mismatch"]


# --- grounded's repaint cases as rows (oracle 5) --------------------------------------

REPAINT_ROWS: list[tuple[str, list[AskFrame], str]] = [
    (
        "terminal repaint drops a stale tail",
        [
            frame(md(["A "], 0)),
            frame(md(["B "], 1)),
            frame(md(["C"], 2), text_completed=True),
            terminal(md(["A ", "B"], 0)),
        ],
        "A B",
    ),
    (
        "chunk past the end keeps its place",
        [
            frame(md(["D"], 3)),
            frame(md(["A "], 0)),
            frame(md(["B "], 1)),
            frame(md(["C "], 2)),
            terminal(),
        ],
        "A B C D",
    ),
    (
        "midstream patch does not truncate",
        [frame(md(["A"], 0)), frame(md(["B"], 1)), frame(md(["C"], 2)), frame(md(["a"], 0))],
        "aBC",
    ),
    (
        "terminal repaint without offset is not appended",
        [frame(md(["A "])), frame(md(["B"])), terminal(md(["A ", "B"]))],
        "A B",
    ),
    (
        "empty terminal repaint keeps the answer",
        [frame(md(["A"], 0)), terminal(md([], 0))],
        "A",
    ),
    (
        "chunks without offsets append",
        [frame(md(["x"])), frame(md(["y"])), frame(md(["z"]))],
        "xyz",
    ),
]


@pytest.mark.parametrize(
    ("frames", "expected"), [r[1:] for r in REPAINT_ROWS], ids=[r[0] for r in REPAINT_ROWS]
)
def test_repaint_row(frames: list[AskFrame], expected: str) -> None:
    s = store()
    feed(s, *frames)
    assert text_of(s, completed=False) == expected


@given(st.lists(st.text(min_size=1, max_size=3), min_size=1, max_size=12), st.randoms())
def test_offset_runs_in_any_order_rebuild_then_repaint(
    parts: list[str], rnd: random.Random
) -> None:
    runs = [(i, parts[i : i + 2]) for i in range(0, len(parts), 2)]
    rnd.shuffle(runs)
    s = store()
    feed(s, *(frame(md(run, i)) for i, run in runs))
    assert text_of(s, completed=False) == "".join(parts).strip()
    head = parts[: len(parts) // 2 or 1]
    feed(s, terminal(md(head, 0)))
    assert text_of(s, completed=False) == "".join(head).strip()


def test_reconnect_snapshot_repaints_from_zero() -> None:
    s = store()
    feed(s, frame(md(["A"])), frame(md(["B"])))
    s.begin_reconnect()
    feed(s, frame(md(["A", "B", "C"])))
    assert text_of(s, completed=False) == "ABC"


# --- tracking (oracle 10) -------------------------------------------------------------


def test_untracked_field_keeps_no_document() -> None:
    s = store()
    chrome = snap("sources_answer_mode", "sources_mode_block", {"rows": [1, 2]})
    feed(s, frame(chrome, diff("answer_tabs", "answer_tabs_block", replace("", {"t": 1}))))
    states = s.fields
    assert isinstance(states[("sources_answer_mode", ("sources_mode_block",))], Untracked)
    assert isinstance(states[("answer_tabs", ("answer_tabs_block",))], Untracked)
    assert s.budget.total_weight == 0


def test_track_all_keeps_chrome_documents_under_the_same_accounting() -> None:
    s = store(track_all=True)
    feed(s, frame(snap("sources_answer_mode", "sources_mode_block", {"rows": [1, 2]})))
    st_ = s.state(("sources_answer_mode", ("sources_mode_block",)))
    assert st_ == Synced({"rows": [1, 2]}, weight({"rows": [1, 2]}))
    assert s.budget.total_weight == held_weight(s)


def test_unknown_block_field_is_drift_once_and_untracked() -> None:
    s = store()
    f = frame(snap("canvas_mode", "gizmo_block", {"a": 1}))
    r1, r2 = feed(s, f, f)
    assert r1.drift == (Drift("unknown_block_field", name_of("gizmo_block")),)
    assert r2.drift == ()
    assert isinstance(s.state(("canvas_mode", ("gizmo_block",))), Untracked)


@pytest.mark.parametrize("path", EVENT_FIXTURES, ids=[p.name for p in EVENT_FIXTURES])
def test_track_all_on_every_fixture_stays_under_default_caps(path: Path) -> None:
    s = store(track_all=True)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and set(record) == {"event", "data"}:
            record = record["data"]
        f, _ = decode_frame(json.dumps(record))
        if isinstance(f, AskFrame):
            ok(s.apply_frame(f))
            assert s.budget.total_weight == held_weight(s)
    assert all(n <= blocks.SEEN_CAP for n in s.seen_sizes())


# --- diffs, desync and resync ---------------------------------------------------------


def test_diff_on_a_missing_field_starts_from_an_empty_object() -> None:
    s = store()
    feed(s, frame(diff("ask_text", "markdown_block", add("/chunks", ["a"]))))
    assert s.get("ask_text") == {"chunks": ["a"]}
    assert s.budget.total_weight == weight({"chunks": ["a"]})


def test_rejected_patch_desyncs_drops_later_ops_and_a_snapshot_resyncs() -> None:
    s = store()
    key = ("ask_text", ("markdown_block",))
    feed(s, frame(diff("ask_text", "markdown_block", replace("", {"chunks": ["a"]}))))
    (r,) = feed(s, frame(diff("ask_text", "markdown_block", remove("/nope"))))
    assert r.drift == (Drift("patch_rejected", name_of("ask_text/markdown_block:missing_target")),)
    assert s.state(key) == Desynced("missing_target", 0)
    assert s.budget.total_weight == 0
    feed(s, frame(diff("ask_text", "markdown_block", add("/chunks/1", "b"), add("/x", 1))))
    assert s.state(key) == Desynced("missing_target", 2)
    assert s.get("ask_text") is None
    feed(s, frame(md(["a", "b"], 0)))
    assert text_of(s, completed=False) == "ab"
    assert s.budget.total_weight == held_weight(s)


def undecodable(block: dict[str, Any]) -> AskFrame:
    f, drift = decode_frame(json.dumps({"status": "PENDING", "blocks": [block]}))
    assert isinstance(f, AskFrame)
    assert [d.kind for d in drift] == ["malformed_block"]
    return f


def md_diff(field: str, patches: JsonValue) -> dict[str, Any]:
    return {"intended_usage": "ask_text", "diff_block": {"field": field, "patches": patches}}


MALFORMED_ROWS: list[tuple[MalformedReason, dict[str, Any]]] = [
    ("bad_op", diff("ask_text", "markdown_block", add("/chunks/1", " world"), add("x", 1))),
    ("pointer_too_deep", md_diff("markdown_block" + ".a" * 127, [add("/b/c", 1)])),
    ("patches_not_list", md_diff("markdown_block", {})),
    ("field_too_deep", md_diff("markdown_block" + ".a" * 200, [])),
    ("snapshot_not_object", {"intended_usage": "ask_text", "markdown_block": ["x"]}),
]


@pytest.mark.parametrize(("reason", "block"), MALFORMED_ROWS, ids=[r[0] for r in MALFORMED_ROWS])
def test_malformed_block_desyncs_its_field_like_a_rejected_patch(
    reason: MalformedReason, block: dict[str, Any]
) -> None:
    """The server applied a change here that pplx could not read, so a later
    diff must not land on the document without it."""
    s = store()
    feed(s, frame(web("https://a.example/")), frame(md(["Hello"])))
    (r,) = feed(s, undecodable(block))
    assert r.drift == ()
    assert s.state(MD_KEY) == Desynced(reason, 0)
    assert s.budget.total_weight == held_weight(s) > 0
    feed(s, frame(diff("ask_text", "markdown_block", add("/chunks/-", "!"))))
    assert s.state(MD_KEY) == Desynced(reason, 1)
    assert text_of(s, completed=False) == ""
    feed(s, frame(md(["Hello", " world", "!"], 0)))
    assert text_of(s, completed=False) == "Hello world!"


# --- caps, checked before the work (§2.4.1) -------------------------------------------


def test_ops_per_frame_cap_applies_nothing_and_ends_the_store() -> None:
    s = BlockStore("ask_text_or_workflow", SMALL)
    ops = [add(f"/k{i}", i) for i in range(SMALL.ops_per_frame + 1)]
    f = frame(diff("ask_text", "markdown_block", *ops))
    r = s.apply_frame(f)
    assert r == CapExceeded("ops_per_frame", SMALL.ops_per_frame, SMALL.ops_per_frame + 1, None)
    assert (s.fields, s.budget.total_weight, s.budget.run_work) == ({}, 0, 0)
    assert s.apply_frame(frame(md(["a"]))) is r


def test_field_weight_cap_drops_the_field_and_keeps_the_count_exact() -> None:
    s = BlockStore("ask_text_or_workflow", SMALL)
    feed(s, frame(web("https://a.example/")))
    big = "x" * SMALL.field_weight
    r = s.apply_frame(frame(md([big])))
    assert isinstance(r, CapExceeded)
    assert (r.cap, r.field) == ("field_weight", ("ask_text", ("markdown_block",)))
    assert s.get("ask_text") is None
    assert s.budget.total_weight == held_weight(s) > 0


def test_retained_sources_are_charged_after_their_document_empties() -> None:
    s = BlockStore("ask_text_or_workflow", SMALL)
    feed(s, frame(web("u" * 200)), frame(web()))
    assert [x.url for x in s.run_sources] == ["u" * 200]
    assert s.budget.total_weight == held_weight(s) > weight({"web_results": []}) + 200


def test_retained_sources_pass_the_total_weight_cap_before_they_are_kept() -> None:
    """The document fits field_weight; it and the list it yields do not fit
    total_weight together."""
    s = BlockStore("ask_text_or_workflow", SMALL)
    r = s.apply_frame(frame(web("u" * 380)))
    assert isinstance(r, CapExceeded)
    assert (r.cap, r.field) == ("total_weight", ("web_results", ("web_result_block",)))
    assert r.observed > SMALL.total_weight
    assert s.run_sources == ()
    assert s.budget.total_weight == held_weight(s) <= SMALL.total_weight


def test_markdown_merge_work_is_charged_before_the_merge() -> None:
    """Each appended chunk re-places every held one, so the charge grows with
    the list; it is checked before the merge and the copy run."""
    lim = Limits(work_per_frame=256)
    s = BlockStore("ask_text_or_workflow", lim)
    for i in range(lim.work_per_frame):
        r = s.apply_frame(frame(md([f"c{i}"])))
        if isinstance(r, CapExceeded):
            break
    else:
        pytest.fail("no cap hit")
    assert (r.cap, r.field) == ("work_per_frame", ("ask_text", ("markdown_block",)))
    assert s.budget.frame_work <= lim.work_per_frame
    assert s.get("ask_text") is None
    assert s.budget.total_weight == held_weight(s) == 0


LOOSE_WEIGHT = Limits(field_weight=2**70, total_weight=2**70)


@pytest.mark.parametrize(
    ("held", "offset", "lim", "cap"),
    [
        ([], 2**63, Limits(), "field_weight"),
        (["HELLO"], sys.maxsize, LOOSE_WEIGHT, "work_per_frame"),
    ],
    ids=["past_index_range", "maxsize_under_loose_weight_caps"],
)
def test_absurd_chunk_offset_is_a_cap_not_an_exception(
    held: list[str], offset: int, lim: Limits, cap: str
) -> None:
    """No list of that length could be built; the cap it would pass is
    reported before one is tried."""
    s = BlockStore("ask_text_or_workflow", lim)
    feed(s, frame(md(held)))
    r = s.apply_frame(frame(md(["TAIL"], offset)))
    assert isinstance(r, CapExceeded)
    assert (r.cap, r.field) == (cap, MD_KEY)
    assert s.state(MD_KEY) is None
    assert s.budget.total_weight == 0


GAP = 1_000_000


@pytest.mark.parametrize(
    ("lim", "cap"),
    [
        (Limits(field_weight=1 << 30, total_weight=1 << 30, work_per_frame=1000), "work_per_frame"),
        (Limits(field_weight=GAP), "field_weight"),
        (Limits(field_weight=1 << 30, total_weight=GAP), "total_weight"),
    ],
    ids=["work", "field_weight", "total_weight"],
)
def test_chunk_gap_passes_its_cap_before_it_is_allocated(lim: Limits, cap: str) -> None:
    """A padded list holds a pointer per slot, so a peak below one byte per
    slot means the padding was never built."""
    s = BlockStore("ask_text_or_workflow", lim)
    feed(s, frame(md(["HELLO"])))
    f = frame(md(["TAIL"], 1 + GAP))
    tracemalloc.start()
    try:
        r = s.apply_frame(f)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert isinstance(r, CapExceeded)
    assert (r.cap, r.field) == (cap, MD_KEY)
    assert peak < GAP


FIELDS = [
    ("ask_text", "markdown_block"),
    ("web_results", "web_result_block"),
    ("unified_assets", "unified_assets_block"),
]


class CountingProbe:
    def __init__(self) -> None:
        self.steps = 0

    def visit(self) -> None:
        self.steps += 1

    def read(self, text: JsonValue) -> None:
        pass

    def shift(self, n: int) -> None:
        self.steps += n

    def op_done(self, index: int, charge: int, doc: JsonValue, weight: int) -> None:
        pass


@st.composite
def frames_for(draw: st.DrawFn, s: BlockStore) -> AskFrame:
    """A frame of snapshots and diffs over several fields, the diffs aimed
    at each field's current document."""
    out: list[dict[str, Any]] = []
    for usage, field in draw(st.lists(st.sampled_from(FIELDS), min_size=1, max_size=3)):
        if draw(st.booleans()):
            doc = draw(st.dictionaries(st.text("ab", max_size=2), VALUES, max_size=3))
            if field == "markdown_block" and draw(st.booleans()):
                # A chunk run merges over the held chunks: the merge has its
                # own work charge and cap path.
                doc["chunks"] = draw(st.lists(st.text("xy", max_size=3), max_size=8))
                if draw(st.booleans()):
                    doc["chunk_starting_offset"] = draw(st.integers(0, 10))
            if field == "web_result_block" and draw(st.booleans()):
                # An empty list keeps the earlier sources retained and charged.
                doc["web_results"] = draw(SOURCE_ROWS)
            out.append(snap(usage, field, doc))
        else:
            state = s.state((usage, (field,)))
            doc = state.doc if isinstance(state, Synced) else {}
            ops = [draw(op_for(doc, VALUES)) for _ in range(draw(st.integers(1, 4)))]
            out.append(diff(usage, field, *ops))
    return frame(*out, status=draw(st.sampled_from(["PENDING", "COMPLETED"])))


@pytest.mark.parametrize(
    "lim", [SMALL, SMALL_WORK, MERGE_WORK], ids=["weight_caps", "work_caps", "merge_work_cap"]
)
@settings(max_examples=150, deadline=None)
@given(data=st.data())
def test_weight_rule_and_caps_hold_across_fields_after_every_result(
    lim: Limits, data: st.DataObject
) -> None:
    """I20 and I21 through `apply_frame`: after any result the budget counts
    exactly the held documents and the retained sources list, no cap is passed, and the real steps never
    exceed the work charged."""
    probe = CountingProbe()
    s = BlockStore("ask_text_or_workflow", lim, track_all=True, probe=probe)
    for _ in range(data.draw(st.integers(1, 30))):
        if data.draw(st.integers(0, 9)) == 0:
            s.begin_reconnect()
        r = s.apply_frame(data.draw(frames_for(s)))
        b = s.budget
        assert b.total_weight == held_weight(s)
        for st_ in s.fields.values():
            if isinstance(st_, Synced):
                assert st_.weight == weight(st_.doc) <= lim.field_weight
        assert b.total_weight <= lim.total_weight
        if s.run_sources:
            event("sources retained")
        assert b.frame_work <= lim.work_per_frame
        assert b.run_work <= b.run_work_limit
        assert probe.steps <= b.run_work
        event(type(r).__name__ if isinstance(r, CapExceeded) else "applied")
        if isinstance(r, CapExceeded):
            event(r.cap)
            assert s.apply_frame(frame()) is r
            break


# --- progress (§2.4) --------------------------------------------------------------------


def test_legacy_text_counts_only_new_values() -> None:
    s = store()
    changes = [r.change for r in feed(s, *(frame(text=t) for t in "ABAB"))]
    assert changes == ["progress", "progress", "idle", "idle"]


def test_content_snapshot_repeat_and_noop_diff_are_idle() -> None:
    s = store()
    f = frame(web("https://a.example/"))
    noop = frame(
        diff("web_results", "web_result_block", replace("/web_results/0/url", "https://a.example/"))
    )
    assert [r.change for r in feed(s, f, f, noop)] == ["progress", "idle", "idle"]


def test_chrome_never_counts_and_report_counts_only_on_body_growth() -> None:
    s = store()
    chrome = frame(snap("answer_tabs", "answer_tabs_block", {"n": 1}))
    body = diff("unified_assets", "unified_assets_block", replace("", report("abc")))
    same = diff("unified_assets", "unified_assets_block", add("/assets/0/uuid", "u"))
    shorter = diff("unified_assets", "unified_assets_block", replace("", report("ab")))
    longer = diff("unified_assets", "unified_assets_block", replace("", report("abcd")))
    got = [
        r.change for r in feed(s, chrome, frame(body), frame(same), frame(shorter), frame(longer))
    ]
    assert got == ["idle", "progress", "idle", "idle", "progress"]


def test_seen_sets_are_bounded_fifo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blocks, "SEEN_CAP", 3)
    s = store()
    changes = [r.change for r in feed(s, *(frame(text=t) for t in "ABCDA"))]
    assert changes == ["progress"] * 5
    assert max(s.seen_sizes()) == 3


# --- terminal and reconnect parity (oracle 3 on hand-built diff streams) ---------------

A, B, C = "https://a.example/", "https://b.example/", "https://c.example/"


def ask_stream(streamed: list[str]) -> list[AskFrame]:
    rows = [{"url": u, "name": u} for u in (A, B)]
    return [
        frame(
            diff(
                "ask_text", "markdown_block", replace("", {"chunks": [], "progress": "IN_PROGRESS"})
            ),
            diff("web_results", "web_result_block", replace("", {"web_results": rows})),
        ),
        *(
            frame(diff("ask_text", "markdown_block", add(f"/chunks/{i}", c)))
            for i, c in enumerate(streamed)
        ),
    ]


def ask_terminal(chunks: list[str], final: str, urls: tuple[str, ...] = (A, B)) -> AskFrame:
    return terminal(md(chunks, 0, answer=final, progress="DONE"), web(*urls, progress="DONE"))


def test_terminal_parity_equal_on_a_consistent_stream() -> None:
    s = store()
    chunks = ["Hi [1]", " there [2]."]
    results = feed(s, *ask_stream(chunks), ask_terminal(chunks, "Hi [1] there [2]."))
    assert results[-1].parity == Parity("terminal", "equal", "equal", "equal")
    assert mismatches(results) == []


def test_terminal_parity_accepts_renumbered_citations_and_returns_the_final() -> None:
    s = store()
    chunks = ["Hi [1]", " there [9]."]
    results = feed(s, *ask_stream(chunks), ask_terminal(chunks, "Hi [2] there [10]."))
    assert results[-1].parity == Parity("terminal", "citations_renumbered", "equal", "equal")
    assert mismatches(results) == []
    assert text_of(s) == "Hi [2] there [10]."


@pytest.mark.parametrize(
    ("final", "urls", "last_chunk", "name"),
    [
        ("Hi [1] there [2]!", (A, B), " there [2].", "answer"),
        ("Hi [1] there [2].", (B, A), " there [2].", "sources"),
        # The terminal chunks differ from the streamed ones while its answer
        # still equals the streamed join.
        ("Hi [1] there [2].", (A, B), " there [2]!", "answer"),
    ],
    ids=["final", "sources", "terminal_chunks"],
)
def test_one_mutation_gives_exactly_one_mismatch(
    final: str, urls: tuple[str, ...], last_chunk: str, name: str
) -> None:
    s = store()
    chunks = ["Hi [1]", " there [2]."]
    results = feed(s, *ask_stream(chunks), ask_terminal([chunks[0], last_chunk], final, urls))
    assert mismatches(results) == [Drift("projection_mismatch", name_of(name))]


def test_empty_web_results_never_erase_earlier_sources() -> None:
    """The latest non-empty list wins, whether the empty one comes from a
    snapshot or a diff, mid-run or in the terminal frame."""
    chunks = ["Hi"]
    s = store()
    results = feed(s, *ask_stream(chunks), ask_terminal(chunks, "Hi", ()))
    assert results[-1].parity == Parity("terminal", "equal", "equal", "equal")
    assert mismatches(results) == []
    assert [x.url for x in s.run_sources] == [A, B]

    mid = store()
    results = feed(
        mid,
        *ask_stream(chunks),
        frame(diff("web_results", "web_result_block", replace("/web_results", []))),
        ask_terminal(chunks, "Hi", ()),
    )
    assert mismatches(results) == []
    assert [x.url for x in mid.run_sources] == [A, B]

    legacy = store()
    feed(legacy, frame(web(A, B)), frame(web()), terminal(md(["x"], 0), web()))
    assert [x.url for x in legacy.run_sources] == [A, B]


def test_a_non_empty_snapshot_list_replaces_the_sources() -> None:
    s = store()
    feed(s, frame(web(A, B)), frame(web(C)))
    assert [x.url for x in s.run_sources] == [C]


def test_an_empty_block_after_a_non_empty_one_in_the_same_frame_keeps_it() -> None:
    s = store()
    feed(s, frame(web(A), web()))
    assert [x.url for x in s.run_sources] == [A]
    assert s.budget.total_weight == held_weight(s)


# Each kind a `web_results` block can take: an empty list, no list, a list
# with a usable URL, and a list with none.
WEB_BLOCKS = st.one_of(
    st.just(web()),
    st.just(snap("web_results", "web_result_block", {"progress": "DONE"})),
    st.lists(st.sampled_from([A, B, C]), min_size=1, max_size=3).map(lambda us: web(*us)),
    st.lists(
        st.sampled_from([{"name": "n"}, {"url": ""}, {"url": 7, "snippet": "s"}, "row"]),
        min_size=1,
        max_size=3,
    ).map(lambda rows: snap("web_results", "web_result_block", {"web_results": rows})),
)


def ask_py_sources(held: list[Any], block: dict[str, Any]) -> list[Any]:
    """The sources `verbs.ask` keeps after an event holding only `block`."""
    raw = extract_web_results({"data": {"blocks": [block]}})
    if not raw:
        return held
    out: list[Any] = []
    for src in filter(None, map(to_source, raw)):
        if src.url not in {o.url for o in out}:
            out.append(src)
    return out


@settings(max_examples=200, deadline=None)
@given(st.lists(st.lists(WEB_BLOCKS, min_size=1, max_size=4), min_size=1, max_size=4))
def test_retained_sources_match_the_ask_verb_after_every_block(
    frames: list[list[dict[str, Any]]],
) -> None:
    s = store()
    expected: list[Any] = []
    for blocks_ in frames:
        ok(s.apply_frame(frame(*blocks_)))
        for b in blocks_:
            expected = ask_py_sources(expected, b)
        got = [(x.url, x.title, x.snippet) for x in s.run_sources]
        assert got == [(x.url, x.title, x.snippet) for x in expected]
        assert s.budget.total_weight == held_weight(s)


def test_retained_sources_pass_the_field_weight_cap_before_they_are_kept() -> None:
    """URL-only rows weigh more as (url, title, snippet) rows than as the
    document, so the list alone can pass field_weight."""
    rows: list[JsonValue] = [{"url": f"u{i}"} for i in range(20)]
    block = snap("web_results", "web_result_block", {"web_results": rows})
    doc_w = weight({"web_results": rows})
    rows_w = weight([[f"u{i}", None, None] for i in range(20)])
    assert doc_w < rows_w
    s = BlockStore("ask_text_or_workflow", Limits(field_weight=doc_w, total_weight=1 << 20))
    r = s.apply_frame(frame(block))
    assert isinstance(r, CapExceeded)
    assert (r.cap, r.limit, r.observed) == ("field_weight", doc_w, rows_w)
    assert r.field == ("web_results", ("web_result_block",))
    assert s.run_sources == ()
    assert s.budget.total_weight == held_weight(s)


def test_workflow_final_may_not_renumber_citations() -> None:
    """Only `ask_text`'s final may differ from the join in citation digits."""
    s = store()
    results = feed(
        s,
        frame(
            diff("workflow_root", "workflow_block", replace("", workflow_text((["a [1]"], None))))
        ),
        terminal(snap("workflow_root", "workflow_block", workflow_text((["a [1]"], "a [2]")))),
    )
    assert results[-1].parity == Parity("terminal", "mismatch", "equal", "equal")
    assert mismatches(results) == [Drift("projection_mismatch", name_of("answer"))]


def test_reconnect_that_switches_answer_source_is_a_mismatch() -> None:
    """A snapshot whose `ask_text` extends the streamed workflow text is
    still a different source, not a snapshot ahead of the stream."""
    s = store()
    feed(
        s,
        frame(diff("workflow_root", "workflow_block", replace("", workflow_text((["ab"], None))))),
    )
    s.begin_reconnect()
    (r,) = feed(s, frame(md(["abc"], 0)))
    assert r.parity is not None and r.parity.answer == "mismatch"
    assert mismatches([r]) == [Drift("projection_mismatch", name_of("answer"))]


def research_stream(body: str) -> list[AskFrame]:
    return [
        frame(diff("ask_text", "markdown_block", replace("", {"chunks": ["Cover"]}))),
        frame(diff("unified_assets", "unified_assets_block", replace("", report(body)))),
    ]


def test_report_body_mutation_gives_exactly_one_mismatch() -> None:
    s = store("ask_text_only")
    results = feed(
        s,
        *research_stream("Body"),
        terminal(md(["Cover"], 0), snap("unified_assets", "unified_assets_block", report("Bodx"))),
    )
    assert mismatches(results) == [Drift("projection_mismatch", name_of("report_body"))]


def test_workflow_final_must_equal_the_streamed_join() -> None:
    s = store()
    wf = workflow_text((["An", "swer"], None))
    done = workflow_text((["An", "swer"], "Answer"))
    bad = workflow_text((["An", "swer"], "Answe2"))
    good = feed(
        s,
        frame(diff("workflow_root", "workflow_block", replace("", wf))),
        terminal(snap("workflow_root", "workflow_block", done)),
    )
    assert good[-1].parity == Parity("terminal", "equal", "equal", "equal")
    s2 = store()
    worse = feed(
        s2,
        frame(diff("workflow_root", "workflow_block", replace("", wf))),
        terminal(snap("workflow_root", "workflow_block", bad)),
    )
    assert mismatches(worse) == [Drift("projection_mismatch", name_of("answer"))]


def test_reconnect_snapshot_ahead_with_reordered_shorter_sources_is_no_mismatch() -> None:
    s = store()
    feed(s, *ask_stream(["Hi"]), frame(web(A, B, C)))
    s.begin_reconnect()
    (r,) = feed(s, frame(md(["Hi", " more"], 0), web(C, A)))
    assert r.parity == Parity("reconnect", "at_or_ahead", "not_compared", "at_or_ahead")
    assert mismatches([r]) == []
    assert [x.url for x in sources(s)] == [C, A]


def test_reconnect_snapshot_behind_is_a_mismatch() -> None:
    s = store()
    feed(s, *ask_stream(["Hi", " there"]))
    s.begin_reconnect()
    (r,) = feed(s, frame(md(["Hi"], 0)))
    assert r.parity is not None and r.parity.answer == "mismatch"
    assert mismatches([r]) == [Drift("projection_mismatch", name_of("answer"))]


def test_reconnect_behind_on_report_body_is_a_mismatch() -> None:
    s = store("ask_text_only")
    feed(s, *research_stream("Body grows"))
    s.begin_reconnect()
    (r,) = feed(s, frame(snap("unified_assets", "unified_assets_block", report("Body"))))
    assert mismatches([r]) == [Drift("projection_mismatch", name_of("report_body"))]


def test_legacy_snapshot_stream_has_no_parity_check() -> None:
    """Without diffs every snapshot replaces the field, so there is no
    accumulated state to compare; legacy asks change sources at the end."""
    s = store()
    results = feed(s, frame(web(A)), frame(md(["x"], 0)), terminal(md(["x"], 0), web(B, A)))
    assert results[-1].parity is None
    assert mismatches(results) == []


def test_mismatch_is_reported_once_per_projection() -> None:
    s = store()
    chunks = ["Hi"]
    feed(s, *ask_stream(chunks))
    first = feed(s, ask_terminal(chunks, "Hi", (B, A)))
    again = feed(s, ask_terminal(chunks, "Hi", (A, B)))
    assert mismatches(first) == [Drift("projection_mismatch", name_of("sources"))]
    assert mismatches(again) == []


# --- terminal-only drift -------------------------------------------------------------------


def test_ambiguity_and_missing_paths_are_reported_at_the_terminal_frame_only() -> None:
    s = store()
    both = (
        md(["A"], 0),
        snap("workflow_root", "workflow_block", workflow_text((["W"], None))),
        snap("web_results", "web_result_block", {"progress": "DONE"}),
    )
    (mid,) = feed(s, frame(*both))
    assert mid.drift == ()
    (end,) = feed(s, terminal(*both))
    assert set(end.drift) == {
        Drift("projection_ambiguous", name_of("answer:both_paths")),
        Drift("projection_missing", name_of("web_results/web_results")),
    }


def test_research_ignores_workflow_text_at_the_terminal() -> None:
    s = store("ask_text_only", expect_text=True)
    both = (md(["A"], 0), snap("workflow_root", "workflow_block", workflow_text((["W"], None))))
    (end,) = feed(s, terminal(*both, text="[]"))
    assert end.drift == ()


def test_research_terminal_without_text_is_projection_missing() -> None:
    s = store("ask_text_only", expect_text=True)
    (end,) = feed(s, terminal(md(["A"], 0)))
    assert end.drift == (Drift("projection_missing", name_of("text")),)
