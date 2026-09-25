"""The sanitized diff-mode fixtures under tests/fixtures/ask-diff (U3 oracles
3, 6 and 11), and the sanitizer that cuts them from raw captures.

Oracles 2, 7 and 10 run over these files through the corpus globs in
test_askstream_fixtures.py and test_askstream_blocks.py."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pplx_agent_tools.askstream import projections as P
from pplx_agent_tools.askstream.blocks import BlockStore, FrameApplied, Synced
from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import AskFrame, BlockSnapshot, decode_frame
from pplx_agent_tools.askstream.patch import Limits
from pplx_agent_tools.verbs.research import decode_research_text
from tests._account_metadata import (
    ACCOUNT_KEYS,
    ACCOUNT_PARENTS,
    ACCOUNT_PLACEHOLDER,
    PLANTED_DOC,
    PLANTED_VALUES,
    account_values,
    identity_values,
)

ROOT = Path(__file__).parent.parent
DIFF = Path(__file__).parent / "fixtures" / "ask-diff"
SANITIZER_SCRIPT = ROOT / "scripts" / "re-sanitize-diff-fixture.py"
RESEARCH_SANITIZER_SCRIPT = ROOT / "scripts" / "re-sanitize-research-fixture.py"

ASK: P.AnswerPaths = "ask_text_or_workflow"
RESEARCH: P.AnswerPaths = "ask_text_only"

# run name → (fixture files in stream order, answer paths)
RUNS: dict[str, tuple[tuple[str, ...], P.AnswerPaths]] = {
    "p2-turbo-1": (("p2-turbo-1",), ASK),
    "p2-claude48opusthinking-1": (("p2-claude48opusthinking-1",), ASK),
    "p2-claude48opusthinking-2": (("p2-claude48opusthinking-2",), ASK),
    "p2-claude48opusthinking-4": (("p2-claude48opusthinking-4",), ASK),
    "p2-fetch": (("p2-fetch",), ASK),
    "p1-A": (("p1-A",), RESEARCH),
    "p1-B": (("p1-B",), RESEARCH),
    "p3-research": (("p3-research-initial", "p3-research-reconnect1"), RESEARCH),
    "p3-ask": (("p3-ask-initial", "p3-ask-reconnect1"), ASK),
}
RENUMBERED = {"p2-claude48opusthinking-1", "p2-claude48opusthinking-4"}
WORKFLOW_RUNS = {"p2-turbo-1", "p2-fetch"}
RESEARCH_RUNS = [name for name, (_, paths) in RUNS.items() if paths == RESEARCH]
RECONNECT_RUNS = {name for name, (files, _) in RUNS.items() if len(files) > 1}


def _load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def san() -> ModuleType:
    return _load_script("re_sanitize_diff_fixture", SANITIZER_SCRIPT)


def _path(stem: str) -> Path:
    return DIFF / f"{stem}.events.jsonl"


def _frames(stem: str) -> Iterator[AskFrame]:
    for line in _path(stem).read_text().splitlines():
        frame, drift = decode_frame(line)
        assert drift == ()
        assert isinstance(frame, AskFrame)
        yield frame


def _applied(r: object) -> FrameApplied:
    assert isinstance(r, FrameApplied), r
    return r


def replay(
    run: str, *, track_all: bool = False, drop_text: bool = False
) -> tuple[BlockStore, list[FrameApplied]]:
    files, paths = RUNS[run]
    store = BlockStore(paths, Limits(), track_all=track_all, expect_text=paths == RESEARCH)
    results: list[FrameApplied] = []
    for i, stem in enumerate(files):
        if i:
            store.begin_reconnect()
        for f in _frames(stem):
            fed = dataclasses.replace(f, text=None) if drop_text else f
            results.append(_applied(store.apply_frame(fed)))
    return store, results


def _drift(results: list[FrameApplied]) -> Counter[Drift]:
    return Counter(d for r in results for d in r.drift)


def test_diff_fixture_set_is_what_the_runs_name() -> None:
    named = {f"{stem}.events.jsonl" for files, _ in RUNS.values() for stem in files}
    assert {p.name for p in DIFF.iterdir()} == named


# --- oracle 3: projection parity -------------------------------------------------------


@pytest.mark.parametrize("run", RUNS)
def test_terminal_parity_holds_on_every_diff_fixture(run: str) -> None:
    _, results = replay(run)
    parities = [r.parity for r in results if r.parity is not None]
    terminal = [p for p in parities if p.kind == "terminal"]
    assert len(terminal) == 1
    (t,) = terminal
    assert t.answer == ("citations_renumbered" if run in RENUMBERED else "equal")
    assert (t.sources, t.report_body) == ("equal", "equal")
    assert _drift(results) == Counter()


@pytest.mark.parametrize("run", sorted(RECONNECT_RUNS))
def test_reconnect_snapshot_is_at_or_ahead(run: str) -> None:
    _, results = replay(run)
    reconnect = [r.parity for r in results if r.parity is not None and r.parity.kind == "reconnect"]
    assert [(p.answer, p.sources, p.report_body) for p in reconnect] == [
        ("at_or_ahead", "not_compared", "at_or_ahead")
    ]


@pytest.mark.parametrize("run", RUNS)
def test_whole_block_comparison_at_the_terminal_frame_differs(run: str) -> None:
    """The terminal snapshots never equal the accumulated blocks byte for
    byte, so parity must compare projections to pass on real runs."""
    files, paths = RUNS[run]
    store = BlockStore(paths, Limits(), track_all=True)
    differing: list[str] = []
    for i, stem in enumerate(files):
        if i:
            store.begin_reconnect()
        for f in _frames(stem):
            if f.stage == "completed" and not differing:
                for u in f.blocks:
                    if not isinstance(u, BlockSnapshot):
                        continue
                    held = store.state(u.key)
                    if isinstance(held, Synced) and held.doc != u.value:
                        differing.append(u.key[0])
                assert differing, "no terminal snapshot differs from its accumulated block"
            _applied(store.apply_frame(f))
    assert differing, "the run has no terminal frame"


# --- oracle 11: answer paths -----------------------------------------------------------


@pytest.mark.parametrize("run", RUNS)
def test_answer_path_on_diff_fixtures(run: str) -> None:
    store, results = replay(run)
    ans, drift = P.answer(store, RUNS[run][1])
    expected = P.WorkflowPath() if run in WORKFLOW_RUNS else P.AskTextPath()
    assert ans.source == expected
    assert drift == ()
    assert not any(d.kind == "projection_ambiguous" for d in _drift(results))


def test_p1a_cut_after_workflow_text_before_ask_text_has_no_answer() -> None:
    frames = list(_frames("p1-A"))
    store = BlockStore(RESEARCH, Limits())
    cut = None
    for i, f in enumerate(frames):
        _applied(store.apply_frame(f))
        if P.answer(store, RESEARCH)[0] != P.NO_ANSWER:
            cut = i
            break
    assert cut is not None, "p1-A never streams ask_text"
    store = BlockStore(RESEARCH, Limits())
    for f in frames[:cut]:
        _applied(store.apply_frame(f))
    assert P.answer(store, RESEARCH) == (P.NO_ANSWER, ())
    # The cut holds workflow text items, which research must not read.
    assert P.answer(store, ASK)[0].source == P.WorkflowPath()


@pytest.mark.parametrize("run", sorted(RENUMBERED))
def test_answer_text_on_renumbered_fixtures(run: str) -> None:
    store, _ = replay(run)
    ans, _ = P.answer(store, ASK)
    assert ans.final is not None
    assert P.answer_text(ans, completed=True) == P.AnswerText(ans.final.strip(), True)
    streamed = P.answer_text(ans, completed=False)
    assert streamed == P.AnswerText(ans.streamed.strip(), False)
    assert P.citations_renumbered(streamed.text, ans.final.strip())


@pytest.mark.parametrize("run", RESEARCH_RUNS)
def test_research_without_terminal_text_builds_from_projections(run: str) -> None:
    _, results = replay(run)
    assert Drift("projection_missing", name_of("text")) not in _drift(results)
    terminal_text = next(
        f.text for stem in RUNS[run][0] for f in _frames(stem) if f.text is not None
    )
    expected, _ = decode_research_text(terminal_text)

    cut, cut_results = replay(run, drop_text=True)
    assert _drift(cut_results) == Counter({Drift("projection_missing", name_of("text")): 1})
    built = P.research_answer(cut, completed=True)
    assert built == P.AnswerText(expected, True)


# --- oracle 6: no account data ---------------------------------------------------------


def _payloads() -> Iterator[tuple[str, dict[str, Any]]]:
    for path in sorted(DIFF.glob("*.events.jsonl")):
        for line in path.read_text().splitlines():
            yield path.name, json.loads(line)


def test_diff_fixtures_carry_no_account_metadata() -> None:
    seen = 0
    for name, payload in _payloads():
        for parent, key, value in account_values(payload, ACCOUNT_KEYS):
            assert value in (None, ACCOUNT_PLACEHOLDER), (name, parent, key)
            seen += 1
    assert seen, "the walk found no account metadata at all; it is not looking"


def test_diff_fixtures_carry_no_identifiers_or_telemetry(san: ModuleType) -> None:
    allowed = {*san.SENTINELS.values(), "REDACTED", None, ""}
    seen = 0
    for name, payload in _payloads():
        for key, value in identity_values(payload):
            assert value in allowed or value in ([], {}), (name, key, value)
            seen += 1
        for key in san.TELEMETRY:
            assert payload.get(key) in (None, {}), (name, key)
    assert seen, "the walk found no identity keys at all; it is not looking"
    for path in sorted(DIFF.glob("*.events.jsonl")):
        assert san.leaks(path.read_text()) == [], path.name


def test_diff_sanitizer_shares_the_research_sanitizer_policy(san: ModuleType) -> None:
    research = _load_script("re_sanitize_research_fixture", RESEARCH_SANITIZER_SCRIPT)
    assert san.SENTINELS == research.SENTINELS
    assert (san.SENTINEL_EMAIL, san.SENTINEL_REPORT_URL) == (
        research.SENTINEL_EMAIL,
        research.SENTINEL_REPORT_URL,
    )
    assert san.ACCOUNT_PARENTS is ACCOUNT_PARENTS


# --- the sanitizer ---------------------------------------------------------------------

REAL = "7f3c9a10-2b4d-4e8f-9a1b-3c5d7e9f1a2b"
THUMB = f"https://d1.cloudfront.net/thumbnails/{REAL}/{REAL}.jpg"


def _envelope(**extra: Any) -> dict[str, Any]:
    return {"status": "PENDING", "text_completed": False, "blocks": [], **extra}


def _body_op(n: int) -> dict[str, Any]:
    return {"op": "replace", "path": "/assets/0/research_report/source_content", "value": "b" * n}


def _assets_diff(*ops: dict[str, Any]) -> dict[str, Any]:
    return {
        "intended_usage": "unified_assets",
        "diff_block": {"field": "unified_assets_block", "patches": list(ops)},
    }


def test_sanitizer_scrubs_planted_values_everywhere(san: ModuleType) -> None:
    op_values = [
        {"op": "replace", "path": "/assets/0/uuid", "value": "planted-user-42"},
        {"op": "add", "path": "/author_id", "value": "planted-user-42"},
        {"op": "add", "path": "/x", "value": {"note": PLANTED_DOC, "email": "me@corp.example"}},
        {"op": "add", "path": "/img", "value": THUMB},
    ]
    payload = _envelope(
        backend_uuid=REAL,
        _extras=json.loads(PLANTED_DOC)["_extras"],
        telemetry_data={"country": "planted-country", "region": "x"},
        classifier_results={"score": 0.5},
        text=json.dumps([{"url": f"https://example.com/{REAL}", "user_id": "planted-user-42"}]),
        blocks=[_assets_diff(*op_values)],
    )
    (out,) = san.sanitize([[payload]], 1)
    text = json.dumps(out)
    for planted in (*PLANTED_VALUES, "me@corp.example"):
        assert planted not in text
    assert out[0]["telemetry_data"] == {}
    assert out[0]["classifier_results"] == {}
    assert out[0]["backend_uuid"] == san.SENTINELS["backend_uuid"]
    # A copy of an identity value elsewhere becomes the same sentinel; the
    # thumbnail path is a public content hash and stays.
    assert f"https://example.com/{san.SENTINELS['backend_uuid']}" in out[0]["text"]
    assert THUMB in text
    assert san.leaks(text) == []


def test_sanitizer_maps_a_uuid_the_same_in_every_file_of_a_run(san: ModuleType) -> None:
    a = _envelope(blocks=[_assets_diff({"op": "add", "path": "/id", "value": REAL})])
    b = _envelope(blocks=[_assets_diff({"op": "add", "path": "/ref", "value": f"x {REAL}"})])
    (first, second) = san.sanitize([[a], [b]], 1)
    mapped = first[0]["blocks"][0]["diff_block"]["patches"][0]["value"]
    assert mapped.startswith(san.MAPPED_PREFIX)
    assert second[0]["blocks"][0]["diff_block"]["patches"][0]["value"] == f"x {mapped}"


def test_sanitizer_thins_only_overwritten_body_replaces(san: ModuleType) -> None:
    payloads = [
        _envelope(blocks=[_assets_diff({"op": "add", "path": "", "value": {"assets": [{}]}})]),
        *(_envelope(blocks=[_assets_diff(_body_op(n))]) for n in range(1, 8)),
        _envelope(status="COMPLETED", text_completed=True),
    ]
    (out,) = san.sanitize([payloads], 3)
    bodies = [
        len(op["value"])
        for p in out
        for b in p["blocks"]
        for op in b["diff_block"]["patches"]
        if op["path"].endswith("source_content")
    ]
    # Of the six overwritten replaces every third stays; the last is never thinned.
    assert bodies == [1, 4, 7]
    assert len(out) == 5
    assert out[-1]["status"] == "COMPLETED"


def test_sanitizer_refuses_an_email_split_across_chunks(san: ModuleType, tmp_path: Path) -> None:
    md = {
        "intended_usage": "ask_text",
        "diff_block": {
            "field": "markdown_block",
            "patches": [
                {"op": "replace", "path": "", "value": {"chunks": []}},
                {"op": "add", "path": "/chunks/0", "value": "write to someone@exa"},
                {"op": "add", "path": "/chunks/1", "value": "mple.org today"},
            ],
        },
    }
    capture = tmp_path / "cap.jsonl"
    record = {"k": "frame", "data": _envelope(blocks=[md])}
    capture.write_text(json.dumps(record) + "\n")
    out_dir = tmp_path / "out"
    assert san.main([str(out_dir), str(capture)]) == 1
    assert not out_dir.exists()


@pytest.mark.parametrize("run", RUNS)
def test_sanitizer_is_stable_on_committed_fixtures(san: ModuleType, run: str) -> None:
    files, _ = RUNS[run]
    payloads = [san.read_payloads(_path(stem)) for stem in files]
    again = san.sanitize(payloads, 1)
    assert again == payloads
