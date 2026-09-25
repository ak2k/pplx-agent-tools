"""Legacy fixtures through the block store and projections give the same
answers and sources as today's extractors (U3 oracle 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pplx_agent_tools.askstream.blocks import BlockStore
from pplx_agent_tools.askstream.frames import AskFrame, decode_frame
from pplx_agent_tools.askstream.patch import Limits
from pplx_agent_tools.askstream.projections import (
    AnswerPaths,
    AskTextPath,
    answer,
    answer_text,
    sources,
)
from pplx_agent_tools.verbs._ask_common import (
    apply_chunk_patch,
    event_marks_completed,
    extract_chunk_patches,
    extract_chunks_from_event,
    extract_web_results,
    status_completed,
    to_source,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _payloads(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if isinstance(record, dict) and set(record) == {"event", "data"}:
                record = record["data"]
            if record:
                out.append(record)
    return out


def _replay(path: Path, paths: AnswerPaths) -> tuple[BlockStore, bool]:
    store = BlockStore(paths, Limits())
    completed = False
    for payload in _payloads(path):
        f, _ = decode_frame(json.dumps(payload))
        assert isinstance(f, AskFrame)
        store.apply_frame(f)
        completed = completed or f.stage == "completed"
    return store, completed


def _today_ask(path: Path) -> tuple[str, list[str]]:
    chunks: dict[int, str] = {}
    urls: list[str] = []
    for payload in _payloads(path):
        event = {"data": payload}
        terminal = status_completed(event)
        for offset, run in extract_chunk_patches(event):
            apply_chunk_patch(chunks, offset, run, terminal=terminal)
        raw = extract_web_results(event)
        if raw:
            urls = list(dict.fromkeys(s.url for r in raw if (s := to_source(r)) is not None))
        if terminal:
            break
    return "".join(chunks[i] for i in sorted(chunks)).strip(), urls


def _today_fetch(path: Path) -> str:
    chunks: list[str] = []
    for payload in _payloads(path):
        event = {"data": payload}
        chunks.extend(extract_chunks_from_event(event))
        if event_marks_completed(event):
            break
    return "".join(chunks).strip()


def _check(today: str, store: BlockStore, completed: bool) -> None:
    """The returned text equals today's. The terminal `answer` wins where it
    is present (§1 decision 15c); no committed legacy fixture has one that
    differs from the chunk join, so none is listed as an intended change."""
    ans, drift = answer(store, "ask_text_or_workflow")
    assert drift == ()
    assert answer_text(ans, completed).text == today


def test_legacy_ask_fixture_answer_and_sources_match_today() -> None:
    path = FIXTURES / "ask/multi-step-sources.events.jsonl"
    store, completed = _replay(path, "ask_text_or_workflow")
    today, urls = _today_ask(path)
    _check(today, store, completed)
    assert [s.url for s in sources(store)] == urls


FETCH = [
    *sorted((FIXTURES / "fetch-url").glob("*prompt.events.jsonl")),
    FIXTURES / "fetch-url/no-completed-marker.events.jsonl",
]


@pytest.mark.parametrize("path", FETCH, ids=[p.name for p in FETCH])
def test_legacy_fetch_fixture_answer_matches_today(path: Path) -> None:
    store, completed = _replay(path, "ask_text_or_workflow")
    _check(_today_fetch(path), store, completed)


RESEARCH = sorted((FIXTURES / "research").glob("*.events.jsonl"))


@pytest.mark.parametrize("path", RESEARCH, ids=[p.name for p in RESEARCH])
def test_legacy_research_fixture_reads_ask_text_with_no_ambiguity(path: Path) -> None:
    """Legacy research returns its terminal `text`; the projections still
    pick `ask_text`, and its final answer is the FINAL cover note."""
    store, completed = _replay(path, "ask_text_only")
    ans, drift = answer(store, "ask_text_only")
    assert (ans.source, drift, completed) == (AskTextPath(), (), True)
    assert ans.final is not None
    final_text = json.loads(_payloads(path)[-1]["text"])
    finals = [b for b in final_text if isinstance(b, dict) and b.get("step_type") == "FINAL"]
    assert ans.final in json.loads(finals[-1]["content"]["answer"])["answer"]
