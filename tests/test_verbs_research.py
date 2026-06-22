"""Unit tests for verbs/research.py — schematized block decode + render + orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from pplx_agent_tools.errors import SchemaError, StreamDeadlineError
from pplx_agent_tools.render import render_research_json, render_research_text
from pplx_agent_tools.verbs.research import (
    ResearchResult,
    ResearchSource,
    _build_research_body,
    _model_for_mode,
    decode_research_text,
    research,
)

from ._doubles import _TestClientBase


def _snapshot(answer: str = "QUIC is a protocol. [1]") -> str:
    """A research `text` snapshot: block list with a JSON-wrapped FINAL answer."""
    blocks = [
        {"step_type": "INITIAL_QUERY", "content": {"query": "q"}},
        {
            "step_type": "SEARCH_RESULTS",
            "content": {"web_results": [{"url": "https://mid", "name": "mid"}]},
        },
        {
            "step_type": "FINAL",
            "content": {
                "answer": json.dumps(
                    {"answer": answer, "web_results": [{"url": "https://cited", "name": "Cited"}]}
                )
            },
        },
    ]
    return json.dumps(blocks)


class _FakeClient(_TestClientBase):
    """Yields canned research SSE events; records delete_thread calls."""

    def __init__(self, events: list[dict[str, Any]], *, raise_deadline: bool = False) -> None:
        super().__init__()
        self._events = events
        self._raise_deadline = raise_deadline
        self.deleted: list[tuple[str, str]] = []

    def sse_post(  # type: ignore[override]
        self, path: str, body: dict[str, Any], *, max_total_seconds: float | None = None
    ) -> Iterator[dict[str, Any]]:
        yield from self._events
        if self._raise_deadline:
            raise StreamDeadlineError("simulated deadline")

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


BLOCKS = [
    {"step_type": "INITIAL_QUERY", "content": {"goal_id": None, "query": "q"}, "uuid": "1"},
    {
        "step_type": "SEARCH_WEB",
        "content": {"goal_id": "0", "queries": [{"query": "q"}]},
        "uuid": "2",
    },
    {
        "step_type": "SEARCH_RESULTS",
        "content": {
            "goal_id": "0",
            "web_results": [
                {"url": "https://a", "name": "A", "snippet": "sa"},
                {"url": "https://b", "name": "B"},
            ],
        },
        "uuid": "3",
    },
    {
        "step_type": "SEARCH_RESULTS",
        "content": {
            "goal_id": "1",
            "web_results": [
                {"url": "https://a", "name": "A dup"},  # duplicate URL
                {"url": "https://c", "title": "C"},  # title (not name)
                {"name": "no url — skipped"},
            ],
        },
        "uuid": "4",
    },
    {
        "step_type": "FINAL",
        "content": {"goal_id": None, "answer": "# Answer\nbody [1][2]"},
        "uuid": "5",
    },
]


def test_decode_answer_and_sources() -> None:
    answer, sources = decode_research_text(json.dumps(BLOCKS))
    assert answer == "# Answer\nbody [1][2]"
    assert [s.url for s in sources] == ["https://a", "https://b", "https://c"]  # deduped, ordered
    assert sources[0].title == "A"
    assert sources[0].snippet == "sa"
    assert sources[2].title == "C"  # picked up from "title" key


def test_decode_unwraps_json_wrapped_final_answer() -> None:
    """The real FINAL shape: content.answer is a JSON string wrapping
    {answer: markdown, web_results: [...cited]}. Cited web_results win over
    the intermediate SEARCH_RESULTS rounds."""
    blocks = [
        {
            "step_type": "SEARCH_RESULTS",
            "content": {"web_results": [{"url": "https://intermediate", "name": "mid"}]},
        },
        {
            "step_type": "FINAL",
            "content": {
                "answer": json.dumps(
                    {
                        "answer": "QUIC is a transport protocol. [1]",
                        "web_results": [{"url": "https://cited", "name": "Cited", "snippet": "s"}],
                    }
                )
            },
        },
    ]
    answer, sources = decode_research_text(json.dumps(blocks))
    assert answer == "QUIC is a transport protocol. [1]"  # unwrapped markdown, not the JSON blob
    assert [s.url for s in sources] == ["https://cited"]  # FINAL cited set, not intermediate


def test_decode_multiple_final_concatenated() -> None:
    blocks = [
        {"step_type": "FINAL", "content": {"answer": "part one"}},
        {"step_type": "FINAL", "content": {"answer": "part two"}},
    ]
    answer, _ = decode_research_text(json.dumps(blocks))
    assert answer == "part one\n\npart two"


def test_decode_ignores_unknown_steps_and_bad_blocks() -> None:
    blocks = ["not a dict", {"step_type": "MYSTERY", "content": {"x": 1}}, {"no": "content"}]
    answer, sources = decode_research_text(json.dumps(blocks))
    assert answer == ""
    assert sources == []


def test_decode_non_json_raises() -> None:
    with pytest.raises(SchemaError):
        decode_research_text("{not json")


def test_decode_non_list_raises() -> None:
    with pytest.raises(SchemaError):
        decode_research_text(json.dumps({"step_type": "FINAL"}))


def test_render_text_has_answer_and_sources() -> None:
    result = ResearchResult(
        query="q",
        answer="The answer.",
        sources=[ResearchSource("https://a", "A", None), ResearchSource("https://b", None, None)],
        mode="research",
    )
    out = render_research_text(result)
    assert "The answer." in out
    assert "— sources (2) —" in out
    assert "[1] A" in out and "https://a" in out
    assert "[2] https://b" in out  # no title → url as label


def test_render_text_incomplete_marker() -> None:
    result = ResearchResult("q", "partial", [], "research", stream_complete=False)
    assert "stream: incomplete" in render_research_text(result)


def test_render_json_envelope() -> None:
    result = ResearchResult(
        query="q",
        answer="A",
        sources=[ResearchSource("https://a", "A", "snip")],
        mode="research",
    )
    j = render_research_json(result)
    assert j["_verb"] == "research"
    assert j["mode"] == "research"
    assert j["answer"] == "A"
    assert j["sources"][0] == {"url": "https://a", "title": "A", "snippet": "snip"}
    assert j["stream_complete"] is True


# ---------- orchestration (research()) with a fake SSE client ----------


def _complete_events() -> list[dict[str, Any]]:
    return [
        {"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": _snapshot()}},
        {"data": {"text": _snapshot(), "status": "COMPLETED"}},
    ]


def test_research_completes_parses_and_cleans_up() -> None:
    client = _FakeClient(_complete_events())
    result = research(client, "what is quic")
    assert result.stream_complete is True
    assert result.answer == "QUIC is a protocol. [1]"  # unwrapped markdown
    assert [s.url for s in result.sources] == ["https://cited"]  # FINAL cited set
    assert result.mode == "research"
    assert client.deleted == [("BU", "RW")]  # incognito thread cleaned up by default


def test_research_keep_thread_skips_cleanup() -> None:
    client = _FakeClient(_complete_events())
    research(client, "q", keep_thread=True)
    assert client.deleted == []


def test_research_partial_on_deadline_returns_incomplete() -> None:
    # One snapshot arrives, then the stream trips the overall deadline.
    client = _FakeClient(
        [{"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": _snapshot()}}],
        raise_deadline=True,
    )
    result = research(client, "q", timeout=30)
    assert result.stream_complete is False  # never saw COMPLETED
    assert result.answer == "QUIC is a protocol. [1]"  # partial answer still returned
    assert client.deleted == [("BU", "RW")]


def test_research_deadline_before_any_content_raises() -> None:
    client = _FakeClient([], raise_deadline=True)
    with pytest.raises(StreamDeadlineError):
        research(client, "q", timeout=30)


def test_research_completed_without_text_raises_schema() -> None:
    client = _FakeClient([{"data": {"status": "COMPLETED"}}])
    with pytest.raises(SchemaError):
        research(client, "q")


def test_research_failed_status_raises_clear_error() -> None:
    # Real shape when model_preference is incompatible with the mode: a single
    # frame with text present but status=FAILED + mode dropped to CONCISE.
    client = _FakeClient(
        [{"data": {"text": "{}", "status": "FAILED", "mode": "CONCISE", "text_completed": False}}]
    )
    with pytest.raises(SchemaError, match="FAILED"):
        research(client, "q", mode="research")
    assert client.deleted == []  # no thread to clean up on a failed request


def test_model_for_mode_maps_to_driving_model() -> None:
    # model_preference (not params.mode) is the real selector.
    assert _model_for_mode("research") == "pplx_alpha"
    assert _model_for_mode("agentic_research") == "pplx_agentic_research"
    assert _model_for_mode("council") == "pplx_agentic_research"  # friendly alias
    assert _model_for_mode("pplx_asi") == "pplx_asi"  # unknown → literal passthrough


def test_build_body_drives_via_model_preference() -> None:
    body = _build_research_body("q", "research")
    assert body["params"]["model_preference"] == "pplx_alpha"  # the deep-research model
    assert body["params"]["mode"] == "copilot"  # coarse; server derives mode from the model
    assert body["params"]["is_incognito"] is True
    assert (
        _build_research_body("q", "council")["params"]["model_preference"]
        == "pplx_agentic_research"
    )


def test_research_passes_model_preference_into_body() -> None:
    captured: dict[str, Any] = {}

    class _BodyCapture(_FakeClient):
        def sse_post(self, path: str, body: dict[str, Any], *, max_total_seconds=None):  # type: ignore[override]
            captured["model_preference"] = body["params"]["model_preference"]
            captured["is_incognito"] = body["params"]["is_incognito"]
            return iter(self._events)

    client = _BodyCapture(_complete_events())
    research(client, "q", mode="agentic_research")
    assert captured["model_preference"] == "pplx_agentic_research"
    assert captured["is_incognito"] is True
