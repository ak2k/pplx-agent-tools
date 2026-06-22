"""Unit tests for verbs/research.py — schematized block decode + render."""

from __future__ import annotations

import json

import pytest

from pplx_agent_tools.errors import SchemaError
from pplx_agent_tools.render import render_research_json, render_research_text
from pplx_agent_tools.verbs.research import ResearchResult, ResearchSource, decode_research_text

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
