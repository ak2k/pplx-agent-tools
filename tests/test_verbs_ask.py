"""Unit tests for verbs/ask.py — copilot stream accumulation + render."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pplx_agent_tools.errors import RateLimitError, SchemaError, StreamDeadlineError
from pplx_agent_tools.render import render_ask_json, render_ask_text
from pplx_agent_tools.verbs.ask import AskResult, _build_ask_body, ask

from ._doubles import _TestClientBase


def _chunk_event(text: str) -> dict[str, Any]:
    """A copilot SSE event carrying one markdown_block chunk."""
    return {
        "data": {
            "backend_uuid": "BU",
            "read_write_token": "RW",
            "blocks": [{"intended_usage": "ask_text", "markdown_block": {"chunks": [text]}}],
        }
    }


def _web_results_event(results: list[dict[str, Any]]) -> dict[str, Any]:
    """A copilot SSE event carrying the cited-sources block."""
    return {
        "data": {
            "blocks": [
                {"intended_usage": "web_results", "web_result_block": {"web_results": results}}
            ]
        }
    }


class _FakeClient(_TestClientBase):
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


def _complete() -> list[dict[str, Any]]:
    return [_chunk_event("Hello "), _chunk_event("world."), {"data": {"status": "COMPLETED"}}]


def test_ask_accumulates_and_cleans_up() -> None:
    client = _FakeClient(_complete())
    result = ask(client, "hi")
    assert result.answer == "Hello world."
    assert result.stream_complete is True
    assert result.model == "turbo"
    assert client.deleted == [("BU", "RW")]  # incognito thread cleaned up by default


def test_ask_keep_thread_skips_cleanup() -> None:
    client = _FakeClient(_complete())
    ask(client, "hi", keep_thread=True)
    assert client.deleted == []


def test_ask_extracts_sources_from_web_results_block() -> None:
    client = _FakeClient(
        [
            _web_results_event(
                [
                    {"url": "https://a", "name": "A", "snippet": "sa"},
                    {"url": "https://b", "name": "B"},
                    {"url": "https://a", "name": "dup"},  # duplicate URL
                    {"name": "no url — skipped"},
                ]
            ),
            _chunk_event("the answer"),
            {"data": {"status": "COMPLETED"}},
        ]
    )
    result = ask(client, "q")
    assert result.answer == "the answer"
    assert [s.url for s in result.sources] == ["https://a", "https://b"]  # deduped, ordered
    assert result.sources[0].title == "A"
    assert result.sources[0].snippet == "sa"


def test_ask_partial_on_deadline() -> None:
    client = _FakeClient([_chunk_event("partial")], raise_deadline=True)
    result = ask(client, "hi", timeout=30)
    assert result.stream_complete is False
    assert result.answer == "partial"


def test_ask_no_content_no_completion_raises() -> None:
    client = _FakeClient([{"data": {"foo": "bar"}}])
    with pytest.raises(SchemaError):
        ask(client, "hi")


def test_ask_failed_status_reaps_thread_then_raises() -> None:
    # FAILED frame carries the thread IDs — cleanup must still run (no leak) even
    # though ask() then raises. (Regression: cleanup used to be skipped by the raise.)
    client = _FakeClient(
        [{"data": {"backend_uuid": "BU", "read_write_token": "RW", "status": "FAILED"}}]
    )
    with pytest.raises(SchemaError, match="FAILED"):
        ask(client, "hi", model="bogus_model")
    assert client.deleted == [("BU", "RW")]


def test_ask_deadline_before_any_content_raises() -> None:
    client = _FakeClient([], raise_deadline=True)
    with pytest.raises(StreamDeadlineError):
        ask(client, "hi", timeout=30)


def test_ask_model_passthrough_to_body() -> None:
    body = _build_ask_body("q", "claude48opusthinking")
    assert body["params"]["model_preference"] == "claude48opusthinking"
    assert body["params"]["mode"] == "copilot"
    assert body["params"]["is_incognito"] is True


def test_ask_family_bodies_share_one_base() -> None:
    """ask/research/fetch --prompt bodies all delegate to base_ask_params, so a
    field added to one without the others is a bug. Lock the shared key set."""
    from pplx_agent_tools.verbs.fetch import _build_chat_body
    from pplx_agent_tools.verbs.research import _build_research_body

    ask_p = _build_ask_body("q", "turbo")["params"]
    research_p = _build_research_body("q", "pplx_alpha")["params"]
    fetch_p = _build_chat_body("q")["params"]
    base_keys = set(ask_p) - {"compare_model_preferences"}
    assert set(research_p) == base_keys
    assert set(fetch_p) == base_keys
    for k in ("mode", "search_focus", "sources", "is_incognito", "use_schematized_api"):
        assert ask_p[k] == research_p[k] == fetch_p[k]


def test_render_ask_text_and_json() -> None:
    result = AskResult(query="q", answer="The answer.", model="turbo")
    assert render_ask_text(result) == "The answer."
    j = render_ask_json(result)
    assert j["_verb"] == "ask"
    assert j["model"] == "turbo"
    assert j["answer"] == "The answer."
    assert j["stream_complete"] is True


def test_render_ask_text_incomplete_marker() -> None:
    result = AskResult("q", "partial", "turbo", stream_complete=False)
    assert "stream: incomplete" in render_ask_text(result)


def test_render_ask_with_sources() -> None:
    from pplx_agent_tools.verbs._ask_common import Source

    result = AskResult("q", "Answer.", "turbo", True, [Source("https://a", "A", "snip")])
    out = render_ask_text(result)
    assert "— sources (1) —" in out and "[1] A" in out and "https://a" in out
    j = render_ask_json(result)
    assert j["sources"][0] == {"url": "https://a", "title": "A", "snippet": "snip"}


# ---------- run_ask_stream 429 retry/exhaustion (exercised via ask) ----------


class _RateLimitClient(_TestClientBase):
    """Raises RateLimitError on the first `fail_times` sse_post calls (retry_after=0
    so backoff is instant), then yields normal complete events."""

    def __init__(self, fail_times: int, events: list[dict[str, Any]]) -> None:
        super().__init__()
        self._fail_times = fail_times
        self._events = events
        self._calls = 0
        self.deleted: list[tuple[str, str]] = []

    def sse_post(  # type: ignore[override]
        self, path: str, body: dict[str, Any], *, max_total_seconds: float | None = None
    ) -> Iterator[dict[str, Any]]:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RateLimitError("429", retry_after=0.0)
        yield from self._events

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


def test_ask_retries_on_rate_limit_then_succeeds() -> None:
    client = _RateLimitClient(2, _complete())  # fail twice, succeed on the 3rd attempt
    result = ask(client, "hi")
    assert result.answer == "Hello world."
    assert result.stream_complete is True


def test_ask_rate_limit_exhausted_reraises() -> None:
    client = _RateLimitClient(3, _complete())  # fail all 3 attempts
    with pytest.raises(RateLimitError):
        ask(client, "hi")
