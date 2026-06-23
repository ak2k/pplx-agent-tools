"""Unit tests for verbs/ask.py — copilot stream accumulation + render."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pplx_agent_tools.errors import SchemaError, StreamDeadlineError
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


def test_ask_partial_on_deadline() -> None:
    client = _FakeClient([_chunk_event("partial")], raise_deadline=True)
    result = ask(client, "hi", timeout=30)
    assert result.stream_complete is False
    assert result.answer == "partial"


def test_ask_no_content_no_completion_raises() -> None:
    client = _FakeClient([{"data": {"foo": "bar"}}])
    with pytest.raises(SchemaError):
        ask(client, "hi")


def test_ask_failed_status_raises() -> None:
    client = _FakeClient([{"data": {"status": "FAILED"}}])
    with pytest.raises(SchemaError, match="FAILED"):
        ask(client, "hi", model="bogus_model")


def test_ask_model_passthrough_to_body() -> None:
    body = _build_ask_body("q", "claude48opusthinking")
    assert body["params"]["model_preference"] == "claude48opusthinking"
    assert body["params"]["mode"] == "copilot"
    assert body["params"]["is_incognito"] is True


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
