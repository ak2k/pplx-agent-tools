"""Unit tests for verbs/fetch.py — chat body shape + SSE chunk accumulation.

The chunk-accumulation logic is the bug-magnet here (each event carries
duplicate `ask_text` / `ask_text_0_markdown` blocks; reading both
double-counts).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pplx_agent_tools.errors import SchemaError
from pplx_agent_tools.verbs.fetch import _build_chat_body, _fetch_with_prompt
from pplx_agent_tools.wire import Client

# ---------- _build_chat_body ----------


def test_build_chat_body_query_str_set() -> None:
    body = _build_chat_body("hello world")
    assert body["query_str"] == "hello world"
    assert body["params"]["dsl_query"] == "hello world"


def test_build_chat_body_search_focus_internet() -> None:
    body = _build_chat_body("q")
    assert body["params"]["search_focus"] == "internet"
    assert body["params"]["sources"] == ["web"]


def test_build_chat_body_uuids_are_per_call() -> None:
    a = _build_chat_body("q")
    b = _build_chat_body("q")
    assert a["params"]["frontend_uuid"] != b["params"]["frontend_uuid"]
    assert a["params"]["frontend_context_uuid"] != b["params"]["frontend_context_uuid"]


def test_build_chat_body_cache_key_matches_frontend_uuid() -> None:
    body = _build_chat_body("q")
    assert body["params"]["client_search_results_cache_key"] == body["params"]["frontend_uuid"]


def test_build_chat_body_no_attachments_or_mentions() -> None:
    body = _build_chat_body("q")
    assert body["params"]["attachments"] == []
    assert body["params"]["mentions"] == []


def test_build_chat_body_strips_ui_widget_lists() -> None:
    # Make sure the heavy UI widget config is NOT in our body
    body = _build_chat_body("q")
    assert "supported_block_use_cases" not in body["params"]
    assert "supported_features" not in body["params"]


# ---------- _fetch_with_prompt chunk accumulation ----------


def _block(intended_usage: str, chunks: list[str]) -> dict[str, Any]:
    return {
        "intended_usage": intended_usage,
        "markdown_block": {"progress": "IN_PROGRESS", "chunks": list(chunks)},
    }


def _ev(blocks: list[dict[str, Any]], *, status: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"blocks": blocks}
    if status:
        payload["status"] = status
    return {"event": "message", "data": payload}


class FakeClient(Client):
    """Client stand-in that yields canned SSE events from sse_post + records
    delete_thread calls for assertion.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._cookies = {"x": "y"}
        self._base_url = "https://www.perplexity.ai"
        self._timeout = 1.0
        self._events = events
        self.deleted: list[tuple[str, str]] = []
        self.delete_should_fail = False

    def sse_post(self, path: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        yield from self._events

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> None:  # type: ignore[override]
        if self.delete_should_fail:
            raise RuntimeError("simulated cleanup failure")
        self.deleted.append((entry_uuid, read_write_token))


def test_chunk_accumulation_only_reads_ask_text_blocks() -> None:
    """Each event has two parallel blocks (ask_text and ask_text_0_markdown)
    with identical chunks. Reading both would double-count the answer.
    """
    events = [
        _ev(
            [
                _block("ask_text", ["Hello "]),
                _block("ask_text_0_markdown", ["Hello "]),
            ]
        ),
        _ev(
            [
                _block("ask_text", ["world."]),
                _block("ask_text_0_markdown", ["world."]),
            ]
        ),
        _ev([], status="COMPLETED"),
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(
        client, "https://example.com", "summarize", "example.com", max_chars=None
    )
    assert result.content == "Hello world."
    assert result.is_extracted is True


def test_chunk_accumulation_stops_at_completed_status() -> None:
    events = [
        _ev([_block("ask_text", ["first."])]),
        _ev([_block("ask_text", ["second."])], status="COMPLETED"),
        # If we don't break on COMPLETED, this leaks into the result
        _ev([_block("ask_text", ["LEAK"])]),
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert "LEAK" not in result.content
    assert result.content == "first.second."


def test_chunk_accumulation_stops_at_text_completed_flag() -> None:
    # Alternative termination signal that the verb honors
    events = [
        _ev([_block("ask_text", ["done."])]),
        {"event": "message", "data": {"text_completed": True, "blocks": []}},
        _ev([_block("ask_text", ["LEAK"])]),
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert "LEAK" not in result.content


def test_chunk_accumulation_handles_empty_chunks_list() -> None:
    events = [
        _ev([_block("ask_text", [])]),
        _ev([_block("ask_text", ["hello"])], status="COMPLETED"),
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert result.content == "hello"


def test_chunk_accumulation_empty_stream_raises() -> None:
    # No content + no COMPLETED signal → SchemaError, not silent ""
    events: list[dict[str, Any]] = []
    client = FakeClient(events)
    with pytest.raises(SchemaError):
        _fetch_with_prompt(client, "u", "p", "d", max_chars=None)


def test_chunk_accumulation_truncates_to_max_chars() -> None:
    events = [
        _ev([_block("ask_text", ["a" * 100])], status="COMPLETED"),
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=50)
    assert len(result.content) == 50
    assert result.truncated is True


def test_chunk_accumulation_no_truncation_when_under_budget() -> None:
    events = [
        _ev([_block("ask_text", ["short"])], status="COMPLETED"),
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=100)
    assert result.content == "short"
    assert result.truncated is False


def test_chunk_accumulation_ignores_non_dict_event_data() -> None:
    # Robust against malformed events without crashing
    events = [
        {"event": "message", "data": None},
        {"event": "message", "data": "not-a-dict"},
        _ev([_block("ask_text", ["ok"])], status="COMPLETED"),
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert result.content == "ok"


# ---------- thread cleanup (default behavior + --keep-thread) ----------


def _ev_with_thread(
    blocks: list[dict[str, Any]],
    *,
    backend_uuid: str = "thread-abc",
    read_write_token: str = "rwt-xyz",
    status: str | None = None,
) -> dict[str, Any]:
    """Event with the thread identifiers that real responses carry."""
    payload: dict[str, Any] = {
        "blocks": blocks,
        "backend_uuid": backend_uuid,
        "read_write_token": read_write_token,
    }
    if status:
        payload["status"] = status
    return {"event": "message", "data": payload}


def test_thread_cleanup_default_deletes_after_call() -> None:
    events = [
        _ev_with_thread(
            [_block("ask_text", ["answer."])],
            status="COMPLETED",
        ),
    ]
    client = FakeClient(events)
    _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert client.deleted == [("thread-abc", "rwt-xyz")]


def test_thread_cleanup_keep_thread_skips_delete() -> None:
    events = [
        _ev_with_thread(
            [_block("ask_text", ["answer."])],
            status="COMPLETED",
        ),
    ]
    client = FakeClient(events)
    _fetch_with_prompt(client, "u", "p", "d", max_chars=None, keep_thread=True)
    assert client.deleted == []


def test_thread_cleanup_skipped_when_no_backend_uuid() -> None:
    # Defensive: an event stream that doesn't expose backend_uuid (e.g. an
    # edge case in Perplexity's response) shouldn't crash — just skip cleanup
    events = [
        _ev([_block("ask_text", ["answer."])], status="COMPLETED"),
    ]
    client = FakeClient(events)
    _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert client.deleted == []


def test_thread_cleanup_first_event_uuid_used() -> None:
    # backend_uuid should be captured from the first event that carries it
    # (real responses keep it stable across the stream)
    events = [
        _ev_with_thread(
            [_block("ask_text", ["part one"])],
            backend_uuid="first-uuid",
            read_write_token="first-token",
        ),
        _ev_with_thread(
            [_block("ask_text", ["part two"])],
            backend_uuid="should-not-overwrite",
            read_write_token="should-not-overwrite",
            status="COMPLETED",
        ),
    ]
    client = FakeClient(events)
    _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert client.deleted == [("first-uuid", "first-token")]


def test_thread_cleanup_failure_does_not_propagate(
    capsys: pytest.CaptureFixture,
) -> None:
    events = [
        _ev_with_thread([_block("ask_text", ["ok"])], status="COMPLETED"),
    ]
    client = FakeClient(events)
    client.delete_should_fail = True
    # Should NOT raise — best-effort cleanup
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert result.content == "ok"
    err = capsys.readouterr().err
    assert "thread cleanup failed" in err
