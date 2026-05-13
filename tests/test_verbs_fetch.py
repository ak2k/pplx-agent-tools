"""Unit tests for verbs/fetch.py — chat body shape + SSE chunk accumulation.

The chunk-accumulation logic is the bug-magnet here (each event carries
duplicate `ask_text` / `ask_text_0_markdown` blocks; reading both
double-counts).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pplx_agent_tools.errors import (
    NetworkError,
    RateLimitError,
    SchemaError,
    StreamDeadlineError,
)
from pplx_agent_tools.verbs.fetch import (
    _build_chat_body,
    _fetch_with_prompt,
    _require_http_url,
    fetch_page,
)
from tests._doubles import _TestClientBase

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


def test_build_chat_body_timezone_is_utc_not_location() -> None:
    # We deliberately send UTC instead of the local timezone — leaking the
    # user's location to Perplexity beyond what cookies already imply is
    # not in scope, and time.tzname returns abbreviations Perplexity may
    # not even accept.
    body = _build_chat_body("q")
    assert body["params"]["timezone"] == "UTC"


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


class FakeClient(_TestClientBase):
    """Client stand-in that yields canned SSE events from sse_post + records
    delete_thread calls for assertion.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        super().__init__()
        self._events = events
        self.deleted: list[tuple[str, str]] = []
        self.delete_should_fail = False

    def sse_post(  # type: ignore[override]
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        yield from self._events

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        if self.delete_should_fail:
            # Real Client.delete_thread is best-effort: it logs to stderr
            # and returns False. Mirror that so the caller stays simple.
            import sys

            print("warning: thread cleanup failed: simulated cleanup failure", file=sys.stderr)
            return False
        self.deleted.append((entry_uuid, read_write_token))
        return True


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


# ---------- SSRF scheme allowlist (P1) ----------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/",
        "javascript:alert(1)",
        "data:text/plain,hello",
        "localhost:8080",  # no scheme — parsed as scheme=localhost, netloc=""
    ],
)
def test_require_http_url_rejects_non_http(url: str) -> None:
    with pytest.raises(NetworkError):
        _require_http_url(url)


@pytest.mark.parametrize(
    "url",
    ["http://example.com/", "https://example.com/path?q=1", "HTTPS://Example.com/"],
)
def test_require_http_url_accepts_http_https(url: str) -> None:
    # Must NOT raise. urlparse lowercases the scheme, so HTTPS works.
    _require_http_url(url.lower() if url.upper() == url else url)


def test_require_http_url_rejects_missing_host() -> None:
    with pytest.raises(NetworkError) as ei:
        _require_http_url("http:///")
    assert "no host" in str(ei.value)


def test_fetch_page_rejects_file_scheme_before_network() -> None:
    # File-scheme URLs must be rejected up front — never reach curl_cffi.
    with pytest.raises(NetworkError) as ei:
        fetch_page("file:///etc/passwd", domain="local", max_chars=None)
    assert "scheme" in str(ei.value)


# ---------- stream_complete signal (P1) ----------


def test_stream_complete_true_when_completed_status_seen() -> None:
    events = [
        _ev([_block("ask_text", ["hello"])], status="COMPLETED"),
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert result.stream_complete is True


def test_stream_complete_true_on_text_completed_flag() -> None:
    events = [
        _ev([_block("ask_text", ["hello"])]),
        {"event": "message", "data": {"text_completed": True, "blocks": []}},
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert result.stream_complete is True


def test_stream_complete_false_when_stream_cuts_mid_flight() -> None:
    # Server emits content but never sends COMPLETED / text_completed.
    # Caller must be able to detect this and decide whether to trust the
    # partial answer.
    events = [
        _ev([_block("ask_text", ["partial..."])]),
        # stream just ends — no terminator
    ]
    client = FakeClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None)
    assert result.content == "partial..."
    assert result.stream_complete is False


# ---------- StreamDeadlineError soft-fail (overall timeout) ----------


class _DeadlineClient(_TestClientBase):
    """Yields a prefix of events, then raises StreamDeadlineError to simulate
    the wall-clock deadline tripping mid-stream.
    """

    def __init__(self, events_before_deadline: list[dict[str, Any]]) -> None:
        super().__init__()
        self._events = events_before_deadline

    def sse_post(  # type: ignore[override]
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        yield from self._events
        raise StreamDeadlineError("simulated deadline")

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        return True


def test_deadline_soft_fails_with_partial_content() -> None:
    events = [
        _ev([_block("ask_text", ["partial-"])]),
        _ev([_block("ask_text", ["answer."])]),
        # then deadline trips before COMPLETED
    ]
    client = _DeadlineClient(events)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None, timeout=5.0)
    assert result.content == "partial-answer."
    assert result.stream_complete is False
    assert result.is_extracted is True


def test_deadline_with_no_content_raises_stream_deadline() -> None:
    # Deadline trips before any chunks arrive — there's nothing to salvage,
    # so caller gets the typed error (exit 4).
    client = _DeadlineClient([])
    with pytest.raises(StreamDeadlineError):
        _fetch_with_prompt(client, "u", "p", "d", max_chars=None, timeout=5.0)


def test_deadline_kwarg_propagates_to_sse_post() -> None:
    seen: dict[str, float | None] = {}

    class _Spy(_TestClientBase):
        def sse_post(  # type: ignore[override]
            self,
            path: str,
            body: dict[str, Any],
            *,
            max_total_seconds: float | None = None,
        ) -> Iterator[dict[str, Any]]:
            seen["max_total_seconds"] = max_total_seconds
            yield from [_ev([_block("ask_text", ["x"])], status="COMPLETED")]

        def delete_thread(self, *_a: Any, **_k: Any) -> bool:  # type: ignore[override]
            return True

    _fetch_with_prompt(_Spy(), "u", "p", "d", max_chars=None, timeout=7.5)
    assert seen["max_total_seconds"] is not None
    # Slightly less than 7.5 because some monotonic time passed between the
    # caller computing the deadline and the spy reading the remaining budget.
    assert 0 < seen["max_total_seconds"] <= 7.5


def test_no_timeout_passes_none_to_sse_post() -> None:
    seen: dict[str, float | None] = {"max_total_seconds": -1.0}

    class _Spy(_TestClientBase):
        def sse_post(  # type: ignore[override]
            self,
            path: str,
            body: dict[str, Any],
            *,
            max_total_seconds: float | None = None,
        ) -> Iterator[dict[str, Any]]:
            seen["max_total_seconds"] = max_total_seconds
            yield from [_ev([_block("ask_text", ["x"])], status="COMPLETED")]

        def delete_thread(self, *_a: Any, **_k: Any) -> bool:  # type: ignore[override]
            return True

    _fetch_with_prompt(_Spy(), "u", "p", "d", max_chars=None, timeout=None)
    assert seen["max_total_seconds"] is None


# ---------- Auto-retry on RateLimitError ----------


class _RateLimitClient(_TestClientBase):
    """Raises RateLimitError on the first N attempts, then yields events."""

    def __init__(
        self,
        fail_attempts: int,
        events_on_success: list[dict[str, Any]],
        retry_after: float | None = 0.0,
    ) -> None:
        super().__init__()
        self._fail_attempts = fail_attempts
        self._events = events_on_success
        self._retry_after = retry_after
        self.attempts = 0

    def sse_post(  # type: ignore[override]
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        self.attempts += 1
        if self.attempts <= self._fail_attempts:
            raise RateLimitError("rate limited", retry_after=self._retry_after)
        yield from self._events

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        return True


def test_rate_limit_retries_then_succeeds() -> None:
    events = [_ev([_block("ask_text", ["ok"])], status="COMPLETED")]
    # Fail twice with 0s retry_after so we don't actually sleep
    client = _RateLimitClient(fail_attempts=2, events_on_success=events, retry_after=0.0)
    result = _fetch_with_prompt(client, "u", "p", "d", max_chars=None, timeout=30.0)
    assert client.attempts == 3
    assert result.content == "ok"
    assert result.stream_complete is True


def test_rate_limit_gives_up_after_max_attempts() -> None:
    # All 3 attempts fail → final RateLimitError propagates (exit 3)
    client = _RateLimitClient(fail_attempts=10, events_on_success=[], retry_after=0.0)
    with pytest.raises(RateLimitError):
        _fetch_with_prompt(client, "u", "p", "d", max_chars=None, timeout=30.0)
    assert client.attempts == 3  # _RATE_LIMIT_MAX_ATTEMPTS


def test_rate_limit_retry_respects_deadline(capsys: pytest.CaptureFixture) -> None:
    # retry_after exceeds remaining timeout → sleep gets capped, then raises.
    client = _RateLimitClient(fail_attempts=10, events_on_success=[], retry_after=60.0)
    with pytest.raises(RateLimitError):
        _fetch_with_prompt(client, "u", "p", "d", max_chars=None, timeout=0.01)
    # Process didn't actually sleep 60s — test would time out long before
    # we got here if it had.


# ---------- Progress heartbeat ----------


def test_progress_emits_stderr_after_event_stride(
    capsys: pytest.CaptureFixture,
) -> None:
    # Need >= _PROGRESS_EVENT_STRIDE events to see any heartbeat. Default = 10.
    events = [_ev([_block("ask_text", [f"c{i}"])]) for i in range(12)] + [
        _ev([], status="COMPLETED")
    ]
    client = FakeClient(events)
    _fetch_with_prompt(client, "u", "p", "d", max_chars=None, progress=True)
    err = capsys.readouterr().err
    # At least one heartbeat dot was emitted, plus a trailing newline.
    assert "." in err
    assert err.endswith("\n")


def test_progress_silent_when_disabled(
    capsys: pytest.CaptureFixture,
) -> None:
    events = [_ev([_block("ask_text", [f"c{i}"])]) for i in range(30)] + [
        _ev([], status="COMPLETED")
    ]
    client = FakeClient(events)
    _fetch_with_prompt(client, "u", "p", "d", max_chars=None, progress=False)
    err = capsys.readouterr().err
    # No heartbeat output. The fake's delete_thread path doesn't emit either.
    assert err == ""
