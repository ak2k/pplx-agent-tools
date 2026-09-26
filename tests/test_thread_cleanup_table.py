"""Cleanup on every way an ask-family run can end: terminate, then delete.

The rule, restated here rather than taken from production code:

- DELETE iff a thread exists with its read_write_token and not keep_thread.
- TERMINATE iff backend_uuid, context_uuid and display_model all arrived and
  the run may still be live when pplx stops reading. It is not live after a
  COMPLETED or FAILED frame, after the `text_completed` frame that ends a
  fetch, or for an ask that saw `text_completed` and then ended by drop,
  deadline, stall, silence or server close. Every other end after the first
  byte is live, KeyboardInterrupt included. keep_thread never suppresses
  terminate; with no ids neither leg is sent.

Each cell runs a verb over the real `Client.sse_post`, `Client.terminate` and
`Client.delete_thread` on a scripted session with a fake clock, and reads the
requests the session received.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from curl_cffi import CurlECode
from curl_cffi.requests import Response
from curl_cffi.requests.exceptions import RequestException

from pplx_agent_tools import wire
from pplx_agent_tools.errors import NetworkError, RateLimitError, SchemaError
from pplx_agent_tools.render import render_ask_json, render_fetch_json, render_research_json
from pplx_agent_tools.verbs._ask_common import COPILOT_STALL_SECONDS, DEFAULT_STALL_SECONDS
from pplx_agent_tools.verbs.ask import ask
from pplx_agent_tools.verbs.fetch import fetch
from pplx_agent_tools.verbs.research import research
from pplx_agent_tools.wire import Client

from ._doubles import _TestClientBase

VERBS = ("ask", "research", "fetch")
STREAM_ENDS = (
    "completed",
    "failed",
    "server_close",
    "drop",
    "deadline",
    "stall",
    "first_content",
    "silence",
    "oversize",
    "keyboard_interrupt",
)
EARLY_ENDS = ("rate_limit", "pre_first_byte")
IDS = ("complete", "no_context", "no_model", "no_token", "none")
MODEL = {"ask": "turbo", "research": "pplx_alpha", "fetch": "turbo"}
DEADLINE = 100.0
_TERMINATE_PATH = "/rest/sse/perplexity_terminate"
_OVERSIZE_CAP = 512

Step = tuple[float, "bytes | BaseException"]


def expected_legs(
    verb: str, end: str, text_completed: bool, ids: str, keep: bool
) -> tuple[bool, bool]:
    """(terminate, delete) from the rule in the module docstring."""
    delete = ids in ("complete", "no_context", "no_model") and not keep
    terminable = ids in ("complete", "no_token")
    settled_ask = verb == "ask" and end in ("server_close", "drop", "deadline", "stall", "silence")
    over = end in ("completed", "failed", "rate_limit", "pre_first_byte")
    live = not (over or (text_completed and (verb == "fetch" or settled_ask)))
    return terminable and live, delete


# ---------- scripted transport ----------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class _SseResp:
    def __init__(self, steps: list[Step], clock: _Clock) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.content = b""
        self._steps = steps
        self._clock = clock

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for advance, item in self._steps:
            self._clock.now += advance
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        pass


class _Plain:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}
        self.content = b""
        self.text = ""


def _undecodable_error() -> Response:
    """A 502 whose declared charset is unknown and whose body is not UTF-8."""
    resp = Response()
    resp.status_code = 502
    resp.headers["Content-Type"] = "text/plain; charset=bogus-charset"
    resp.content = b"\xff\xfe\xfa oops"
    return resp


class _Session:
    """Answers the SSE POST from a script and records terminate and delete."""

    def __init__(
        self,
        sse: Callable[[], _SseResp | _Plain],
        *,
        cleanup_fails: bool = False,
        cleanup_answer: Callable[[], Any] = lambda: _Plain(200),
    ) -> None:
        self._sse = sse
        self._cleanup_fails = cleanup_fails
        self._cleanup_answer = cleanup_answer
        self.calls: list[tuple[str, dict[str, Any], dict[str, str], Any]] = []

    def post(self, url: str, **kw: Any) -> Any:
        if url.endswith(_TERMINATE_PATH):
            self.calls.append(("terminate", kw["json"], kw.get("headers") or {}, kw["timeout"]))
            if self._cleanup_fails:
                raise RuntimeError("terminate transport down")
            return self._cleanup_answer()
        return self._sse()

    def request(self, method: str, url: str, **kw: Any) -> Any:
        self.calls.append(("delete", kw["json"], kw.get("headers") or {}, kw["timeout"]))
        if self._cleanup_fails:
            raise RuntimeError("delete transport down")
        return self._cleanup_answer()


class _CleanupClient(_TestClientBase):
    """Sends the real cleanup requests, so the table sees what goes on the wire."""

    def __init__(self, session: _Session) -> None:
        super().__init__()
        self.session = session
        vars(self)["_session"] = session

    def terminate(self, entry_uuid: str, context_uuid: str, model_preference: str) -> bool:
        return Client.terminate(self, entry_uuid, context_uuid, model_preference)

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:
        return Client.delete_thread(self, entry_uuid, read_write_token)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(wire, "time", SimpleNamespace(monotonic=c.monotonic, time=lambda: 1.7e9))
    monkeypatch.setattr(wire, "_MAX_SSE_BUFFER_BYTES", _OVERSIZE_CAP)
    return c


def _ids(verb: str, ids: str) -> dict[str, str]:
    full = {
        "backend_uuid": "BU",
        "read_write_token": "RW",
        "context_uuid": "CTX",
        "display_model": MODEL[verb],
    }
    drop = {
        "complete": (),
        "no_context": ("context_uuid",),
        "no_model": ("display_model",),
        "no_token": ("read_write_token",),
        "none": tuple(full),
    }[ids]
    return {k: v for k, v in full.items() if k not in drop}


def _frame(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def _content(verb: str, text: str) -> dict[str, Any]:
    if verb == "research":
        final = {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": text})}}
        return {"text": json.dumps([final])}
    return {"blocks": [{"intended_usage": "ask_text", "markdown_block": {"chunks": [text]}}]}


def _script(verb: str, end: str, text_completed: bool, ids: str) -> list[Step]:
    base = _ids(verb, ids)

    def frame(**extra: Any) -> bytes:
        return _frame({**base, "status": "PENDING", **extra})

    if end == "first_content":
        return [(0, frame()), *[(15, b": ping\n\n") for _ in range(14)]]
    steps: list[Step] = [(0, frame(**_content(verb, "partial ")))]
    if text_completed:
        steps.append((0, frame(text_completed=True, **_content(verb, "whole answer"))))
    stall = DEFAULT_STALL_SECONDS if verb == "research" else COPILOT_STALL_SECONDS
    tail: list[Step] = {
        "completed": [(0, _frame({**base, "status": "COMPLETED"}))],
        "failed": [(0, _frame({**base, "status": "FAILED"}))],
        "server_close": [],
        "drop": [(0, RequestException("curl: (56) simulated", CurlECode.RECV_ERROR))],
        "deadline": [(10, frame(**_content(verb, f"more {i} "))) for i in range(20)],
        "stall": [(15, b": ping\n\n") for _ in range(int(stall // 15) + 4)],
        "silence": [(60, RequestException("curl: (28) simulated", CurlECode.OPERATION_TIMEDOUT))],
        "oversize": [(0, b"data: " + b"x" * (_OVERSIZE_CAP * 2))],
        "keyboard_interrupt": [(0, KeyboardInterrupt())],
    }[end]
    return steps + tail


def _run(
    clock: _Clock,
    verb: str,
    end: str,
    text_completed: bool,
    ids: str,
    keep: bool,
    *,
    cleanup_fails: bool = False,
    cleanup_answer: Callable[[], Any] = lambda: _Plain(200),
) -> tuple[_Session, Any]:
    def sse() -> _SseResp | _Plain:
        if end == "rate_limit":
            return _Plain(429, {"retry-after": "0"})
        if end == "pre_first_byte":
            raise ConnectionError("connection refused")
        return _SseResp(_script(verb, end, text_completed, ids), clock)

    session = _Session(sse, cleanup_fails=cleanup_fails, cleanup_answer=cleanup_answer)
    client = _CleanupClient(session)
    kwargs: dict[str, Any] = {
        "keep_thread": keep,
        "timeout": DEADLINE if end == "deadline" else None,
        "stall_seconds": DEFAULT_STALL_SECONDS if verb == "research" else COPILOT_STALL_SECONDS,
    }
    outcome: Any
    try:
        if verb == "ask":
            outcome = render_ask_json(ask(client, "q", **kwargs))
        elif verb == "research":
            outcome = render_research_json(research(client, "q", **kwargs))
        else:
            outcome = render_fetch_json(fetch(client, "https://example.com", prompt="p", **kwargs))
    except BaseException as e:  # KeyboardInterrupt is one of the ends under test
        outcome = e
    return session, outcome


def _expected_outcome(verb: str, end: str, text_completed: bool) -> Any:
    """How the verb ends, so each cell provably reached the end it names:
    an exception type, or (stream_complete, cut_by)."""
    if verb == "fetch" and text_completed and end not in EARLY_ENDS:
        return (True, None)
    raised = {
        "failed": SchemaError,
        "first_content": NetworkError,
        "oversize": SchemaError,
        "keyboard_interrupt": KeyboardInterrupt,
        "rate_limit": RateLimitError,
        "pre_first_byte": NetworkError,
    }
    if end in raised:
        return raised[end]
    if end == "completed" or (verb == "ask" and text_completed):
        return (True, None)
    return {
        "server_close": (False, None),
        "drop": (False, "drop"),
        "deadline": (False, "deadline"),
        "stall": (False, "stall"),
        "silence": (False, "stall"),
    }[end]


def _check_cell(
    session: _Session, outcome: Any, verb: str, end: str, text_completed: bool, ids: str, keep: bool
) -> tuple[bool, bool]:
    expected_end = _expected_outcome(verb, end, text_completed)
    if isinstance(expected_end, type):
        assert isinstance(outcome, expected_end), outcome
        if end == "first_content":
            assert "no first content" in str(outcome), outcome
    else:
        assert isinstance(outcome, dict), outcome
        assert (outcome["stream_complete"], outcome["cut_by"]) == expected_end
        if end == "silence" and expected_end[1] == "stall":
            assert any("no bytes for" in w for w in outcome["warnings"]), outcome["warnings"]

    kinds = [c[0] for c in session.calls]
    assert kinds.count("terminate") <= 1 and kinds.count("delete") <= 1, kinds
    if kinds == ["delete", "terminate"]:
        pytest.fail("delete sent before terminate")
    for kind, body, headers, timeout in session.calls:
        assert timeout < wire.DEFAULT_TIMEOUT
        if kind == "terminate":
            assert body == {
                "entry_uuid": "BU",
                "context_uuid": "CTX",
                "model_preference": MODEL[verb],
                "terminate_requested_at_ms": 1_700_000_000_000,
            }
            assert headers == {"X-Perplexity-Request-Reason": "thread-floating-footer"}
        else:
            assert body == {"entry_uuid": "BU", "read_write_token": "RW"}
    return "terminate" in kinds, "delete" in kinds


def _cells() -> Iterator[tuple[str, str, bool, str, bool]]:
    for verb, keep in itertools.product(VERBS, (False, True)):
        for end, tc, ids in itertools.product(STREAM_ENDS, (False, True), IDS):
            if end == "first_content" and tc:
                continue  # a text_completed frame carries content
            yield verb, end, tc, ids, keep
        for end in EARLY_ENDS:
            yield verb, end, False, "none", keep  # no frame, so no ids


def test_cleanup_table_over_every_end(clock: _Clock) -> None:
    failures: list[str] = []
    cells = list(_cells())
    for verb, end, tc, ids, keep in cells:
        clock.now = 1000.0
        session, outcome = _run(clock, verb, end, tc, ids, keep)
        try:
            legs = _check_cell(session, outcome, verb, end, tc, ids, keep)
            assert legs == expected_legs(verb, end, tc, ids, keep), legs
        except AssertionError as e:
            failures.append(f"{verb} {end} tc={tc} ids={ids} keep={keep}: {e}")
    assert len(cells) == 3 * 2 * (10 * 2 * 5 - 5 + 2)
    assert not failures, "\n".join(failures[:40]) + f"\n({len(failures)} of {len(cells)} cells)"


@pytest.mark.parametrize(
    ("verb", "end", "tc", "ids", "keep", "legs"),
    [
        ("ask", "completed", False, "complete", False, (False, True)),
        ("ask", "drop", True, "complete", False, (False, True)),
        ("ask", "keyboard_interrupt", True, "complete", False, (True, True)),
        ("research", "server_close", True, "complete", False, (True, True)),
        ("ask", "deadline", False, "complete", False, (True, True)),
        ("research", "deadline", False, "complete", False, (True, True)),
        ("fetch", "deadline", False, "complete", False, (True, True)),
        ("fetch", "server_close", True, "complete", False, (False, True)),
        ("ask", "stall", False, "no_context", False, (False, True)),
        ("research", "stall", False, "complete", True, (True, False)),
        ("ask", "rate_limit", False, "none", False, (False, False)),
        ("fetch", "drop", False, "no_token", False, (True, False)),
    ],
    ids=str,
)
def test_cleanup_anchor_rows(
    clock: _Clock, verb: str, end: str, tc: bool, ids: str, keep: bool, legs: tuple[bool, bool]
) -> None:
    assert expected_legs(verb, end, tc, ids, keep) == legs
    session, outcome = _run(clock, verb, end, tc, ids, keep)
    assert _check_cell(session, outcome, verb, end, tc, ids, keep) == legs


@pytest.mark.parametrize("verb", VERBS)
def test_cleanup_failures_never_replace_the_exception_in_flight(
    clock: _Clock, verb: str, capsys: pytest.CaptureFixture[str]
) -> None:
    session, outcome = _run(
        clock, verb, "keyboard_interrupt", False, "complete", False, cleanup_fails=True
    )
    assert isinstance(outcome, KeyboardInterrupt)
    assert [c[0] for c in session.calls] == ["terminate", "delete"]
    err = capsys.readouterr().err
    assert "terminate failed" in err and "cleanup failed" in err


def test_a_failed_terminate_still_deletes_and_returns_the_partial(clock: _Clock) -> None:
    session, outcome = _run(
        clock, "research", "stall", False, "complete", False, cleanup_fails=True
    )
    assert isinstance(outcome, dict)
    assert outcome["cut_by"] == "stall"
    assert [c[0] for c in session.calls] == ["terminate", "delete"]


def test_the_undecodable_answer_is_one_curl_cffi_cannot_read() -> None:
    with pytest.raises(UnicodeDecodeError):
        _ = _undecodable_error().text


@pytest.mark.parametrize("verb", VERBS)
def test_an_undecodable_cleanup_answer_never_replaces_the_interrupt(
    clock: _Clock, verb: str, capsys: pytest.CaptureFixture[str]
) -> None:
    session, outcome = _run(
        clock,
        verb,
        "keyboard_interrupt",
        False,
        "complete",
        False,
        cleanup_answer=_undecodable_error,
    )
    assert isinstance(outcome, KeyboardInterrupt), outcome
    assert [c[0] for c in session.calls] == ["terminate", "delete"]
    err = capsys.readouterr().err
    assert "terminate failed: BU returned 502" in err
    assert "cleanup failed: DELETE BU returned 502" in err


@pytest.mark.parametrize("verb", VERBS)
def test_an_undecodable_cleanup_answer_keeps_the_salvaged_partial(clock: _Clock, verb: str) -> None:
    session, outcome = _run(
        clock, verb, "drop", False, "complete", False, cleanup_answer=_undecodable_error
    )
    assert isinstance(outcome, dict), outcome
    assert outcome["cut_by"] == "drop"
    assert "partial" in json.dumps(outcome)
    assert [c[0] for c in session.calls] == ["terminate", "delete"]


def test_both_cleanup_requests_share_one_short_timeout(clock: _Clock) -> None:
    session, _ = _run(clock, "ask", "keyboard_interrupt", False, "complete", False)
    timeouts = {kind: timeout for kind, _, _, timeout in session.calls}
    assert set(timeouts) == {"terminate", "delete"}
    assert timeouts["delete"] == timeouts["terminate"] < wire.DEFAULT_TIMEOUT
