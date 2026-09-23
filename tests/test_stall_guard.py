"""Stall guard for the ask-family SSE stream: no data event for N seconds cuts it.

Runs the real `Client.sse_post` over a scripted stream whose clock is fake, so
multi-minute scenarios run instantly. Heartbeats are SSE comment frames, which
Perplexity sends every ~15 s whether or not the backend is making progress.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from curl_cffi import CurlECode
from curl_cffi.requests.exceptions import RequestException

from pplx_agent_tools import cli_ask, cli_fetch, cli_research, cli_runner, wire
from pplx_agent_tools.errors import (
    EXIT_NETWORK,
    EXIT_PARTIAL,
    NetworkError,
    StreamDeadlineError,
    StreamStallError,
)
from pplx_agent_tools.render import render_fetch_json
from pplx_agent_tools.verbs._ask_common import DEFAULT_STALL_SECONDS, no_content_error
from pplx_agent_tools.verbs.ask import ask
from pplx_agent_tools.verbs.fetch import FetchResult, fetch
from pplx_agent_tools.verbs.research import research

from ._doubles import _TestClientBase

HEARTBEAT = b": ping\n\n"
# (seconds to advance the fake clock, then the chunk to deliver or exception to raise)
Step = tuple[float, "bytes | BaseException"]


def _frame(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def _chunk(text: str) -> bytes:
    return _frame(
        {
            "backend_uuid": "BU",
            "read_write_token": "RW",
            "blocks": [{"intended_usage": "ask_text", "markdown_block": {"chunks": [text]}}],
        }
    )


def _snapshot(answer: str) -> bytes:
    final = {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": answer})}}
    return _frame({"backend_uuid": "BU", "read_write_token": "RW", "text": json.dumps([final])})


COMPLETED = _frame({"status": "COMPLETED"})


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class _ScriptedResp:
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


class _Session:
    def __init__(self, resp: _ScriptedResp) -> None:
        self._resp = resp
        self.timeout: Any = None

    def post(self, url: str, **kwargs: Any) -> _ScriptedResp:
        self.timeout = kwargs["timeout"]
        return self._resp


class _StreamClient(_TestClientBase):
    """Real `sse_post` over a scripted stream; records thread deletions."""

    def __init__(self, steps: list[Step], clock: _Clock) -> None:
        super().__init__()
        self.session = _Session(_ScriptedResp(steps, clock))
        self._session = self.session  # type: ignore[assignment]
        self.deleted: list[tuple[str, str]] = []

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(wire, "time", SimpleNamespace(monotonic=c.monotonic))
    return c


def _heartbeats(seconds: float, every: float = 15.0) -> list[Step]:
    return [(every, HEARTBEAT) for _ in range(int(seconds // every))]


def _curl_error(code: CurlECode) -> RequestException:
    # The shape curl_cffi queues when a streaming transfer dies mid-body.
    return RequestException(f"curl: ({int(code)}) simulated", code)


# ---------- wire: stall detection ----------


def test_heartbeats_do_not_reset_the_stall_clock(clock: _Clock) -> None:
    client = _StreamClient([(0, _chunk("a")), *_heartbeats(200)], clock)
    data_events = 0
    with pytest.raises(StreamStallError) as exc:
        for event in client.sse_post("/x", {}, max_total_seconds=1800, stall_seconds=120):
            data_events += event["data"] is not None
    assert exc.value.seconds == 120
    assert "no data for 120.0s" in str(exc.value)
    assert data_events == 1
    # Heartbeats arrive every 15 s, so the trip lands within one of the window.
    assert 120 < clock.now - 1000.0 <= 135


def test_long_healthy_stream_outlives_the_old_research_deadline(clock: _Clock) -> None:
    steps: list[Step] = [(60, _chunk(str(i))) for i in range(10)]
    client = _StreamClient([*steps, (60, COMPLETED)], clock)
    events = list(client.sse_post("/x", {}, max_total_seconds=1800, stall_seconds=120))
    assert len(events) == 11
    assert clock.now - 1000.0 > 300


def test_deadline_tighter_than_stall_reports_as_deadline(clock: _Clock) -> None:
    client = _StreamClient([(10, _chunk(str(i))) for i in range(20)], clock)
    with pytest.raises(StreamDeadlineError) as exc:
        list(client.sse_post("/x", {}, max_total_seconds=50, stall_seconds=120))
    assert not isinstance(exc.value, StreamStallError)
    assert "exceeded 50.0s deadline" in str(exc.value)


def test_curl_timeout_mid_stream_is_a_stall(clock: _Clock) -> None:
    client = _StreamClient(
        [(0, _chunk("a")), (0, _curl_error(CurlECode.OPERATION_TIMEDOUT))], clock
    )
    with pytest.raises(StreamStallError, match=r"no data for 120\.0s"):
        list(client.sse_post("/x", {}, max_total_seconds=1800, stall_seconds=120))


def test_curl_timeout_is_a_deadline_when_the_deadline_is_the_tighter_bound(clock: _Clock) -> None:
    client = _StreamClient(
        [(0, _chunk("a")), (0, _curl_error(CurlECode.OPERATION_TIMEDOUT))], clock
    )
    with pytest.raises(StreamDeadlineError) as exc:
        list(client.sse_post("/x", {}, max_total_seconds=60, stall_seconds=120))
    assert not isinstance(exc.value, StreamStallError)


def test_curl_timeout_without_stall_guard_is_a_stall_of_the_backstop(clock: _Clock) -> None:
    client = _StreamClient(
        [(0, _chunk("a")), (0, _curl_error(CurlECode.OPERATION_TIMEDOUT))], clock
    )
    with pytest.raises(StreamStallError, match=r"no data for 90\.0s"):
        list(client.sse_post("/x", {}, max_total_seconds=180))


def test_other_curl_errors_stay_network_errors(clock: _Clock) -> None:
    client = _StreamClient([(0, _chunk("a")), (0, _curl_error(CurlECode.RECV_ERROR))], clock)
    with pytest.raises(NetworkError, match="mid-stream") as exc:
        list(client.sse_post("/x", {}, max_total_seconds=1800, stall_seconds=120))
    assert not isinstance(exc.value, StreamDeadlineError)


@pytest.mark.parametrize(
    ("stall", "deadline", "expected"),
    [
        (120.0, 1800.0, (30.0, 90.0)),  # low-speed abort after ~120 s of silence
        (120.0, 50.0, (30.0, 20.0)),  # capped by the remaining deadline
        (10.0, None, (30.0, 1.0)),  # window shorter than the connect leg
        (None, 180.0, (30.0, 60.0)),  # guard disabled: today's 90 s backstop
    ],
)
def test_transport_read_leg_matches_the_silence_window(
    clock: _Clock,
    stall: float | None,
    deadline: float | None,
    expected: tuple[float, float],
) -> None:
    client = _StreamClient([(0, COMPLETED)], clock)
    list(client.sse_post("/x", {}, max_total_seconds=deadline, stall_seconds=stall))
    assert client.session.timeout == expected


# ---------- verbs + CLI: exit codes and warnings ----------


def _run_cli(
    monkeypatch: pytest.MonkeyPatch, main: Callable[[list[str]], int], argv: list[str], client: Any
) -> int:
    monkeypatch.setattr(
        cli_runner.Client, "from_default_cookies", classmethod(lambda cls, **_: client)
    )
    return main(argv)


def test_stall_after_partial_content_returns_the_partial(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient([(0, _chunk("partial answer")), *_heartbeats(300)], clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q"], client)
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    assert "partial answer" in cap.out
    assert "stalled: no data for 120.0s" in cap.err
    assert client.deleted == [("BU", "RW")]


def test_stall_warning_reaches_the_json_envelope(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient([(0, _snapshot("partial report")), *_heartbeats(300)], clock)
    rc = _run_cli(monkeypatch, cli_research.main, ["q", "--json"], client)
    out = json.loads(capsys.readouterr().out)
    assert rc == EXIT_PARTIAL
    assert out["stream_complete"] is False
    assert out["content_shortfall"] is False
    assert any("stalled: no data for 120.0s" in w for w in out["warnings"])


def test_stall_before_any_content_exits_network(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient(_heartbeats(300), clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q"], client)
    cap = capsys.readouterr()
    assert rc == EXIT_NETWORK
    assert "no data for 120.0s before the first content arrived" in cap.err


def test_curl_timeout_after_content_keeps_the_partial(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    steps: list[Step] = [(0, _chunk("kept")), (0, _curl_error(CurlECode.OPERATION_TIMEDOUT))]
    client = _StreamClient(steps, clock)
    rc = _run_cli(monkeypatch, cli_fetch.main, ["https://example.com", "--prompt", "p"], client)
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    assert "kept" in cap.out
    assert "stalled" in cap.err


def test_non_timeout_transport_error_after_content_exits_network(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient([(0, _chunk("lost")), (0, _curl_error(CurlECode.RECV_ERROR))], clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q"], client)
    assert rc == EXIT_NETWORK
    assert "mid-stream" in capsys.readouterr().err
    assert client.deleted == [("BU", "RW")]


def test_hard_cap_with_data_still_flowing_returns_a_deadline_partial(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient([(60, _snapshot(f"report v{i}")) for i in range(40)], clock)
    rc = _run_cli(monkeypatch, cli_research.main, ["q"], client)
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    assert "report v" in cap.out
    assert "s deadline" in cap.err
    assert "stalled" not in cap.err


def test_no_content_messages_name_the_bound_that_fired() -> None:
    stall = no_content_error(
        label="ask", endpoint="/e", timeout=180, cutoff=StreamStallError("x", 120)
    )
    deadline = no_content_error(
        label="ask", endpoint="/e", timeout=180, cutoff=StreamDeadlineError("x")
    )
    assert isinstance(stall, StreamStallError)
    assert "no data for 120.0s" in str(stall)
    assert type(deadline) is StreamDeadlineError
    assert "exceeded 180.0s deadline" in str(deadline)


# ---------- thread cleanup on KeyboardInterrupt ----------


_VERBS: list[Callable[..., Any]] = [
    lambda c, **kw: ask(c, "q", **kw),
    lambda c, **kw: research(c, "q", **kw),
    lambda c, **kw: fetch(c, "https://example.com", prompt="p", **kw),
]


@pytest.mark.parametrize("run", _VERBS, ids=["ask", "research", "fetch"])
def test_keyboard_interrupt_mid_stream_still_reaps_the_thread(
    clock: _Clock, run: Callable[..., Any]
) -> None:
    client = _StreamClient([(0, _chunk("a")), (0, KeyboardInterrupt())], clock)
    with pytest.raises(KeyboardInterrupt):
        run(client, stall_seconds=120)
    assert client.deleted == [("BU", "RW")]


@pytest.mark.parametrize("run", _VERBS, ids=["ask", "research", "fetch"])
def test_keyboard_interrupt_with_keep_thread_leaves_the_thread(
    clock: _Clock, run: Callable[..., Any]
) -> None:
    client = _StreamClient([(0, _chunk("a")), (0, KeyboardInterrupt())], clock)
    with pytest.raises(KeyboardInterrupt):
        run(client, keep_thread=True)
    assert client.deleted == []


# ---------- CLI: --stall-timeout / $PPLX_STALL_TIMEOUT ----------


def _captured_stall(
    monkeypatch: pytest.MonkeyPatch, module: Any, verb: str, argv: list[str]
) -> float | None:
    seen: dict[str, Any] = {}

    def _spy(*_a: Any, **kw: Any) -> Any:
        seen["stall"] = kw["stall_seconds"]
        raise StreamStallError("stop", 1)

    monkeypatch.setattr(module, verb, _spy)
    monkeypatch.setattr(
        cli_runner.Client, "from_default_cookies", classmethod(lambda cls, **_: object())
    )
    module.main(argv)
    return seen["stall"]


@pytest.mark.parametrize(
    ("argv_extra", "env", "expected"),
    [
        ([], None, DEFAULT_STALL_SECONDS),
        (["--stall-timeout", "45"], None, 45.0),
        (["--stall-timeout", "0"], None, None),
        (["--stall-timeout", "-1"], None, None),
        ([], "30", 30.0),
        ([], "0", None),
        (["--stall-timeout", "45"], "30", 45.0),
        ([], "soon", DEFAULT_STALL_SECONDS),
    ],
)
@pytest.mark.parametrize(
    ("module", "verb", "argv"),
    [
        (cli_ask, "ask", ["q"]),
        (cli_research, "research", ["q"]),
        (cli_fetch, "fetch", ["https://example.com", "--prompt", "p"]),
    ],
    ids=["ask", "research", "fetch"],
)
def test_stall_timeout_resolution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module: Any,
    verb: str,
    argv: list[str],
    argv_extra: list[str],
    env: str | None,
    expected: float | None,
) -> None:
    if env is None:
        monkeypatch.delenv("PPLX_STALL_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("PPLX_STALL_TIMEOUT", env)
    assert _captured_stall(monkeypatch, module, verb, argv + argv_extra) == expected
    capsys.readouterr()


def test_stall_timeout_ignored_in_plain_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPLX_STALL_TIMEOUT", "30")
    stall = _captured_stall(
        monkeypatch, cli_fetch, "fetch", ["https://example.com", "--stall-timeout", "45"]
    )
    assert stall is None


def test_fetch_json_carries_warnings() -> None:
    result = FetchResult(
        url="u", title=None, domain="d", content="c", is_extracted=True, warnings=["w"]
    )
    assert render_fetch_json(result)["warnings"] == ["w"]
