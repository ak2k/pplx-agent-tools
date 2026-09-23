"""Stall guard for the ask-family SSE stream: no new content for N seconds cuts it.

Runs the real `Client.sse_post` over a scripted stream whose clock is fake, so
multi-minute scenarios run instantly. Heartbeats are SSE comment frames, which
Perplexity sends every ~15 s whether or not the backend is making progress; it
also keeps sending envelope-only and repeated-snapshot data frames, which the
ask-family verbs must not count as progress either.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from curl_cffi import CurlECode
from curl_cffi.requests.exceptions import RequestException

from pplx_agent_tools import cli_ask, cli_fetch, cli_research, cli_runner, wire
from pplx_agent_tools.errors import (
    EXIT_NETWORK,
    EXIT_PARTIAL,
    NetworkError,
    RateLimitError,
    StreamDeadlineError,
    StreamStallError,
)
from pplx_agent_tools.render import (
    render_ask_json,
    render_ask_text,
    render_fetch_json,
    render_fetch_text,
    render_research_json,
    render_research_text,
)
from pplx_agent_tools.verbs._ask_common import (
    COPILOT_SETTLE_SECONDS,
    COPILOT_STALL_SECONDS,
    DEFAULT_STALL_SECONDS,
    AskStreamState,
    blocks_changed,
    cutoff_cause,
    no_content_error,
    run_ask_stream,
)
from pplx_agent_tools.verbs.ask import (
    SOURCES_FRAME_MISSING,
    AskResult,
    Cut,
    Finished,
    FinishedWithoutSources,
    ask,
)
from pplx_agent_tools.verbs.fetch import FetchResult, fetch
from pplx_agent_tools.verbs.research import ResearchResult, _text_changed, research

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


def _research_steps(*steps: dict[str, Any]) -> bytes:
    """A research snapshot made of the given blocks, without a FINAL answer."""
    return _frame({"backend_uuid": "BU", "read_write_token": "RW", "text": json.dumps(steps)})


INITIAL_QUERY = {"step_type": "INITIAL_QUERY", "content": {"query": "q"}}
SEARCH_RESULTS = {
    "step_type": "SEARCH_RESULTS",
    "content": {"web_results": [{"url": "https://found", "name": "Found"}]},
}
COMPLETED = _frame({"status": "COMPLETED"})
# ask and fetch --prompt default to a 180 s deadline, inside the default stall
# window, so their stall scenarios need the deadline lifted out of the way.
LONG_DEADLINE = ["--timeout", "1800"]
# A data frame with no `blocks`: the bulk of a live ask stream.
ENVELOPE = _frame({"backend_uuid": "BU", "read_write_token": "RW", "status": "PENDING"})


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
        # Swap the transport Client.__init__ built without redefining the
        # attribute, so the real `sse_post` streams from the script.
        vars(self)["_session"] = self.session
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


def _repeat(frame: bytes, seconds: float, every: float = 14.0) -> list[Step]:
    return [(every, frame) for _ in range(int(seconds // every))]


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
    assert "no new content for 120.0s" in str(exc.value)
    assert data_events == 1
    # Heartbeats arrive every 15 s, so the trip lands within one of the window.
    assert 120 < clock.now - 1000.0 <= 135


def test_default_predicate_counts_any_data_event(clock: _Clock) -> None:
    client = _StreamClient([(0, _chunk("a")), *_repeat(ENVELOPE, 500), (14, COMPLETED)], clock)
    events = list(client.sse_post("/x", {}, max_total_seconds=1800, stall_seconds=120))
    assert events[-1]["data"] == {"status": "COMPLETED"}
    assert clock.now - 1000.0 > 500


def test_predicate_decides_what_resets_the_stall_clock(clock: _Clock) -> None:
    client = _StreamClient([(0, _chunk("a")), *_repeat(ENVELOPE, 500)], clock)
    with pytest.raises(StreamStallError, match=r"no new content for 120\.0s"):
        list(
            client.sse_post(
                "/x", {}, max_total_seconds=1800, stall_seconds=120, is_progress=blocks_changed()
            )
        )
    assert 120 < clock.now - 1000.0 <= 134


def test_heartbeats_never_reach_the_predicate(clock: _Clock) -> None:
    seen: list[Any] = []

    def _always(event: dict[str, Any]) -> bool:
        seen.append(event)
        return True

    client = _StreamClient([(0, _chunk("a")), *_heartbeats(200)], clock)
    with pytest.raises(StreamStallError):
        list(client.sse_post("/x", {}, stall_seconds=120, is_progress=_always))
    assert len(seen) == 1


def test_blocks_changed_counts_only_new_blocks() -> None:
    is_progress = blocks_changed()
    first = {"data": {"blocks": [{"x": 1}]}}
    assert not is_progress({"data": {"status": "PENDING"}})
    assert not is_progress({"data": "raw"})
    assert is_progress(first)
    assert not is_progress({"data": {"blocks": [{"x": 1}], "status": "PENDING"}})
    assert is_progress({"data": {"blocks": [{"x": 2}]}})
    assert not is_progress(first)  # a replay of earlier blocks is not progress


def test_text_changed_counts_only_new_text() -> None:
    is_progress = _text_changed()
    assert not is_progress({"data": {"status": "PENDING"}})
    assert not is_progress({"data": {"text": None}})
    assert is_progress({"data": {"text": "a"}})
    assert not is_progress({"data": {"text": "a", "status": "PENDING"}})
    assert is_progress({"data": {"text": "ab"}})
    assert not is_progress({"data": {"text": "a"}})  # replayed snapshot


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
    with pytest.raises(StreamStallError, match=r"no new content for 120\.0s"):
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
    with pytest.raises(StreamStallError, match=r"no new content for 90\.0s"):
        list(client.sse_post("/x", {}, max_total_seconds=180))


def test_curl_timeout_after_the_deadline_passed_reports_the_deadline(clock: _Clock) -> None:
    # Last progress at 1700: the deadline (1800) comes due before the stall
    # (1940), and curl's abort, counted from the last byte, lands after both.
    client = _StreamClient(
        [
            (0, _chunk("a")),
            (1700, _chunk("b")),
            (200, _curl_error(CurlECode.OPERATION_TIMEDOUT)),
        ],
        clock,
    )
    with pytest.raises(StreamDeadlineError, match=r"exceeded 1800\.0s deadline") as exc:
        list(client.sse_post("/x", {}, max_total_seconds=1800, stall_seconds=240))
    assert not isinstance(exc.value, StreamStallError)


def test_curl_timeout_after_the_deadline_is_a_stall_when_the_stall_came_due_first(
    clock: _Clock,
) -> None:
    # Progress at 0, a heartbeat at 80, then silence: the stall was due at 240,
    # inside the 300 s deadline, though curl's abort only lands at ~326.
    client = _StreamClient(
        [
            (0, _chunk("a")),
            (80, HEARTBEAT),
            (246, _curl_error(CurlECode.OPERATION_TIMEDOUT)),
        ],
        clock,
    )
    with pytest.raises(StreamStallError):
        list(client.sse_post("/x", {}, max_total_seconds=300, stall_seconds=240))


def test_research_alternating_replayed_snapshots_still_stalls(clock: _Clock) -> None:
    steps: list[Step] = [(0, _snapshot("one")), (14, _snapshot("two"))]
    steps += [(14, _snapshot("one" if i % 2 else "two")) for i in range(40)]
    client = _StreamClient(steps, clock)
    result = research(client, "q", timeout=1800, stall_seconds=240)
    assert result.cut_by == "stall"


def test_other_curl_errors_stay_network_errors(clock: _Clock) -> None:
    client = _StreamClient([(0, _chunk("a")), (0, _curl_error(CurlECode.RECV_ERROR))], clock)
    with pytest.raises(NetworkError, match="mid-stream") as exc:
        list(client.sse_post("/x", {}, max_total_seconds=1800, stall_seconds=120))
    assert not isinstance(exc.value, StreamDeadlineError)


@pytest.mark.parametrize(
    ("stall", "deadline", "expected"),
    [
        (DEFAULT_STALL_SECONDS, 1800.0, (30.0, 210.0)),  # abort at the default window
        (120.0, 1800.0, (30.0, 90.0)),  # low-speed abort after ~120 s of silence
        (120.0, 50.0, (30.0, 20.0)),  # capped by the remaining deadline
        (10.0, None, (10.0, 0.0)),  # window shorter than the connect leg shrinks it
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
    client = _StreamClient([(0, _chunk("partial answer")), *_heartbeats(600)], clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q", *LONG_DEADLINE], client)
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    assert "partial answer" in cap.out
    assert "stream: incomplete (stall: no new content)" in cap.out
    assert "stalled: no new content for 480.0s" in cap.err
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
    assert out["cut_by"] == "stall"
    assert any("stalled: no new content for 240.0s" in w for w in out["warnings"])


def test_stall_before_any_content_exits_network(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient(_heartbeats(600), clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q", *LONG_DEADLINE], client)
    cap = capsys.readouterr()
    assert rc == EXIT_NETWORK
    assert "no new content for 480.0s before the first content arrived" in cap.err


def test_research_repeating_its_snapshot_stalls_with_the_partial(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = _snapshot("partial report")
    client = _StreamClient([(0, frame), *_repeat(frame, 600)], clock)
    rc = _run_cli(monkeypatch, cli_research.main, ["q", "--json"], client)
    out = json.loads(capsys.readouterr().out)
    assert rc == EXIT_PARTIAL
    assert "partial report" in out["answer"]
    assert out["stream_complete"] is False
    assert any("stalled: no new content for 240.0s" in w for w in out["warnings"])
    assert 240 < clock.now - 1000.0 <= 254
    assert client.deleted == [("BU", "RW")]


PARTIAL_CHUNK = _chunk("partial answer")


@pytest.mark.parametrize(
    ("main", "argv", "filler"),
    [
        (cli_ask.main, ["q"], ENVELOPE),
        (cli_fetch.main, ["https://example.com", "--prompt", "p"], ENVELOPE),
        (cli_ask.main, ["q"], PARTIAL_CHUNK),
    ],
    ids=["ask-envelope", "fetch-envelope", "ask-repeated-blocks"],
)
def test_non_progress_frames_after_content_stall_with_the_partial(
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    main: Callable[[list[str]], int],
    argv: list[str],
    filler: bytes,
) -> None:
    client = _StreamClient([(0, PARTIAL_CHUNK), *_repeat(filler, 900)], clock)
    rc = _run_cli(monkeypatch, main, [*argv, *LONG_DEADLINE], client)
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    assert "partial answer" in cap.out
    assert "stalled: no new content for 480.0s" in cap.err
    assert 480 < clock.now - 1000.0 <= 494


def test_envelope_frames_before_content_exit_network(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient(_repeat(ENVELOPE, 900), clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q", *LONG_DEADLINE], client)
    cap = capsys.readouterr()
    assert rc == EXIT_NETWORK
    assert "no new content for 480.0s before the first content arrived" in cap.err


def test_ask_whose_blocks_keep_changing_runs_past_the_window(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    steps: list[Step] = []
    for i in range(5):
        steps += [(0, _chunk(f"part{i} ")), *_repeat(ENVELOPE, 200)]
    client = _StreamClient([*steps, (0, COMPLETED)], clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q", *LONG_DEADLINE], client)
    cap = capsys.readouterr()
    assert rc == 0
    assert "part4" in cap.out
    assert clock.now - 1000.0 > 900


def test_thinking_ask_that_answers_at_the_end_completes_under_defaults(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A thinking model on a long prompt: one plan block, ~300 s of envelope-only
    # frames, then the whole answer at once.
    plan = _frame({"backend_uuid": "BU", "read_write_token": "RW", "blocks": [{"plan": 1}]})
    steps: list[Step] = [
        (0, plan),
        *_repeat(ENVELOPE, 300),
        (0, _chunk("the answer")),
        (0, COMPLETED),
    ]
    client = _StreamClient(steps, clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q"], client)
    cap = capsys.readouterr()
    assert rc == 0
    assert "the answer" in cap.out
    assert clock.now - 1000.0 > 290


def test_stall_window_hook_tightens_the_window_mid_stream(clock: _Clock) -> None:
    window: list[float] = [120.0]
    client = _StreamClient([(0, _chunk("a")), *_heartbeats(200)], clock)
    with pytest.raises(StreamStallError) as exc:
        for _ in client.sse_post("/x", {}, stall_seconds=120, stall_window=lambda: window[0]):
            window[0] = 20.0
    assert exc.value.seconds == 20.0
    assert 20 < clock.now - 1000.0 <= 35


def _text_completed(text: str) -> bytes:
    return _frame(
        {
            "backend_uuid": "BU",
            "read_write_token": "RW",
            "status": "PENDING",
            "text_completed": True,
            "blocks": [{"intended_usage": "ask_text", "markdown_block": {"chunks": [text]}}],
        }
    )


@pytest.mark.parametrize("stall_seconds", [COPILOT_STALL_SECONDS, None])
def test_ask_silent_after_text_completed_returns_within_the_settle_window(
    clock: _Clock, stall_seconds: float | None
) -> None:
    client = _StreamClient(
        [(0, _chunk("the ")), (0, _text_completed("answer")), *_heartbeats(600)], clock
    )
    result = ask(client, "q", timeout=None, stall_seconds=stall_seconds)
    assert result.answer == "the answer"
    assert result.completion == FinishedWithoutSources()
    assert result.warnings == [SOURCES_FRAME_MISSING]
    # Heartbeats drive the check, so the cut lands within one of the window.
    assert COPILOT_SETTLE_SECONDS < clock.now - 1000.0 <= COPILOT_SETTLE_SECONDS + 15
    assert client.deleted == [("BU", "RW")]


def test_settle_window_counts_from_the_text_completed_frame(clock: _Clock) -> None:
    """A `text_completed` frame that repeats blocks already seen is not new
    content, yet it starts the settle window; the COMPLETED frame 5 s later
    must still be read even though the last new block is 30 s old."""
    client = _StreamClient(
        [
            (0, _chunk("the answer")),
            *_heartbeats(30),
            (0, _text_completed("the answer")),
            (5, COMPLETED),
        ],
        clock,
    )
    result = ask(client, "q", timeout=None, stall_seconds=COPILOT_STALL_SECONDS)
    assert result.completion == Finished()
    assert result.warnings == []


def test_ask_cli_silent_after_text_completed_exits_zero(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient([(0, _text_completed("the answer")), *_heartbeats(600)], clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q"], client)
    cap = capsys.readouterr()
    assert rc == 0
    assert "the answer" in cap.out
    assert "stream: incomplete" not in cap.out
    assert SOURCES_FRAME_MISSING in cap.err
    assert clock.now - 1000.0 <= COPILOT_SETTLE_SECONDS + 15


def test_ask_fully_silent_after_text_completed_ends_at_the_stall_window(clock: _Clock) -> None:
    """With no COMPLETED frame and no heartbeats nothing reaches the in-loop
    settle check, so curl's low-speed abort, sized to the stall window, ends the
    read; the whole answer is still returned."""
    client = _StreamClient(
        [
            (0, _chunk("the ")),
            (0, _text_completed("answer")),
            (COPILOT_STALL_SECONDS, _curl_error(CurlECode.OPERATION_TIMEDOUT)),
        ],
        clock,
    )
    result = ask(client, "q", timeout=None, stall_seconds=COPILOT_STALL_SECONDS)
    assert result.answer == "the answer"
    assert result.completion == FinishedWithoutSources()
    assert result.warnings == [SOURCES_FRAME_MISSING]
    assert clock.now - 1000.0 == COPILOT_STALL_SECONDS
    assert client.session.timeout[0] + client.session.timeout[1] == COPILOT_STALL_SECONDS


def test_research_whose_text_keeps_changing_runs_past_the_window(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    steps: list[Step] = []
    for i in range(5):
        frame = _snapshot(f"report v{i}")
        steps += [(0, frame), *_repeat(frame, 200)]
    client = _StreamClient([*steps, (0, COMPLETED)], clock)
    rc = _run_cli(monkeypatch, cli_research.main, ["q"], client)
    cap = capsys.readouterr()
    assert rc == 0
    assert "report v4" in cap.out
    assert clock.now - 1000.0 > 900


def test_curl_timeout_after_content_keeps_the_partial(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    steps: list[Step] = [(0, _chunk("kept")), (0, _curl_error(CurlECode.OPERATION_TIMEDOUT))]
    client = _StreamClient(steps, clock)
    argv = ["https://example.com", "--prompt", "p", *LONG_DEADLINE]
    rc = _run_cli(monkeypatch, cli_fetch.main, argv, client)
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
    client = _StreamClient([(60, _snapshot(f"report v{i}")) for i in range(70)], clock)
    rc = _run_cli(monkeypatch, cli_research.main, ["q"], client)
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    assert "report v" in cap.out
    assert "stream: incomplete (deadline)" in cap.out
    assert "s deadline" in cap.err
    assert "stalled" not in cap.err


def test_research_cut_with_only_the_initial_query_exits_network(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient([(0, _research_steps(INITIAL_QUERY)), *_heartbeats(300)], clock)
    rc = _run_cli(monkeypatch, cli_research.main, ["q"], client)
    cap = capsys.readouterr()
    assert rc == EXIT_NETWORK
    assert "no new content for 240.0s before the first content arrived" in cap.err
    assert "(no answer)" not in cap.out
    assert client.deleted == [("BU", "RW")]


def test_research_cut_with_sources_but_no_answer_is_a_partial(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = _research_steps(INITIAL_QUERY, SEARCH_RESULTS)
    client = _StreamClient([(0, frame), *_heartbeats(300)], clock)
    rc = _run_cli(monkeypatch, cli_research.main, ["q", "--json"], client)
    out = json.loads(capsys.readouterr().out)
    assert rc == EXIT_PARTIAL
    assert out["answer"] == ""
    assert [s["url"] for s in out["sources"]] == ["https://found"]
    assert out["cut_by"] == "stall"
    assert client.deleted == [("BU", "RW")]


def test_server_cut_names_no_bound(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _StreamClient([(0, _chunk("cut short"))], clock)
    rc = _run_cli(monkeypatch, cli_ask.main, ["q"], client)
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    assert "stream: incomplete (server cut)" in cap.out


@pytest.mark.parametrize(
    ("cutoff", "expected"),
    [
        (StreamStallError("x", 240), "stall"),
        (StreamDeadlineError("x"), "deadline"),
        (None, None),
    ],
    ids=["stall", "deadline", "none"],
)
def test_cutoff_cause_names_the_bound(
    cutoff: StreamDeadlineError | None, expected: str | None
) -> None:
    assert cutoff_cause(AskStreamState(cutoff=cutoff)) == expected


_Bound = Literal["stall", "deadline"]
_CUT_RESULTS: list[Callable[[_Bound | None], Any]] = [
    lambda cut_by: AskResult("q", "partial", "turbo", Cut(cut_by or "server")),
    lambda cut_by: ResearchResult("q", "partial", [], "research", False, cut_by=cut_by),
    lambda cut_by: FetchResult(
        url="https://example.com",
        title=None,
        domain="example.com",
        content="partial",
        is_extracted=True,
        stream_complete=False,
        cut_by=cut_by,
    ),
]
_RENDERERS = [
    (render_ask_text, render_ask_json),
    (render_research_text, render_research_json),
    (render_fetch_text, render_fetch_json),
]


@pytest.mark.parametrize("verb", range(3), ids=["ask", "research", "fetch"])
@pytest.mark.parametrize(
    ("cut_by", "marker"),
    [
        ("stall", "stream: incomplete (stall: no new content)"),
        ("deadline", "stream: incomplete (deadline)"),
        (None, "stream: incomplete (server cut)"),
    ],
    ids=["stall", "deadline", "server"],
)
def test_incomplete_marker_and_json_name_the_cause(
    verb: int, cut_by: _Bound | None, marker: str
) -> None:
    result = _CUT_RESULTS[verb](cut_by)
    render_text, render_json = _RENDERERS[verb]
    assert marker in render_text(result)
    j = render_json(result)
    assert j["stream_complete"] is False
    assert j["cut_by"] == cut_by


def test_no_content_messages_name_the_bound_that_fired() -> None:
    stall = no_content_error(
        label="ask", endpoint="/e", timeout=180, cutoff=StreamStallError("x", 120)
    )
    deadline = no_content_error(
        label="ask", endpoint="/e", timeout=180, cutoff=StreamDeadlineError("x")
    )
    assert isinstance(stall, StreamStallError)
    assert "no new content for 120.0s" in str(stall)
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
        ([], None, "default"),
        (["--stall-timeout", "45"], None, 45.0),
        (["--stall-timeout", "0"], None, None),
        (["--stall-timeout", "-1"], None, None),
        ([], "30", 30.0),
        ([], "0", None),
        (["--stall-timeout", "45"], "30", 45.0),
        ([], "soon", "default"),
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
    expected: float | str | None,
) -> None:
    if expected == "default":
        expected = DEFAULT_STALL_SECONDS if verb == "research" else COPILOT_STALL_SECONDS
    if env is None:
        monkeypatch.delenv("PPLX_STALL_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("PPLX_STALL_TIMEOUT", env)
    assert _captured_stall(monkeypatch, module, verb, argv + argv_extra) == expected
    capsys.readouterr()


def test_stall_timeout_ignored_in_plain_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Plain mode goes through fetch_plain, which has no stream bounds to pass.
    monkeypatch.setenv("PPLX_STALL_TIMEOUT", "30")
    seen: list[str] = []

    def _plain(url: str, **_kw: Any) -> FetchResult:
        seen.append(url)
        return FetchResult(url=url, title=None, domain="d", content="c", is_extracted=False)

    def _prompt(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("plain fetch must not take the --prompt path")

    monkeypatch.setattr(cli_fetch, "fetch_plain", _plain)
    monkeypatch.setattr(cli_fetch, "fetch", _prompt)
    assert cli_fetch.main(["https://example.com", "--stall-timeout", "45"]) == 0
    assert seen == ["https://example.com"]


def test_fetch_json_carries_warnings() -> None:
    result = FetchResult(
        url="u", title=None, domain="d", content="c", is_extracted=True, warnings=["w"]
    )
    assert render_fetch_json(result)["warnings"] == ["w"]


class _RetryThenDeadlineClient(_TestClientBase):
    """429 on the first attempt, then a deadline cut on the retry."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def sse_post(  # type: ignore[override]
        self, path: str, body: dict[str, Any], **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        self.calls += 1
        if self.calls == 1:
            raise RateLimitError("429", retry_after=0.0)
        raise StreamDeadlineError(f"SSE stream on {path} exceeded 170.0s deadline")
        yield {}  # pragma: no cover


def test_deadline_after_a_rate_limit_retry_names_the_callers_timeout() -> None:
    client = _RetryThenDeadlineClient()
    state = AskStreamState()
    run_ask_stream(
        client,
        "/x",
        {},
        state,
        on_event=lambda _e: None,
        timeout=180.0,
        stall_seconds=240.0,
        progress=False,
        label="ask",
    )
    assert client.calls == 2
    assert "exceeded 180.0s deadline" in str(state.cutoff)
