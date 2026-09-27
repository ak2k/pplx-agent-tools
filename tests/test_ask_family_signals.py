"""What the ask-family verbs tell a caller when a result is not what was asked
for: a first-content or silence cut, a dropped connection, a model the server
swapped, clarifying questions nobody answered. Plus the request flags that
keep a run from waiting on the web UI.

Streams run through the real `Client.sse_post` on a fake clock (see
test_stall_guard), or replay committed fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from curl_cffi import CurlECode

from pplx_agent_tools import cli_ask, cli_fetch, cli_research, wire
from pplx_agent_tools.errors import EXIT_NETWORK, EXIT_PARTIAL, StreamDeadlineError
from pplx_agent_tools.render import (
    render_ask_json,
    render_ask_text,
    render_fetch_json,
    render_fetch_text,
    render_research_json,
    render_research_text,
)
from pplx_agent_tools.verbs._ask_common import (
    COPILOT_STALL_SECONDS,
    DEFAULT_STALL_SECONDS,
    no_content_error,
)
from pplx_agent_tools.verbs.ask import ask
from pplx_agent_tools.verbs.fetch import fetch
from pplx_agent_tools.verbs.research import research

from ._doubles import _TestClientBase
from .test_stall_guard import (
    ENVELOPE,
    HEARTBEAT,
    INITIAL_QUERY,
    LONG_DEADLINE,
    Step,
    _chunk,
    _Clock,
    _curl_error,
    _frame,
    _heartbeats,
    _run_cli,
    _snapshot,
    _StreamClient,
)

FIXTURES = Path(__file__).parent / "fixtures"
# Pinned, not imported: SKILL.md documents 90 s. Heartbeats every 15 s drive the check.
FIRST_CONTENT_S = 90.0

_CLI: dict[str, tuple[Callable[[list[str]], int], list[str]]] = {
    "ask": (cli_ask.main, ["q"]),
    "research": (cli_research.main, ["q"]),
    "fetch": (cli_fetch.main, ["https://example.com", "--prompt", "p"]),
}


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(wire, "time", SimpleNamespace(monotonic=c.monotonic, time=lambda: 1.7e9))
    return c


def _content(verb: str, text: str) -> bytes:
    return _snapshot(text) if verb == "research" else _chunk(text)


def _cli_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
    steps: list[Step],
    clock: _Clock,
    *extra: str,
) -> tuple[int, dict[str, Any], str, _StreamClient]:
    main, argv = _CLI[verb]
    client = _StreamClient(steps, clock)
    rc = _run_cli(monkeypatch, main, [*argv, *LONG_DEADLINE, "--json", *extra], client)
    cap = capsys.readouterr()
    return rc, json.loads(cap.out), cap.err, client


# ---------- first content ----------


@pytest.mark.parametrize("verb", _CLI)
def test_no_first_content_is_cut_at_the_bound_with_its_own_message(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    steps: list[Step] = [(0, ENVELOPE), *_heartbeats(600)]
    rc, out, err, client = _cli_json(monkeypatch, capsys, verb, steps, clock)
    assert rc == EXIT_NETWORK
    assert out["error"]["exit_code"] == EXIT_NETWORK
    assert f"no first content within {FIRST_CONTENT_S:.1f}s" in err
    assert "stall" not in err
    assert FIRST_CONTENT_S < clock.now - 1000.0 <= FIRST_CONTENT_S + 15
    # ENVELOPE names no context uuid, so pplx cannot stop the run, and research
    # keeps its thread for `pplx resume` rather than delete it.
    kept = verb == "research"
    assert client.deleted == ([] if kept else [("BU", "RW")])
    assert (out.get("resume") == "pplx resume --profile default BU") == kept


# ---------- deadline before content ----------


def _non_content_progress(verb: str, i: int) -> bytes:
    """A frame the verb's progress rule counts that still carries no content."""
    ids = {"backend_uuid": "BU", "read_write_token": "RW"}
    if verb == "research":
        step = {"step_type": "INITIAL_QUERY", "content": {"query": f"q{i}"}}
        return _frame({**ids, "text": json.dumps([step])})
    return _frame({**ids, "blocks": [{"plan": i}]})


@pytest.mark.parametrize("verb", _CLI)
@pytest.mark.parametrize(
    ("scenario", "timeout", "expected"),
    [
        ("working", "100", "the last progress event came 20.0s before the cut"),
        ("echo only", "60", "the last progress event came 60.0s before the cut"),
        ("none", "60", "no progress event arrived"),
    ],
)
def test_a_deadline_before_content_says_when_progress_last_came(
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
    scenario: str,
    timeout: str,
    expected: str,
) -> None:
    if scenario == "working":
        steps: list[Step] = [
            (0 if i == 0 else 20, _non_content_progress(verb, i)) for i in range(10)
        ]
    elif scenario == "echo only":
        steps = [(0, _non_content_progress(verb, 0)), *_heartbeats(300)]
    else:
        steps = [(0, ENVELOPE), *_heartbeats(300)]
    rc, out, err, _ = _cli_json(monkeypatch, capsys, verb, steps, clock, "--timeout", timeout)
    assert rc == EXIT_NETWORK
    assert out["error"]["exit_code"] == EXIT_NETWORK
    assert f"exceeded {float(timeout):.1f}s deadline before the first content arrived" in err
    assert expected in err


@pytest.mark.parametrize("verb", _CLI)
@pytest.mark.parametrize(
    ("scenario", "advice"),
    [
        (
            "working",
            "the last progress event came 4.0s before the cut, so the run was still working: "
            "raise --timeout",
        ),
        (
            "stuck",
            "the last progress event came 60.0s before the cut, longer than a healthy run "
            "goes without one: retry once",
        ),
        ("none", "no progress event arrived: retry once"),
    ],
)
def test_a_deadline_before_content_says_which_retry_fits(
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
    scenario: str,
    advice: str,
) -> None:
    # Progress every 4 s, like a healthy thinking ask; "stuck" stops at 40 s.
    steps: list[Step] = [(0, _non_content_progress(verb, 0))]
    timeout = "100"
    if scenario == "working":
        steps += [(4, _non_content_progress(verb, i)) for i in range(1, 40)]
    elif scenario == "stuck":
        steps += [(4, _non_content_progress(verb, i)) for i in range(1, 11)]
        steps += _heartbeats(300)
    else:
        # Under the 90 s first-content bound, so the deadline is what cuts.
        steps, timeout = [(0, ENVELOPE), *_heartbeats(300)], "60"
    rc, out, err, _ = _cli_json(monkeypatch, capsys, verb, steps, clock, "--timeout", timeout)
    assert rc == EXIT_NETWORK
    assert advice in out["error"]["message"]
    assert advice in err


@pytest.mark.parametrize("verb", _CLI)
@pytest.mark.parametrize("ending", ["heartbeat", "silence"])
def test_a_deadline_before_content_counts_the_gap_to_when_the_cut_is_seen(
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
    ending: str,
) -> None:
    # The last progress comes 20 s before the 100 s deadline, but the cut is
    # seen only at the next heartbeat or at the transport's silence abort.
    steps: list[Step] = [(0, _non_content_progress(verb, 0)), (80, _non_content_progress(verb, 1))]
    if ending == "heartbeat":
        gap = 30.0
        steps += [(15, HEARTBEAT), (15, HEARTBEAT)]
    else:
        gap = 90.0 if verb == "research" else 45.0
        steps.append((gap, _curl_error(CurlECode.OPERATION_TIMEDOUT)))
    rc, out, err, _ = _cli_json(monkeypatch, capsys, verb, steps, clock, "--timeout", "100")
    advice = (
        f"the last progress event came {gap:.1f}s before the cut, longer than a healthy run "
        "goes without one: retry once"
    )
    assert rc == EXIT_NETWORK
    assert out["error"]["type"] == "StreamDeadlineError"
    assert advice in out["error"]["message"]
    assert advice in err


# Pinned, not imported: SKILL.md documents the 20 s turn.
@pytest.mark.parametrize(
    ("since", "advice"),
    [
        (None, "retry once"),
        (0.0, "raise --timeout"),
        (20.0, "raise --timeout"),
        (20.1, "retry once"),
        (300.0, "retry once"),
    ],
)
def test_the_deadline_advice_turns_past_twenty_seconds_without_progress(
    since: float | None, advice: str
) -> None:
    err = no_content_error(
        label="ask", endpoint="/e", timeout=100.0, cutoff=StreamDeadlineError("cut", since)
    )
    assert type(err) is StreamDeadlineError
    assert str(err).endswith(advice)


# ---------- silence ----------


@pytest.mark.parametrize("verb", _CLI)
def test_silence_after_content_is_a_stall_cut_that_says_no_bytes(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    steps: list[Step] = [
        (0, _content(verb, "kept")),
        (40, _curl_error(CurlECode.OPERATION_TIMEDOUT)),
    ]
    rc, out, err, client = _cli_json(monkeypatch, capsys, verb, steps, clock)
    connect, read = client.session.timeout
    stall = DEFAULT_STALL_SECONDS if verb == "research" else COPILOT_STALL_SECONDS
    assert rc == EXIT_PARTIAL
    assert out["cut_by"] == "stall"
    assert any(f"no bytes for {connect + read:.1f}s" in w for w in out["warnings"])
    assert "no bytes for" in err
    # The transport's silence abort is its own window, far inside the stall window.
    assert connect + read <= stall / 2


@pytest.mark.parametrize("verb", _CLI)
def test_a_silence_cut_names_itself_on_stdout(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    main, argv = _CLI[verb]
    steps: list[Step] = [
        (0, _content(verb, "kept")),
        (40, _curl_error(CurlECode.OPERATION_TIMEDOUT)),
    ]
    client = _StreamClient(steps, clock)
    rc = _run_cli(monkeypatch, main, [*argv, *LONG_DEADLINE], client)
    out = capsys.readouterr().out
    connect, read = client.session.timeout
    assert rc == EXIT_PARTIAL
    assert f"stream: incomplete (stall: no bytes for {connect + read:.1f}s)" in out
    assert "no new content" not in out


# Pinned, not imported: SKILL.md documents these windows.
@pytest.mark.parametrize(("verb", "window"), [("ask", 45.0), ("fetch", 45.0), ("research", 90.0)])
def test_research_alone_gets_a_silence_window_past_a_clarifying_wait(
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verb: str,
    window: float,
) -> None:
    steps: list[Step] = [
        (0, _content(verb, "kept")),
        (40, _curl_error(CurlECode.OPERATION_TIMEDOUT)),
    ]
    _, out, _, client = _cli_json(monkeypatch, capsys, verb, steps, clock)
    connect, read = client.session.timeout
    assert connect + read == window
    assert any(f"no bytes for {window:.1f}s" in w for w in out["warnings"])


@pytest.mark.parametrize("verb", _CLI)
def test_silence_before_content_exits_network_saying_no_bytes(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    steps: list[Step] = [(0, ENVELOPE), (40, _curl_error(CurlECode.OPERATION_TIMEDOUT))]
    rc, out, err, _ = _cli_json(monkeypatch, capsys, verb, steps, clock)
    assert rc == EXIT_NETWORK
    assert out["error"]["exit_code"] == EXIT_NETWORK
    assert "no bytes for" in err
    assert "before the first content arrived" in err


def test_research_that_sends_no_bytes_is_a_silence_cut_not_a_first_content_cut(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Research's silence window equals the first-content bound, so both come due at once.
    steps: list[Step] = [(90, _curl_error(CurlECode.OPERATION_TIMEDOUT))]
    rc, out, err, _ = _cli_json(monkeypatch, capsys, "research", steps, clock)
    assert rc == EXIT_NETWORK
    assert out["error"]["type"] == "StreamSilenceError"
    assert "no bytes for 90.0s before the first content arrived" in out["error"]["message"]
    assert "no first content" not in err


# ---------- drop salvage ----------


@pytest.mark.parametrize("verb", _CLI)
def test_a_drop_after_content_returns_the_partial_as_a_drop_cut(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    steps: list[Step] = [(0, _content(verb, "kept part")), (0, _curl_error(CurlECode.RECV_ERROR))]
    rc, out, _, client = _cli_json(monkeypatch, capsys, verb, steps, clock)
    assert rc == EXIT_PARTIAL
    assert out["stream_complete"] is False
    assert out["cut_by"] == "drop"
    assert "kept part" in json.dumps(out)
    assert any("failed mid-stream" in w for w in out["warnings"])
    # Research keeps the thread of a dropped run for `pplx resume`.
    assert client.deleted == ([] if verb == "research" else [("BU", "RW")])


@pytest.mark.parametrize("verb", _CLI)
def test_a_drop_after_content_names_itself_on_stdout(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    main, argv = _CLI[verb]
    steps: list[Step] = [(0, _content(verb, "kept part")), (0, _curl_error(CurlECode.RECV_ERROR))]
    rc = _run_cli(monkeypatch, main, [*argv, *LONG_DEADLINE], _StreamClient(steps, clock))
    assert rc == EXIT_PARTIAL
    assert "stream: incomplete (connection dropped)" in capsys.readouterr().out


@pytest.mark.parametrize("verb", _CLI)
def test_a_drop_before_content_still_exits_network(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    steps: list[Step] = [(0, ENVELOPE), (0, _curl_error(CurlECode.RECV_ERROR))]
    rc, out, err, _ = _cli_json(monkeypatch, capsys, verb, steps, clock)
    assert rc == EXIT_NETWORK
    assert out["error"]["type"] == "NetworkError"
    assert "mid-stream" in err


# ---------- downgrade ----------


class _Replay(_TestClientBase):
    """Yields events without a transport and records the request body."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        super().__init__()
        self._events = events
        self.bodies: list[dict[str, Any]] = []
        self.deleted: list[tuple[str, str]] = []

    def sse_post(  # type: ignore[override]
        self, path: str, body: dict[str, Any], **_bounds: Any
    ) -> Iterator[dict[str, Any]]:
        self.bodies.append(body)
        yield from self._events

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


def _fixture_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        is_event = isinstance(raw, dict) and set(raw) == {"event", "data"}
        events.append(raw if is_event else {"event": "message", "data": raw})
    return events


def _run_json(verb: str, client: _Replay, **kw: Any) -> dict[str, Any]:
    if verb == "ask":
        return render_ask_json(ask(client, "q", **kw))
    if verb == "research":
        return render_research_json(research(client, "q", **kw))
    return render_fetch_json(fetch(client, "https://example.com", prompt="p", **kw))


def _run_text(verb: str, client: _Replay) -> str:
    """What a text-mode caller that discards stderr sees."""
    if verb == "ask":
        return render_ask_text(ask(client, "q"))
    if verb == "research":
        return render_research_text(research(client, "q"))
    return render_fetch_text(fetch(client, "https://example.com", prompt="p"))


_FIXTURE_DOWNGRADE = [
    ("ask", "ask/multi-step-sources.events.jsonl", None),
    ("fetch", "fetch-url/example-com-prompt.events.jsonl", False),
    ("fetch", "fetch-url/multi-block-prompt.events.jsonl", False),
    ("fetch", "fetch-url/no-completed-marker.events.jsonl", None),
    ("fetch", "fetch-url/no-results-prompt.events.jsonl", False),
    ("fetch", "fetch-url/paywalled-article-prompt.events.jsonl", False),
    ("research", "research/ocio-fees-final-only.events.jsonl", False),
    ("research", "research/weather-nowcasting-apis.events.jsonl", False),
]


def test_the_downgrade_table_covers_every_stream_fixture() -> None:
    committed = {
        str(p.relative_to(FIXTURES))
        for d in ("ask", "fetch-url", "research")
        for p in (FIXTURES / d).glob("*.events.jsonl")
    }
    # The empty stream carries no frame, so it raises before any verdict.
    assert committed - {name for _, name, _ in _FIXTURE_DOWNGRADE} == {
        "fetch-url/empty-stream.events.jsonl"
    }


@pytest.mark.parametrize(("verb", "name", "expected"), _FIXTURE_DOWNGRADE)
def test_no_committed_fixture_reads_as_downgraded(
    verb: str, name: str, expected: bool | None
) -> None:
    out = _run_json(verb, _Replay(_fixture_events(FIXTURES / name)))
    assert out.get("downgraded", "absent") is expected
    assert not any("downgraded" in w for w in out.get("warnings", []))


_REQUESTED = {"ask": "turbo", "research": "pplx_alpha", "fetch": "turbo"}


def _served_by(verb: str, models: list[str | None]) -> list[dict[str, Any]]:
    """Frames whose `display_model` walks through `models`, then COMPLETED."""
    events: list[dict[str, Any]] = []
    for i, m in enumerate(models):
        data: dict[str, Any] = {"backend_uuid": "BU", "read_write_token": "RW", "status": "PENDING"}
        if m is not None:
            data["display_model"] = m
        if verb == "research":
            final = {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": f"r{i}"})}}
            data["text"] = json.dumps([final])
        else:
            data["blocks"] = [
                {"intended_usage": "ask_text", "markdown_block": {"chunks": [f"c{i} "]}}
            ]
        events.append({"data": data})
    events.append({"data": {"backend_uuid": "BU", "status": "COMPLETED"}})
    return events


@pytest.mark.parametrize("verb", _CLI)
def test_a_swapped_model_is_flagged_with_both_names(verb: str) -> None:
    requested = _REQUESTED[verb]
    out = _run_json(verb, _Replay(_served_by(verb, [requested, "sonar", None])))
    assert out.get("downgraded", "absent") is True
    assert any(requested in w and "sonar" in w for w in out.get("warnings", []))


@pytest.mark.parametrize("verb", _CLI)
def test_a_swapped_model_is_named_on_stdout(verb: str) -> None:
    requested = _REQUESTED[verb]
    text = _run_text(verb, _Replay(_served_by(verb, [requested, "sonar", None])))
    assert "model: downgraded (the server ran 'sonar')" in text


@pytest.mark.parametrize("verb", _CLI)
def test_the_requested_model_puts_no_model_line_on_stdout(verb: str) -> None:
    text = _run_text(verb, _Replay(_served_by(verb, [_REQUESTED[verb]])))
    assert "model: downgraded" not in text


@pytest.mark.parametrize("verb", _CLI)
def test_the_last_named_model_decides(verb: str) -> None:
    requested = _REQUESTED[verb]
    out = _run_json(verb, _Replay(_served_by(verb, ["sonar", requested, None])))
    assert out.get("downgraded", "absent") is False


def test_model_council_is_never_judged() -> None:
    out = _run_json(
        "research",
        _Replay(_served_by("research", ["gpt55_thinking"])),
        mode="council",
    )
    assert "downgraded" in out
    assert out["downgraded"] is None
    assert not any("downgraded" in w for w in out.get("warnings", []))


# ---------- clarifying questions ----------


def _fixture_questions(path: Path) -> list[str]:
    for event in _fixture_events(path):
        data = event["data"]
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str):
            continue
        for block in json.loads(text):
            if block.get("step_type") == "RESEARCH_CLARIFYING_QUESTIONS":
                return [q["question_text"] for q in block["content"]["questions"]]
    raise AssertionError("fixture has no clarifying step")


def test_the_captured_clarifying_questions_are_surfaced() -> None:
    path = FIXTURES / "research/ocio-fees-final-only.events.jsonl"
    expected = _fixture_questions(path)
    out = _run_json("research", _Replay(_fixture_events(path)))
    assert len(expected) == 3
    assert out.get("clarifying_questions") == expected
    assert any("default answers" in w and expected[0] in w for w in out.get("warnings", []))


def test_the_captured_clarifying_questions_reach_stdout() -> None:
    path = FIXTURES / "research/ocio-fees-final-only.events.jsonl"
    expected = _fixture_questions(path)
    text = _run_text("research", _Replay(_fixture_events(path)))
    assert "clarifying questions: unanswered" in text
    for question in expected:
        assert f"  - {question}" in text


def test_a_run_without_clarifying_questions_says_nothing_about_them() -> None:
    path = FIXTURES / "research/weather-nowcasting-apis.events.jsonl"
    out = _run_json("research", _Replay(_fixture_events(path)))
    assert out.get("clarifying_questions", "absent") == []
    assert not any("default answers" in w for w in out.get("warnings", []))
    assert "clarifying questions" not in _run_text("research", _Replay(_fixture_events(path)))


def _clarifying_run(content: Any) -> list[dict[str, Any]]:
    step = {"step_type": "RESEARCH_CLARIFYING_QUESTIONS", "content": content}
    final = {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": "report"})}}
    ids = {"backend_uuid": "BU", "read_write_token": "RW"}
    return [
        {"data": {**ids, "text": json.dumps([step, final])}},
        {"data": {"status": "COMPLETED"}},
    ]


@pytest.mark.parametrize(
    "content",
    [{"questions": [{"text": "Which region?"}], "auto_skip_seconds": 60}, {"questions": "?"}, None],
    ids=["no question_text", "questions not a list", "no content"],
)
def test_a_clarifying_step_whose_questions_cannot_be_read_still_warns(content: Any) -> None:
    out = _run_json("research", _Replay(_clarifying_run(content)))
    assert out.get("clarifying_questions") == []
    assert any("could not read" in w and "default answers" in w for w in out.get("warnings", []))
    text = _run_text("research", _Replay(_clarifying_run(content)))
    assert "clarifying questions: unanswered" in text


def test_malformed_clarifying_entries_are_skipped() -> None:
    step = {
        "step_type": "RESEARCH_CLARIFYING_QUESTIONS",
        "content": {
            "questions": [
                {"question_text": "Which region?", "options": ["EU", "US"]},
                {"question_text": ""},
                {"options": ["x"]},
                "not a dict",
                {"question_text": "Which year?"},
            ],
            "auto_skip_seconds": 60,
        },
    }
    final = {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": "report"})}}
    events = [
        {
            "data": {
                "backend_uuid": "BU",
                "read_write_token": "RW",
                "text": json.dumps([step, final]),
            }
        },
        {"data": {"status": "COMPLETED"}},
    ]
    out = _run_json("research", _Replay(events))
    assert out.get("clarifying_questions") == ["Which region?", "Which year?"]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            {
                "questions": [{"question_text": "Which region?"}, {"question_text": "Which year?"}],
                "auto_skip_seconds": 60,
            },
            ["asked clarifying questions", "Which region?", "Which year?"],
        ),
        (None, ["asked clarifying questions pplx could not read"]),
    ],
    ids=["readable", "unreadable"],
)
def test_a_run_cut_while_asking_clarifying_questions_names_them_in_the_error(
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: Any,
    expected: list[str],
) -> None:
    step = {"step_type": "RESEARCH_CLARIFYING_QUESTIONS", "content": content}
    ids = {
        "backend_uuid": "BU",
        "read_write_token": "RW",
        "context_uuid": "CTX",
        "display_model": "pplx_alpha",
    }
    steps: list[Step] = [
        (0, _frame({**ids, "text": json.dumps([INITIAL_QUERY, step])})),
        *_heartbeats(300),
    ]
    rc, out, err, client = _cli_json(
        monkeypatch, capsys, "research", steps, clock, "--timeout", "45"
    )
    message = out["error"]["message"]
    assert rc == EXIT_NETWORK
    assert out["error"]["type"] == "StreamDeadlineError"
    for fragment in expected:
        assert fragment in message
        assert fragment in err
    # pplx cut the run and stopped it, so it did not go on with default answers.
    assert "default answers" not in message
    assert client.terminated == [("BU", "CTX", "pplx_alpha")]


# ---------- UI-wait flags ----------


@pytest.mark.parametrize("verb", _CLI)
def test_the_ask_body_declines_ui_confirmations(verb: str) -> None:
    client = _Replay(_served_by(verb, [_REQUESTED[verb]]))
    _run_json(verb, client)
    params = client.bodies[0]["params"]
    assert params.get("should_ask_for_mcp_tool_confirmation", "absent") is False
    assert params.get("supports_tool_approval_modal", "absent") is False
