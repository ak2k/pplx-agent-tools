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
from pplx_agent_tools.errors import EXIT_NETWORK, EXIT_PARTIAL
from pplx_agent_tools.render import render_ask_json, render_fetch_json, render_research_json
from pplx_agent_tools.verbs._ask_common import COPILOT_STALL_SECONDS, DEFAULT_STALL_SECONDS
from pplx_agent_tools.verbs.ask import ask
from pplx_agent_tools.verbs.fetch import fetch
from pplx_agent_tools.verbs.research import research

from ._doubles import _TestClientBase
from .test_stall_guard import (
    ENVELOPE,
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
FIRST_CONTENT_S = 90.0  # the bound the brief fixes; heartbeats every 15 s drive the check

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
    assert client.deleted == [("BU", "RW")]


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
    assert client.deleted == [("BU", "RW")]


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


def test_a_run_without_clarifying_questions_says_nothing_about_them() -> None:
    path = FIXTURES / "research/weather-nowcasting-apis.events.jsonl"
    out = _run_json("research", _Replay(_fixture_events(path)))
    assert out.get("clarifying_questions", "absent") == []
    assert not any("default answers" in w for w in out.get("warnings", []))


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


# ---------- UI-wait flags ----------


@pytest.mark.parametrize("verb", _CLI)
def test_the_ask_body_declines_ui_confirmations(verb: str) -> None:
    client = _Replay(_served_by(verb, [_REQUESTED[verb]]))
    _run_json(verb, client)
    params = client.bodies[0]["params"]
    assert params.get("should_ask_for_mcp_tool_confirmation", "absent") is False
    assert params.get("supports_tool_approval_modal", "absent") is False
