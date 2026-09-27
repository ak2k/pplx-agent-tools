"""Research keeps its thread when the connection drops or goes silent.

The run goes on server-side (a research run closed at 29 s finished with no
listener), so neither terminate nor delete is sent, the local record says
`kept`, and every channel an agent reads names the command that fetches the
finished report. Every other end still terminates and deletes.

Streams run through the real `Client.sse_post` on a fake clock.
"""

from __future__ import annotations

import contextlib
import errno
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from curl_cffi import CurlECode

from pplx_agent_tools import cli_research, handles, wire
from pplx_agent_tools.askstream.driver import ConnBounds
from pplx_agent_tools.errors import EXIT_NETWORK, EXIT_PARTIAL
from pplx_agent_tools.handles import ThreadStore
from pplx_agent_tools.render import render_research_json, render_research_text
from pplx_agent_tools.verbs import _research_stream
from pplx_agent_tools.verbs._ask_common import DEFAULT_STALL_SECONDS, AskStreamState
from pplx_agent_tools.verbs.research import DECODER, ENDPOINT, research

from ._doubles import CTX
from .test_stall_guard import (
    HEARTBEAT,
    LONG_DEADLINE,
    Step,
    _Clock,
    _curl_error,
    _frame,
    _run_cli,
    _StreamClient,
)

UUID = "0f0e0d0c-aaaa-4bbb-8ccc-123456789abc"
TOKEN = "SECRET-RW-TOKEN-7f3a"
IDS = {
    "backend_uuid": UUID,
    "read_write_token": TOKEN,
    "context_uuid": CTX,
    "display_model": "pplx_alpha",
}
RESUME = f"pplx resume --profile default {UUID}"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(wire, "time", SimpleNamespace(monotonic=c.monotonic, time=lambda: 1.7e9))
    monkeypatch.setattr(
        _research_stream, "time", SimpleNamespace(monotonic=c.monotonic, sleep=c.sleep)
    )
    return c


@pytest.fixture(autouse=True)
def state_home(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(home))
    monkeypatch.delenv("PPLX_PROFILE", raising=False)
    return home


def _text(answer: str) -> str:
    final = {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": answer})}}
    return json.dumps([final])


def _report(answer: str, **ids: Any) -> bytes:
    return _frame({**(ids or IDS), "status": "PENDING", "text": _text(answer)})


def _completed(answer: str) -> bytes:
    return _frame({**IDS, "status": "COMPLETED", "text": _text(answer)})


def _no_content(**ids: Any) -> bytes:
    step = {"step_type": "INITIAL_QUERY", "content": {"query": "q"}}
    return _frame({**(ids or IDS), "status": "PENDING", "text": json.dumps([step])})


DROP = _curl_error(CurlECode.RECV_ERROR)
SILENCE = _curl_error(CurlECode.OPERATION_TIMEDOUT)


def _status() -> str | None:
    record = ThreadStore().load(UUID)
    return record.status if record is not None else None


def _cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    steps: list[Step],
    clock: _Clock,
    *extra: str,
) -> tuple[int, str, str, _StreamClient]:
    client = _StreamClient(steps, clock)
    rc = _run_cli(monkeypatch, cli_research.main, ["q", *LONG_DEADLINE, *extra], client)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err, client


@pytest.mark.parametrize("cut", [DROP, SILENCE], ids=["drop", "silence"])
def test_a_cut_after_content_keeps_the_thread_and_names_the_resume_command(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cut: Any
) -> None:
    steps: list[Step] = [(0, _report("kept part")), (40, cut)]
    rc, out, err, client = _cli(monkeypatch, capsys, steps, clock, "--json")
    assert rc == EXIT_PARTIAL
    doc = json.loads(out)
    assert doc["resume"] == RESUME
    assert any(RESUME in w and "do not re-run" in w for w in doc["warnings"])
    assert "kept part" in doc["answer"]
    assert client.terminated == [] and client.deleted == []
    assert _status() == "kept"
    assert TOKEN not in out + err

    rc, out, err, _ = _cli(monkeypatch, capsys, [(0, _report("kept part")), (40, cut)], clock)
    assert rc == EXIT_PARTIAL
    assert f"thread kept: the run continues server-side; get the report with: {RESUME}" in out
    assert RESUME in err
    assert TOKEN not in out + err


@pytest.mark.parametrize("cut", [DROP, SILENCE], ids=["drop", "silence"])
def test_a_cut_before_content_exits_network_naming_the_resume_command(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cut: Any
) -> None:
    steps: list[Step] = [(0, _no_content()), (40, cut)]
    rc, out, err, client = _cli(monkeypatch, capsys, steps, clock, "--json")
    assert rc == EXIT_NETWORK
    doc = json.loads(out)
    assert doc["error"]["exit_code"] == EXIT_NETWORK
    assert RESUME in doc["error"]["message"]
    assert doc["resume"] == RESUME
    assert RESUME in err
    assert client.terminated == [] and client.deleted == []
    assert _status() == "kept"
    assert TOKEN not in out + err


def test_the_resume_command_names_a_non_default_profile(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    steps: list[Step] = [(0, _report("kept part")), (0, DROP)]
    _, out, _, _ = _cli(monkeypatch, capsys, steps, clock, "--json", "--profile", "work")
    doc = json.loads(out)
    assert doc["resume"] == f"pplx resume --profile work {UUID}"
    record = ThreadStore("work").load(UUID)
    assert record is not None and record.status == "kept"
    assert ThreadStore().load(UUID) is None


def test_the_resume_command_names_the_default_profile_over_the_environment(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PPLX_PROFILE", "work")
    steps: list[Step] = [(0, _report("kept part")), (0, DROP)]
    _, out, _, _ = _cli(monkeypatch, capsys, steps, clock, "--json", "--profile", "default")
    assert json.loads(out)["resume"] == f"pplx resume --profile default {UUID}"
    record = ThreadStore("default").load(UUID)
    assert record is not None and record.status == "kept"


def test_a_drop_before_the_token_still_leaves_a_resumable_record(clock: _Clock) -> None:
    no_token = {k: v for k, v in IDS.items() if k != "read_write_token"}
    client = _StreamClient([(0, _report("part", **no_token)), (0, DROP)], clock)
    result = research(client, "q")
    assert result.resume == RESUME
    record = ThreadStore().load(UUID)
    assert record is not None
    assert record.status == "kept" and record.read_write_token is None
    assert ThreadStore().pick_last().record == record


def test_the_record_is_written_at_the_first_frame_and_gains_the_token(clock: _Clock) -> None:
    no_token = {k: v for k, v in IDS.items() if k != "read_write_token"}
    seen: list[Any] = []

    class _Watching(_StreamClient):
        def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:
            seen.append(ThreadStore().load(UUID))
            return super().delete_thread(entry_uuid, read_write_token)

    steps: list[Step] = [
        (0, _no_content(**no_token)),
        (0, _report("whole")),
        (0, _completed("whole")),
    ]
    client = _Watching(steps, clock)
    result = research(client, "q")
    assert result.stream_complete and result.resume is None
    [before_delete] = seen
    assert before_delete.status == "running" and before_delete.read_write_token == TOKEN
    assert _status() is None


@pytest.mark.parametrize("end", ["deadline", "stall", "interrupt"])
def test_every_other_cut_still_terminates_and_deletes(clock: _Clock, end: str) -> None:
    tail: list[Step] = {
        "deadline": [(10, _report(f"more {i}")) for i in range(20)],
        "stall": [(15, HEARTBEAT) for _ in range(int(DEFAULT_STALL_SECONDS // 15) + 4)],
        "interrupt": [(0, KeyboardInterrupt())],
    }[end]
    client = _StreamClient([(0, _report("part")), *tail], clock)
    kwargs: dict[str, Any] = {"timeout": 100.0 if end == "deadline" else None}
    if end == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            research(client, "q", **kwargs)
    else:
        result = research(client, "q", stall_seconds=DEFAULT_STALL_SECONDS, **kwargs)
        assert result.resume is None
        assert result.cut_by == ("deadline" if end == "deadline" else "stall")
        assert not any("pplx resume" in w for w in result.warnings)
    assert client.terminated == [(UUID, CTX, "pplx_alpha")]
    assert client.deleted == [(UUID, TOKEN)]
    assert _status() is None


def test_keep_thread_on_a_completed_run_leaves_no_resumable_record(clock: _Clock) -> None:
    client = _StreamClient([(0, _completed("whole"))], clock)
    result = research(client, "q", keep_thread=True)
    assert result.resume is None
    assert client.deleted == []
    assert _status() is None
    assert ThreadStore().pick_last().record is None


def test_an_unwritable_state_dir_is_a_warning_and_a_full_result(
    clock: _Clock, state_home: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    state_home.mkdir(parents=True)
    (state_home / "perplexity").write_text("not a directory")
    client = _StreamClient([(0, _completed("whole"))], clock)
    result = research(client, "q")
    assert result.stream_complete and result.answer == "whole"
    assert [w for w in result.warnings if "could not record" in w] == result.warnings
    assert len(result.warnings) == 1
    assert client.deleted == [(UUID, TOKEN)]
    rendered = json.dumps(render_research_json(result)) + render_research_text(result)
    assert TOKEN not in rendered + capsys.readouterr().err


def test_an_unwritable_state_dir_on_a_kept_run_still_names_the_resume_command(
    clock: _Clock, state_home: Any
) -> None:
    state_home.mkdir(parents=True)
    (state_home / "perplexity").write_text("not a directory")
    client = _StreamClient([(0, _report("part")), (0, DROP)], clock)
    result = research(client, "q")
    assert result.resume == RESUME
    assert any("could not record" in w for w in result.warnings)
    assert any(RESUME in w for w in result.warnings)


def _interrupt(seconds: float) -> None:
    raise KeyboardInterrupt


def test_an_interrupt_while_reconnecting_after_a_drop_deletes_and_is_not_offered_to_last(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ctrl-C in the backoff before the first reconnect.
    monkeypatch.setattr(_research_stream.time, "sleep", _interrupt)
    client = _StreamClient([(0, _report("part")), (0, DROP)], clock)
    with pytest.raises(KeyboardInterrupt):
        research(client, "q")
    assert client.deleted == [(UUID, TOKEN)]
    assert ThreadStore().pick_last().record is None


@pytest.mark.parametrize("raised", [False, True])
def test_cleanup_records_its_keep_decision_in_the_state(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, raised: bool
) -> None:
    if raised:
        monkeypatch.setattr(_research_stream.time, "sleep", _interrupt)
    client = _StreamClient([(0, _report("part")), (0, DROP)], clock)

    def post(b: ConnBounds) -> Iterator[dict[str, Any]]:
        return client.sse_post(
            ENDPOINT,
            {},
            max_total_seconds=b.max_total_seconds,
            stall_seconds=b.stall_seconds,
            silence_seconds=b.silence_seconds,
        )

    state = AskStreamState()
    with contextlib.suppress(KeyboardInterrupt):
        _research_stream.research_stream(client, post, DECODER, endpoint=ENDPOINT, state=state)
    assert (state.kept, state.deleted) == ((False, True) if raised else (True, False))


class _RefusingTerminate(_StreamClient):
    def terminate(self, entry_uuid: str, context_uuid: str, model_preference: str) -> bool:
        super().terminate(entry_uuid, context_uuid, model_preference)
        return False


@pytest.mark.parametrize("terminate", ["refused", "unsendable"])
def test_a_stall_cut_that_pplx_could_not_stop_keeps_the_thread(
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    terminate: str,
) -> None:
    ids = IDS if terminate == "refused" else {k: v for k, v in IDS.items() if k != "context_uuid"}
    stall = [(15.0, HEARTBEAT) for _ in range(int(DEFAULT_STALL_SECONDS // 15) + 4)]
    steps: list[Step] = [(0, _report("part", **ids)), *stall]
    client = (_RefusingTerminate if terminate == "refused" else _StreamClient)(steps, clock)
    rc = _run_cli(monkeypatch, cli_research.main, ["q", *LONG_DEADLINE, "--json"], client)
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    doc = json.loads(cap.out)
    assert doc["cut_by"] == "stall" and doc["resume"] == RESUME
    assert any("may still be running on the server" in w for w in doc["warnings"])
    assert any(RESUME in w and "do not re-run" in w for w in doc["warnings"])
    assert client.deleted == []
    assert _status() == "kept"
    assert TOKEN not in cap.out + cap.err


def test_a_first_content_cut_that_pplx_could_not_stop_names_the_resume_command(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    no_model = {k: v for k, v in IDS.items() if k != "display_model"}
    steps: list[Step] = [(0, _frame({**no_model, "status": "PENDING"}))]
    steps += [(15.0, HEARTBEAT) for _ in range(14)]
    rc, out, _, client = _cli(monkeypatch, capsys, steps, clock, "--json")
    assert rc == EXIT_NETWORK
    doc = json.loads(out)
    assert "no first content" in doc["error"]["message"]
    assert doc["resume"] == RESUME and RESUME in doc["error"]["message"]
    assert client.terminated == [] and client.deleted == []
    assert _status() == "kept"


def test_a_finished_run_leaves_no_record_and_no_token_on_disk(clock: _Clock) -> None:
    client = _StreamClient([(0, _completed("whole"))], clock)
    research(client, "q")
    assert client.deleted == [(UUID, TOKEN)]
    directory = ThreadStore().directory
    assert list(directory.iterdir()) == []


def test_a_settle_after_writes_start_failing_still_removes_the_record(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = handles.atomic_write_0600
    calls = {"n": 0}

    def fails_after_first(dest: Any, content: str) -> None:
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        real(dest, content)

    monkeypatch.setattr(handles, "atomic_write_0600", fails_after_first)
    result = research(_StreamClient([(0, _completed("whole"))], clock), "q")
    monkeypatch.setattr(handles, "_alive", lambda pid: False)
    assert ThreadStore().load(UUID) is None
    assert ThreadStore().pick_last().record is None
    assert not any("could not record" in w for w in result.warnings)


def test_a_record_that_cannot_be_removed_is_one_warning_saying_so(
    clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cannot_remove(self: ThreadStore, backend_uuid: str) -> None:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(ThreadStore, "remove", cannot_remove)
    result = research(_StreamClient([(0, _completed("whole"))], clock), "q")
    assert len(result.warnings) == 1
    assert "could not remove" in result.warnings[0]
    assert "Permission denied" in result.warnings[0]
    assert "could not record" not in result.warnings[0]
