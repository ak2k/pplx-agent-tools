"""`pplx resume`: reattach to a research thread and read its report to the end.

A reconnect to a finished thread returns one COMPLETED frame whose `text` is
byte-equal to the live terminal's, then `end_of_stream`, so the resumed report
must equal what `research()` returns replaying the whole stream. A still
running thread sends snapshots until COMPLETED. The cleanup policy is
research's (delete after COMPLETED, keep on a drop, terminate and delete on a
stall), except that nothing is deleted or removed before the report is out.
"""

from __future__ import annotations

import io
import json
import os
import sys
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from pplx_agent_tools import cli, cli_resume, cli_runner
from pplx_agent_tools.errors import (
    EXIT_GENERIC,
    EXIT_NETWORK,
    EXIT_OK,
    EXIT_PARTIAL,
    NetworkError,
    SchemaError,
    StreamSilenceError,
    StreamStallError,
    ThreadGoneError,
)
from pplx_agent_tools.handles import ThreadHandle, ThreadRecord, ThreadStore, hash_prompt
from pplx_agent_tools.render import render_resume_json, render_resume_text
from pplx_agent_tools.verbs import _research_stream
from pplx_agent_tools.verbs.research import research
from pplx_agent_tools.verbs.resume import resume

from ._doubles import FakeTime, _TestClientBase
from ._driver import fixture_items
from .test_fixture_replay_research import (
    FIXTURES,
    SENTINEL_BACKEND_UUID,
    SENTINEL_RW_TOKEN,
    FixtureClient,
)

WEATHER = FIXTURES / "weather-nowcasting-apis.events.jsonl"
OCIO = FIXTURES / "ocio-fees-final-only.events.jsonl"
UUID = SENTINEL_BACKEND_UUID
TOKEN = SENTINEL_RW_TOKEN


@pytest.fixture(autouse=True)
def state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(home))
    monkeypatch.delenv("PPLX_PROFILE", raising=False)
    return home


@pytest.fixture(autouse=True)
def _fake_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_research_stream, "time", FakeTime())


def _payloads(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _completed(path: Path) -> dict[str, Any]:
    return next(p for p in _payloads(path) if p.get("status") == "COMPLETED")


class _Reconnect(_TestClientBase):
    """Answers `sse_reconnect` from a script; any ask would be a bug."""

    def __init__(self, frames: list[dict[str, Any]], *, then: BaseException | None = None) -> None:
        super().__init__()
        self._frames = frames
        self._then = then
        self.reconnected: list[str] = []
        self.deleted: list[tuple[str, str]] = []

    def sse_reconnect(  # type: ignore[override]
        self,
        backend_uuid: str,
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        is_progress: Callable[[dict[str, Any]], bool] | None = None,
        stall_window: Callable[[], float | None] | None = None,
        silence_seconds: float | None = None,
        first_content_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        self.reconnected.append(backend_uuid)
        for frame in self._frames:
            yield {"event": "message", "data": frame}
        if self._then is not None:
            raise self._then
        yield {"event": "end_of_stream", "data": {}}

    def sse_post(self, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        raise AssertionError("resume must not create a thread")

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


def _replayed(path: Path) -> tuple[str, list[str]]:
    result = research(FixtureClient(path), "q")
    assert result.stream_complete
    return result.answer, [s.url for s in result.sources]


def _save(status: str = "kept", *, token: str | None = TOKEN, age: float = 1.0, **kw: Any) -> None:
    started = datetime.now(timezone.utc) - timedelta(hours=age)
    ThreadStore().save(
        ThreadRecord(UUID, started, status, read_write_token=token, **kw)  # type: ignore[arg-type]
    )


def _status() -> str | None:
    record = ThreadStore().load(UUID)
    return record.status if record is not None else None


@pytest.mark.parametrize("fixture", [WEATHER, OCIO], ids=["weather", "ocio"])
def test_a_finished_thread_resumes_to_the_whole_report_and_is_deleted(fixture: Path) -> None:
    _save(mode="research", model="pplx_alpha")
    client = _Reconnect([_completed(fixture)])
    result, held = resume(client, UUID, store=ThreadStore())
    # Nothing goes before the caller has put the report out.
    assert client.deleted == [] and _status() == "running"
    held.release()
    assert client.deleted == [(UUID, TOKEN)]
    assert _status() is None
    answer, urls = _replayed(fixture)
    assert result.answer == answer
    assert [s.url for s in result.sources] == urls
    assert len(result.answer) > 5000 and urls
    assert result.stream_complete and result.resume is None
    assert result.mode == "research" and result.downgraded is False
    assert client.reconnected == [UUID]


def test_a_running_thread_resumes_through_snapshots_to_completed() -> None:
    payloads = _payloads(WEATHER)
    pending = next(p for p in payloads if p.get("status") == "PENDING" and "read_write_token" in p)
    client = _Reconnect([pending, _completed(WEATHER)])
    result, held = resume(client, UUID, store=ThreadStore())
    held.release()
    answer, urls = _replayed(WEATHER)
    assert result.answer == answer
    assert [s.url for s in result.sources] == urls
    assert client.deleted == [(UUID, TOKEN)]


def test_without_a_record_the_token_comes_from_the_frames() -> None:
    client = _Reconnect([_completed(WEATHER)])
    result, held = resume(client, UUID, store=ThreadStore())
    held.release()
    assert result.stream_complete
    assert client.deleted == [(UUID, TOKEN)]
    assert _status() is None
    assert result.query.startswith("Compare five weather nowcasting APIs")


def test_with_no_token_anywhere_a_warning_says_the_thread_was_not_deleted() -> None:
    _save(token=None)
    frame = {k: v for k, v in _completed(WEATHER).items() if k != "read_write_token"}
    client = _Reconnect([frame])
    result, held = resume(client, UUID, store=ThreadStore())
    held.release()
    assert client.deleted == []
    assert any("not deleted" in w for w in result.warnings)
    assert _status() is None


def test_keep_thread_leaves_a_finished_thread() -> None:
    _save()
    client = _Reconnect([_completed(WEATHER)])
    result, held = resume(client, UUID, store=ThreadStore(), keep_thread=True)
    held.release()
    assert result.stream_complete and client.deleted == []
    assert _status() is None


def test_a_resume_that_drops_keeps_the_thread_and_names_itself_again() -> None:
    _save()
    pending = _payloads(WEATHER)[2]
    client = _Reconnect([pending], then=NetworkError("SSE stream failed mid-stream: reset"))
    result, held = resume(client, UUID, store=ThreadStore())
    held.release()
    assert result.cut_by == "drop" and result.resume == f"pplx resume --profile default {UUID}"
    assert client.deleted == [] and client.terminated == []
    assert _status() == "kept"


def test_a_resume_that_stalls_terminates_and_deletes() -> None:
    _save()
    pending = _payloads(WEATHER)[2]
    client = _Reconnect([pending], then=StreamStallError("stalled", 60.0))
    # The driver's own stall timer decides the cut, so the bound must come due
    # before research's 90 s silence window.
    result, held = resume(client, UUID, store=ThreadStore(), stall_seconds=60.0)
    held.release()
    assert result.cut_by == "stall" and result.resume is None
    assert client.terminated == [(UUID, pending["context_uuid"], "pplx_alpha")]
    assert client.deleted == [(UUID, TOKEN)]
    assert _status() is None


def test_a_resume_stall_that_pplx_could_not_stop_keeps_the_thread() -> None:
    _save()
    pending = _payloads(WEATHER)[2]

    class _Refusing(_Reconnect):
        def terminate(self, entry_uuid: str, context_uuid: str, model_preference: str) -> bool:
            super().terminate(entry_uuid, context_uuid, model_preference)
            return False

    client = _Refusing([pending], then=StreamStallError("stalled", 60.0))
    result, held = resume(client, UUID, store=ThreadStore(), stall_seconds=60.0)
    held.release()
    assert result.cut_by == "stall" and result.resume == f"pplx resume --profile default {UUID}"
    assert len(client.terminated) == 1 and client.deleted == []
    assert any("may still be running on the server" in w for w in result.warnings)
    assert _status() == "kept"


def test_an_interrupt_during_resume_keeps_the_thread_and_prints_the_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    client = _Reconnect([_payloads(WEATHER)[2]], then=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        resume(client, UUID, store=ThreadStore())
    assert client.terminated == [] and client.deleted == []
    assert _status() == "kept"
    err = capsys.readouterr().err
    assert f"`pplx resume --profile default {UUID}`" in err
    assert TOKEN not in err


def test_any_error_during_resume_keeps_the_thread_and_names_the_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _save()
    client = _Reconnect([_payloads(WEATHER)[2]], then=SchemaError("SSE stream exceeded"))
    rc, out, err = _run(monkeypatch, capsys, [UUID, "-j"], client)
    assert rc == EXIT_GENERIC
    doc = json.loads(out)
    assert doc["error"]["type"] == "SchemaError"
    assert doc["resume"] == f"pplx resume --profile default {UUID}"
    assert f"`pplx resume --profile default {UUID}`" in err
    assert client.terminated == [] and client.deleted == []
    assert _status() == "kept"


def test_a_gone_thread_on_a_full_disk_is_not_offered_again(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pplx_agent_tools import handles

    _save()

    def full_disk(dest: Path, content: str) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(handles, "atomic_write_0600", full_disk)
    first = _Reconnect([], then=ThreadGoneError("thread gone (status 403)"))
    rc, _, _ = _run(monkeypatch, capsys, ["--last", "-j"], first)
    assert rc == EXIT_GENERIC and first.reconnected == [UUID]
    second = _Reconnect([], then=ThreadGoneError("thread gone (status 403)"))
    rc, out, _ = _run(monkeypatch, capsys, ["--last", "-j"], second)
    assert rc == EXIT_GENERIC and second.reconnected == []
    assert "no resumable research thread" in json.loads(out)["error"]["message"]


def test_a_gone_thread_loses_its_record_and_is_not_offered_again() -> None:
    _save()
    client = _Reconnect([], then=ThreadGoneError("thread gone"))
    with pytest.raises(ThreadGoneError):
        resume(client, UUID, store=ThreadStore())
    assert _status() is None
    assert ThreadStore().pick_last().record is None
    assert client.deleted == [] and client.terminated == []


class _FailsToOpen(_Reconnect):
    """Resume's first `failures` opens raise `first` before any event; the
    open after them streams `frames`."""

    def __init__(self, frames: list[dict[str, Any]], first: NetworkError, failures: int) -> None:
        super().__init__(frames)
        self._first = first
        self._failures = failures

    def sse_reconnect(self, backend_uuid: str, **kwargs: Any) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        if len(self.reconnected) < self._failures:
            self.reconnected.append(backend_uuid)
            raise self._first
        return super().sse_reconnect(backend_uuid, **kwargs)


@pytest.mark.parametrize(
    "first",
    [NetworkError("reset"), StreamSilenceError("went silent", 90.0)],
    ids=["drop", "silence"],
)
def test_a_first_open_that_fails_transiently_is_retried_like_a_later_drop(
    first: NetworkError,
) -> None:
    _save()
    client = _FailsToOpen([_completed(WEATHER)], first, failures=1)
    result, held = resume(client, UUID, store=ThreadStore())
    held.release()
    assert client.reconnected == [UUID, UUID]
    assert result.stream_complete and result.resume is None
    assert client.deleted == [(UUID, TOKEN)]


def test_first_opens_that_keep_failing_stop_at_the_reconnect_bound_and_keep_the_thread() -> None:
    _save()
    first = StreamSilenceError("went silent", 90.0)
    client = _FailsToOpen([_completed(WEATHER)], first, failures=99)
    with pytest.raises(StreamSilenceError):
        resume(client, UUID, store=ThreadStore())
    # The initial open and the three consecutive reconnects a drop may spend.
    assert client.reconnected == [UUID] * 4
    assert client.deleted == [] and _status() == "kept"


class _GoneOnReconnect(_Reconnect):
    """Resume's own open streams `frames` and drops; the in-call reconnect
    after it is refused as gone."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        super().__init__(frames, then=NetworkError("SSE stream failed mid-stream: reset"))

    def sse_reconnect(self, backend_uuid: str, **kwargs: Any) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        if self.reconnected:
            self.reconnected.append(backend_uuid)
            raise ThreadGoneError("thread gone on reconnect (status 403)")
        return super().sse_reconnect(backend_uuid, **kwargs)


def _p3() -> list[dict[str, Any]]:
    return [item["data"] for item in fixture_items("p3-research-initial")]


def test_a_thread_gone_on_the_in_call_reconnect_is_not_deleted_or_kept_after_the_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    client = _GoneOnReconnect(_p3())
    result, held = resume(client, UUID, store=ThreadStore())
    assert result.sources and result.resume is None
    assert _status() is None
    held.release()
    held.keep(OSError(32, "Broken pipe"))
    assert client.reconnected == [UUID, UUID]
    assert client.deleted == [] and client.terminated == []
    assert _status() is None
    assert "thread was kept" not in capsys.readouterr().err


def test_a_thread_gone_on_the_in_call_reconnect_with_nothing_read_says_re_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    client = _GoneOnReconnect(_p3()[:1])
    with pytest.raises(ThreadGoneError) as raised:
        resume(client, UUID, store=ThreadStore())
    assert raised.value.resume is None
    assert "re-run" in str(raised.value) and "pplx resume" not in str(raised.value)
    assert client.reconnected == [UUID, UUID]
    assert client.deleted == [] and client.terminated == []
    assert _status() is None
    assert "thread was kept" not in capsys.readouterr().err


def test_resume_renders_as_its_own_verb_and_names_the_query() -> None:
    result, _ = resume(_Reconnect([_completed(WEATHER)]), UUID, store=ThreadStore())
    doc = render_resume_json(result)
    assert doc["_verb"] == "resume"
    assert doc["answer"] == result.answer and doc["resume"] is None
    assert doc["query"] == result.query and result.query.startswith("Compare five weather")
    first, blank, rest = render_resume_text(result).split("\n", 2)
    assert first == f"query: {result.query}" and blank == ""
    assert rest.startswith(result.answer[:40])


# ---------- CLI ----------


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    client: Any,
) -> tuple[int, str, str]:
    monkeypatch.setattr(
        cli_runner.Client, "from_default_cookies", classmethod(lambda cls, **_: client)
    )
    rc = cli.main(["resume", *argv])
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def test_cli_resume_last_json_returns_the_report_without_the_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    other = "00000000-0000-4000-8000-00000000abcd"
    ThreadStore().save(ThreadRecord(other, datetime.now(timezone.utc) - timedelta(hours=3), "kept"))
    _save(age=1.0)
    client = _Reconnect([_completed(WEATHER)])
    rc, out, err = _run(monkeypatch, capsys, ["--last", "-j"], client)
    assert rc == EXIT_OK
    doc = json.loads(out)
    assert doc["_verb"] == "resume" and doc["stream_complete"] is True
    assert doc["answer"] == _replayed(WEATHER)[0]
    assert client.reconnected == [UUID]
    assert TOKEN not in out + err


def test_cli_resume_by_uuid_with_a_drop_exits_partial(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _Reconnect([_payloads(WEATHER)[2]], then=NetworkError("reset"))
    rc, out, err = _run(monkeypatch, capsys, [UUID, "--profile", "work"], client)
    assert rc == EXIT_PARTIAL
    command = f"pplx resume --profile work {UUID}"
    assert f"get the report with: {command}" in out
    assert command in err


def test_cli_resume_last_with_nothing_to_resume_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out, err = _run(monkeypatch, capsys, ["--last", "--json"], _Reconnect([]))
    assert rc == EXIT_GENERIC
    doc = json.loads(out)
    assert "no resumable research thread" in doc["error"]["message"]
    assert "no resumable research thread" in err


def test_cli_resume_last_skips_an_undecodable_record_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _save()
    store = ThreadStore()
    for name in ("e" * 32, "f" * 32):
        (store.directory / f"{name}.json").write_bytes(b"\xff\xfe\x00 not utf-8")
    client = _Reconnect([_completed(WEATHER)])
    rc, out, _ = _run(monkeypatch, capsys, ["--last", "-j"], client)
    assert rc == EXIT_OK and client.reconnected == [UUID]
    notes = [w for w in json.loads(out)["warnings"] if "could not be read" in w]
    assert len(notes) == 1 and "2 record file(s)" in notes[0]


def test_cli_resume_by_uuid_with_an_undecodable_record_resumes_without_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _save()
    ThreadStore()._path(UUID).write_bytes(b"\xff\xfe corrupt")
    client = _Reconnect([_completed(WEATHER)])
    rc, out, _ = _run(monkeypatch, capsys, [UUID, "-j"], client)
    assert rc == EXIT_OK and client.reconnected == [UUID]
    assert client.deleted == [(UUID, TOKEN)]
    assert any("record of this thread" in w for w in json.loads(out)["warnings"])
    assert _status() is None


def test_cli_resume_last_with_an_unreadable_state_dir_says_so_and_not_re_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _save()
    directory = ThreadStore().directory
    directory.chmod(0)
    try:
        client = _Reconnect([_completed(WEATHER)])
        rc, out, _ = _run(monkeypatch, capsys, ["--last", "-j"], client)
    finally:
        directory.chmod(0o700)
    assert rc == EXIT_GENERIC and client.reconnected == []
    error = json.loads(out)["error"]
    assert error["type"] == "ThreadRecordsError"
    assert str(directory) in error["message"] and "Permission denied" in error["message"]
    assert "no resumable" not in error["message"] and "re-run" not in error["message"]


def test_resume_by_uuid_with_an_unreadable_state_dir_never_says_no_token_was_recorded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save()
    directory = ThreadStore().directory
    frame = {k: v for k, v in _completed(WEATHER).items() if k != "read_write_token"}
    directory.chmod(0)
    try:
        result, _ = resume(_Reconnect([frame]), UUID, store=ThreadStore())
    finally:
        directory.chmod(0o700)
    assert any("could not be read" in w and "Permission denied" in w for w in result.warnings)
    assert not any("no read_write_token was recorded" in w for w in result.warnings)
    assert any("not deleted" in w for w in result.warnings)


def test_cli_resume_a_failed_session_check_on_a_403_is_reported_as_that(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pplx_agent_tools.wire import Client

    from .test_wire_reconnect import _ProbeFails

    _save()
    client = Client({"any": "cookie"})
    client._session = _ProbeFails("network")  # type: ignore[assignment]
    rc, out, _ = _run(monkeypatch, capsys, [UUID, "-j"], client)
    doc = json.loads(out)
    assert rc == EXIT_NETWORK and doc["error"]["type"] == "SessionCheckError"
    assert "403" in doc["error"]["message"]
    assert "goes on server-side" not in doc["error"]["message"]
    assert "going on server-side" not in doc["error"]["message"]
    assert _status() == "kept"


def test_cli_resume_gone_json_is_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _Reconnect([], then=ThreadGoneError("thread gone"))
    rc, out, _ = _run(monkeypatch, capsys, [UUID, "-j"], client)
    assert rc == EXIT_GENERIC
    assert json.loads(out)["error"]["type"] == "ThreadGoneError"


_IDS = {
    "backend_uuid": UUID,
    "read_write_token": TOKEN,
    "context_uuid": "CTX",
    "display_model": "pplx_alpha",
}
_QUERY_ONLY = json.dumps([{"step_type": "INITIAL_QUERY", "content": {"query": "q"}}])


@pytest.mark.parametrize(
    "frame",
    [
        {**_IDS, "status": "COMPLETED", "text": "not-json"},
        {**_IDS, "status": "PENDING", "text": _QUERY_ONLY},
    ],
    ids=["undecodable", "no-content"],
)
def test_cli_resume_with_no_report_to_print_keeps_the_thread_and_record(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], frame: dict[str, Any]
) -> None:
    _save()
    client = _Reconnect([frame])
    rc, out, err = _run(monkeypatch, capsys, [UUID, "-j"], client)
    command = f"pplx resume --profile default {UUID}"
    doc = json.loads(out)
    assert rc == EXIT_GENERIC and doc["error"]["type"] == "SchemaError"
    assert doc["resume"] == command and f"`{command}`" in err
    assert client.deleted == []
    assert _status() == "kept"
    assert TOKEN not in out + err


def test_cli_resume_an_interrupt_during_the_delete_comes_after_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _save()

    class _InterruptedDelete(_Reconnect):
        def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:
            super().delete_thread(entry_uuid, read_write_token)
            raise KeyboardInterrupt

    client = _InterruptedDelete([_completed(WEATHER)])
    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, capsys, [UUID, "-j"], client)
    cap = capsys.readouterr()
    assert client.deleted == [(UUID, TOKEN)]
    assert _status() is None and "thread was kept" not in cap.err
    assert json.loads(cap.out)["answer"] == _replayed(WEATHER)[0]


class _BrokenPipe(io.StringIO):
    def flush(self) -> None:
        raise BrokenPipeError(32, "Broken pipe")


@pytest.mark.parametrize("failure", ["interrupt", "broken-pipe"])
def test_cli_resume_keeps_the_thread_and_record_until_the_report_is_out(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    _save()
    if failure == "interrupt":

        def interrupted(result: Any) -> dict[str, Any]:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli_resume, "render_resume_json", interrupted)
    else:
        monkeypatch.setattr(sys, "stdout", _BrokenPipe())
    client = _Reconnect([_completed(WEATHER)])
    try:
        _, _, err = _run(monkeypatch, capsys, [UUID, "-j"], client)
    except KeyboardInterrupt:
        err = capsys.readouterr().err
    assert client.deleted == []
    assert _status() == "kept"
    assert f"`pplx resume --profile default {UUID}`" in err
    assert TOKEN not in err


@pytest.mark.parametrize(
    "argv",
    [[], [UUID, "--last"], ["../../rest/thread/list_recent"], ["short"]],
    ids=["neither", "both", "path", "short"],
)
def test_cli_resume_usage_errors_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    client = _Reconnect([])
    with pytest.raises(SystemExit) as ei:
        _run(monkeypatch, capsys, [*argv, "--json"], client)
    assert ei.value.code == EXIT_GENERIC
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "UsageError"
    assert client.reconnected == []


ALPHA, BETA = "alpha: which CDNs ship HTTP/3?", "beta: which CDNs ship QUIC?"
OTHER = "11111111-2222-4333-8444-555555555555"


class _ByUuid(_Reconnect):
    """Answers each thread with its own prompt and report."""

    def sse_reconnect(self, backend_uuid: str, **_kw: Any) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        self.reconnected.append(backend_uuid)
        prompt = BETA if backend_uuid == OTHER else ALPHA
        blocks = [
            {"step_type": "INITIAL_QUERY", "content": {"query": prompt}},
            {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": f"on {prompt}"})}},
        ]
        frame = {"backend_uuid": backend_uuid, "status": "COMPLETED", "text": json.dumps(blocks)}
        yield {"event": "message", "data": frame}
        yield {"event": "end_of_stream", "data": {}}


def _two_killed_runs() -> None:
    """A fan-out of two research runs, both killed: ALPHA, then BETA a
    second later, each recorded the way research records it."""
    for uuid, prompt in ((UUID, ALPHA), (OTHER, BETA)):
        ThreadHandle(ThreadStore(), prompt=prompt).observe(uuid, TOKEN)
    store = ThreadStore()
    for uuid, secs in ((UUID, 60), (OTHER, 59)):
        record = store.load(uuid)
        assert record is not None
        started = datetime.now(timezone.utc) - timedelta(seconds=secs)
        store.save(
            ThreadRecord(
                uuid,
                started,
                "running",
                read_write_token=TOKEN,
                prompt_sha256=record.prompt_sha256,
                pid=None,
            )
        )


def test_cli_resume_last_query_picks_the_run_with_that_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _two_killed_runs()
    client = _ByUuid([])
    rc, out, err = _run(monkeypatch, capsys, ["--last", "--query", ALPHA], client)
    assert rc == EXIT_OK and client.reconnected == [UUID]
    assert out.splitlines()[0] == f"query: {ALPHA}"
    assert f"on {ALPHA}" in out and BETA not in out
    assert "--query" not in err

    client = _ByUuid([])
    rc, out, _ = _run(monkeypatch, capsys, ["--last", "--query", BETA, "-j"], client)
    doc = json.loads(out)
    assert rc == EXIT_OK and client.reconnected == [OTHER] and doc["query"] == BETA


def test_cli_resume_last_among_several_warns_and_names_the_query(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _two_killed_runs()
    client = _ByUuid([])
    rc, out, _ = _run(monkeypatch, capsys, ["--last", "-j"], client)
    doc = json.loads(out)
    assert rc == EXIT_OK and client.reconnected == [OTHER] and doc["query"] == BETA
    [note] = [w for w in doc["warnings"] if "--query" in w]
    assert "2 resumable" in note and "uuid" in note


def test_cli_resume_last_query_with_no_match_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _two_killed_runs()
    client = _ByUuid([])
    rc, out, _ = _run(monkeypatch, capsys, ["--last", "--query", "gamma", "-j"], client)
    assert rc == EXIT_GENERIC and client.reconnected == []
    message = json.loads(out)["error"]["message"]
    assert "no resumable research thread" in message and "--query" in message


def _live_run(uuid: str, prompt: str, *, age: timedelta) -> None:
    """A research run another pplx process is still reading."""
    ThreadStore().save(
        ThreadRecord(
            uuid,
            datetime.now(timezone.utc) - age,
            "running",
            read_write_token=TOKEN,
            prompt_sha256=hash_prompt(prompt),
            pid=os.getppid(),
        )
    )


def test_cli_resume_last_without_query_does_not_pass_a_newer_live_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _save(age=1.0, prompt_sha256=hash_prompt(ALPHA))
    _live_run(OTHER, BETA, age=timedelta(minutes=5))
    client = _ByUuid([])
    rc, out, err = _run(monkeypatch, capsys, ["--last", "-j"], client)
    assert rc == EXIT_GENERIC
    assert client.reconnected == [] and client.deleted == []
    message = json.loads(out)["error"]["message"]
    assert "1 newer research run(s)" in message and "live pplx process" in message
    assert "--query" in message and "uuid" in message
    assert "no resumable research thread" not in message + err
    assert _status() == "kept"
    live = ThreadStore().load(OTHER)
    assert live is not None and live.status == "running"

    client = _ByUuid([])
    rc, out, _ = _run(monkeypatch, capsys, ["--last", "--query", ALPHA, "-j"], client)
    assert rc == EXIT_OK and client.reconnected == [UUID]
    assert json.loads(out)["query"] == ALPHA


@pytest.mark.parametrize("query", [[], ["--query", BETA]], ids=["last", "last-query"])
def test_cli_resume_last_with_only_live_runs_says_wait_and_not_re_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], query: list[str]
) -> None:
    _live_run(OTHER, BETA, age=timedelta(minutes=5))
    client = _ByUuid([])
    rc, out, err = _run(monkeypatch, capsys, ["--last", *query, "-j"], client)
    assert rc == EXIT_GENERIC and client.reconnected == []
    message = json.loads(out)["error"]["message"]
    assert "no resumable research thread" not in message + err
    assert "still being read by a live pplx process" in message
    assert "wait" in message and "uuid" in message
    live = ThreadStore().load(OTHER)
    assert live is not None and live.status == "running"


def test_the_query_hash_is_the_one_research_records() -> None:
    prompt = "  Compare HTTP/3 adoption\nacross CDNs, 2026 — ünïcode  "
    ThreadHandle(ThreadStore(), prompt=prompt).observe(UUID, None)
    record = ThreadStore().load(UUID)
    assert record is not None and record.prompt_sha256 == hash_prompt(prompt)
    pick = ThreadStore().pick_last(prompt_hash=hash_prompt(prompt))
    assert pick.record is not None and pick.record.backend_uuid == UUID


def test_cli_resume_query_without_last_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as ei:
        _run(monkeypatch, capsys, [UUID, "--query", ALPHA, "--json"], _Reconnect([]))
    assert ei.value.code == EXIT_GENERIC


def test_resume_is_listed_in_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    assert "resume" in capsys.readouterr().out
