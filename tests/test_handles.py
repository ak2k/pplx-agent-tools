"""The local record of each research thread, which `pplx resume` reads.

Round trip, file modes, atomic writes, many writers at once, pruning past the
thread's expiry, `--last` selection, and a state dir that cannot be written.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pplx_agent_tools import handles
from pplx_agent_tools.handles import ThreadHandle, ThreadRecord, ThreadStore

UUID = "0f0e0d0c-aaaa-4bbb-8ccc-123456789abc"
TOKEN = "SECRET-RW-TOKEN"


@pytest.fixture
def state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(home))
    monkeypatch.delenv("PPLX_PROFILE", raising=False)
    return home


def _uuid(i: int) -> str:
    return f"00000000-0000-4000-8000-{i:012d}"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_store_lives_under_xdg_state_scoped_by_profile(
    state_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ThreadStore().directory == state_home / "perplexity" / "default" / "threads"
    assert ThreadStore("work").directory == state_home / "perplexity" / "work" / "threads"
    monkeypatch.setenv("PPLX_PROFILE", "env")
    assert ThreadStore().directory == state_home / "perplexity" / "env" / "threads"


def test_default_state_home_is_dot_local_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert (
        ThreadStore("p").directory == tmp_path / ".local" / "state" / "perplexity" / "p" / "threads"
    )


def test_round_trip_with_file_0600_in_dir_0700(state_home: Path) -> None:
    store = ThreadStore()
    started = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
    record = ThreadRecord(
        backend_uuid=UUID,
        started=started,
        status="running",
        read_write_token=TOKEN,
        prompt_sha256="ab" * 32,
        mode="research",
        model="pplx_alpha",
        pid=4242,
    )
    store.save(record, now=started)
    assert store.load(UUID) == record
    [path] = list(store.directory.iterdir())
    assert _mode(path) == 0o600
    assert _mode(store.directory) == 0o700
    assert UUID not in path.name


def test_a_loose_existing_directory_is_tightened(state_home: Path) -> None:
    store = ThreadStore()
    store.directory.mkdir(parents=True, mode=0o755)
    store.directory.chmod(0o755)
    store.save(ThreadRecord(UUID, datetime.now(timezone.utc), "running"))
    assert _mode(store.directory) == 0o700


def test_a_save_that_fails_before_the_rename_leaves_the_old_record_whole(
    state_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ThreadStore()
    now = datetime.now(timezone.utc)
    store.save(ThreadRecord(UUID, now, "running"))

    def no_rename(self: Path, target: Path) -> Path:
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m, pytest.raises(OSError):
        m.setattr(Path, "replace", no_rename)
        store.save(ThreadRecord(UUID, now, "kept", read_write_token=TOKEN))
    loaded = store.load(UUID)
    assert loaded is not None and loaded.status == "running"
    assert [p.suffix for p in store.directory.iterdir()] == [".json"]


def test_unreadable_records_are_skipped(state_home: Path) -> None:
    store = ThreadStore()
    store.save(ThreadRecord(UUID, datetime.now(timezone.utc), "running"))
    (store.directory / "junk.json").write_text("{not json")
    (store.directory / "list.json").write_text("[]")
    (store.directory / "shape.json").write_text(json.dumps({"backend_uuid": 3}))
    assert [r.backend_uuid for r in store.records()] == [UUID]


def _write_many(home: str, start: int, count: int) -> None:
    os.environ["XDG_STATE_HOME"] = home
    store = ThreadStore()
    for i in range(start, start + count):
        store.save(ThreadRecord(_uuid(i), datetime.now(timezone.utc), "running"))


def test_concurrent_threads_lose_no_record(state_home: Path) -> None:
    store = ThreadStore()
    now = datetime.now(timezone.utc)

    def write(i: int) -> None:
        handle = ThreadHandle(store, prompt=f"q{i}")
        handle.observe(_uuid(i), None)
        handle.observe(_uuid(i), f"tok{i}")
        handle.keep()
        assert handle.warnings == []

    workers = [threading.Thread(target=write, args=(i,)) for i in range(20)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    records = {r.backend_uuid: r for r in store.records(now=now)}
    assert set(records) == {_uuid(i) for i in range(20)}
    assert all(r.status == "kept" and r.read_write_token for r in records.values())
    assert not list(store.directory.glob("*.tmp"))


def test_concurrent_processes_lose_no_record(state_home: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_write_many, args=(str(state_home), i * 5, 5)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert {r.backend_uuid for r in ThreadStore().records()} == {_uuid(i) for i in range(20)}


def test_records_past_25_hours_are_pruned_on_write(state_home: Path) -> None:
    store = ThreadStore()
    now = datetime(2026, 9, 26, 12, 0, 0, tzinfo=timezone.utc)
    store.save(ThreadRecord(_uuid(1), now - timedelta(hours=25, minutes=1), "kept"), now=now)
    store.save(ThreadRecord(_uuid(2), now - timedelta(hours=24, minutes=59), "kept"), now=now)
    stale_tmp = store.directory / "x.json.abc.tmp"
    stale_tmp.write_text("{}")
    old = (now - timedelta(hours=26)).timestamp()
    os.utime(stale_tmp, (old, old))
    store.save(ThreadRecord(_uuid(3), now, "running"), now=now)
    assert {r.backend_uuid for r in store.records(now=now)} == {_uuid(2), _uuid(3)}
    assert not stale_tmp.exists()


def test_last_picks_the_newest_resumable(state_home: Path) -> None:
    store = ThreadStore()
    now = datetime.now(timezone.utc)
    dead_pid = _dead_pid()
    store.save(ThreadRecord(_uuid(1), now - timedelta(hours=3), "kept"))
    store.save(ThreadRecord(_uuid(2), now - timedelta(hours=2), "running", pid=dead_pid))
    # Still streaming in a live process: not offered.
    store.save(ThreadRecord(_uuid(6), now - timedelta(minutes=10), "running", pid=os.getppid()))
    pick = store.pick_last()
    assert pick.record is not None and pick.record.backend_uuid == _uuid(2)
    assert pick.live == 1


def test_last_with_nothing_resumable(state_home: Path) -> None:
    store = ThreadStore()
    assert store.pick_last().record is None
    store.save(ThreadRecord(_uuid(1), datetime.now(timezone.utc), "kept"))
    store.remove(_uuid(1))
    assert store.pick_last().record is None


def test_profiles_do_not_see_each_others_records(state_home: Path) -> None:
    ThreadStore("a").save(ThreadRecord(_uuid(1), datetime.now(timezone.utc), "kept"))
    assert ThreadStore("b").pick_last().record is None
    pick = ThreadStore("a").pick_last()
    assert pick.record is not None


def test_handle_writes_at_the_first_id_and_adds_the_token_later(state_home: Path) -> None:
    store = ThreadStore()
    handle = ThreadHandle(store, prompt="what", mode="research", model="pplx_alpha")
    handle.observe(None, None)
    assert store.records() == []
    handle.observe(UUID, None)
    first = store.load(UUID)
    assert first is not None
    assert first.status == "running" and first.read_write_token is None
    assert first.pid == os.getpid()
    assert first.prompt_sha256 is not None and "what" not in json.dumps(first.prompt_sha256)
    handle.observe(UUID, TOKEN)
    second = store.load(UUID)
    assert second is not None and second.read_write_token == TOKEN
    handle.keep()
    third = store.load(UUID)
    assert third is not None and third.status == "kept" and third.read_write_token == TOKEN
    handle.forget()
    assert store.load(UUID) is None and list(store.directory.iterdir()) == []


def test_an_unwritable_state_dir_is_one_warning_and_never_raises(
    state_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_home.mkdir(parents=True)
    (state_home / "perplexity").write_text("a file where the directory should be")
    handle = ThreadHandle(ThreadStore(), prompt="q")
    handle.observe(UUID, None)
    handle.observe(UUID, TOKEN)
    handle.keep()
    handle.forget()
    assert len(handle.warnings) == 1
    assert "could not record" in handle.warnings[0]
    assert TOKEN not in handle.warnings[0] and UUID not in handle.warnings[0]
    assert capsys.readouterr().err == ""


def test_resume_command_names_a_non_default_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PPLX_PROFILE", raising=False)
    assert handles.resume_command(UUID, None) == f"pplx resume {UUID}"
    assert handles.resume_command(UUID, "default") == f"pplx resume {UUID}"
    assert handles.resume_command(UUID, "work") == f"pplx resume --profile work {UUID}"
    assert handles.resume_command(UUID, "my work") == f"pplx resume --profile 'my work' {UUID}"
    monkeypatch.setenv("PPLX_PROFILE", "env")
    assert handles.resume_command(UUID, None) == f"pplx resume --profile env {UUID}"


def _dead_pid() -> int:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=int)
    p.start()
    p.join()
    assert p.pid is not None
    return p.pid


def test_no_token_bearing_field_shows_in_a_repr() -> None:
    from pplx_agent_tools.verbs._ask_common import AskStreamState

    record = ThreadRecord(UUID, datetime.now(timezone.utc), "kept", read_write_token=TOKEN)
    state = AskStreamState(backend_uuid=UUID, read_write_token=TOKEN)
    assert TOKEN not in repr(record) and TOKEN not in repr(state)
    assert UUID in repr(record)


@pytest.mark.parametrize(
    "junk", [b"\xff\xfe\x00 not utf-8", b"\xff not utf-8", b"{not json"], ids=["bom", "raw", "json"]
)
def test_an_undecodable_record_file_is_skipped_counted_and_never_breaks_a_save(
    state_home: Path, junk: bytes
) -> None:
    store = ThreadStore()
    store.save(ThreadRecord(UUID, datetime.now(timezone.utc), "kept"))
    (store.directory / ("f" * 32 + ".json")).write_bytes(junk)
    pick = store.pick_last()
    assert pick.record is not None and pick.record.backend_uuid == UUID
    assert pick.unreadable == 1
    handle = ThreadHandle(store, prompt="q")
    handle.observe(_uuid(7), None)
    assert handle.warnings == []
    assert store.load(_uuid(7)) is not None


def test_a_record_that_is_not_one_raises_on_load(state_home: Path) -> None:
    store = ThreadStore()
    store.save(ThreadRecord(UUID, datetime.now(timezone.utc), "kept"))
    store._path(UUID).write_bytes(b"\xff\xfe corrupt")
    with pytest.raises(ValueError):
        store.load(UUID)
    assert store.load(_uuid(8)) is None


def test_an_unlistable_directory_raises_rather_than_reading_as_empty(state_home: Path) -> None:
    store = ThreadStore()
    store.save(ThreadRecord(UUID, datetime.now(timezone.utc), "kept"))
    store.directory.chmod(0)
    try:
        with pytest.raises(PermissionError):
            store.pick_last()
        with pytest.raises(PermissionError):
            store.load(UUID)
        store.prune(datetime.now(timezone.utc))
    finally:
        store.directory.chmod(0o700)
