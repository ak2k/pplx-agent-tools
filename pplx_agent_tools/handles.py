# pyright: strict
"""Local handles for the research threads pplx starts, so `pplx resume` can
find a run whose client went away before the report arrived.

One JSON file per thread in `$XDG_STATE_HOME/perplexity/<profile>/threads/`
(`$XDG_STATE_HOME` defaults to `~/.local/state`): mode 0600 in a 0700
directory, replaced whole by an atomic rename. A file per thread is what lets
many pplx processes write at once without a lock. A file exists only while
its thread can be resumed: it is written when the stream first names the
thread and removed once the thread is deleted, gone or finished with, so a
client killed outright (SIGKILL, a machine sleep, a closed terminal) leaves
it `running`, which is the case `pplx resume --last` exists for. Records
older than the thread's expiry plus a margin are pruned on every write.

The read_write_token is kept here because deleting a thread needs it. It never
leaves the file: no message, warning or error carries it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shlex
import stat
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast, get_args

from .auth import DEFAULT_PROFILE, atomic_write_0600, resolve_profile

# Incognito threads expire 24 h after they start; the extra hour keeps a record
# a little past that so a resume near the edge still finds its token.
MAX_AGE = timedelta(hours=25)

# running: a pplx process is reading the stream, or was until it died without
#   cleaning up. kept: left going server-side on purpose. Only `pick_last`
#   tells them apart, to pass over a thread a live process is still reading.
Status = Literal["running", "kept"]


@dataclass(frozen=True)
class ThreadRecord:
    backend_uuid: str
    started: datetime
    status: Status
    read_write_token: str | None = field(default=None, repr=False)
    prompt_sha256: str | None = None
    mode: str | None = None
    model: str | None = None
    pid: int | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "backend_uuid": self.backend_uuid,
                "started": self.started.isoformat(),
                "status": self.status,
                "read_write_token": self.read_write_token,
                "prompt_sha256": self.prompt_sha256,
                "mode": self.mode,
                "model": self.model,
                "pid": self.pid,
            },
            indent=2,
            sort_keys=True,
        )


def hash_prompt(prompt: str) -> str:
    """The digest a record keeps of its research prompt; `--query` matches it."""
    return hashlib.sha256(prompt.encode()).hexdigest()


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _parse(content: bytes) -> ThreadRecord | None:
    """A record file's bytes → the record, or None when it is not one. Bytes
    that are not UTF-8 are one more way not to be a record: `json.loads`
    raises UnicodeDecodeError, a ValueError, for them."""
    try:
        raw: object = json.loads(content)
    except ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    data = cast("dict[str, object]", raw)
    uuid, started, status, pid = (
        data.get("backend_uuid"),
        data.get("started"),
        data.get("status"),
        data.get("pid"),
    )
    if not isinstance(uuid, str) or not uuid or not isinstance(started, str):
        return None
    if status not in get_args(Status):
        return None
    try:
        when = datetime.fromisoformat(started)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return ThreadRecord(
        backend_uuid=uuid,
        started=when,
        status=cast("Status", status),
        read_write_token=_opt_str(data.get("read_write_token")),
        prompt_sha256=_opt_str(data.get("prompt_sha256")),
        mode=_opt_str(data.get("mode")),
        model=_opt_str(data.get("model")),
        pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _state_home() -> Path:
    # The XDG spec has a relative value ignored.
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg and Path(xdg).is_absolute():
        return Path(xdg)
    return Path.home() / ".local" / "state"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return None


def resume_command(backend_uuid: str, profile: str | None) -> str:
    """The command that resumes `backend_uuid`, naming the profile unless it is the default."""
    name = resolve_profile(profile)
    flag = "" if name == DEFAULT_PROFILE else f"--profile {shlex.quote(name)} "
    return f"pplx resume {flag}{shlex.quote(backend_uuid)}"


@dataclass(frozen=True)
class LastPick:
    """What `--last` found: the newest resumable record, if any; how many
    newer `running` records it passed over because a live process is still
    reading them; how many files were not readable records; and how many
    records it could have picked."""

    record: ThreadRecord | None
    live: int
    unreadable: int = 0
    candidates: int = 0


class ThreadStore:
    """The records of one cookie profile."""

    def __init__(self, profile: str | None = None) -> None:
        self.profile = resolve_profile(profile)
        self.directory = _state_home() / "perplexity" / self.profile / "threads"

    def _path(self, backend_uuid: str) -> Path:
        # Hashed so a thread id from the wire or the command line can never
        # name a path outside the directory.
        return self.directory / f"{hashlib.sha256(backend_uuid.encode()).hexdigest()[:32]}.json"

    def _ensure_directory(self) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if stat.S_IMODE(self.directory.stat().st_mode) != 0o700:
            self.directory.chmod(0o700)

    def save(self, record: ThreadRecord, *, now: datetime | None = None) -> None:
        """Write `record` whole, then prune expired records. Raises OSError."""
        self._ensure_directory()
        atomic_write_0600(self._path(record.backend_uuid), record.to_json())
        self.prune(now or _now())

    def remove(self, backend_uuid: str) -> None:
        """Remove the record of `backend_uuid`, if any. Raises OSError."""
        self._path(backend_uuid).unlink(missing_ok=True)

    def load(self, backend_uuid: str) -> ThreadRecord | None:
        """The record of `backend_uuid`, or None when there is none. Raises
        OSError when it cannot be read and ValueError when the file there is
        not its record."""
        try:
            content = self._path(backend_uuid).read_bytes()
        except FileNotFoundError:
            return None
        record = _parse(content)
        if record is None or record.backend_uuid != backend_uuid:
            raise ValueError("not a thread record")
        return record

    def records(self, *, now: datetime | None = None) -> list[ThreadRecord]:
        """Every readable record younger than MAX_AGE, newest first. Raises
        OSError when the directory cannot be listed."""
        return self._scan(now or _now())[0]

    def _scan(self, now: datetime) -> tuple[list[ThreadRecord], int]:
        """`records`, and how many files were not readable records."""
        cutoff = now - MAX_AGE
        try:
            # Not `glob`, which reads an unlistable directory as empty.
            paths = [p for p in self.directory.iterdir() if p.suffix == ".json"]
        except FileNotFoundError:
            return [], 0
        found: list[ThreadRecord] = []
        unreadable = 0
        for path in paths:
            try:
                record = _parse(path.read_bytes())
            except FileNotFoundError:
                continue  # removed since the listing
            except OSError:
                record = None
            if record is None:
                unreadable += 1
            elif record.started >= cutoff:
                found.append(record)
        return sorted(found, key=lambda r: r.started, reverse=True), unreadable

    def pick_last(self, *, now: datetime | None = None, prompt_hash: str | None = None) -> LastPick:
        """The newest resumable record, only among those whose prompt hashes
        to `prompt_hash` when given. Raises OSError when the directory cannot
        be listed."""
        live = candidates = 0
        chosen: ThreadRecord | None = None
        records, unreadable = self._scan(now or _now())
        for record in records:
            if prompt_hash is not None and record.prompt_sha256 != prompt_hash:
                continue
            pid = record.pid
            if (
                record.status == "running"
                and pid is not None
                and pid != os.getpid()
                and _alive(pid)
            ):
                if chosen is None:
                    live += 1
                continue
            candidates += 1
            if chosen is None:
                chosen = record
        return LastPick(chosen, live, unreadable, candidates)

    def prune(self, now: datetime) -> None:
        """Remove records, and temp files a killed writer left, older than
        MAX_AGE. Best-effort: another process may be pruning the same files."""
        cutoff = now - MAX_AGE
        try:
            entries = list(self.directory.iterdir())
        except OSError:
            return
        for path in entries:
            if path.suffix == ".json":
                try:
                    record = _parse(path.read_bytes())
                except OSError:
                    continue
                started = record.started if record is not None else _mtime(path)
            elif path.suffix == ".tmp":
                started = _mtime(path)
            else:
                continue
            if started is not None and started < cutoff:
                with contextlib.suppress(OSError):
                    path.unlink()


class ThreadHandle:
    """One run's record: written when the stream first names its thread,
    rewritten when the token first arrives, marked `kept` or removed once
    cleanup is done with it.

    A write or removal that fails costs the run nothing: it becomes the one
    entry in `warnings`, saying what the record was left as.
    """

    def __init__(
        self,
        store: ThreadStore,
        *,
        prompt: str | None = None,
        mode: str | None = None,
        model: str | None = None,
        record: ThreadRecord | None = None,
    ) -> None:
        self._store = store
        self._prompt_sha256 = hash_prompt(prompt) if prompt is not None else None
        self._mode = mode
        self._model = model
        # A record this run resumes; it is claimed at the first frame.
        self._record = record
        self._claimed = False
        # Whether a file for the thread may exist, so removal is worth trying.
        self._on_disk = record is not None
        self.warnings: list[str] = []

    def observe(self, backend_uuid: str | None, read_write_token: str | None) -> None:
        """Record what the stream has named so far; a no-op when nothing is new."""
        if backend_uuid is None:
            return
        record = self._record
        if record is None or record.backend_uuid != backend_uuid:
            self._write(
                ThreadRecord(
                    backend_uuid=backend_uuid,
                    started=_now(),
                    status="running",
                    read_write_token=read_write_token,
                    prompt_sha256=self._prompt_sha256,
                    mode=self._mode,
                    model=self._model,
                    pid=os.getpid(),
                )
            )
        elif not self._claimed:
            # This process is now the one reading the thread, so `--last`
            # elsewhere passes over it while it lives.
            self._write(
                replace(
                    record,
                    status="running",
                    pid=os.getpid(),
                    read_write_token=record.read_write_token or read_write_token,
                )
            )
        elif read_write_token and record.read_write_token is None:
            self._write(replace(record, read_write_token=read_write_token))

    def keep(self) -> None:
        """Mark the record `kept`: the thread was left going on purpose."""
        if self._record is not None and self._record.status != "kept":
            self._write(replace(self._record, status="kept"))

    def forget(self) -> None:
        """Remove the record: the thread was deleted, is gone, or pplx is done
        with it, so there is nothing left to resume."""
        if self._record is None or not self._on_disk:
            return
        try:
            self._store.remove(self._record.backend_uuid)
        except Exception as e:
            self._fail(
                "could not remove this research thread's record, so `pplx resume --last` "
                "may still offer it",
                e,
            )
        else:
            self._on_disk = False

    @property
    def record(self) -> ThreadRecord | None:
        return self._record

    def _write(self, record: ThreadRecord) -> None:
        self._record = record
        self._claimed = True
        try:
            self._store.save(record)
        except Exception as e:
            self._fail(
                "could not update this research thread's record for `pplx resume`"
                if self._on_disk
                else "could not record this research thread for `pplx resume`",
                e,
            )
        else:
            self._on_disk = True

    def _fail(self, what: str, e: Exception) -> None:
        reason = e.strerror if isinstance(e, OSError) and e.strerror else type(e).__name__
        # One entry: the latest failure is the one that says what the record
        # was left as.
        self.warnings[:] = [f"{what} (under {self._store.directory}: {reason})"]
