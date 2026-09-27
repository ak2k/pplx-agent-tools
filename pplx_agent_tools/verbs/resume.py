"""pplx resume verb: reattach to a research thread and read its report to the end.

A research run whose connection dropped, or whose client was killed, goes on
server-side and finishes without a listener; its thread holds the report for
about 24 h. `POST /rest/sse/perplexity_ask/reconnect/{backend_uuid}` returns
a snapshot of the thread first: a finished thread sends one COMPLETED frame
carrying the whole report, a running one sends snapshots until COMPLETED.
Reconnecting creates no thread, so this verb is stateless in the sense of
CLAUDE.md's endpoint principle; deleting the thread afterward is the same
cleanup research does, but only once the report is out (`HeldThread`).

The stream is read by research's driver (`research_stream`), fresh for
every resume: the server renumbers step ids between snapshots, so nothing
from the dropped run is merged in. A drop is reconnected within the call,
as for research.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

from ..askstream.driver import reconnecting
from ..errors import PplxError, ThreadGoneError, ThreadRecordsError
from ..handles import ThreadHandle, ThreadRecord, ThreadStore, hash_prompt, resume_command
from ..wire import RECONNECT_PATH, Client, thread_ref
from ._ask_common import AskStreamState, error_notes
from ._research_stream import research_stream
from .research import _COUNCIL_MODEL, DECODER, DEFAULT_MODE, ResearchResult, finish_report

# The user-facing mode a served model stands for, for a thread with no record.
_MODEL_MODE = {"pplx_alpha": "research", _COUNCIL_MODEL: "agentic_research"}


def last_resumable(
    store: ThreadStore, *, query: str | None = None
) -> tuple[ThreadRecord, list[str]]:
    """The newest resumable record in `store`, for `--last`, and the notes
    the result should carry about how it was chosen. `query`, the exact
    prompt the run was started with, narrows the choice to that run; without
    it, a newer run a live process is still reading makes this an error
    rather than a pick of an older one."""
    try:
        pick = store.pick_last(prompt_hash=hash_prompt(query) if query is not None else None)
    except OSError as e:
        raise ThreadRecordsError(
            f"cannot read the research thread records under {store.directory}: "
            f"{_reason(e)}; fix the directory, or resume by the uuid `pplx research` printed"
        ) from e
    unreadable = (
        [f"{pick.unreadable} record file(s) under {store.directory} could not be read"]
        if pick.unreadable
        else []
    )
    matching = " whose prompt matches --query exactly" if query is not None else ""
    if pick.record is not None and query is None and pick.live:
        # The live run is likely the caller's own; the older kept thread past it
        # is then another run's, and resuming it would delete that run's report.
        raise PplxError(
            f"--last found {pick.live} newer research run(s) for profile {store.profile!r} "
            "still being read by a live pplx process, and will not take an older thread "
            "past them: pass --query with the exact prompt of the run to resume, or its "
            f"uuid{error_notes(unreadable)}"
        )
    if pick.record is not None:
        # In a fan-out the newest run is as likely another agent's as this one's.
        several = (
            [
                f"--last took the newest of {pick.candidates} resumable research threads "
                f"recorded for profile {store.profile!r}; to get a specific run's report, "
                "pass --query with its exact prompt, or its uuid"
            ]
            if query is None and pick.candidates > 1
            else []
        )
        return pick.record, unreadable + several
    if pick.live:
        # Worded apart from the nothing-recorded error below, which tells an
        # agent to re-run: here the run is going, and a re-run pays for it twice.
        raise PplxError(
            f"the {pick.live} research run(s) recorded for profile {store.profile!r} in the "
            f"last 25 h{matching} are still being read by a live pplx process: wait for it "
            "to finish, or pass the run's uuid; do not re-run, which spends another research "
            f"unit{error_notes(unreadable)}"
        )
    raise PplxError(
        f"no resumable research thread recorded for profile {store.profile!r} in the "
        f"last 25 h{matching}{error_notes(unreadable)}"
    )


def _reason(e: Exception) -> str:
    return e.strerror if isinstance(e, OSError) and e.strerror else str(e) or type(e).__name__


class HeldThread:
    """A resumed thread and its record, held until the caller has put the
    report out: `release` then deletes and removes them as research's cleanup
    would, and `keep` leaves both for a report that never got out. The thread
    is the only other copy of a report already paid for."""

    def __init__(
        self,
        client: Client,
        state: AskStreamState,
        handle: ThreadHandle,
        *,
        command: str,
        keep_thread: bool,
    ) -> None:
        self._client = client
        self._state = state
        self._handle = handle
        self._keep_thread = keep_thread
        self.command = command

    def release(self) -> None:
        """The report is out: delete the thread unless it was kept, is gone
        or `keep_thread`, then remove its record."""
        state = self._state
        if state.kept:
            return
        before = list(self._handle.warnings)
        try:
            if (
                not (self._keep_thread or state.gone)
                and state.backend_uuid
                and state.read_write_token
            ):
                state.deleted = self._client.delete_thread(
                    state.backend_uuid, state.read_write_token
                )
        finally:
            self._handle.forget()
            self._warn_since(before)

    def keep(self, error: BaseException | None = None) -> None:
        """The report did not get out: mark the record `kept` and name the
        command on stderr and on `error`. A gone thread has nothing to keep."""
        if self._state.gone:
            return
        before = list(self._handle.warnings)
        self._handle.keep()
        if isinstance(error, PplxError):
            error.resume = self.command
        print(
            f"warning: the thread was kept: resume it with `{self.command}` within about "
            "24 h of its start; do not re-run, which spends another research unit",
            file=sys.stderr,
        )
        self._warn_since(before)

    def _warn_since(self, before: list[str]) -> None:
        # The report, or the error, already carried the earlier warnings.
        if self._handle.warnings != before:
            for warning in self._handle.warnings:
                print(f"warning: {warning}", file=sys.stderr)


def resume(
    client: Client,
    backend_uuid: str,
    *,
    store: ThreadStore,
    keep_thread: bool = False,
    timeout: float | None = None,
    stall_seconds: float | None = None,
    progress: bool = False,
    notes: Sequence[str] = (),
    prompt: str | None = None,
) -> tuple[ResearchResult, HeldThread]:
    """Reconnect to `backend_uuid` and return its report as `research` would,
    with the thread held until the caller has put the report out.

    The local record in `store`, when there is one, supplies the
    read_write_token the delete needs and the mode and model the run asked
    for; without it the token comes from the frames. Cleanup is research's,
    except that nothing is deleted or removed here: releasing the returned
    `HeldThread` deletes a thread the read is done with unless `keep_thread`,
    and removes its record. A run that may go on (a drop, or a terminate that
    failed) keeps its thread and names this command again, and a deadline or
    stall terminates it. An exception (Ctrl-C included), a report that does
    not decode, or no content at all keeps the thread and its record and
    prints this command on stderr, unless the server reported the thread gone
    (at the open or on a reconnect): nothing is then sent, kept or named, and
    with nothing read the error is `ThreadGoneError`. `notes` lead the
    result's warnings.
    `prompt`, the run's prompt when the caller knows it, stands in for a
    snapshot that does not echo it.
    """
    notes = list(notes)
    unread = False
    try:
        record = store.load(backend_uuid)
    except (OSError, ValueError) as e:
        record, unread = None, True
        notes.append(
            f"the record of this thread under {store.directory} could not be read "
            f"({_reason(e)}); resuming without it"
        )
    state = AskStreamState(read_write_token=record.read_write_token if record else None)
    handle = ThreadHandle(store, record=record)
    where = RECONNECT_PATH + thread_ref(backend_uuid)
    command = resume_command(backend_uuid, store.profile)
    held = HeldThread(client, state, handle, command=command, keep_thread=keep_thread)
    try:
        run = research_stream(
            client,
            reconnecting(client, backend_uuid),
            DECODER,
            endpoint=where,
            label="resume",
            keep_thread=keep_thread,
            keep_on_raise=True,
            hold=True,
            timeout=timeout,
            stall_seconds=stall_seconds,
            progress=progress,
            state=state,
            handle=handle,
        )
        if (
            not (state.kept or state.gone or keep_thread)
            and state.backend_uuid
            and not state.read_write_token
        ):
            why = (
                "its record could not be read and the stream sent no read_write_token"
                if unread
                else "no read_write_token was recorded for it or sent on the stream"
            )
            state.cleanup_warnings.append(
                f"the thread was not deleted: {why}; it expires about 24 h after it started"
            )
        model = record.model if record else None
        mode = (record.mode if record else None) or _MODEL_MODE.get(
            state.display_model or "", DEFAULT_MODE
        )
        result = finish_report(
            run,
            handle,
            label="resume",
            endpoint=where,
            query=_initial_query(run.consumer.text) or prompt or "",
            mode=mode,
            requested_model=None if model == _COUNCIL_MODEL else model,
            timeout=timeout,
            resume=command if state.kept else None,
            notes=notes,
        )
    except ThreadGoneError:
        raise  # nothing left to keep or resume
    except BaseException as e:
        held.keep(e)
        raise
    return result, held


def _initial_query(text: str | None) -> str | None:
    """The question a research snapshot echoes in its INITIAL_QUERY step."""
    if text is None:
        return None
    try:
        blocks: Any = json.loads(text)
    except ValueError:
        return None
    for blk in blocks if isinstance(blocks, list) else []:
        if isinstance(blk, dict) and blk.get("step_type") == "INITIAL_QUERY":
            content = blk.get("content")
            query = content.get("query") if isinstance(content, dict) else None
            if isinstance(query, str):
                return query
    return None
