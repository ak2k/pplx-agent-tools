"""pplx resume verb: reattach to a research thread and read its report to the end.

A research run whose connection dropped, or whose client was killed, goes on
server-side and finishes without a listener; its thread holds the report for
about 24 h. `POST /rest/sse/perplexity_ask/reconnect/{backend_uuid}` returns
a snapshot of the thread first: a finished thread sends one COMPLETED frame
carrying the whole report, a running one sends snapshots until COMPLETED.
Reconnecting creates no thread, so this verb is stateless in the sense of
CLAUDE.md's endpoint principle; deleting the thread afterward is the same
cleanup research does.

The stream is read by the research driver and consumer (`read_report`,
`SnapshotReport`), fresh for every resume: the server renumbers step ids
between snapshots, so nothing from the dropped run is merged in.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from ..errors import PplxError, ThreadGoneError, ThreadRecordsError
from ..handles import ThreadHandle, ThreadRecord, ThreadStore, hash_prompt, resume_command
from ..wire import RECONNECT_PATH, Client, thread_ref
from ._ask_common import AskStreamState, error_notes
from .research import (
    _COUNCIL_MODEL,
    DEFAULT_MODE,
    ResearchResult,
    SnapshotReport,
    finish_report,
    read_report,
)

# The user-facing mode a served model stands for, for a thread with no record.
_MODEL_MODE = {"pplx_alpha": "research", _COUNCIL_MODEL: "agentic_research"}


def last_resumable(
    store: ThreadStore, *, query: str | None = None
) -> tuple[ThreadRecord, list[str]]:
    """The newest resumable record in `store`, for `--last`, and the notes
    the result should carry about how it was chosen. `query`, the exact
    prompt the run was started with, narrows the choice to that run."""
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
    live = (
        [f"{pick.live} newer run(s) are still being read by a live pplx process"]
        if pick.live
        else []
    )
    matching = " whose prompt matches --query exactly" if query is not None else ""
    raise PplxError(
        f"no resumable research thread recorded for profile {store.profile!r} in the "
        f"last 25 h{matching}{error_notes(live + unreadable)}"
    )


def _reason(e: Exception) -> str:
    return e.strerror if isinstance(e, OSError) and e.strerror else str(e) or type(e).__name__


def resume(
    client: Client,
    backend_uuid: str,
    *,
    store: ThreadStore,
    keep_thread: bool = False,
    timeout: float | None = None,
    stall_seconds: float | None = None,
    progress: bool = False,
    new_consumer: Callable[[], SnapshotReport] = SnapshotReport,
    notes: Sequence[str] = (),
    prompt: str | None = None,
) -> ResearchResult:
    """Reconnect to `backend_uuid` and return its report as `research` would.

    The local record in `store`, when there is one, supplies the
    read_write_token the delete needs and the mode and model the run asked
    for; without it the token comes from the frames. Cleanup is research's:
    a COMPLETED thread is deleted unless `keep_thread`, a run that may go on
    (a drop, or a terminate that failed) keeps it and names this command
    again, and a deadline or stall terminates and deletes it. Unlike
    research, an exception (Ctrl-C included) keeps it too, since the thread
    is the only copy of a report already paid for, and prints this command on
    stderr. `new_consumer` builds the consumer the frames are read into.
    `notes` lead the result's warnings. `prompt`, the run's prompt when
    the caller knows it, stands in for a snapshot that does not echo it.
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
    report = new_consumer()
    state = AskStreamState(read_write_token=record.read_write_token if record else None)
    handle = ThreadHandle(store, record=record)
    where = RECONNECT_PATH + thread_ref(backend_uuid)
    try:
        read_report(
            client,
            state,
            handle,
            report.on_event,
            endpoint=where,
            body={},
            label="resume",
            keep_thread=keep_thread,
            timeout=timeout,
            stall_seconds=stall_seconds,
            progress=progress,
            opener=partial(client.sse_reconnect, backend_uuid),
            keep_on_raise=True,
        )
    except ThreadGoneError:
        raise  # nothing left to keep or resume
    except BaseException as e:
        if state.kept and not state.deleted:
            command = resume_command(backend_uuid, store.profile)
            if isinstance(e, PplxError):
                e.resume = command
            print(
                f"warning: the thread was kept: resume it with `{command}` within about "
                "24 h of its start; do not re-run, which spends another research unit",
                file=sys.stderr,
            )
        raise
    kept = state.kept and not state.deleted
    if not kept and not keep_thread and state.backend_uuid and not state.read_write_token:
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
    return finish_report(
        report,
        state,
        handle,
        label="resume",
        endpoint=where,
        query=_initial_query(report.text) or prompt or "",
        mode=mode,
        requested_model=None if model == _COUNCIL_MODEL else model,
        timeout=timeout,
        resume=resume_command(backend_uuid, store.profile) if kept else None,
        notes=notes,
    )


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
