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
from collections.abc import Callable
from functools import partial
from typing import Any

from ..errors import PplxError, ThreadGoneError
from ..handles import ThreadHandle, ThreadRecord, ThreadStore, resume_command
from ..wire import RECONNECT_PATH, Client, thread_ref
from ._ask_common import AskStreamState
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


def last_resumable(store: ThreadStore) -> ThreadRecord:
    """The newest resumable record in `store`, for `--last`."""
    pick = store.pick_last()
    if pick.record is not None:
        return pick.record
    live = (
        f"; {pick.live} newer run(s) are still being read by a live pplx process"
        if pick.live
        else ""
    )
    raise PplxError(
        f"no resumable research thread recorded for profile {store.profile!r} in the "
        f"last 25 h{live}"
    )


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
    """
    record = store.load(backend_uuid)
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
        handle.settle("gone")
        raise
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
        state.cleanup_warnings.append(
            "the thread was not deleted: no read_write_token was recorded for it or sent "
            "on the stream; it expires about 24 h after it started"
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
        query=_initial_query(report.text) or "",
        mode=mode,
        requested_model=None if model == _COUNCIL_MODEL else model,
        timeout=timeout,
        resume=resume_command(backend_uuid, store.profile) if kept else None,
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
