"""Research over the diff-mode stream: the run, its salvage and its cleanup.

`research_stream` runs the lifecycle driver under research's policy, keeps
v0.8's bookkeeping over each frame's `text` snapshot, and maps the run's
outcome onto the cutoff errors v0.8's transport raised, so `cutoff_warnings`,
`cutoff_cause`, `cutoff_silence` and `no_content_error` report it unchanged.
A diff-mode stream carries `text` only on its COMPLETED frame, so a run that
ends without one is answered from the block store's projections.

Cleanup keeps the thread of a run that may go on server-side once the read
ends, for `pplx resume`, and keeps the run's record (handles.py) in step.

research.py imports this module, so it passes its decoding in (`Decoder`).
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, TextIO, TypeAlias

from typing_extensions import assert_never

from ..askstream.blocks import BlockStore, Desynced
from ..askstream.cleanup import (
    Delete,
    DeleteKept,
    DeleteNotNeeded,
    DeleteNoToken,
    Terminate,
    TerminateNotNeeded,
    TerminateUnsupported,
    cleanup_plan,
    kept_on_loss,
)
from ..askstream.drift import Drift
from ..askstream.driver import Driver, Opener, StreamClient, delete
from ..askstream.frames import AskFrame
from ..askstream.fsm import AUTH_NOTICE, Done, ReconnectReason, State
from ..askstream.outcome import (
    Completed,
    Cut,
    EndedEarly,
    Lost,
    Outcome,
    Rejected,
    ServerFailed,
    SettledWithoutTerminal,
)
from ..askstream.policy import (
    Bounded,
    FirstContentWithin,
    PolicyError,
    deadline_of,
    for_verb,
    stall_of,
)
from ..askstream.projections import (
    READS,
    ReadKey,
    citation_warnings,
    report_body,
    research_answer,
)
from ..errors import (
    AuthError,
    NetworkError,
    PplxError,
    SchemaError,
    StreamDeadlineError,
    StreamFirstContentError,
    StreamSilenceError,
    StreamStallError,
    ThreadGoneError,
)
from ..handles import ThreadHandle
from ._ask_common import (
    _RUN_MAY_BE_LIVE,
    FIRST_CONTENT_SECONDS,
    RESEARCH_SILENCE_SECONDS,
    AskStreamState,
    Source,
    error_notes,
    no_content_error,
)

RECONNECT = Bounded(3, 8)

# (cover note parts, report body parts, sources, clarifying question texts)
Parts: TypeAlias = tuple[list[str], list[str], list[Source], list[str] | None]
# "deleted": the delete succeeded; "cleaned": released without one.
ThreadStatus = Literal["none", "gone", "kept", "deleted", "cleaned"]
# The fields a projected answer is read from, as a warning names them.
_READ_LABELS: dict[ReadKey, str] = {
    "ask_text": "cover note",
    "unified_assets": "report",
    "web_results": "source list",
}


@dataclass(frozen=True)
class Decoder:
    """research.py's snapshot decoding. `parts` raises SchemaError for a
    `text` it cannot read; `unanswered` words the clarifying questions of a
    run that returned no answer."""

    parts: Callable[[str], Parts]
    join: Callable[[list[str], list[str]], str]
    unanswered: Callable[[list[str] | None], list[str]]


class SnapshotConsumer:
    """v0.8's bookkeeping over each frame's `text`: the newest snapshot that
    decoded, and the high-water marks that show whether it lost content.
    It also keeps the store's report body high-water, the measure for an
    answer read from the projections."""

    def __init__(self, decoder: Decoder, store: BlockStore) -> None:
        self._decoder = decoder
        self._store = store
        self.text: str | None = None
        self.answer = ""
        self.sources: list[Source] = []
        self.body_len = 0
        self.questions: list[str] | None = None
        self.best = {"body": 0, "total": 0}
        self.saw = {"body": False, "last_frame_decoded": False}
        self.decode_error: SchemaError | None = None
        self.report_high = 0

    def on_frame(self, frame: AskFrame) -> None:
        self.report_high = max(self.report_high, len(report_body(self._store)))
        text = frame.text
        if text is None:
            return
        try:
            cover, report, sources, questions = self._decoder.parts(text)
        except SchemaError as e:
            self.decode_error = e
            self.saw["last_frame_decoded"] = False
            return
        # The body is the report asset itself, not what the join keeps: a
        # cover note that quotes the report would otherwise measure zero.
        body = "\n\n".join(report).strip()
        answer = self._decoder.join(cover, report)
        self.text, self.answer, self.sources = text, answer, sources
        self.body_len, self.questions = len(body), questions
        self.saw["last_frame_decoded"] = True
        if body:
            self.saw["body"] = True
        self.best["body"] = max(self.best["body"], len(body))
        self.best["total"] = max(self.best["total"], len(answer))


@dataclass
class ResearchRun:
    """A finished run: what research builds its result from, and the parts
    a caller may inspect. `body_len`, `best` and `saw` feed v0.8's
    shortfall verdict; `warnings` holds the projection's citation warning,
    and `unread` a warning for each field of a projected answer the store
    could not keep whole, which marks the answer short."""

    driver: Driver
    store: BlockStore
    consumer: SnapshotConsumer
    state: AskStreamState
    thread: ThreadStatus
    outcome: Outcome
    answer: str
    sources: list[Source]
    body_len: int
    best: dict[str, int]
    saw: dict[str, bool]
    questions: list[str] | None
    warnings: list[str]
    unread: list[str] = field(default_factory=list[str])

    @property
    def terminal_text(self) -> str | None:
        return self.driver.terminal_text


def research_stream(
    client: StreamClient,
    opener: Opener,
    decoder: Decoder,
    *,
    endpoint: str,
    label: str = "research",
    keep_thread: bool = False,
    keep_on_raise: bool = False,
    hold: bool = False,
    timeout: float | None = None,
    stall_seconds: float | None = None,
    progress: bool = False,
    state: AskStreamState | None = None,
    handle: ThreadHandle | None = None,
    on_data: Callable[[dict[str, object]], None] | None = None,
    observe: Callable[[ResearchRun], None] | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    rand: Callable[[], float] = random.random,
    err: TextIO | None = None,
) -> ResearchRun:
    """Run one research stream to its end and release its thread.

    Returns the salvaged run, a FAILED or empty one included, for the caller
    to build its result or error from (`raise_if_empty`). Raises only what
    ended the read itself: an exception (Ctrl-C included), or a rejection
    with no partial to salvage. `observe` sees the run before it returns.

    Cleanup (`release`) keeps the thread of a run that may go on server-side
    once the read ends. `keep_on_raise` keeps it, sending nothing, when the
    read raises, which otherwise terminates and deletes. `hold`
    sends no delete and leaves the record of a thread the read is done with,
    for a caller that deletes it only once the report is out.

    `state` is filled in, when given, instead of a fresh one; the thread
    ids, `kept`, `deleted` and `gone` are set even when the read raises. `handle`
    records the thread before the frame that first names it is applied, and
    is settled from what cleanup did.
    """
    policy = for_verb(
        "research",
        deadline=deadline_of(timeout),
        stall=stall_of(stall_seconds),
        reconnect=RECONNECT,
        first_content=FirstContentWithin(FIRST_CONTENT_SECONDS),
        silence_s=RESEARCH_SILENCE_SECONDS,
    )
    if isinstance(policy, PolicyError):
        raise ValueError(policy.reason)
    state = AskStreamState() if state is None else state
    store = BlockStore("ask_text_only", policy.limits, expect_text=True)
    consumer = SnapshotConsumer(decoder, store)

    def seen(data: dict[str, object]) -> None:
        _note_ids(state, data)
        if handle is not None:
            handle.observe(state.backend_uuid, state.read_write_token)
        if on_data is not None:
            on_data(data)

    driver = Driver(
        policy,
        client,
        opener,
        store,
        label=f"pplx {label}",
        on_frame=consumer.on_frame,
        on_data=seen,
        progress=progress,
        # Looked up per call rather than bound at import, so a test's
        # stand-in for this module's clock reaches the driver.
        clock=time.monotonic if clock is None else clock,
        sleep=time.sleep if sleep is None else sleep,
        rand=rand,
        err=err,
    )
    raised, rejected = True, False
    try:
        done = driver.run()
        raised, rejected = False, _unsalvageable(done.outcome) is not None
    finally:
        thread, warnings = release(
            client,
            driver.state,
            driver.trigger,
            gone=driver.gone,
            display_model=driver.display_model,
            keep_thread=keep_thread or hold,
            raised=raised,
            keep_on_raise=keep_on_raise,
        )
        state.kept, state.deleted = thread == "kept", thread == "deleted"
        state.gone = thread == "gone"
        state.cleanup_warnings += warnings
        if handle is not None:
            _settle(handle, thread, hold=hold)
        if raised or rejected:
            _warn([*warnings, *(handle.warnings if handle is not None else [])], err)
    outcome = done.outcome
    unsalvageable = _unsalvageable(outcome)
    if unsalvageable is not None:
        raise unsalvageable
    state.display_model = driver.display_model
    state.saw_completed = isinstance(outcome, Completed)
    state.failed = isinstance(outcome, ServerFailed)
    state.cutoff = _cutoff(outcome, driver, policy.silence_s, endpoint, label)
    run = _salvage(driver, store, consumer, state, thread, outcome)
    if observe is not None:
        observe(run)
    return run


def _unsalvageable(outcome: Outcome) -> PplxError | None:
    """The error of a rejection with nothing to salvage: a cap or an
    oversized event leaves the store unreadable, and no other rejection
    after the first byte has content."""
    if isinstance(outcome, Rejected) and not isinstance(outcome.error, NetworkError):
        return outcome.error
    return None


def _note_ids(state: AskStreamState, data: dict[str, object]) -> None:
    """The first thread ids the frames carried, raw: the record and the
    delete of a held thread need the token itself."""
    uuid, token, context = (
        data.get(k) for k in ("backend_uuid", "read_write_token", "context_uuid")
    )
    if state.backend_uuid is None and isinstance(uuid, str):
        state.backend_uuid = uuid
    if state.read_write_token is None and isinstance(token, str):
        state.read_write_token = token
    if state.context_uuid is None and isinstance(context, str):
        state.context_uuid = context


def release(
    client: StreamClient,
    last: State | None,
    trigger: ReconnectReason | None,
    *,
    gone: bool,
    display_model: str | None,
    keep_thread: bool,
    raised: bool = False,
    keep_on_raise: bool = False,
) -> tuple[ThreadStatus, list[str]]:
    """Clean up after a run, on every exit path; `raised` when the read
    ended in an exception.

    Nothing is sent for a thread the server reports gone, nor for one kept
    for resume: a run lost to a drop or silence, or with `keep_on_raise` any
    read that raised, a rejection with nothing to salvage included.
    Otherwise the plan's legs, except that a run pplx could not stop (its
    terminate could not be sent, or failed) keeps its thread unless an
    exception is in flight: the run goes on without a listener, and its
    thread is then how its report is got. Returns the thread's status and
    the warnings for a run that may still be going.
    """
    if last is None:
        return "none", []
    if gone:
        return "gone", []
    rejected = isinstance(last, Done) and _unsalvageable(last.outcome) is not None
    if (keep_on_raise and (raised or rejected)) or (
        not raised and kept_on_loss(last, trigger, gone)
    ):
        return "kept", []
    terminate, delete_leg = cleanup_plan(last, keep_thread, display_model)
    warnings: list[str] = []
    stopped = True
    match terminate:
        case Terminate(ref):
            stopped = client.terminate(ref.uuid, ref.context, ref.model_preference)
            if not stopped:
                warnings.append(f"{_RUN_MAY_BE_LIVE}: the request to stop it failed")
        case TerminateUnsupported(missing):
            stopped = False
            warnings.append(
                f"{_RUN_MAY_BE_LIVE}: no {missing} arrived, so pplx could not ask it to stop"
            )
        case TerminateNotNeeded():
            pass
        case _:
            assert_never(terminate)
    if not stopped and not raised:
        return "kept", warnings
    match delete_leg:
        case Delete(ref):
            return ("deleted" if delete(client, ref) else "cleaned"), warnings
        case DeleteNotNeeded() | DeleteKept() | DeleteNoToken():
            return "cleaned", warnings
        case _:
            assert_never(delete_leg)


def _settle(handle: ThreadHandle, thread: ThreadStatus, *, hold: bool) -> None:
    """Bring the record in line with cleanup: kept while the thread can be
    resumed, removed once it is deleted, gone or finished with (unless
    `hold`). A run that never named its thread has no record to change."""
    if thread == "kept":
        handle.keep()
    elif thread in ("gone", "deleted") or not hold:
        handle.forget()


def _cutoff(
    outcome: Outcome, driver: Driver, silence_s: float, endpoint: str, label: str
) -> NetworkError | None:
    """The error v0.8's transport raised for the same end."""
    match outcome:
        case Cut():
            return _cut_error(outcome, driver, silence_s, endpoint, label)
        case Lost(msg):
            e = driver.last_error
            # A failed reconnect can leave a silence or deadline error last,
            # which would report the drop as a stall or a deadline.
            if isinstance(e, NetworkError) and not isinstance(e, StreamDeadlineError):
                return e
            return NetworkError(msg)
        case Rejected(e) if isinstance(e, NetworkError):
            return e
        case Completed() | EndedEarly() | SettledWithoutTerminal() | ServerFailed() | Rejected():
            return None
        case _:
            assert_never(outcome)


def _cut_error(
    cut: Cut, driver: Driver, silence_s: float, endpoint: str, label: str
) -> StreamDeadlineError:
    seconds = cut.seconds
    match cut.cause:
        case "deadline":
            return StreamDeadlineError(
                f"{label} stream on {endpoint} exceeded {seconds:.1f}s deadline",
                driver.since_progress,
            )
        case "stall" if driver.trigger == "silence":
            return StreamSilenceError(
                f"SSE stream on {endpoint} went silent: no bytes for {silence_s:.1f}s", silence_s
            )
        case "stall":
            return StreamStallError(
                f"SSE stream on {endpoint} stalled: no new content for {seconds:.1f}s", seconds
            )
        case "first_content":
            return StreamFirstContentError(
                f"SSE stream on {endpoint} sent no content within {seconds:.1f}s", seconds
            )
        case _:
            assert_never(cut.cause)


def _salvage(
    driver: Driver,
    store: BlockStore,
    consumer: SnapshotConsumer,
    state: AskStreamState,
    thread: ThreadStatus,
    outcome: Outcome,
) -> ResearchRun:
    """The answer from the newest decoded `text`, or else from the
    projections, whose sources are the run's retained list."""
    if consumer.text is not None:
        # In diff mode only the terminal frame carries `text`, so the body
        # the patches built is the one earlier measure it can fall short of.
        best = {**consumer.best, "body": max(consumer.best["body"], consumer.report_high)}
        saw = {**consumer.saw, "body": consumer.saw["body"] or consumer.report_high > 0}
        return ResearchRun(
            driver,
            store,
            consumer,
            state,
            thread,
            outcome,
            answer=consumer.answer,
            sources=consumer.sources,
            body_len=consumer.body_len,
            best=best,
            saw=saw,
            questions=consumer.questions,
            warnings=[],
        )
    projected = research_answer(store, state.saw_completed)
    return ResearchRun(
        driver,
        store,
        consumer,
        state,
        thread,
        outcome,
        answer=projected.text,
        sources=[Source(s.url, s.title, s.snippet) for s in store.run_sources],
        body_len=len(report_body(store)),
        best={"body": consumer.report_high, "total": 0},
        saw={"body": consumer.report_high > 0, "last_frame_decoded": consumer.decode_error is None},
        questions=None,
        warnings=list(citation_warnings(projected)),
        unread=_unread(store, driver.drift),
    )


def _unread(store: BlockStore, drift: Counter[Drift]) -> list[str]:
    """A warning for each field a projected answer reads that a rejected
    patch left out of sync, or that arrived without the list it is read
    from: the answer may be missing what it held."""
    out: list[str] = []
    for key, label in _READ_LABELS.items():
        state = store.state(READS[key])
        if isinstance(state, Desynced):
            out.append(
                f"a patch to the {label} could not be applied ({state.reason}), so it "
                "stopped updating and the answer may be missing content"
            )
        out += [
            f"the {label} arrived in a shape pplx could not read ({d.name}), so the "
            "answer may be missing content"
            for d in drift
            if d.kind == "projection_missing" and d.name.split("/")[0] == key
        ]
    return out


def raise_if_empty(
    run: ResearchRun,
    decoder: Decoder,
    *,
    label: str,
    endpoint: str,
    timeout: float | None,
    notes: Sequence[str],
) -> None:
    """v0.8's errors for a run with nothing to return; a FAILED run's error
    is its caller's. `notes` end the text: cleanup has run by then, so a run
    it could not stop, or a thread it kept, is named there. A thread the
    server reported gone cannot be resumed, so its error says to re-run."""
    outcome = run.outcome
    if run.thread == "gone" and not (run.answer or run.sources):
        e = run.driver.last_error
        gone = (
            e if isinstance(e, ThreadGoneError) else ThreadGoneError(f"thread gone on {endpoint}")
        )
        raise ThreadGoneError(
            f"{gone}; nothing was read from it and it cannot be resumed, so re-run the "
            f"research query{error_notes(notes)}"
        ) from gone
    undecodable = run.consumer.decode_error
    if run.consumer.text is None and undecodable is not None and not (run.answer or run.sources):
        raise SchemaError(f"{undecodable}{error_notes(notes)}") from undecodable
    if run.answer or run.sources or (run.consumer.text is not None and run.state.saw_completed):
        return
    if isinstance(outcome, (EndedEarly, SettledWithoutTerminal)) and outcome.by == "auth":
        e = run.driver.last_error
        refused = e if isinstance(e, AuthError) else AuthError(AUTH_NOTICE)
        if notes:
            raise AuthError(f"{refused}{error_notes(notes)}") from refused
        raise refused
    raise no_content_error(
        label=label,
        endpoint=endpoint,
        timeout=timeout,
        cutoff=run.state.cutoff,
        warnings=[*decoder.unanswered(run.questions), *notes],
    )


def _warn(warnings: list[str], err: TextIO | None) -> None:
    """Warnings with an exception in flight, which carries no warnings list."""
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr if err is None else err)
