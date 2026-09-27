"""Research over the diff-mode stream: the run, its salvage and its cleanup.

`research_stream` runs the lifecycle driver under research's policy, keeps
v0.8's bookkeeping over each frame's `text` snapshot, and maps the run's
outcome onto the cutoff errors v0.8's transport raised, so `cutoff_warnings`,
`cutoff_cause`, `cutoff_silence` and `no_content_error` report it unchanged.
A diff-mode stream carries `text` only on its COMPLETED frame, so a run that
ends without one is answered from the block store's projections.

research.py imports this module, so it passes its decoding in (`Decoder`).
"""

from __future__ import annotations

import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TextIO, TypeAlias

from typing_extensions import assert_never

from ..askstream.blocks import BlockStore
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
from ..askstream.driver import Driver, Opener, StreamClient, delete
from ..askstream.frames import AskFrame
from ..askstream.fsm import AUTH_NOTICE, ReconnectReason, State
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
from ..askstream.projections import citation_warnings, report_body, research_answer
from ..errors import (
    AuthError,
    NetworkError,
    SchemaError,
    StreamDeadlineError,
    StreamFirstContentError,
    StreamSilenceError,
    StreamStallError,
)
from ._ask_common import (
    _RUN_MAY_BE_LIVE,
    FIRST_CONTENT_SECONDS,
    RESEARCH_SILENCE_SECONDS,
    AskStreamState,
    Source,
    error_notes,
    no_content_error,
)

LABEL = "research"
RECONNECT = Bounded(3, 8)

# (cover note parts, report body parts, sources, clarifying question texts)
Parts: TypeAlias = tuple[list[str], list[str], list[Source], list[str] | None]
ThreadStatus = Literal["none", "gone", "kept", "cleaned"]


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
    shortfall verdict; `warnings` holds the projection's citation warning."""

    driver: Driver
    store: BlockStore
    consumer: SnapshotConsumer
    state: AskStreamState
    thread: ThreadStatus
    answer: str
    sources: list[Source]
    body_len: int
    best: dict[str, int]
    saw: dict[str, bool]
    questions: list[str] | None
    warnings: list[str]

    @property
    def terminal_text(self) -> str | None:
        return self.driver.terminal_text


def research_stream(
    client: StreamClient,
    opener: Opener,
    decoder: Decoder,
    *,
    endpoint: str,
    keep_thread: bool = False,
    timeout: float | None = None,
    stall_seconds: float | None = None,
    progress: bool = False,
    on_data: Callable[[dict[str, object]], None] | None = None,
    observe: Callable[[ResearchRun], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
    err: TextIO | None = None,
) -> ResearchRun:
    """Run one research stream to its end and release its thread.

    Returns the salvaged run, a FAILED one included (`state.failed`), for
    research to build its result or its FAILED error from. Raises when
    nothing can be salvaged, with v0.8's errors and texts. `observe` sees
    the run before any such raise.
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
    store = BlockStore("ask_text_only", policy.limits, expect_text=True)
    consumer = SnapshotConsumer(decoder, store)
    driver = Driver(
        policy,
        client,
        opener,
        store,
        label=f"pplx {LABEL}",
        on_frame=consumer.on_frame,
        on_data=on_data,
        progress=progress,
        clock=clock,
        sleep=sleep,
        rand=rand,
        err=err,
    )
    raised = True
    try:
        done = driver.run()
        raised = False
    finally:
        thread, warnings = release(
            client,
            driver.state,
            driver.trigger,
            gone=driver.gone,
            display_model=driver.display_model,
            keep_thread=keep_thread,
        )
        if raised:
            _warn(warnings, err)
    outcome = done.outcome
    # A cap or an oversized event leaves the store unreadable, and no other
    # rejection after the first byte has content to salvage.
    if isinstance(outcome, Rejected) and not isinstance(outcome.error, NetworkError):
        _warn(warnings, err)
        raise outcome.error
    state = AskStreamState(
        display_model=driver.display_model,
        saw_completed=isinstance(outcome, Completed),
        failed=isinstance(outcome, ServerFailed),
        cutoff=_cutoff(outcome, driver, policy.silence_s, endpoint),
        cleanup_warnings=warnings,
    )
    run = _salvage(driver, store, consumer, state, thread)
    if observe is not None:
        observe(run)
    if not state.failed:
        _raise_if_empty(run, outcome, decoder, endpoint, timeout)
    return run


def release(
    client: StreamClient,
    last: State | None,
    trigger: ReconnectReason | None,
    *,
    gone: bool,
    display_model: str | None,
    keep_thread: bool,
) -> tuple[ThreadStatus, list[str]]:
    """Clean up after a run, on every exit path: nothing for a thread the
    server reports gone or one kept for resume, otherwise the plan's legs.
    Returns the thread's status and the warnings for a run that may still
    be going."""
    if last is None:
        return "none", []
    if gone:
        return "gone", []
    if kept_on_loss(last, trigger, gone):
        return "kept", []
    terminate, delete_leg = cleanup_plan(last, keep_thread, display_model)
    warnings: list[str] = []
    match terminate:
        case Terminate(ref):
            if not client.terminate(ref.uuid, ref.context, ref.model_preference):
                warnings.append(f"{_RUN_MAY_BE_LIVE}: the request to stop it failed")
        case TerminateUnsupported(missing):
            warnings.append(
                f"{_RUN_MAY_BE_LIVE}: no {missing} arrived, so pplx could not ask it to stop"
            )
        case TerminateNotNeeded():
            pass
        case _:
            assert_never(terminate)
    match delete_leg:
        case Delete(ref):
            delete(client, ref)
        case DeleteNotNeeded() | DeleteKept() | DeleteNoToken():
            pass
        case _:
            assert_never(delete_leg)
    return "cleaned", warnings


def _cutoff(
    outcome: Outcome, driver: Driver, silence_s: float, endpoint: str
) -> NetworkError | None:
    """The error v0.8's transport raised for the same end."""
    match outcome:
        case Cut():
            return _cut_error(outcome, driver, silence_s, endpoint)
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


def _cut_error(cut: Cut, driver: Driver, silence_s: float, endpoint: str) -> StreamDeadlineError:
    seconds = cut.seconds
    match cut.cause:
        case "deadline":
            return StreamDeadlineError(
                f"{LABEL} stream on {endpoint} exceeded {seconds:.1f}s deadline",
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
) -> ResearchRun:
    """The answer from the newest decoded `text`, or else from the
    projections, whose sources are the run's retained list."""
    if consumer.text is not None:
        return ResearchRun(
            driver,
            store,
            consumer,
            state,
            thread,
            answer=consumer.answer,
            sources=consumer.sources,
            body_len=consumer.body_len,
            best=dict(consumer.best),
            saw=dict(consumer.saw),
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
        answer=projected.text,
        sources=[Source(s.url, s.title, s.snippet) for s in store.run_sources],
        body_len=len(report_body(store)),
        best={"body": consumer.report_high, "total": 0},
        saw={"body": consumer.report_high > 0, "last_frame_decoded": consumer.decode_error is None},
        questions=None,
        warnings=list(citation_warnings(projected)),
    )


def _raise_if_empty(
    run: ResearchRun, outcome: Outcome, decoder: Decoder, endpoint: str, timeout: float | None
) -> None:
    """v0.8's errors for a run with nothing to return. Cleanup has run by
    now, so a run it could not stop is named at the end of the text."""
    notes = run.state.cleanup_warnings
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
        label=LABEL,
        endpoint=endpoint,
        timeout=timeout,
        cutoff=run.state.cutoff,
        warnings=decoder.unanswered(run.questions) + notes,
    )


def _warn(warnings: list[str], err: TextIO | None) -> None:
    """Warnings with an exception in flight, which carries no warnings list."""
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr if err is None else err)
