"""The ask-family lifecycle as a pure state machine.

`step(policy, state, event, u)` returns the next state and the effects the
driver must carry out. It does no I/O and reads no clock: time enters only
as each event's `now`, and the one random input, the backoff jitter draw
`u` in [0, 1), is passed in. The FSM sees a `FrameSummary`, never block
contents.

The start phase (`Starting`, `StartBackoff`) is the only one that can emit
`Open[InitialPost]`; `step_live`'s return type admits only
`Open[ReconnectTarget]`, so no second billed POST can follow the first
response.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Generic, Literal, TypeAlias, TypeVar, final

from typing_extensions import assert_never

from pplx_agent_tools.askstream.ids import (
    BackendUuid,
    ConnId,
    ContextUuid,
    Cursor,
    ReadWriteToken,
    ThreadRef,
)
from pplx_agent_tools.askstream.outcome import (
    Completed,
    Cut,
    EndedEarly,
    Lost,
    Outcome,
    Rejected,
    ServerFailed,
    SettleBy,
    SettledWithoutTerminal,
)
from pplx_agent_tools.askstream.policy import (
    At,
    AtCompleted,
    AtTextComplete,
    Bounded,
    Deadline,
    FirstContentOff,
    FirstContentWithin,
    Off,
    Policy,
    SettleAfterText,
    StallAfter,
    StallOff,
    Unbounded,
    rate_limit_delay,
    reconnect_delay,
)
from pplx_agent_tools.errors import (
    AuthError,
    NetworkError,
    PplxError,
    RateLimitError,
    ResourceLimitError,
    SchemaError,
)

# --- frame summary -------------------------------------------------------------------------------

# Derived once per frame from (status, text_completed).
Stage = Literal["pending", "text_complete", "completed", "failed", "other"]
# Whether the frame changed content the progress rule counts.
Change = Literal["idle", "progress"]
Reconnectable = Literal["yes", "no", "absent"]


@final
@dataclass(frozen=True, slots=True)
class FrameSummary:
    """What the lifecycle reads from one decoded frame; the id fields are
    per-frame presence, merged into the state by `merge_ids`."""

    stage: Stage
    change: Change
    uuid: BackendUuid | None = None
    token: ReadWriteToken | None = None
    cursor: Cursor | None = None
    context: ContextUuid | None = None
    reconnectable: Reconnectable = "absent"
    raw_status: str | None = None


# --- ids -----------------------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class NoIds:
    pass


@final
@dataclass(frozen=True, slots=True)
class UuidOnly:
    uuid: BackendUuid
    context: ContextUuid | None


@final
@dataclass(frozen=True, slots=True)
class Known:
    ref: ThreadRef
    context: ContextUuid | None


Ids: TypeAlias = NoIds | UuidOnly | Known


def merge_ids(ids: Ids, s: FrameSummary) -> Ids:
    """Ids only move forward: NoIds < UuidOnly < Known, and a uuid, token or
    context once held is kept. A token or context seen before any uuid has
    no thread to belong to and is not kept."""
    match ids:
        case Known(ref, context):
            return ids if context is not None or s.context is None else Known(ref, s.context)
        case UuidOnly(uuid, context):
            ctx = context if context is not None else s.context
            if s.token is not None:
                return Known(ThreadRef(uuid, s.token), ctx)
            return ids if ctx is context else UuidOnly(uuid, ctx)
        case NoIds():
            if s.uuid is None:
                return ids
            if s.token is not None:
                return Known(ThreadRef(s.uuid, s.token), s.context)
            return UuidOnly(s.uuid, s.context)
        case _:
            assert_never(ids)


def ids_uuid(ids: UuidOnly | Known) -> BackendUuid:
    match ids:
        case UuidOnly(uuid, _):
            return uuid
        case Known(ref, _):
            return ref.uuid
        case _:
            assert_never(ids)


# --- states --------------------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class AwaitingFirst:
    """No progress frame yet."""


@final
@dataclass(frozen=True, slots=True)
class Producing:
    last_progress_at: float


@final
@dataclass(frozen=True, slots=True)
class TextComplete:
    """`text_completed` seen (SettleAfterText only); `at` is when, and the
    settle window runs from it whatever arrives later."""

    last_progress_at: float
    at: float


Phase: TypeAlias = AwaitingFirst | Producing | TextComplete


@final
@dataclass(frozen=True, slots=True)
class NoGrace:
    pass


@final
@dataclass(frozen=True, slots=True)
class GraceUntil:
    """A reconnect opened; its snapshot has until `t` before stall or
    silence can fire."""

    t: float


Grace: TypeAlias = NoGrace | GraceUntil


@final
@dataclass(frozen=True, slots=True)
class InitialPost:
    pass


@final
@dataclass(frozen=True, slots=True)
class ReconnectTarget:
    uuid: BackendUuid


ReconnectReason = Literal["drop", "eof", "silence", "stall", "first_content", "settle"]


@final
@dataclass(frozen=True, slots=True)
class Starting:
    started_at: float
    deadline: Deadline
    conn: ConnId
    rl_attempts: int
    open_due_at: float


@final
@dataclass(frozen=True, slots=True)
class StartBackoff:
    started_at: float
    deadline: Deadline
    until: float
    rl_attempts: int
    next_conn: ConnId


@final
@dataclass(frozen=True, slots=True)
class Live:
    started_at: float
    deadline: Deadline
    ids: Ids
    cursor: Cursor | None
    reconnectable: Reconnectable
    phase: Phase
    rc_consecutive: int
    rc_total: int
    next_conn: ConnId


@final
@dataclass(frozen=True, slots=True)
class Streaming:
    live: Live
    conn: ConnId
    last_byte_at: float
    grace: Grace


@final
@dataclass(frozen=True, slots=True)
class Reconnecting:
    live: Live
    conn: ConnId
    target: ReconnectTarget
    reason: ReconnectReason
    open_due_at: float


@final
@dataclass(frozen=True, slots=True)
class ReconnectBackoff:
    live: Live
    until: float
    target: ReconnectTarget
    reason: ReconnectReason


@final
@dataclass(frozen=True, slots=True)
class Done:
    outcome: Outcome
    ids: Ids


StartState: TypeAlias = Starting | StartBackoff
LiveState: TypeAlias = Streaming | Reconnecting | ReconnectBackoff
State: TypeAlias = Starting | StartBackoff | Streaming | Reconnecting | ReconnectBackoff | Done

# --- events --------------------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class RateLimited:
    retry_after: float | None


@final
@dataclass(frozen=True, slots=True)
class Transient:
    msg: str


@final
@dataclass(frozen=True, slots=True)
class Gone:
    status: int


@final
@dataclass(frozen=True, slots=True)
class Fatal:
    err: PplxError


Failure: TypeAlias = RateLimited | Transient | Gone | Fatal
BrokeKind = Literal["transport", "oversize", "cap"]


@final
@dataclass(frozen=True, slots=True)
class Opened:
    conn: ConnId
    now: float


@final
@dataclass(frozen=True, slots=True)
class OpenFailed:
    conn: ConnId
    now: float
    f: Failure


@final
@dataclass(frozen=True, slots=True)
class FrameIn:
    conn: ConnId
    now: float
    s: FrameSummary


@final
@dataclass(frozen=True, slots=True)
class HeartbeatIn:
    conn: ConnId
    now: float


@final
@dataclass(frozen=True, slots=True)
class StreamEnded:
    conn: ConnId
    now: float


@final
@dataclass(frozen=True, slots=True)
class StreamBroke:
    conn: ConnId
    now: float
    kind: BrokeKind
    msg: str


@final
@dataclass(frozen=True, slots=True)
class Tick:
    now: float


ConnEvent: TypeAlias = Opened | OpenFailed | FrameIn | HeartbeatIn | StreamEnded | StreamBroke
Event: TypeAlias = Opened | OpenFailed | FrameIn | HeartbeatIn | StreamEnded | StreamBroke | Tick

# --- effects -------------------------------------------------------------------------------------

TargetT = TypeVar("TargetT", InitialPost, ReconnectTarget)


@final
@dataclass(frozen=True, slots=True)
class Open(Generic[TargetT]):
    conn: ConnId
    target: TargetT
    low_speed_s: float


@final
@dataclass(frozen=True, slots=True)
class Close:
    conn: ConnId


NoticeKind = Literal["rate_limited", "reconnect", "reconnect_failed", "open_timeout", "auth"]
AUTH_NOTICE = "session cookies expired; refresh with pplx auth"


@final
@dataclass(frozen=True, slots=True)
class Notice:
    kind: NoticeKind
    detail: str


StartEffect: TypeAlias = Open[InitialPost] | Close | Notice
LiveEffect: TypeAlias = Open[ReconnectTarget] | Close | Notice
Effect: TypeAlias = Open[InitialPost] | Open[ReconnectTarget] | Close | Notice

StartStep: TypeAlias = tuple[Starting | StartBackoff | Streaming | Done, tuple[StartEffect, ...]]
LiveStep: TypeAlias = tuple[
    Streaming | Reconnecting | ReconnectBackoff | Done, tuple[LiveEffect, ...]
]
Step: TypeAlias = tuple[State, tuple[Effect, ...]]

# --- timers --------------------------------------------------------------------------------------

DueTag = Literal[
    "deadline", "open_due", "backoff", "settle", "first_content", "stall", "silence", "none"
]


def timers(policy: Policy, s: State) -> list[tuple[DueTag, float]]:
    """Every timer the state has, in tie-break order: the first one due wins,
    so the deadline beats everything and settle beats stall."""
    out: list[tuple[DueTag, float]] = []
    match s:
        case Done():
            return out
        case Starting() | StartBackoff():
            deadline = s.deadline
        case Streaming() | Reconnecting() | ReconnectBackoff():
            deadline = s.live.deadline
        case _:
            assert_never(s)
    match deadline:
        case At(t):
            out.append(("deadline", t))
        case Unbounded():
            pass
    match s:
        case Starting(open_due_at=due) | Reconnecting(open_due_at=due):
            out.append(("open_due", due))
        case StartBackoff(until=until) | ReconnectBackoff(until=until):
            out.append(("backoff", until))
        case Streaming(live, _, last_byte_at, grace):
            floor = grace.t if isinstance(grace, GraceUntil) else float("-inf")
            phase = live.phase
            match phase:
                case AwaitingFirst():
                    last_progress = live.started_at
                case Producing(lp) | TextComplete(lp, _):
                    last_progress = lp
                case _:
                    assert_never(phase)
            if isinstance(phase, TextComplete) and isinstance(policy.completion, SettleAfterText):
                out.append(("settle", phase.at + policy.completion.settle_s))
            if isinstance(phase, AwaitingFirst) and isinstance(
                policy.first_content, FirstContentWithin
            ):
                out.append(("first_content", live.started_at + policy.first_content.s))
            if isinstance(policy.stall, StallAfter):
                out.append(("stall", max(last_progress + policy.stall.s, floor)))
            out.append(("silence", max(last_byte_at + policy.silence_s, floor)))
        case _:
            assert_never(s)
    return out


def due_class(policy: Policy, s: State, now: float) -> DueTag:
    """Which timer a Tick at `now` fires; shared by the driver and `step`."""
    for tag, t in timers(policy, s):
        if now >= t:
            return tag
    return "none"


def next_wake(policy: Policy, s: State) -> float | None:
    """When the driver must send the next Tick; None only for `Done`."""
    times = [t for _, t in timers(policy, s)]
    return min(times) if times else None


def _remaining(deadline: Deadline, now: float) -> float:
    match deadline:
        case At(t):
            return t - now
        case Unbounded():
            return float("inf")
        case _:
            assert_never(deadline)


def _deadline_span(started_at: float, deadline: Deadline) -> float:
    match deadline:
        case At(t):
            return t - started_at
        case Unbounded():
            return float("inf")
        case _:
            assert_never(deadline)


# --- outcomes ------------------------------------------------------------------------------------


def ended_after_text(phase: Phase) -> bool:
    """Once `text_completed` is seen the answer is whole, so every early end
    keeps it as a settled result; the one rule `fallback`,
    `deadline_outcome` and `auth_outcome` share."""
    return isinstance(phase, TextComplete)


_SETTLE_BY: dict[ReconnectReason, SettleBy] = {
    "settle": "settle",
    "stall": "stall",
    "silence": "stall",
    "first_content": "stall",
    "drop": "server",
    "eof": "server",
}


def fallback(policy: Policy, reason: ReconnectReason, live: Live) -> Outcome:
    """The outcome when a reconnect trigger cannot (or may no longer)
    reconnect."""
    n = live.rc_total
    if ended_after_text(live.phase):
        return SettledWithoutTerminal(n, _SETTLE_BY[reason])
    match reason:
        case "drop":
            return Lost("stream dropped")
        case "eof":
            return EndedEarly(n, "server")
        # Settle is due only in TextComplete; a stall is the nearest meaning.
        case "stall" | "silence" | "settle":
            return Cut("stall", stall_window(policy), n)
        case "first_content":
            fc = policy.first_content
            match fc:
                case FirstContentWithin(s):
                    return Cut("first_content", s, n)
                case FirstContentOff():
                    return Cut("first_content", 0.0, n)
                case _:
                    assert_never(fc)
        case _:
            assert_never(reason)


def stall_window(policy: Policy) -> float:
    """The seconds a stall cut reports; with the stall check off only
    silence can cut, so it reports the silence window."""
    stall = policy.stall
    match stall:
        case StallAfter(s):
            return s
        case StallOff():
            return policy.silence_s
        case _:
            assert_never(stall)


def deadline_outcome(live: Live) -> Outcome:
    if ended_after_text(live.phase):
        return SettledWithoutTerminal(live.rc_total, "deadline")
    return Cut("deadline", _deadline_span(live.started_at, live.deadline), live.rc_total)


def auth_outcome(live: Live, err: PplxError) -> Outcome:
    """A reconnect refused for expired cookies keeps what streamed."""
    if ended_after_text(live.phase):
        return SettledWithoutTerminal(live.rc_total, "auth")
    if isinstance(live.phase, AwaitingFirst):
        return Rejected(err)
    return EndedEarly(live.rc_total, "auth")


def _failure_error(f: Failure) -> PplxError:
    match f:
        case RateLimited(retry_after):
            return RateLimitError("HTTP 429: rate limited", retry_after=retry_after)
        case Transient(msg):
            return NetworkError(msg)
        case Gone(status):
            return SchemaError(f"HTTP {status} on the initial request")
        case Fatal(err):
            return err
        case _:
            assert_never(f)


# --- start phase ---------------------------------------------------------------------------------


def initial(policy: Policy, now: float) -> tuple[Starting, tuple[Open[InitialPost]]]:
    """The first state and its one effect: the initial POST on conn 1."""
    deadline: Deadline
    match policy.deadline:
        case At(s):
            deadline = At(now + s)
        case Unbounded():
            deadline = Unbounded()
        case _:
            assert_never(policy.deadline)
    conn = ConnId(1)
    state = Starting(now, deadline, conn, 1, now + policy.open_s)
    return state, (Open(conn, InitialPost(), policy.low_speed_s),)


def _cut_before_first_byte(s: StartState) -> Done:
    return Done(Cut("deadline", _deadline_span(s.started_at, s.deadline), 0), NoIds())


def step_start(policy: Policy, s: StartState, e: Event, u: float) -> StartStep:
    if isinstance(e, Tick):
        return _tick_start(policy, s, e.now)
    match s:
        case StartBackoff():
            return s, ()
        case Starting():
            pass
        case _:
            assert_never(s)
    if e.conn != s.conn:
        return s, ()
    return _starting(policy, s, e, u)


def _starting(policy: Policy, s: Starting, e: ConnEvent, u: float) -> StartStep:
    match e:
        case Opened(conn, now):
            live = Live(
                started_at=s.started_at,
                deadline=s.deadline,
                ids=NoIds(),
                cursor=None,
                reconnectable="absent",
                phase=AwaitingFirst(),
                rc_consecutive=0,
                rc_total=0,
                next_conn=ConnId(conn + 1),
            )
            return Streaming(live, conn, now, NoGrace()), ()
        case OpenFailed(conn, now, f):
            remaining = _remaining(s.deadline, now)
            if (
                isinstance(f, RateLimited)
                and s.rl_attempts < policy.rate_limit_attempts
                and remaining > 0
            ):
                delay = rate_limit_delay(f.retry_after, remaining, u)
                notice = Notice(
                    "rate_limited",
                    f"rate limited (attempt {s.rl_attempts}/{policy.rate_limit_attempts}); "
                    f"retrying in {delay:.1f}s",
                )
                backoff = StartBackoff(
                    s.started_at, s.deadline, now + delay, s.rl_attempts, ConnId(conn + 1)
                )
                return backoff, (notice,)
            return Done(Rejected(_failure_error(f)), NoIds()), ()
        # The per-conn reader emits Opened or OpenFailed before any body item.
        case FrameIn() | HeartbeatIn() | StreamEnded() | StreamBroke():
            return Done(Rejected(SchemaError("event before Opened")), NoIds()), (Close(s.conn),)
        case _:
            assert_never(e)


def _tick_start(policy: Policy, s: StartState, now: float) -> StartStep:
    due = due_class(policy, s, now)
    match due:
        case "deadline":
            effects: tuple[StartEffect, ...] = (Close(s.conn),) if isinstance(s, Starting) else ()
            return _cut_before_first_byte(s), effects
        case "open_due" if isinstance(s, Starting):
            err = NetworkError(f"no response headers within {policy.open_s:g}s")
            return Done(Rejected(err), NoIds()), (Close(s.conn),)
        case "backoff" if isinstance(s, StartBackoff):
            conn = s.next_conn
            starting = Starting(
                s.started_at, s.deadline, conn, s.rl_attempts + 1, now + policy.open_s
            )
            return starting, (Open(conn, InitialPost(), policy.low_speed_s),)
        case "open_due" | "backoff" | "settle" | "first_content" | "stall" | "silence" | "none":
            return s, ()
        case _:
            assert_never(due)


# --- live phase ----------------------------------------------------------------------------------


def lifecycle_valid(policy: Policy, s: State) -> bool:
    """False only for a state no history can build: TextComplete is entered
    only under SettleAfterText."""
    match s:
        case Streaming(live=live) | Reconnecting(live=live) | ReconnectBackoff(live=live):
            return not isinstance(live.phase, TextComplete) or isinstance(
                policy.completion, SettleAfterText
            )
        case Starting() | StartBackoff() | Done():
            return True
        case _:
            assert_never(s)


def _close_current(s: LiveState) -> tuple[LiveEffect, ...]:
    conn = current_conn(s)
    return () if conn is None else (Close(conn),)


def _reconnect_or_fallback(
    policy: Policy,
    live: Live,
    conn: ConnId,
    reason: ReconnectReason,
    now: float,
    u: float,
    *,
    retry_after: float | None = None,
    notices: tuple[Notice, ...] = (),
) -> LiveStep:
    """R(reason): reconnect when the policy, ids, server flag, counters and
    remaining time all allow it; otherwise fall back to an outcome."""
    rc = policy.reconnect
    ids = live.ids
    allowed = False
    match rc:
        case Bounded(consecutive, total):
            allowed = (
                not isinstance(ids, NoIds)
                and live.reconnectable != "no"
                and live.rc_consecutive < consecutive
                and live.rc_total < total
                and _remaining(live.deadline, now) >= policy.min_useful_s
            )
        case Off():
            pass
        case _:
            assert_never(rc)
    if allowed and not isinstance(ids, NoIds):
        delay = reconnect_delay(policy, live.rc_consecutive, retry_after, u)
        nxt = replace(live, rc_consecutive=live.rc_consecutive + 1, rc_total=live.rc_total + 1)
        state = ReconnectBackoff(nxt, now + delay, ReconnectTarget(ids_uuid(ids)), reason)
        return state, (Close(conn), *notices, Notice("reconnect", f"{reason}; reconnecting"))
    return Done(fallback(policy, reason, live), ids), (Close(conn), *notices)


def _advance_phase(policy: Policy, live: Live, fs: FrameSummary, now: float) -> Phase:
    """The phase only moves forward, and TextComplete keeps its `at`."""
    phase = live.phase
    progress = fs.change == "progress"
    enters_text = fs.stage == "text_complete" and isinstance(policy.completion, SettleAfterText)
    match phase:
        case TextComplete(_, at):
            return TextComplete(now, at) if progress else phase
        case AwaitingFirst():
            if enters_text:
                return TextComplete(now if progress else live.started_at, now)
            return Producing(now) if progress else phase
        case Producing(lp):
            if enters_text:
                return TextComplete(now if progress else lp, now)
            return Producing(now) if progress else phase
        case _:
            assert_never(phase)


def _frame_completes(policy: Policy, stage: Stage) -> bool:
    if stage == "completed":
        return True
    completion = policy.completion
    match completion:
        case AtTextComplete():
            return stage == "text_complete"
        case SettleAfterText() | AtCompleted():
            return False
        case _:
            assert_never(completion)


def _frame_in(policy: Policy, s: Streaming, fs: FrameSummary, now: float) -> LiveStep:
    live = s.live
    ids = merge_ids(live.ids, fs)
    if fs.stage == "failed":
        return Done(ServerFailed(fs.raw_status), ids), (Close(s.conn),)
    if _frame_completes(policy, fs.stage):
        return Done(Completed(live.rc_total), ids), (Close(s.conn),)
    progress = fs.change == "progress"
    nxt = replace(
        live,
        ids=ids,
        cursor=fs.cursor if fs.cursor is not None else live.cursor,
        reconnectable=fs.reconnectable if fs.reconnectable != "absent" else live.reconnectable,
        phase=_advance_phase(policy, live, fs, now),
        rc_consecutive=0 if progress else live.rc_consecutive,
    )
    grace: Grace = NoGrace() if progress else s.grace
    return Streaming(nxt, s.conn, now, grace), ()


def step_live(policy: Policy, s: LiveState, e: Event, u: float) -> LiveStep:
    live = s.live
    if not lifecycle_valid(policy, s):
        err = SchemaError("lifecycle state the policy cannot reach")
        return Done(Rejected(err), live.ids), _close_current(s)
    if isinstance(e, Tick):
        return _tick_live(policy, s, e.now, u)
    if e.conn != current_conn(s):
        return s, ()
    match s:
        case ReconnectBackoff():
            return s, ()
        case Reconnecting():
            return _reconnecting(policy, s, e, u)
        case Streaming():
            return _streaming(policy, s, e, u)
        case _:
            assert_never(s)


def _reconnecting(policy: Policy, s: Reconnecting, e: ConnEvent, u: float) -> LiveStep:
    live = s.live
    match e:
        case Opened(conn, now):
            return Streaming(live, conn, now, GraceUntil(now + policy.grace_s)), ()
        case OpenFailed(conn, now, f):
            return _reconnect_failed(policy, s, conn, now, f, u)
        # The per-conn reader emits Opened or OpenFailed before any body item.
        case FrameIn(conn, now) | HeartbeatIn(conn, now) | StreamEnded(conn, now):
            return _reconnect_or_fallback(policy, live, conn, s.reason, now, u)
        case StreamBroke(conn, now):
            return _reconnect_or_fallback(policy, live, conn, s.reason, now, u)
        case _:
            assert_never(e)


def _refused(policy: Policy, s: Reconnecting) -> Outcome:
    """The server refused the reconnect, so it, not the trigger, ended an
    answer that was already whole."""
    o = fallback(policy, s.reason, s.live)
    return replace(o, by="server") if isinstance(o, SettledWithoutTerminal) else o


def _reconnect_failed(
    policy: Policy, s: Reconnecting, conn: ConnId, now: float, f: Failure, u: float
) -> LiveStep:
    """T9 retries within the bounds; T10 ends the run, keeping the partial
    when the cookies were refused."""
    live = s.live
    match f:
        case RateLimited(retry_after):
            notice = Notice("reconnect_failed", "reconnect rate limited")
            return _reconnect_or_fallback(
                policy, live, conn, s.reason, now, u, retry_after=retry_after, notices=(notice,)
            )
        case Transient(msg):
            notice = Notice("reconnect_failed", f"reconnect failed: {msg}")
            return _reconnect_or_fallback(policy, live, conn, s.reason, now, u, notices=(notice,))
        case Fatal(err) if isinstance(err, AuthError):
            # Fixed text: no class name or status, whichever check refused.
            return Done(auth_outcome(live, err), live.ids), (
                Close(conn),
                Notice("auth", AUTH_NOTICE),
            )
        case Fatal(err):
            notice = Notice("reconnect_failed", f"reconnect refused: {type(err).__name__}")
            return Done(_refused(policy, s), live.ids), (Close(conn), notice)
        case Gone(status):
            notice = Notice("reconnect_failed", f"reconnect refused: Gone (HTTP {status})")
            return Done(_refused(policy, s), live.ids), (Close(conn), notice)
        case _:
            assert_never(f)


def _streaming(policy: Policy, s: Streaming, e: ConnEvent, u: float) -> LiveStep:
    live = s.live
    match e:
        # One Opened or OpenFailed per conn, before its body.
        case Opened() | OpenFailed():
            return s, ()
        case FrameIn(_, now, fs):
            return _frame_in(policy, s, fs, now)
        case HeartbeatIn(_, now):
            return replace(s, last_byte_at=now), ()
        case StreamEnded(conn, now):
            return _reconnect_or_fallback(policy, live, conn, "eof", now, u)
        case StreamBroke():
            return _broke(policy, live, e, u)
        case _:
            assert_never(e)


def _broke(policy: Policy, live: Live, e: StreamBroke, u: float) -> LiveStep:
    match e.kind:
        case "transport":
            return _reconnect_or_fallback(policy, live, e.conn, "drop", e.now, u)
        case "oversize":
            return Done(Rejected(SchemaError(e.msg)), live.ids), (Close(e.conn),)
        case "cap":
            return Done(Rejected(ResourceLimitError(e.msg)), live.ids), (Close(e.conn),)
        case _:
            assert_never(e.kind)


def _tick_live(policy: Policy, s: LiveState, now: float, u: float) -> LiveStep:
    live = s.live
    due = due_class(policy, s, now)
    match due:
        case "deadline":
            return Done(deadline_outcome(live), live.ids), _close_current(s)
        case "open_due" if isinstance(s, Reconnecting):
            notice = Notice("open_timeout", "reconnect open timeout")
            return _reconnect_or_fallback(policy, live, s.conn, s.reason, now, u, notices=(notice,))
        case "backoff" if isinstance(s, ReconnectBackoff):
            conn = live.next_conn
            nxt = replace(live, next_conn=ConnId(conn + 1))
            state = Reconnecting(nxt, conn, s.target, s.reason, now + policy.open_s)
            return state, (Open(conn, s.target, policy.low_speed_s),)
        case "settle" | "first_content" | "stall" | "silence" if isinstance(s, Streaming):
            return _reconnect_or_fallback(policy, live, s.conn, due, now, u)
        case "open_due" | "backoff" | "settle" | "first_content" | "stall" | "silence" | "none":
            return s, ()
        case _:
            assert_never(due)


# --- dispatch ------------------------------------------------------------------------------------


def step(policy: Policy, s: State, e: Event, u: float) -> Step:
    """One transition. `u` in [0, 1) is the jitter draw for any backoff this
    event starts."""
    match s:
        case Done():
            return s, ()
        case Starting() | StartBackoff():
            return step_start(policy, s, e, u)
        case Streaming() | Reconnecting() | ReconnectBackoff():
            return step_live(policy, s, e, u)
        case _:
            assert_never(s)


# --- classification for the coverage test and the trace ------------------------------------------

ConnAxis = Literal["current", "stale"]
EventSub: TypeAlias = "type[Failure] | tuple[Stage, Change] | BrokeKind | DueTag | None"


def state_class(s: State) -> tuple[type[State], type[Phase] | None]:
    match s:
        case Streaming(live=live) | Reconnecting(live=live) | ReconnectBackoff(live=live):
            return type(s), type(live.phase)
        case Starting() | StartBackoff() | Done():
            return type(s), None
        case _:
            assert_never(s)


def current_conn(s: State) -> ConnId | None:
    match s:
        case Starting(conn=conn) | Streaming(conn=conn) | Reconnecting(conn=conn):
            return conn
        case StartBackoff() | ReconnectBackoff() | Done():
            return None
        case _:
            assert_never(s)


def event_class(
    policy: Policy, e: Event, s: State
) -> tuple[type[Event], EventSub, ConnAxis | None]:
    sub: EventSub = None
    match e:
        case Tick(now):
            return Tick, due_class(policy, s, now), None
        case Opened() | HeartbeatIn() | StreamEnded():
            pass
        case OpenFailed(f=f):
            sub = type(f)
        case FrameIn(s=fs):
            sub = (fs.stage, fs.change)
        case StreamBroke(kind=kind):
            sub = kind
        case _:
            assert_never(e)
    return type(e), sub, "current" if e.conn == current_conn(s) else "stale"
