"""A bounded ring of FSM transitions, emitted with failed runs.

Every `TraceEntry` field is an int, a Literal, or a tuple of Literals, so no
server string (payload, id, token, cursor, model name) can reach a trace.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias, final

from typing_extensions import assert_never

from pplx_agent_tools.askstream.fsm import (
    AwaitingFirst,
    BrokeKind,
    Change,
    Close,
    Done,
    DueTag,
    Effect,
    Event,
    Fatal,
    FrameIn,
    Gone,
    HeartbeatIn,
    InitialPost,
    Notice,
    Open,
    Opened,
    OpenFailed,
    Producing,
    RateLimited,
    ReconnectBackoff,
    Reconnecting,
    ReconnectReason,
    ReconnectTarget,
    Stage,
    StartBackoff,
    Starting,
    State,
    StreamBroke,
    StreamEnded,
    Streaming,
    TextComplete,
    Tick,
    Transient,
    current_conn,
    due_class,
)
from pplx_agent_tools.askstream.outcome import (
    Completed,
    Cut,
    EndedEarly,
    Lost,
    Outcome,
    Rejected,
    ServerFailed,
    SettledWithoutTerminal,
)
from pplx_agent_tools.askstream.policy import Policy

RING = 256

StateTag = Literal[
    "Starting", "StartBackoff", "Streaming", "Reconnecting", "ReconnectBackoff", "Done"
]
PhaseTag = Literal["AwaitingFirst", "Producing", "TextComplete", "-"]
EventTag = Literal[
    "Opened", "OpenFailed", "FrameIn", "HeartbeatIn", "StreamEnded", "StreamBroke", "Tick"
]
FailureTag = Literal["RateLimited", "Transient", "Gone", "Fatal"]
SubTag: TypeAlias = (
    Stage | Change | FailureTag | BrokeKind | DueTag | ReconnectReason | Literal["stale"]
)
EffectTag = Literal["Open:initial", "Open:reconnect", "Close", "Notice"]
OutcomeTag = Literal[
    "-",
    "Completed",
    "SettledWithoutTerminal",
    "Cut:stall",
    "Cut:deadline",
    "Cut:first_content",
    "EndedEarly",
    "Lost",
    "ServerFailed",
    "Rejected",
]


@final
@dataclass(frozen=True, slots=True)
class TraceEntry:
    t_ms: int  # monotonic milliseconds since the run started
    conn: int  # the FSM's own ordinal, never server data
    state: StateTag
    phase: PhaseTag
    event: EventTag
    sub: tuple[SubTag, ...]
    status: int  # the HTTP status of an OpenFailed, else 0
    to_state: StateTag
    to_phase: PhaseTag
    outcome: OutcomeTag
    effects: tuple[EffectTag, ...]
    n: int  # consecutive repeats merged into this entry


def _state_tag(s: State) -> StateTag:
    match s:
        case Starting():
            return "Starting"
        case StartBackoff():
            return "StartBackoff"
        case Streaming():
            return "Streaming"
        case Reconnecting():
            return "Reconnecting"
        case ReconnectBackoff():
            return "ReconnectBackoff"
        case Done():
            return "Done"
        case _:
            assert_never(s)


def _phase_tag(s: State) -> PhaseTag:
    match s:
        case Streaming(live=live) | Reconnecting(live=live) | ReconnectBackoff(live=live):
            phase = live.phase
            match phase:
                case AwaitingFirst():
                    return "AwaitingFirst"
                case Producing():
                    return "Producing"
                case TextComplete():
                    return "TextComplete"
                case _:
                    assert_never(phase)
        case Starting() | StartBackoff() | Done():
            return "-"
        case _:
            assert_never(s)


def _outcome_tag(o: Outcome) -> OutcomeTag:  # noqa: PLR0911 - one return per variant
    match o:
        case Completed():
            return "Completed"
        case SettledWithoutTerminal():
            return "SettledWithoutTerminal"
        case Cut(cause="stall"):
            return "Cut:stall"
        case Cut(cause="deadline"):
            return "Cut:deadline"
        case Cut(cause="first_content"):
            return "Cut:first_content"
        case EndedEarly():
            return "EndedEarly"
        case Lost():
            return "Lost"
        case ServerFailed():
            return "ServerFailed"
        case Rejected():
            return "Rejected"
        case Cut():
            raise AssertionError(o.cause)
        case _:
            assert_never(o)


def _effect_tag(e: Effect) -> EffectTag:
    match e:
        case Open(target=InitialPost()):
            return "Open:initial"
        case Open(target=ReconnectTarget()):
            return "Open:reconnect"
        case Close():
            return "Close"
        case Notice():
            return "Notice"
        case _:
            assert_never(e)


def _event_parts(  # noqa: PLR0911 - one return per variant
    policy: Policy, s: State, e: Event
) -> tuple[EventTag, tuple[SubTag, ...], int]:
    """(event tag, sub-tags, HTTP status) for one event."""
    match e:
        case Opened():
            return "Opened", (), 0
        case OpenFailed(f=f):
            match f:
                case RateLimited():
                    return "OpenFailed", ("RateLimited",), 429
                case Transient():
                    return "OpenFailed", ("Transient",), 0
                case Gone(status):
                    return "OpenFailed", ("Gone",), status
                case Fatal():
                    return "OpenFailed", ("Fatal",), 0
                case _:
                    assert_never(f)
        case FrameIn(s=fs):
            return "FrameIn", (fs.stage, fs.change), 0
        case HeartbeatIn():
            return "HeartbeatIn", (), 0
        case StreamEnded():
            return "StreamEnded", (), 0
        case StreamBroke(kind=kind):
            return "StreamBroke", (kind,), 0
        case Tick(now):
            return "Tick", (due_class(policy, s, now),), 0
        case _:
            assert_never(e)


def entry(
    policy: Policy,
    started_at: float,
    before: State,
    e: Event,
    after: State,
    effects: tuple[Effect, ...],
) -> TraceEntry:
    event, sub, status = _event_parts(policy, before, e)
    if not isinstance(e, Tick) and e.conn != current_conn(before):
        sub = (*sub, "stale")
    if isinstance(after, ReconnectBackoff) and not isinstance(before, ReconnectBackoff):
        sub = (*sub, after.reason)
    conn = current_conn(before) if isinstance(e, Tick) else e.conn
    return TraceEntry(
        t_ms=max(0, round((e.now - started_at) * 1000)),
        conn=0 if conn is None else int(conn),
        state=_state_tag(before),
        phase=_phase_tag(before),
        event=event,
        sub=sub,
        status=status,
        to_state=_state_tag(after),
        to_phase=_phase_tag(after),
        outcome=_outcome_tag(after.outcome) if isinstance(after, Done) else "-",
        effects=tuple(_effect_tag(x) for x in effects),
        n=1,
    )


@final
class TraceRing:
    """The last `RING` entries. A run of entries equal apart from `t_ms`
    (a heartbeat flood) is one entry with a count and the first time."""

    __slots__ = ("_entries", "dropped")

    def __init__(self) -> None:
        self._entries: deque[TraceEntry] = deque(maxlen=RING)
        self.dropped = 0

    def add(self, e: TraceEntry) -> None:
        if self._entries:
            last = self._entries[-1]
            if replace(last, t_ms=e.t_ms, n=e.n) == e:
                self._entries[-1] = replace(last, n=last.n + e.n)
                return
        if len(self._entries) == RING:
            self.dropped += 1
        self._entries.append(e)

    @property
    def entries(self) -> tuple[TraceEntry, ...]:
        return tuple(self._entries)
