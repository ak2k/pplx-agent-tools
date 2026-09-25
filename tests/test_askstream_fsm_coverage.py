"""Exhaustive state x event coverage (plan §3.8, I17).

The axes come from `typing.get_args` over the unions the FSM itself uses,
with field types read through `typing.get_type_hints`, so a new variant
enlarges the product with no edit here. Every cell must be claimed by
exactly one rule: a `Row` (checked over drawn members of the cell), an
`Unreachable` (the defensive result is checked), or a `NotConstructible`
(`due_class` never yields that Tick for that state).
"""

from __future__ import annotations

import itertools
import math
import random
import typing
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, ForwardRef, Literal, get_args, get_origin, get_type_hints

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pplx_agent_tools.askstream import fsm, policy
from pplx_agent_tools.askstream.fsm import (
    AUTH_NOTICE,
    AwaitingFirst,
    Close,
    Done,
    Fatal,
    FrameIn,
    FrameSummary,
    Gone,
    GraceUntil,
    HeartbeatIn,
    InitialPost,
    Known,
    Live,
    NoGrace,
    NoIds,
    Notice,
    Open,
    Opened,
    OpenFailed,
    Producing,
    RateLimited,
    ReconnectBackoff,
    Reconnecting,
    ReconnectTarget,
    StartBackoff,
    Starting,
    StreamBroke,
    StreamEnded,
    Streaming,
    TextComplete,
    Tick,
    Transient,
    UuidOnly,
)
from pplx_agent_tools.askstream.ids import ConnId, ThreadRef
from pplx_agent_tools.askstream.outcome import (
    Completed,
    Cut,
    EndedEarly,
    Lost,
    Rejected,
    ServerFailed,
    SettledWithoutTerminal,
)
from pplx_agent_tools.askstream.policy import (
    At,
    AtCompleted,
    AtTextComplete,
    Bounded,
    FirstContentOff,
    FirstContentWithin,
    Off,
    Policy,
    PolicyError,
    SettleAfterText,
    StallAfter,
    StallOff,
    Unbounded,
)
from pplx_agent_tools.errors import (
    AuthError,
    NetworkError,
    RateLimitError,
    ResourceLimitError,
    SchemaError,
)
from tests._fsm import (
    CTX,
    UUID,
    UUID2,
    UnexpectedRedirect,
    expected_deadline_outcome,
    expected_fallback,
    expected_ids,
    expected_phase,
    expected_reconnects,
    token,
)

# --- axes ----------------------------------------------------------------------------------------

Sub = Any
Cell = tuple[type, type | None, type, type, Sub, str | None]


def _hints(cls: type) -> dict[str, object]:
    hints = get_type_hints(cls)
    for name, h in hints.items():
        assert not isinstance(h, (str, ForwardRef)), (cls, name, h)
    return hints


def _union_members(tp: object) -> tuple[object, ...]:
    args = get_args(tp)
    assert args, tp
    return args


STATE_CLASSES: tuple[type, ...] = _union_members(fsm.State)  # pyright: ignore[reportAssignmentType]
PHASE_CLASSES: tuple[type, ...] = _union_members(_hints(Live)["phase"])  # pyright: ignore[reportAssignmentType]
COMPLETION_CLASSES: tuple[type, ...] = _union_members(_hints(Policy)["completion"])  # pyright: ignore[reportAssignmentType]
EVENT_CLASSES: tuple[type, ...] = _union_members(fsm.Event)  # pyright: ignore[reportAssignmentType]
DUE_TAGS: tuple[str, ...] = _union_members(fsm.DueTag)  # pyright: ignore[reportAssignmentType]


def state_axis() -> list[tuple[type, type | None]]:
    out: list[tuple[type, type | None]] = []
    for cls in STATE_CLASSES:
        if _hints(cls).get("live") is Live:
            out.extend((cls, ph) for ph in PHASE_CLASSES)
        else:
            out.append((cls, None))
    return out


def event_axis() -> list[tuple[type, Sub, str | None]]:
    out: list[tuple[type, Sub, str | None]] = []
    for cls in EVENT_CLASSES:
        hints = _hints(cls)
        subs: list[Sub]
        if cls is Tick:
            subs = list(DUE_TAGS)
        elif "f" in hints:
            subs = list(_union_members(hints["f"]))
        elif "s" in hints:
            fs = _hints(hints["s"])  # pyright: ignore[reportArgumentType]
            subs = list(
                itertools.product(_union_members(fs["stage"]), _union_members(fs["change"]))
            )
        elif "kind" in hints:
            subs = list(_union_members(hints["kind"]))
        else:
            subs = [None]
        conns: tuple[str | None, ...] = ("current", "stale") if "conn" in hints else (None,)
        out.extend((cls, sub, c) for sub in subs for c in conns)
    return out


def all_cells() -> list[Cell]:
    return [
        (k, ph, comp, ev, sub, conn)
        for (k, ph) in state_axis()
        for comp in COMPLETION_CLASSES
        for (ev, sub, conn) in event_axis()
    ]


def test_axes_resolve_from_type_hints() -> None:
    """If an annotation stayed a string, an axis would collapse to one value."""
    for axis in (STATE_CLASSES, PHASE_CLASSES, COMPLETION_CLASSES, EVENT_CLASSES, DUE_TAGS):
        assert len(axis) >= 2
    for cls in (*STATE_CLASSES, *EVENT_CLASSES, Live, FrameSummary, Policy):
        _hints(cls)
    for h in (_hints(OpenFailed)["f"], _hints(StreamBroke)["kind"], _hints(FrameSummary)["stage"]):
        assert len(get_args(h)) >= 2
        assert get_origin(h) in (Literal, typing.Union) or type(h).__name__ == "UnionType"


def test_product_size_is_computed_from_the_axes() -> None:
    cells = all_cells()
    n_states = sum(
        len(PHASE_CLASSES) if _hints(k).get("live") is Live else 1 for k in STATE_CLASSES
    )
    n_events = len(event_axis())
    assert len(set(cells)) == len(cells) == n_states * len(COMPLETION_CLASSES) * n_events
    assert len(cells) == 1728  # today's value, recorded in plan §3.8


def test_classifiers_return_the_axis_classes() -> None:
    rng = random.Random(1)
    seen_states: set[tuple[type, type | None]] = set()
    for k, ph in state_axis():
        for _ in range(3):
            p = _policy(rng, SettleAfterText)
            s = _state(rng, p, k, ph)
            seen_states.add(fsm.state_class(s))
    assert seen_states == set(state_axis())


# --- members -------------------------------------------------------------------------------------

LIVE_KINDS = (Streaming, Reconnecting, ReconnectBackoff)
BODY_EVENTS = (FrameIn, HeartbeatIn, StreamEnded, StreamBroke)
REASONS: tuple[fsm.ReconnectReason, ...] = get_args(fsm.ReconnectReason)


def _policy(rng: random.Random, completion_cls: type) -> Policy:
    completion: policy.Completion
    if completion_cls is SettleAfterText:
        completion = SettleAfterText(rng.choice([15.0, 1.0, 40.0]))
    elif completion_cls is AtTextComplete:
        completion = AtTextComplete()
    else:
        assert completion_cls is AtCompleted
        completion = AtCompleted()
    deadline: policy.Deadline = rng.choice([At(rng.uniform(60, 4000)), Unbounded()])
    cap = deadline.t if isinstance(deadline, At) else 600.0
    p = Policy.make(
        deadline=deadline,
        # Mostly on: T24 cells need the stall term to come first.
        stall=rng.choice([StallAfter(rng.uniform(5, cap))] * 3 + [StallOff()]),
        completion=completion,
        answer_paths="ask_text_or_workflow",
        first_content=rng.choice([FirstContentOff(), FirstContentWithin(rng.uniform(1, 120))]),
        reconnect=rng.choice([Off(), Bounded(3, 6), Bounded(1, 1), Bounded(3, 8)]),
    )
    assert not isinstance(p, PolicyError)
    return p


def _ids(rng: random.Random) -> fsm.Ids:
    ctx = rng.choice([CTX, None])
    return rng.choice([NoIds(), UuidOnly(UUID, ctx), Known(ThreadRef(UUID, token()), ctx)])


def _phase(rng: random.Random, ph: type | None, started: float) -> fsm.Phase:
    if ph is AwaitingFirst:
        return AwaitingFirst()
    if ph is Producing:
        return Producing(started + rng.uniform(0, 300))
    assert ph is TextComplete
    return TextComplete(started + rng.uniform(0, 300), started + rng.uniform(0, 300))


def _state(rng: random.Random, p: Policy, k: type, ph: type | None) -> fsm.State:
    started = rng.uniform(0, 1000)
    dl: policy.Deadline = At(started + p.deadline.t) if isinstance(p.deadline, At) else Unbounded()
    conn = ConnId(rng.randint(1, 6))
    if k is Starting:
        return Starting(
            started, dl, conn, rng.randint(1, p.rate_limit_attempts), started + rng.uniform(0, 60)
        )
    if k is StartBackoff:
        return StartBackoff(started, dl, started + rng.uniform(0, 80), rng.randint(1, 3), conn)
    if k is Done:
        outcome = rng.choice(
            [
                Completed(0),
                Cut("stall", 1.0, 1),
                Lost("x"),
                ServerFailed(None),
                Rejected(SchemaError("x")),
            ]
        )
        return Done(outcome, _ids(rng))
    ids = _ids(rng)
    lv = Live(
        started_at=started,
        deadline=dl,
        ids=ids,
        cursor=None,
        reconnectable=rng.choice(["yes", "no", "absent"]),
        phase=_phase(rng, ph, started),
        rc_consecutive=rng.randint(0, 3),
        rc_total=rng.randint(3, 8) if rng.random() < 0.3 else rng.randint(0, 3),
        next_conn=ConnId(conn + 1),
    )
    reason = rng.choice(REASONS)
    target = ReconnectTarget(fsm.ids_uuid(ids) if not isinstance(ids, NoIds) else UUID)
    if k is Streaming:
        grace = rng.choice([NoGrace(), GraceUntil(started + rng.uniform(0, 400))])
        return Streaming(lv, conn, started + rng.uniform(0, 400), grace)
    if k is Reconnecting:
        return Reconnecting(lv, conn, target, reason, started + rng.uniform(0, 400))
    assert k is ReconnectBackoff
    return ReconnectBackoff(lv, started + rng.uniform(0, 400), target, reason)


def _anchor(s: fsm.State) -> float:
    if isinstance(s, (Starting, StartBackoff)):
        return s.started_at
    if isinstance(s, Done):
        return 0.0
    return s.live.started_at


def _failure(rng: random.Random, cls: object) -> fsm.Failure:
    if cls is RateLimited:
        return RateLimited(rng.choice([None, rng.uniform(0, 200)]))
    if cls is Transient:
        return Transient("connection reset")
    if cls is Gone:
        return Gone(rng.choice([403, 404, 410]))
    assert cls is Fatal
    return Fatal(rng.choice([AuthError("expired"), UnexpectedRedirect("302"), SchemaError("x")]))


def _frame(rng: random.Random, stage: fsm.Stage, change: fsm.Change) -> FrameSummary:
    return FrameSummary(
        stage=stage,
        change=change,
        uuid=rng.choice([None, UUID, UUID2]),
        token=rng.choice([None, token(), token("other-token-ABCDEFGHIJKLMNOP")]),
        cursor=rng.choice([None, "c-1"]),  # pyright: ignore[reportArgumentType]
        context=rng.choice([None, CTX]),
        reconnectable=rng.choice(["yes", "no", "absent"]),
        raw_status=rng.choice([None, "FAILED", "COMPLETED"]),
    )


def _tick_for(rng: random.Random, p: Policy, s: fsm.State, tag: str) -> Tick | None:
    times = [t for _, t in fsm.timers(p, s)] or [_anchor(s)]
    cands = [t + d for t in times for d in (0.0, -0.001, 0.001)]
    cands += [rng.uniform(min(times) - 50, max(times) + 50) for _ in range(8)]
    rng.shuffle(cands)
    for now in cands:
        if fsm.due_class(p, s, now) == tag:
            return Tick(now)
    return None


def _last_conn(s: fsm.State) -> int:
    """The conn most recently opened; a backoff state has none current."""
    match s:
        case Starting(conn=c) | Streaming(conn=c) | Reconnecting(conn=c):
            return c
        case StartBackoff(next_conn=n):
            return n - 1
        case ReconnectBackoff(live=lv):
            return lv.next_conn - 1
        case Done():
            return 1


def _event(  # noqa: PLR0911 - one return per event class
    rng: random.Random, p: Policy, s: fsm.State, ev: type, sub: Sub, conn_axis: str | None
) -> fsm.Event | None:
    if ev is Tick:
        if isinstance(s, Done):
            return Tick(rng.uniform(0, 1e5))
        return _tick_for(rng, p, s, str(sub))
    base = _last_conn(s)
    conn = ConnId(base if conn_axis == "current" else base - rng.randint(1, 3))
    now = _anchor(s) + rng.uniform(0, 500)
    if ev is Opened:
        return Opened(conn, now)
    if ev is OpenFailed:
        return OpenFailed(conn, now, _failure(rng, sub))
    if ev is FrameIn:
        stage, change = sub  # pyright: ignore[reportGeneralTypeIssues]
        return FrameIn(conn, now, _frame(rng, stage, change))
    if ev is HeartbeatIn:
        return HeartbeatIn(conn, now)
    if ev is StreamEnded:
        return StreamEnded(conn, now)
    assert ev is StreamBroke
    return StreamBroke(conn, now, sub, "detail")  # pyright: ignore[reportArgumentType]


# --- rules ---------------------------------------------------------------------------------------

Check = Callable[[Policy, fsm.State, fsm.Event, fsm.State, tuple[fsm.Effect, ...]], None]


@dataclass(frozen=True)
class Row:
    name: str
    match: Callable[[Cell], bool]
    check: Check


@dataclass(frozen=True)
class Unreachable:
    name: str
    reason: str
    pinned_by: str
    match: Callable[[Cell], bool]
    defensive: Check


@dataclass(frozen=True)
class NotConstructible:
    name: str
    reason: str
    match: Callable[[Cell], bool]


Rule = Row | Unreachable | NotConstructible


def _valid(c: Cell) -> bool:
    k, ph, comp = c[0], c[1], c[2]
    return k is not Done and not (ph is TextComplete and comp is not SettleAfterText)


def _conn_cur(c: Cell, k: type, ev: tuple[type, ...]) -> bool:
    return _valid(c) and c[0] is k and c[3] in ev and c[5] == "current"


def _tick(c: Cell, k: tuple[type, ...], tags: tuple[str, ...]) -> bool:
    return _valid(c) and c[3] is Tick and c[0] in k and c[4] in tags


def _unchanged(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert s2 is s
    assert eff == ()


def _live(s: fsm.State) -> Live:
    assert isinstance(s, (Streaming, Reconnecting, ReconnectBackoff))
    return s.live


def _conn_of(e: fsm.Event) -> ConnId:
    assert not isinstance(e, Tick)
    return e.conn


def _check_r(
    p: Policy,
    lv: Live,
    conn: ConnId,
    reason: fsm.ReconnectReason,
    now: float,
    s2: fsm.State,
    eff: tuple[fsm.Effect, ...],
    retry_after: float | None = None,
) -> None:
    """R(reason): reconnect exactly when every guard holds, else fall back."""
    assert eff[0] == Close(conn)
    if expected_reconnects(p, lv, now):
        assert isinstance(s2, ReconnectBackoff)
        assert s2.live == _counted(lv)
        assert s2.reason == reason
        assert not isinstance(lv.ids, NoIds)
        assert s2.target == ReconnectTarget(fsm.ids_uuid(lv.ids))
        low = min(p.backoff_cap_s, p.backoff_base_s * 2**lv.rc_consecutive * policy.JITTER_LOW)
        high = min(p.backoff_cap_s, p.backoff_base_s * 2**lv.rc_consecutive * policy.JITTER_HIGH)
        if retry_after is not None:
            floor = min(max(retry_after, 0.0), policy.RATE_LIMIT_CAP_S)
            low, high = max(low, floor), max(high, floor)
        assert low - 1e-9 <= s2.until - now <= high + 1e-9
        assert eff[-1] == Notice("reconnect", f"{reason}; reconnecting")
    else:
        assert s2 == Done(expected_fallback(p, reason, lv), lv.ids)


def _counted(lv: Live) -> Live:
    return replace(lv, rc_consecutive=lv.rc_consecutive + 1, rc_total=lv.rc_total + 1)


def _chk_t1(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    _unchanged(p, s, e, s2, eff)


def _chk_unreachable_state(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s2, Done)
    assert isinstance(s2.outcome, Rejected)
    assert type(s2.outcome.error) is SchemaError
    assert s2.ids == _live(s).ids
    cur = fsm.current_conn(s)
    assert eff == (() if cur is None else (Close(cur),))


def _chk_t3(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Starting)
    assert isinstance(s2, Streaming)
    assert eff == ()
    assert (s2.conn, s2.last_byte_at, s2.grace) == (s.conn, e.now, NoGrace())
    lv = s2.live
    assert (lv.started_at, lv.deadline, lv.ids, lv.phase, lv.rc_consecutive, lv.rc_total) == (
        s.started_at,
        s.deadline,
        NoIds(),
        AwaitingFirst(),
        0,
        0,
    )
    assert lv.next_conn == s.conn + 1


def _remaining(dl: policy.Deadline, now: float) -> float:
    return dl.t - now if isinstance(dl, At) else math.inf


def _chk_t4_t5(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Starting)
    assert isinstance(e, OpenFailed)
    assert isinstance(e.f, RateLimited)
    rem = _remaining(s.deadline, e.now)
    if s.rl_attempts < p.rate_limit_attempts and rem > 0:
        assert isinstance(s2, StartBackoff)
        assert e.now <= s2.until <= e.now + min(rem, policy.RATE_LIMIT_CAP_S * policy.JITTER_HIGH)
        assert (s2.rl_attempts, s2.next_conn, s2.deadline) == (
            s.rl_attempts,
            s.conn + 1,
            s.deadline,
        )
        assert len(eff) == 1
        assert isinstance(eff[0], Notice)
    else:
        assert isinstance(s2, Done)
        assert s2.ids == NoIds()
        assert isinstance(s2.outcome, Rejected)
        assert isinstance(s2.outcome.error, RateLimitError)
        assert eff == ()


def _chk_t5(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(e, OpenFailed)
    assert isinstance(s2, Done)
    assert s2.ids == NoIds()
    assert eff == ()
    assert isinstance(s2.outcome, Rejected)
    err = s2.outcome.error
    match e.f:
        case Transient():
            assert type(err) is NetworkError
        case Gone():
            assert type(err) is SchemaError
        case Fatal(f_err):
            assert err is f_err
        case RateLimited():
            raise AssertionError


def _chk_start_body(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Starting)
    assert isinstance(s2, Done)
    assert isinstance(s2.outcome, Rejected)
    assert type(s2.outcome.error) is SchemaError
    assert s2.ids == NoIds()
    assert eff == (Close(s.conn),)


def _chk_t8(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Reconnecting)
    assert s2 == Streaming(s.live, s.conn, e.now, GraceUntil(e.now + p.grace_s))
    assert eff == ()


def _chk_t9(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Reconnecting)
    assert isinstance(e, OpenFailed)
    retry = e.f.retry_after if isinstance(e.f, RateLimited) else None
    _check_r(p, s.live, s.conn, s.reason, e.now, s2, eff, retry)
    assert isinstance(eff[1], Notice)
    assert eff[1].kind == "reconnect_failed"


def _chk_t10(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Reconnecting)
    assert isinstance(e, OpenFailed)
    assert isinstance(s2, Done)
    assert s2.ids == s.live.ids
    ph = s.live.phase
    if isinstance(e.f, Fatal) and isinstance(e.f.err, AuthError):
        assert eff == (Close(s.conn), Notice("auth", AUTH_NOTICE))
        expected = {
            AwaitingFirst: Rejected(e.f.err),
            Producing: EndedEarly(s.live.rc_total, "auth"),
            TextComplete: SettledWithoutTerminal(s.live.rc_total, "auth"),
        }[type(ph)]
        assert s2.outcome == expected
    else:
        assert eff[0] == Close(s.conn)
        assert isinstance(eff[1], Notice)
        assert eff[1].detail != AUTH_NOTICE
        assert s2.outcome == expected_fallback(p, s.reason, s.live)
        assert getattr(s2.outcome, "by", None) != "auth"


def _chk_reconnecting_body(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Reconnecting)
    _check_r(p, s.live, s.conn, s.reason, e.now, s2, eff)


def _chk_t13(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Streaming)
    assert isinstance(e, FrameIn)
    assert s2 == Done(ServerFailed(e.s.raw_status), expected_ids(s.live.ids, e.s))
    assert eff == (Close(s.conn),)


def _chk_t14(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Streaming)
    assert isinstance(e, FrameIn)
    assert s2 == Done(Completed(s.live.rc_total), expected_ids(s.live.ids, e.s))
    assert eff == (Close(s.conn),)


def _chk_t15_t16(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Streaming)
    assert isinstance(e, FrameIn)
    assert isinstance(s2, Streaming)
    assert eff == ()
    lv, lv2 = s.live, s2.live
    progress = e.s.change == "progress"
    assert s2.conn == s.conn
    assert s2.last_byte_at == e.now
    assert s2.grace == s.grace
    assert lv2.phase == expected_phase(p, lv, e.s, e.now)
    assert lv2.ids == expected_ids(lv.ids, e.s)
    assert lv2.rc_consecutive == (0 if progress else lv.rc_consecutive)
    assert (lv2.rc_total, lv2.deadline, lv2.started_at, lv2.next_conn) == (
        lv.rc_total,
        lv.deadline,
        lv.started_at,
        lv.next_conn,
    )


def _chk_t17(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Streaming)
    assert s2 == Streaming(s.live, s.conn, e.now, s.grace)
    assert eff == ()


def _chk_streamed_r(reason: fsm.ReconnectReason) -> Check:
    def chk(
        p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
    ) -> None:
        assert isinstance(s, Streaming)
        _check_r(p, s.live, s.conn, reason, e.now, s2, eff)

    return chk


def _chk_t19(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Streaming)
    assert isinstance(e, StreamBroke)
    assert isinstance(s2, Done)
    assert s2.ids == s.live.ids
    assert isinstance(s2.outcome, Rejected)
    assert type(s2.outcome.error) is {"oversize": SchemaError, "cap": ResourceLimitError}[e.kind]
    assert eff == (Close(s.conn),)


def _chk_t20(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    _chk_streamed_r("drop")(p, s, e, s2, eff)


def _chk_t21(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s2, Done)
    cur = fsm.current_conn(s)
    assert eff == (() if cur is None else (Close(cur),))
    if isinstance(s, (Starting, StartBackoff)):
        assert isinstance(s.deadline, At)
        assert s2 == Done(Cut("deadline", s.deadline.t - s.started_at, 0), NoIds())
    else:
        lv = _live(s)
        assert s2 == Done(expected_deadline_outcome(lv), lv.ids)


def _chk_t6(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Starting)
    assert isinstance(s2, Done)
    assert s2.ids == NoIds()
    assert isinstance(s2.outcome, Rejected)
    assert type(s2.outcome.error) is NetworkError
    assert eff == (Close(s.conn),)


def _chk_t7(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, StartBackoff)
    assert s2 == Starting(
        s.started_at, s.deadline, s.next_conn, s.rl_attempts + 1, e.now + p.open_s
    )
    assert eff == (Open(s.next_conn, InitialPost(), p.low_speed_s),)


def _chk_t11(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Reconnecting)
    _check_r(p, s.live, s.conn, s.reason, e.now, s2, eff)
    assert Notice("open_timeout", "reconnect open timeout") in eff


def _chk_t12(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, ReconnectBackoff)
    assert isinstance(s2, Reconnecting)
    conn = s.live.next_conn
    assert (s2.conn, s2.target, s2.reason, s2.open_due_at) == (
        conn,
        s.target,
        s.reason,
        e.now + p.open_s,
    )
    assert s2.live == _replace_next(s.live, conn + 1)
    assert eff == (Open(conn, s.target, p.low_speed_s),)


def _replace_next(lv: Live, n: int) -> Live:
    return replace(lv, next_conn=ConnId(n))


def _chk_tick_r(reason: fsm.ReconnectReason) -> Check:
    def chk(
        p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
    ) -> None:
        assert isinstance(s, Streaming)
        # I26: a timer never reconnects before a new conn's grace ends.
        if isinstance(s.grace, GraceUntil):
            assert e.now >= s.grace.t
        _check_r(p, s.live, s.conn, reason, e.now, s2, eff)

    return chk


def _chk_t22(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Streaming)
    assert isinstance(s.live.phase, TextComplete)
    assert isinstance(p.completion, SettleAfterText)
    assert e.now >= s.live.phase.at + p.completion.settle_s
    _chk_tick_r("settle")(p, s, e, s2, eff)


def _chk_t23(
    p: Policy, s: fsm.State, e: fsm.Event, s2: fsm.State, eff: tuple[fsm.Effect, ...]
) -> None:
    assert isinstance(s, Streaming)
    assert isinstance(p.first_content, FirstContentWithin)
    assert e.now >= s.live.started_at + p.first_content.s
    _chk_tick_r("first_content")(p, s, e, s2, eff)


BODY = BODY_EVENTS
DUE_STREAM = ("settle", "first_content", "stall", "silence")

RULES: list[Rule] = [
    Row("T2", lambda c: c[0] is Done, _chk_t1),
    Unreachable(
        "I14-state",
        "TextComplete is entered only under SettleAfterText",
        "I14 (model under each policy)",
        lambda c: c[0] in LIVE_KINDS and c[1] is TextComplete and c[2] is not SettleAfterText,
        _chk_unreachable_state,
    ),
    Row(
        "T1",
        lambda c: (
            _valid(c)
            and c[3] is not Tick
            and (c[5] == "stale" or c[0] in (StartBackoff, ReconnectBackoff))
        ),
        _chk_t1,
    ),
    Row("T3", lambda c: _conn_cur(c, Starting, (Opened,)), _chk_t3),
    Row(
        "T4|T5", lambda c: _conn_cur(c, Starting, (OpenFailed,)) and c[4] is RateLimited, _chk_t4_t5
    ),
    Row(
        "T5",
        lambda c: _conn_cur(c, Starting, (OpenFailed,)) and c[4] in (Transient, Gone, Fatal),
        _chk_t5,
    ),
    Unreachable(
        "Starting-body",
        "the per-conn reader emits Opened or OpenFailed before any body item",
        "model (no body event before Opened)",
        lambda c: _conn_cur(c, Starting, BODY),
        _chk_start_body,
    ),
    Row("T8", lambda c: _conn_cur(c, Reconnecting, (Opened,)), _chk_t8),
    Row(
        "T9",
        lambda c: _conn_cur(c, Reconnecting, (OpenFailed,)) and c[4] in (RateLimited, Transient),
        _chk_t9,
    ),
    Row(
        "T10",
        lambda c: _conn_cur(c, Reconnecting, (OpenFailed,)) and c[4] in (Gone, Fatal),
        _chk_t10,
    ),
    Unreachable(
        "Reconnecting-body",
        "the per-conn reader emits Opened or OpenFailed before any body item",
        "model (no body event before Opened)",
        lambda c: _conn_cur(c, Reconnecting, BODY),
        _chk_reconnecting_body,
    ),
    Unreachable(
        "Streaming-open",
        "a conn reports Opened or OpenFailed once, before its body",
        "model (one open result per conn)",
        lambda c: _conn_cur(c, Streaming, (Opened, OpenFailed)),
        _chk_t1,
    ),
    Row("T13", lambda c: _conn_cur(c, Streaming, (FrameIn,)) and c[4][0] == "failed", _chk_t13),  # pyright: ignore[reportIndexIssue]
    Row(
        "T14",
        lambda c: (
            _conn_cur(c, Streaming, (FrameIn,))
            and (c[4][0] == "completed" or (c[4][0] == "text_complete" and c[2] is AtTextComplete))
        ),  # pyright: ignore[reportIndexIssue]
        _chk_t14,
    ),
    Row(
        "T15",
        lambda c: (
            _conn_cur(c, Streaming, (FrameIn,))
            and c[4][1] == "progress"  # pyright: ignore[reportIndexIssue]
            and (
                c[4][0] in ("pending", "other")
                or (c[4][0] == "text_complete" and c[2] is not AtTextComplete)
            )
        ),  # pyright: ignore[reportIndexIssue]
        _chk_t15_t16,
    ),
    Row(
        "T16",
        lambda c: (
            _conn_cur(c, Streaming, (FrameIn,))
            and c[4][1] == "idle"  # pyright: ignore[reportIndexIssue]
            and (
                c[4][0] in ("pending", "other")
                or (c[4][0] == "text_complete" and c[2] is not AtTextComplete)
            )
        ),  # pyright: ignore[reportIndexIssue]
        _chk_t15_t16,
    ),
    Row("T17", lambda c: _conn_cur(c, Streaming, (HeartbeatIn,)), _chk_t17),
    Row("T18", lambda c: _conn_cur(c, Streaming, (StreamEnded,)), _chk_streamed_r("eof")),
    Row(
        "T19",
        lambda c: _conn_cur(c, Streaming, (StreamBroke,)) and c[4] in ("oversize", "cap"),
        _chk_t19,
    ),
    Row("T20", lambda c: _conn_cur(c, Streaming, (StreamBroke,)) and c[4] == "transport", _chk_t20),
    Row("T21", lambda c: _tick(c, (Starting, StartBackoff, *LIVE_KINDS), ("deadline",)), _chk_t21),
    Row("T6", lambda c: _tick(c, (Starting,), ("open_due",)), _chk_t6),
    Row("T7", lambda c: _tick(c, (StartBackoff,), ("backoff",)), _chk_t7),
    Row("T11", lambda c: _tick(c, (Reconnecting,), ("open_due",)), _chk_t11),
    Row("T12", lambda c: _tick(c, (ReconnectBackoff,), ("backoff",)), _chk_t12),
    Row("T22", lambda c: _tick(c, (Streaming,), ("settle",)) and c[1] is TextComplete, _chk_t22),
    Row(
        "T23",
        lambda c: _tick(c, (Streaming,), ("first_content",)) and c[1] is AwaitingFirst,
        _chk_t23,
    ),
    Row("T24", lambda c: _tick(c, (Streaming,), ("stall",)), _chk_tick_r("stall")),
    Row("T25", lambda c: _tick(c, (Streaming,), ("silence",)), _chk_tick_r("silence")),
    Row("T26", lambda c: _tick(c, (Starting, StartBackoff, *LIVE_KINDS), ("none",)), _chk_t1),
    NotConstructible(
        "no-stream-timers-before-first-byte",
        "Starting has only the deadline and open_due timers",
        lambda c: _tick(c, (Starting,), ("backoff", *DUE_STREAM)),
    ),
    NotConstructible(
        "start-backoff-timers",
        "StartBackoff has only the deadline and its until",
        lambda c: _tick(c, (StartBackoff,), ("open_due", *DUE_STREAM)),
    ),
    NotConstructible(
        "reconnecting-timers",
        "Reconnecting has only the deadline and open_due",
        lambda c: _tick(c, (Reconnecting,), ("backoff", *DUE_STREAM)),
    ),
    NotConstructible(
        "reconnect-backoff-timers",
        "ReconnectBackoff has only the deadline and its until",
        lambda c: _tick(c, (ReconnectBackoff,), ("open_due", *DUE_STREAM)),
    ),
    NotConstructible(
        "streaming-open-timers",
        "Streaming has no open or backoff timer",
        lambda c: _tick(c, (Streaming,), ("open_due", "backoff")),
    ),
    NotConstructible(
        "settle-outside-text-complete",
        "the settle term exists only in TextComplete",
        lambda c: _tick(c, (Streaming,), ("settle",)) and c[1] is not TextComplete,
    ),
    NotConstructible(
        "first-content-after-first-progress",
        "the first-content term exists only in AwaitingFirst",
        lambda c: _tick(c, (Streaming,), ("first_content",)) and c[1] is not AwaitingFirst,
    ),
]


def _rules_for(cell: Cell, rules: list[Rule]) -> list[Rule]:
    return [r for r in rules if r.match(cell)]


def coverage_errors(rules: list[Rule]) -> list[str]:
    errors: list[str] = []
    used: set[str] = set()
    for cell in all_cells():
        hits = _rules_for(cell, rules)
        if len(hits) != 1:
            errors.append(f"{cell}: {[r.name for r in hits]}")
        used.update(r.name for r in hits)
    errors.extend(f"orphan rule {r.name}" for r in rules if r.name not in used)
    return errors


def test_every_cell_has_exactly_one_rule() -> None:
    assert coverage_errors(RULES) == []


@pytest.mark.parametrize("i", range(len(RULES)))
def test_deleting_any_rule_leaves_a_cell_uncovered(i: int) -> None:
    """The mutation spot-check for the rule table."""
    assert coverage_errors(RULES[:i] + RULES[i + 1 :]) != []


# --- checking members ----------------------------------------------------------------------------


def _members(rng: random.Random, cell: Cell) -> Iterator[tuple[Policy, fsm.State, fsm.Event]]:
    k, ph, comp, ev, sub, conn_axis = cell
    for _ in range(40):
        p = _policy(rng, comp)
        s = _state(rng, p, k, ph)
        e = _event(rng, p, s, ev, sub, conn_axis)
        if e is not None:
            yield p, s, e


def _check_member(rule: Rule, cell: Cell, p: Policy, s: fsm.State, e: fsm.Event, u: float) -> None:
    assert fsm.state_class(s) == (cell[0], cell[1])
    if cell[0] is not Done and fsm.current_conn(s) is not None:
        assert fsm.event_class(p, e, s) == (cell[3], cell[4], cell[5])
    s2, eff = fsm.step(p, s, e, u)
    assert isinstance(rule, (Row, Unreachable))
    (rule.check if isinstance(rule, Row) else rule.defensive)(p, s, e, s2, eff)


def _check_not_constructible(rng: random.Random, cell: Cell) -> None:
    k, ph, comp, _, tag, _ = cell
    for _ in range(8):
        p = _policy(rng, comp)
        s = _state(rng, p, k, ph)
        times = [t for _, t in fsm.timers(p, s)]
        nows = [t + d for t in times for d in (0.0, -0.001, 0.001, 1e6)]
        nows += [rng.uniform(-1e3, 1e5) for _ in range(20)]
        assert all(fsm.due_class(p, s, now) != tag for now in nows)


CELLS = all_cells()


def test_every_cell_checked_over_members() -> None:
    """Every Row and Unreachable cell is exercised on drawn members; a Row
    cell that no draw can build fails, since its rule would test nothing."""
    for idx, cell in enumerate(CELLS):
        (rule,) = _rules_for(cell, RULES)
        rng = random.Random(idx)
        if isinstance(rule, NotConstructible):
            _check_not_constructible(rng, cell)
            continue
        members = list(itertools.islice(_members(rng, cell), 4))
        if isinstance(rule, Row) or cell[3] is not Tick:
            assert members, (rule.name, cell)
        for p, s, e in members:
            _check_member(rule, cell, p, s, e, rng.random())


@settings(max_examples=400, deadline=None)
@given(
    idx=st.integers(0, len(CELLS) - 1),
    rnd=st.randoms(use_true_random=False),
    u=st.floats(0, 1, exclude_max=True),
)
def test_cells_hypothesis(idx: int, rnd: random.Random, u: float) -> None:
    cell = CELLS[idx]
    (rule,) = _rules_for(cell, RULES)
    if isinstance(rule, NotConstructible):
        _check_not_constructible(rnd, cell)
        return
    for p, s, e in itertools.islice(_members(rnd, cell), 2):
        _check_member(rule, cell, p, s, e, u)
