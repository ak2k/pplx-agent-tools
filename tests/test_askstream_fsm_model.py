"""Model test (plan §3.7): random histories through the lifecycle, with the
invariants checked after every step.

Events are the ones a real per-conn reader can produce (one open result per
conn, body items only after Opened), plus stale-conn events of every kind,
Ticks at `next_wake`, and clock jumps.
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from pplx_agent_tools.askstream import fsm, trace
from pplx_agent_tools.askstream.cleanup import (
    Delete,
    DeleteKept,
    DeleteNotNeeded,
    DeleteNoToken,
    Terminate,
    TerminateNotNeeded,
    TerminateUnsupported,
    cleanup_plan,
)
from pplx_agent_tools.askstream.fsm import (
    AUTH_NOTICE,
    AwaitingFirst,
    Close,
    Done,
    Fatal,
    FrameIn,
    FrameSummary,
    Gone,
    HeartbeatIn,
    InitialPost,
    Known,
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
from pplx_agent_tools.askstream.ids import ConnId
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
    RATE_LIMIT_CAP_S,
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
from pplx_agent_tools.errors import AuthError, SchemaError
from tests._fsm import CTX, TOKEN_RAW, UUID, UUID2, expected_ids, token


@st.composite
def policies(draw: st.DrawFn) -> Policy:
    deadline = draw(st.sampled_from([At(draw(st.floats(40, 700))), Unbounded()]))
    cap = deadline.t if isinstance(deadline, At) else 700.0
    p = Policy.make(
        deadline=deadline,
        stall=draw(st.sampled_from([StallAfter(draw(st.floats(5, cap))), StallOff()])),
        completion=draw(
            st.sampled_from(
                [SettleAfterText(15.0), SettleAfterText(3.0), AtTextComplete(), AtCompleted()]
            )
        ),
        answer_paths="ask_text_or_workflow",
        first_content=draw(
            st.sampled_from([FirstContentOff(), FirstContentWithin(90.0), FirstContentWithin(4.0)])
        ),
        reconnect=draw(st.sampled_from([Off(), Bounded(3, 6), Bounded(3, 8), Bounded(1, 2)])),
        silence_s=draw(st.sampled_from([25.0, 8.0])),
    )
    assert not isinstance(p, PolicyError)
    return p


FRAMES = st.builds(
    FrameSummary,
    stage=st.sampled_from(
        ["pending"] * 5 + ["text_complete"] * 2 + ["completed", "failed", "other"]
    ),
    change=st.sampled_from(["progress", "progress", "idle"]),
    uuid=st.sampled_from([UUID, UUID, None, UUID2]),
    token=st.sampled_from([token(), None, token("second-token-ABCDEFGHIJKLMNOPQR")]),
    cursor=st.none(),
    context=st.sampled_from([None, CTX]),
    reconnectable=st.sampled_from(["yes", "absent", "absent", "no"]),
    raw_status=st.sampled_from([None, "FAILED"]),
)
FAILURES = st.one_of(
    st.builds(RateLimited, st.none() | st.floats(0, 90)),
    st.just(Transient("reset")),
    st.just(Gone(403)),
    st.sampled_from([Fatal(AuthError("expired")), Fatal(SchemaError("redirect"))]),
)
STEP = st.floats(0, 6)
U = st.floats(0, 1, exclude_max=True)


def _rank(ids: fsm.Ids) -> int:
    return {NoIds: 0, UuidOnly: 1, Known: 2}[type(ids)]


def _phase_rank(ph: fsm.Phase) -> int:
    return {AwaitingFirst: 0, Producing: 1, TextComplete: 2}[type(ph)]


def _lp(s: fsm.State) -> float | None:
    if not isinstance(s, (Streaming, Reconnecting, ReconnectBackoff)):
        return None
    ph = s.live.phase
    return s.live.started_at if isinstance(ph, AwaitingFirst) else ph.last_progress_at


def _deadline(s: fsm.State) -> float:
    if isinstance(s, Done):
        return math.inf
    dl = s.deadline if isinstance(s, (Starting, StartBackoff)) else s.live.deadline
    return dl.t if isinstance(dl, At) else math.inf


class LifecycleModel(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.p: Policy
        self.s: fsm.State

    @initialize(p=policies(), t0=st.floats(0, 1e6))
    def start(self, p: Policy, t0: float) -> None:
        self.p = p
        self.s, eff = fsm.initial(p, t0)
        self.t0 = self.now = t0
        self.ring = trace.TraceRing()
        self.live_conns: set[int] = {1}
        self.max_conn = 1
        self.results: set[int] = set()  # conns that reported Opened or OpenFailed
        self.opened: set[int] = set()
        self.shadow: fsm.Ids = NoIds()
        self.initial_posts = 1
        self.reconnect_opens = 0
        self.went_live = False
        self.late = False
        self.last_progress = t0
        self.saw_429 = False
        assert eff == (Open(ConnId(1), InitialPost(), p.low_speed_s),)

    # --- event sources ---------------------------------------------------------------------------

    def _advance(self, dt: float) -> float:
        wake = fsm.next_wake(self.p, self.s)
        self.now = self.now + dt if wake is None else min(self.now + dt, max(self.now, wake))
        return self.now

    def _cur(self) -> int | None:
        return fsm.current_conn(self.s)

    @precondition(
        lambda self: (
            isinstance(self.s, (Starting, Reconnecting)) and self._cur() not in self.results
        )
    )
    @rule(dt=STEP, fail=st.integers(0, 4), f=FAILURES, u=U)
    def open_result(self, dt: float, fail: int, f: fsm.Failure, u: float) -> None:
        c = ConnId(self._cur() or 0)
        now = self._advance(dt)
        self._apply(OpenFailed(c, now, f) if fail == 0 else Opened(c, now), u)

    def _can_body(self) -> bool:
        return isinstance(self.s, Streaming) and self.s.conn in self.opened

    @precondition(_can_body)
    @rule(dt=STEP, frames=st.lists(FRAMES, min_size=1, max_size=4), u=U)
    def frames(self, dt: float, frames: list[FrameSummary], u: float) -> None:
        for fs in frames:
            if not self._can_body():
                return
            self._apply(FrameIn(ConnId(self._cur() or 0), self._advance(dt), fs), u)

    @precondition(_can_body)
    @rule(dt=STEP, u=U)
    def heartbeat(self, dt: float, u: float) -> None:
        self._apply(HeartbeatIn(ConnId(self._cur() or 0), self._advance(dt)), u)

    @precondition(_can_body)
    @rule(dt=STEP, kind=st.sampled_from(["end", "transport", "oversize", "cap"]), u=U)
    def stream_end(self, dt: float, kind: str, u: float) -> None:
        c, now = ConnId(self._cur() or 0), self._advance(dt)
        self._apply(StreamEnded(c, now) if kind == "end" else StreamBroke(c, now, kind, "m"), u)  # pyright: ignore[reportArgumentType]

    @precondition(
        lambda self: (
            not isinstance(self.s, Done)
            and (self.max_conn > 1 or isinstance(self.s, (StartBackoff, ReconnectBackoff)))
        )
    )
    @rule(which=st.integers(0, 5), f=FAILURES, fs=FRAMES, back=st.integers(0, 3), u=U)
    def stale(self, which: int, f: fsm.Failure, fs: FrameSummary, back: int, u: float) -> None:
        cur = self._cur()
        olds = [c for c in range(1, self.max_conn + 1) if c != cur] or [self.max_conn - back]
        c = ConnId(olds[back % len(olds)])
        now = self.now
        e: fsm.Event = [
            Opened(c, now),
            OpenFailed(c, now, f),
            FrameIn(c, now, fs),
            HeartbeatIn(c, now),
            StreamEnded(c, now),
            StreamBroke(c, now, "transport", "m"),
        ][which]
        self._apply(e, u)

    @precondition(lambda self: not isinstance(self.s, Done))
    @rule(u=U, skip=st.integers(0, 2))
    def tick_at_wake(self, u: float, skip: int) -> None:
        # Skips keep timers from ending most runs before any content arrives.
        if skip:
            return
        wake = fsm.next_wake(self.p, self.s)
        assert wake is not None and math.isfinite(wake)  # I11
        self.now = max(self.now, wake)
        self._apply(Tick(self.now), u)

    @precondition(lambda self: not isinstance(self.s, Done))
    @rule(dt=st.floats(0, 120) | st.floats(0, 3000), u=U, skip=st.integers(0, 3))
    def clock_jump(self, dt: float, u: float, skip: int) -> None:
        if skip:
            return
        wake = fsm.next_wake(self.p, self.s)
        self.now += dt
        if wake is not None and self.now > wake:
            self.late = True
        self._apply(Tick(self.now), u)

    # --- one step and its invariants -------------------------------------------------------------

    def _apply(self, e: fsm.Event, u: float) -> None:  # noqa: PLR0912, PLR0915
        p, before = self.p, self.s
        assert fsm.lifecycle_valid(p, before)
        s2, eff = fsm.step(p, before, e, u)
        self.ring.add(trace.entry(p, self.t0, before, e, s2, eff))
        cur = fsm.current_conn(before)
        is_conn = not isinstance(e, Tick)
        current = is_conn and e.conn == cur  # pyright: ignore[reportAttributeAccessIssue]

        if isinstance(before, Done) or (is_conn and not current):  # I1, I3
            assert s2 is before
            assert eff == ()
            return

        # I2: one live conn; ids strictly increase; each Open after the previous conn ended.
        if current and isinstance(e, (Opened, OpenFailed)):
            self.results.add(e.conn)
        if current and isinstance(e, Opened):
            self.opened.add(e.conn)
        if current and isinstance(e, (OpenFailed, StreamEnded, StreamBroke)):
            self.live_conns.discard(e.conn)
        for x in eff:
            if isinstance(x, Close):
                self.live_conns.discard(x.conn)
            elif isinstance(x, Open):
                assert not self.live_conns
                assert x.conn > self.max_conn
                self.max_conn = x.conn
                self.live_conns.add(x.conn)
                if isinstance(x.target, InitialPost):
                    self.initial_posts += 1
                    assert not self.went_live  # I6
                else:
                    self.reconnect_opens += 1
        assert self.initial_posts <= p.rate_limit_attempts  # I6
        if isinstance(p.reconnect, Bounded):  # I7
            assert self.reconnect_opens <= p.reconnect.total
        else:
            assert self.reconnect_opens == 0

        # I4, I16
        dl = _deadline(before)
        if isinstance(e, Tick) and e.now >= dl:
            assert isinstance(s2, Done)
            assert not any(isinstance(x, Open) for x in eff)
        if not isinstance(s2, Done):
            wake = fsm.next_wake(p, s2)
            assert wake is not None and math.isfinite(wake)  # I11
            assert wake <= _deadline(s2)
            assert _deadline(s2) == dl
        # I15 shadow merge of every frame delivered on a current conn
        if current and isinstance(e, FrameIn) and isinstance(before, Streaming):
            self.shadow = expected_ids(self.shadow, e.s)
        if isinstance(s2, (Streaming, Reconnecting, ReconnectBackoff)):
            self.went_live = True
            lv2 = s2.live
            assert lv2.ids == self.shadow
            if isinstance(p.reconnect, Bounded):
                assert lv2.rc_consecutive <= p.reconnect.consecutive  # I7
            assert isinstance(p.completion, SettleAfterText) or not isinstance(
                lv2.phase, TextComplete
            )  # I14
        if isinstance(s2, ReconnectBackoff) and not isinstance(before, ReconnectBackoff):
            assert isinstance(before, (Streaming, Reconnecting))
            assert before.live.reconnectable != "no"  # I7
            assert s2.target == ReconnectTarget(fsm.ids_uuid(before.live.ids))  # pyright: ignore[reportArgumentType]
        if isinstance(before, (Streaming, Reconnecting, ReconnectBackoff)) and isinstance(
            s2, (Streaming, Reconnecting, ReconnectBackoff)
        ):
            assert _rank(s2.live.ids) >= _rank(before.live.ids)  # I15
            ph, ph2 = before.live.phase, s2.live.phase
            assert _phase_rank(ph2) >= _phase_rank(ph)  # I18
            if isinstance(ph, TextComplete):
                assert isinstance(ph2, TextComplete)
                assert ph2.at == ph.at
            if _lp(s2) != _lp(before):  # I5
                assert current and isinstance(e, FrameIn) and e.s.change == "progress"
                lp2 = _lp(s2)
                assert lp2 is not None
                self.last_progress = lp2
        # I8: never early, and bounded late when every Tick came on time.
        if (
            isinstance(e, Tick)
            and isinstance(before, Streaming)
            and fsm.due_class(p, before, e.now) == "stall"
        ):
            lp = _lp(before)
            assert lp is not None
            assert isinstance(p.stall, StallAfter)
            assert e.now >= lp + p.stall.s
        if current and isinstance(e, OpenFailed) and isinstance(e.f, RateLimited):
            self.saw_429 = True
        cut = s2.outcome if isinstance(s2, Done) else None
        # With the stall check off only silence cuts, and bytes without progress defer it.
        stall_on = isinstance(p.stall, StallAfter)
        if isinstance(cut, Cut) and cut.cause == "stall" and not self.late and stall_on:
            k = p.reconnect.consecutive if isinstance(p.reconnect, Bounded) else 0
            # A 429 on a reconnect may wait its retry-after, capped, instead of the backoff.
            wait = max(p.backoff_cap_s, RATE_LIMIT_CAP_S) if self.saw_429 else p.backoff_cap_s
            assert isinstance(p.stall, StallAfter)
            bound = self.last_progress + p.stall.s + k * (wait + p.open_s + p.grace_s)
            assert self.now <= bound + 1e-6

        if isinstance(s2, Done):
            self._check_done(before, e, s2, eff)
        self._check_cleanup(s2)
        assert TOKEN_RAW not in repr((e, s2, eff))  # I13
        self.s = s2

    def _check_done(
        self, before: fsm.State, e: fsm.Event, s2: Done, eff: tuple[fsm.Effect, ...]
    ) -> None:
        p, o = self.p, s2.outcome
        if isinstance(before, (Starting, StartBackoff)):
            assert s2.ids == NoIds()  # I15
        else:
            assert s2.ids == self.shadow  # I15
        settle = isinstance(p.completion, SettleAfterText)
        assert settle or not isinstance(o, SettledWithoutTerminal)  # I14
        if isinstance(o, Cut) and o.cause == "first_content":  # I10
            assert isinstance(before, (Streaming, Reconnecting, ReconnectBackoff))
            assert isinstance(before.live.phase, AwaitingFirst)
        live_before = (
            before.live if isinstance(before, (Streaming, Reconnecting, ReconnectBackoff)) else None
        )
        if live_before is not None and isinstance(live_before.phase, TextComplete):  # I22
            assert isinstance(o, (SettledWithoutTerminal, Completed, ServerFailed, Rejected))
        auth_refusal = (
            isinstance(before, Reconnecting)
            and isinstance(e, OpenFailed)
            and isinstance(e.f, Fatal)
            and isinstance(e.f.err, AuthError)
        )
        by = getattr(o, "by", None)
        assert (by == "auth") == (auth_refusal and not isinstance(o, Rejected))  # I23
        if auth_refusal:
            assert live_before is not None
            assert not isinstance(o, (Lost, Cut))
            assert Notice("auth", AUTH_NOTICE) in eff
            if isinstance(live_before.phase, AwaitingFirst):
                assert isinstance(o, Rejected) and isinstance(o.error, AuthError)
            else:
                assert isinstance(o, (EndedEarly, SettledWithoutTerminal))

    def _check_cleanup(self, s: fsm.State) -> None:
        """I9: Terminate only where §3.5 lists it; never after the run is over."""
        for keep in (False, True):
            term, dele = cleanup_plan(s, keep, "m")
            ids: fsm.Ids
            if isinstance(s, Done):
                ids = s.ids
                over = isinstance(s.outcome, (Completed, SettledWithoutTerminal, ServerFailed))
            elif isinstance(s, (Starting, StartBackoff)):
                ids, over = NoIds(), True
            else:
                ids, over = s.live.ids, False
            if over or isinstance(ids, NoIds):
                assert term == TerminateNotNeeded()
            else:
                assert isinstance(term, (Terminate, TerminateUnsupported))
                assert isinstance(term, Terminate) == (ids.context is not None)
            expected: Any = (
                DeleteNotNeeded()
                if isinstance(ids, NoIds)
                else DeleteKept()
                if keep
                else Delete(ids.ref)
                if isinstance(ids, Known)
                else DeleteNoToken()
            )
            assert dele == expected

    @precondition(lambda self: isinstance(self.s, Done))
    @rule(dt=STEP, fs=FRAMES, u=U)
    def done_absorbs(self, dt: float, fs: FrameSummary, u: float) -> None:
        self.now += dt
        self._apply(FrameIn(ConnId(self.max_conn), self.now, fs), u)
        self._apply(Tick(self.now + 1e6), u)

    @invariant()
    def ring_bounded(self) -> None:  # I19
        if hasattr(self, "ring"):
            assert len(self.ring.entries) <= trace.RING


LifecycleModel.TestCase.settings = settings(
    max_examples=500,
    stateful_step_count=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
TestLifecycleModel = LifecycleModel.TestCase
