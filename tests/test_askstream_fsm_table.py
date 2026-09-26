"""The lifecycle truth table: one or more rows per T1-T26, the boundary rows,
and the rows plan §9 U4 names (settle, deadline in TextComplete, T10 with an
AuthError, the `by` values, the round 5 table, the R guard and fallback
enumerations)."""

from __future__ import annotations

import itertools
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pplx_agent_tools.askstream import frames, fsm
from pplx_agent_tools.askstream.fsm import (
    AUTH_NOTICE,
    AwaitingFirst,
    Close,
    Done,
    Fatal,
    FrameIn,
    Gone,
    GraceUntil,
    HeartbeatIn,
    InitialPost,
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
    ReconnectReason,
    ReconnectTarget,
    StartBackoff,
    Starting,
    StreamBroke,
    StreamEnded,
    Streaming,
    TextComplete,
    Tick,
    Transient,
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
    for_verb,
    jitter,
)
from pplx_agent_tools.errors import (
    AuthError,
    NetworkError,
    RateLimitError,
    ResourceLimitError,
    SchemaError,
)
from tests._fsm import (
    ASK,
    ASK_RC,
    CTX,
    KNOWN,
    UUID,
    UUID_ONLY,
    UnexpectedRedirect,
    expected_fallback,
    frame,
    live,
    policy,
    streaming,
    token,
)

C1 = ConnId(1)
C2 = ConnId(2)
REASONS: tuple[ReconnectReason, ...] = (
    "drop",
    "eof",
    "silence",
    "stall",
    "first_content",
    "settle",
)
PHASES: tuple[fsm.Phase, ...] = (AwaitingFirst(), Producing(10.0), TextComplete(10.0, 20.0))


def step(p: Policy, s: fsm.State, e: fsm.Event, u: float = 0.5) -> fsm.Step:
    return fsm.step(p, s, e, u)


def starting(conn: int = 1, rl: int = 1, open_due_at: float = 30.0) -> Starting:
    return Starting(0.0, At(540.0), ConnId(conn), rl, open_due_at)


def reconnecting(
    lv: fsm.Live, reason: ReconnectReason = "drop", open_due_at: float = 130.0
) -> Reconnecting:
    return Reconnecting(lv, ConnId(lv.next_conn - 1), ReconnectTarget(UUID), reason, open_due_at)


def rc_backoff(
    lv: fsm.Live, until: float = 101.0, reason: ReconnectReason = "drop"
) -> ReconnectBackoff:
    return ReconnectBackoff(lv, until, ReconnectTarget(UUID), reason)


# --- T1, T2 --------------------------------------------------------------------------------------

CONN_EVENTS = (
    lambda c: Opened(c, 5.0),
    lambda c: OpenFailed(c, 5.0, Transient("x")),
    lambda c: FrameIn(c, 5.0, frame("completed")),
    lambda c: HeartbeatIn(c, 5.0),
    lambda c: StreamEnded(c, 5.0),
    lambda c: StreamBroke(c, 5.0, "cap", "m"),
)


@pytest.mark.parametrize("make", CONN_EVENTS)
@pytest.mark.parametrize(
    "state",
    [
        starting(conn=2),
        streaming(live(next_conn=3)),
        reconnecting(live(next_conn=3, ids=KNOWN)),
        rc_backoff(live(next_conn=3, ids=KNOWN)),
        StartBackoff(0.0, At(540.0), 50.0, 1, C2),
    ],
)
def test_t1_stale_conn_is_a_no_op(state: fsm.State, make: object) -> None:
    e = make(C1)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
    assert step(ASK, state, e) == (state, ())  # pyright: ignore[reportUnknownArgumentType]


@pytest.mark.parametrize("make", [*CONN_EVENTS, lambda _c: Tick(10_000.0)])
def test_t2_done_absorbs(make: object) -> None:
    d = Done(Completed(0), KNOWN)
    e = make(C1)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
    assert step(ASK, d, e) == (d, ())  # pyright: ignore[reportUnknownArgumentType]


# --- start phase: T3-T7 --------------------------------------------------------------------------


def test_initial_emits_one_initial_post() -> None:
    s, effects = fsm.initial(ASK, 100.0)
    assert s == Starting(100.0, At(640.0), C1, 1, 130.0)
    assert effects == (Open(C1, InitialPost(), 35.0),)


def test_initial_unbounded_deadline() -> None:
    s, _ = fsm.initial(policy(deadline=Unbounded()), 0.0)
    assert s.deadline == Unbounded()


def test_t3_opened_starts_streaming() -> None:
    s, eff = step(ASK, starting(), Opened(C1, 2.0))
    assert eff == ()
    assert s == Streaming(live(next_conn=2), C1, 2.0, NoGrace())


@given(retry_after=st.none() | st.floats(0, 1000), u=st.floats(0, 1, exclude_max=True))
def test_t4_rate_limited_backs_off(retry_after: float | None, u: float) -> None:
    s, eff = step(ASK, starting(), OpenFailed(C1, 2.0, RateLimited(retry_after)), u)
    assert isinstance(s, StartBackoff)
    base = 5.0 if retry_after is None else min(retry_after, 60.0)
    assert s.until == pytest.approx(2.0 + base * jitter(u))
    assert (s.rl_attempts, s.next_conn) == (1, C2)
    assert len(eff) == 1
    assert isinstance(eff[0], Notice)
    assert eff[0].kind == "rate_limited"


def test_t4_backoff_never_passes_the_deadline() -> None:
    s, _ = step(ASK, starting(open_due_at=600.0), OpenFailed(C1, 530.0, RateLimited(60.0)))
    assert isinstance(s, StartBackoff)
    assert s.until == 540.0


@pytest.mark.parametrize(
    ("f", "err_type"),
    [
        (Transient("connect refused"), NetworkError),
        (Gone(404), SchemaError),
        (Fatal(AuthError("cookies")), AuthError),
        (Fatal(UnexpectedRedirect("302")), UnexpectedRedirect),
    ],
)
def test_t5_open_failed_rejects(f: fsm.Failure, err_type: type[Exception]) -> None:
    s, eff = step(ASK, starting(), OpenFailed(C1, 2.0, f))
    assert eff == ()
    assert isinstance(s, Done)
    assert s.ids == NoIds()
    assert isinstance(s.outcome, Rejected)
    assert type(s.outcome.error) is err_type
    if isinstance(f, Fatal):
        assert s.outcome.error is f.err


def test_t5_rate_limit_exhausted() -> None:
    s, eff = step(ASK, starting(rl=3), OpenFailed(C1, 2.0, RateLimited(3.0)))
    assert eff == ()
    assert isinstance(s, Done)
    assert isinstance(s.outcome, Rejected)
    assert isinstance(s.outcome.error, RateLimitError)
    assert s.outcome.error.retry_after == 3.0


def test_t5_rate_limited_with_no_time_left() -> None:
    s, _ = step(ASK, starting(open_due_at=600.0), OpenFailed(C1, 540.0, RateLimited(3.0)))
    assert isinstance(s, Done)
    assert isinstance(s.outcome, Rejected)


def test_t6_open_timeout() -> None:
    s, eff = step(ASK, starting(), Tick(30.0))
    assert eff == (Close(C1),)
    assert isinstance(s, Done)
    assert s.ids == NoIds()
    assert isinstance(s.outcome, Rejected)
    assert type(s.outcome.error) is NetworkError


def test_t6_boundary_one_ms_early() -> None:
    st0 = starting()
    assert step(ASK, st0, Tick(29.999)) == (st0, ())


def test_t7_backoff_elapsed_reopens() -> None:
    s, eff = step(ASK, StartBackoff(0.0, At(540.0), 50.0, 1, C2), Tick(50.0))
    assert s == Starting(0.0, At(540.0), C2, 2, 80.0)
    assert eff == (Open(C2, InitialPost(), 35.0),)


def test_start_body_event_before_opened_is_defensive() -> None:
    s, eff = step(ASK, starting(), FrameIn(C1, 2.0, frame()))
    assert eff == (Close(C1),)
    assert isinstance(s, Done)
    assert isinstance(s.outcome, Rejected)
    assert s.ids == NoIds()


# --- reconnect states: T8-T12 --------------------------------------------------------------------


def test_t8_reconnect_opened_starts_grace() -> None:
    lv = live(ids=KNOWN, next_conn=3, phase=Producing(10.0))
    s, eff = step(ASK_RC, reconnecting(lv), Opened(C2, 105.0))
    assert eff == ()
    assert s == Streaming(lv, C2, 105.0, GraceUntil(135.0))


@pytest.mark.parametrize("f", [RateLimited(None), Transient("reset")])
def test_t9_retryable_failure_reconnects_again(f: fsm.Failure) -> None:
    lv = live(ids=KNOWN, next_conn=3, rc_consecutive=1, rc_total=1, phase=Producing(10.0))
    s, eff = step(ASK_RC, reconnecting(lv, "stall"), OpenFailed(C2, 105.0, f), 0.0)
    assert isinstance(s, ReconnectBackoff)
    assert (s.live.rc_consecutive, s.live.rc_total, s.reason) == (2, 2, "stall")
    assert s.until == pytest.approx(105.0 + 2 * 0.85)
    assert eff[0] == Close(C2)
    assert [type(x) for x in eff[1:]] == [Notice, Notice]


def test_t9_429_waits_for_retry_after() -> None:
    lv = live(ids=KNOWN, next_conn=3, rc_consecutive=1, rc_total=1)
    s, _ = step(ASK_RC, reconnecting(lv), OpenFailed(C2, 105.0, RateLimited(500.0)))
    assert isinstance(s, ReconnectBackoff)
    assert s.until == 165.0


@pytest.mark.parametrize("reason", REASONS)
def test_t9_exhausted_falls_back_with_the_starting_reason(reason: ReconnectReason) -> None:
    lv = live(ids=KNOWN, next_conn=3, rc_consecutive=3, rc_total=3, phase=Producing(10.0))
    s, eff = step(ASK_RC, reconnecting(lv, reason), OpenFailed(C2, 105.0, Transient("x")))
    assert isinstance(s, Done)
    assert s.outcome == expected_fallback(ASK_RC, reason, lv)
    assert s.ids == KNOWN
    assert eff[0] == Close(C2)


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize(
    "f", [Gone(403), Fatal(UnexpectedRedirect("302 other")), Fatal(SchemaError("x"))]
)
def test_t10_gone_or_fatal_falls_back(
    reason: ReconnectReason, phase: fsm.Phase, f: fsm.Failure
) -> None:
    lv = live(ids=KNOWN, next_conn=3, phase=phase)
    s, eff = step(ASK_RC, reconnecting(lv, reason), OpenFailed(C2, 105.0, f))
    assert isinstance(s, Done)
    assert s.outcome == expected_fallback(ASK_RC, reason, lv)
    assert s.ids == KNOWN
    assert eff[0] == Close(C2)
    assert isinstance(eff[1], Notice)
    assert eff[1].kind == "reconnect_failed"
    assert eff[1].detail != AUTH_NOTICE


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        (AwaitingFirst(), "rejected"),
        (Producing(10.0), EndedEarly(0, "auth")),
        (TextComplete(10.0, 20.0), SettledWithoutTerminal(0, "auth")),
    ],
)
def test_t10_auth_error_keeps_the_partial(
    reason: ReconnectReason, phase: fsm.Phase, expected: object
) -> None:
    err = AuthError("sign-in redirect to www.perplexity.ai (HTTP 302)")
    lv = live(ids=KNOWN, next_conn=3, phase=phase)
    s, eff = step(ASK_RC, reconnecting(lv, reason), OpenFailed(C2, 105.0, Fatal(err)))
    assert isinstance(s, Done)
    assert s.ids == KNOWN
    if expected == "rejected":
        assert s.outcome == Rejected(err)
    else:
        assert s.outcome == expected
    assert eff == (Close(C2), Notice("auth", AUTH_NOTICE))
    assert AUTH_NOTICE == "session cookies expired; refresh with pplx auth"


@pytest.mark.parametrize("reason", REASONS)
def test_t11_reconnect_open_timeout(reason: ReconnectReason) -> None:
    lv = live(ids=KNOWN, next_conn=3, rc_consecutive=1, rc_total=1)
    s, eff = step(ASK_RC, reconnecting(lv, reason), Tick(130.0))
    assert isinstance(s, ReconnectBackoff)
    assert s.reason == reason
    assert eff[0] == Close(C2)
    assert Notice("open_timeout", "reconnect open timeout") in eff


def test_t11_boundary_one_ms_early() -> None:
    s0 = reconnecting(live(ids=KNOWN, next_conn=3))
    assert step(ASK_RC, s0, Tick(129.999)) == (s0, ())


def test_t12_backoff_elapsed_opens_reconnect() -> None:
    lv = live(ids=KNOWN, next_conn=3)
    s, eff = step(ASK_RC, rc_backoff(lv, 101.0, "eof"), Tick(101.0))
    assert s == Reconnecting(
        replace(lv, next_conn=ConnId(4)), ConnId(3), ReconnectTarget(UUID), "eof", 131.0
    )
    assert eff == (Open(ConnId(3), ReconnectTarget(UUID), 35.0),)


def test_reconnecting_body_event_before_opened_is_defensive() -> None:
    lv = live(ids=KNOWN, next_conn=3, rc_consecutive=1, rc_total=1)
    s, eff = step(ASK_RC, reconnecting(lv, "silence"), HeartbeatIn(C2, 104.0))
    assert isinstance(s, ReconnectBackoff)
    assert s.reason == "silence"
    assert eff[0] == Close(C2)


# --- streaming: T13-T20 --------------------------------------------------------------------------


def test_t13_failed_frame() -> None:
    s, eff = step(
        ASK,
        streaming(),
        FrameIn(C1, 5.0, frame("failed", uuid=UUID, token=token(), raw_status="FAILED")),
    )
    assert s == Done(ServerFailed("FAILED"), fsm.Known(KNOWN.ref, None))
    assert eff == (Close(C1),)


@pytest.mark.parametrize("p", [ASK, policy(AtTextComplete()), policy(AtCompleted())])
def test_t14_completed_frame_completes_every_policy(p: Policy) -> None:
    s, eff = step(
        p,
        streaming(live(rc_total=2)),
        FrameIn(C1, 5.0, frame("completed", "idle", uuid=UUID, token=token(), context=CTX)),
    )
    assert s == Done(Completed(2), KNOWN)
    assert eff == (Close(C1),)


def test_t14_text_complete_completes_fetch_only() -> None:
    fetch = policy(AtTextComplete())
    s, _ = step(fetch, streaming(), FrameIn(C1, 5.0, frame("text_complete")))
    assert isinstance(s, Done)
    assert isinstance(s.outcome, Completed)
    for p in (ASK, policy(AtCompleted())):
        s, _ = step(p, streaming(), FrameIn(C1, 5.0, frame("text_complete")))
        assert isinstance(s, Streaming)


def test_t15_progress() -> None:
    lv = live(ids=UUID_ONLY, rc_consecutive=2, rc_total=2)
    s0 = streaming(lv, grace=GraceUntil(50.0))
    s, eff = step(
        ASK_RC, s0, FrameIn(C1, 7.0, frame(token=token(), cursor="c1", reconnectable="yes"))
    )  # pyright: ignore[reportArgumentType]
    assert eff == ()
    assert isinstance(s, Streaming)
    assert s.live.phase == Producing(7.0)
    assert (s.live.rc_consecutive, s.live.rc_total) == (0, 2)
    assert s.grace == GraceUntil(50.0)
    assert s.last_byte_at == 7.0
    assert s.live.ids == KNOWN
    assert (s.live.cursor, s.live.reconnectable) == ("c1", "yes")


def test_t15_text_complete_progress_enters_text_complete() -> None:
    s, _ = step(
        ASK, streaming(live(phase=Producing(3.0))), FrameIn(C1, 7.0, frame("text_complete"))
    )
    assert isinstance(s, Streaming)
    assert s.live.phase == TextComplete(7.0, 7.0)


@pytest.mark.parametrize(("phase", "lp"), [(AwaitingFirst(), 0.0), (Producing(3.0), 3.0)])
def test_t16_idle_text_complete_starts_the_window(phase: fsm.Phase, lp: float) -> None:
    """A `text_completed` frame with no new blocks still starts settle."""
    s, eff = step(
        ASK,
        streaming(live(phase=phase), grace=GraceUntil(50.0)),
        FrameIn(C1, 7.0, frame("text_complete", "idle")),
    )
    assert eff == ()
    assert isinstance(s, Streaming)
    assert s.live.phase == TextComplete(lp, 7.0)
    assert s.grace == GraceUntil(50.0)
    assert fsm.next_wake(ASK, s) == 50.0  # settle at 22, raised to the grace
    assert fsm.next_wake(ASK, replace(s, grace=NoGrace())) == 22.0


def test_t16_idle_frame_moves_only_last_byte() -> None:
    s0 = streaming(live(phase=Producing(3.0), rc_consecutive=1))
    s, eff = step(ASK, s0, FrameIn(C1, 7.0, frame("pending", "idle")))
    assert eff == ()
    assert s == replace(s0, last_byte_at=7.0)


def test_t17_heartbeat() -> None:
    s0 = streaming(live(phase=Producing(3.0)))
    assert step(ASK, s0, HeartbeatIn(C1, 9.0)) == (replace(s0, last_byte_at=9.0), ())


def test_t18_eof_without_reconnect_ends_early() -> None:
    s, eff = step(ASK, streaming(live(phase=Producing(3.0), ids=KNOWN)), StreamEnded(C1, 9.0))
    assert s == Done(EndedEarly(0, "server"), KNOWN)
    assert eff == (Close(C1),)


@pytest.mark.parametrize(
    ("kind", "err_type"), [("oversize", SchemaError), ("cap", ResourceLimitError)]
)
def test_t19_oversize_and_cap_reject(kind: fsm.BrokeKind, err_type: type[Exception]) -> None:
    s, eff = step(
        ASK,
        streaming(live(phase=Producing(3.0), ids=KNOWN)),
        StreamBroke(C1, 9.0, kind, "cap total_weight"),
    )
    assert isinstance(s, Done)
    assert s.ids == KNOWN
    assert isinstance(s.outcome, Rejected)
    assert type(s.outcome.error) is err_type
    assert eff == (Close(C1),)


def test_t20_drop_without_reconnect_is_lost() -> None:
    s, _ = step(
        ASK,
        streaming(live(phase=Producing(3.0), ids=KNOWN)),
        StreamBroke(C1, 9.0, "transport", "reset"),
    )
    assert s == Done(Lost("stream dropped"), KNOWN)


def test_t20_drop_with_reconnect() -> None:
    lv = live(phase=Producing(3.0), ids=KNOWN)
    s, eff = step(ASK_RC, streaming(lv), StreamBroke(C1, 9.0, "transport", "reset"), 0.5)
    assert s == ReconnectBackoff(
        replace(lv, rc_consecutive=1, rc_total=1), 10.0, ReconnectTarget(UUID), "drop"
    )
    assert eff[0] == Close(C1)
    assert isinstance(eff[-1], Notice)


# --- ticks: T21-T26 and boundaries ---------------------------------------------------------------

LIVE_KINDS = ("streaming", "reconnecting", "backoff")


def _live_state(kind: str, phase: fsm.Phase) -> fsm.LiveState:
    lv = live(phase=phase, ids=KNOWN, next_conn=3)
    if kind == "streaming":
        return streaming(lv, last_byte_at=530.0)
    if kind == "reconnecting":
        return reconnecting(lv, open_due_at=560.0)
    return rc_backoff(lv, until=550.0)


@pytest.mark.parametrize("kind", LIVE_KINDS)
@pytest.mark.parametrize("phase", PHASES)
def test_t21_deadline_live(kind: str, phase: fsm.Phase) -> None:
    s0 = _live_state(kind, phase)
    s, eff = step(ASK_RC, s0, Tick(540.0))
    assert isinstance(s, Done)
    assert s.ids == KNOWN
    if isinstance(phase, TextComplete):
        assert s.outcome == SettledWithoutTerminal(0, "deadline")
    else:
        assert s.outcome == Cut("deadline", 540.0, 0)
    assert eff == (() if kind == "backoff" else (Close(C2),))


@pytest.mark.parametrize("kind", LIVE_KINDS)
@pytest.mark.parametrize("phase", PHASES)
def test_t21_boundary_one_ms_before_deadline(kind: str, phase: fsm.Phase) -> None:
    s0 = _live_state(kind, phase)
    s, _ = step(ASK_RC, s0, Tick(539.999))
    assert not isinstance(s, Done) or s.outcome != Cut("deadline", 540.0, 0)


@pytest.mark.parametrize(
    ("s0", "effects"),
    [(starting(open_due_at=600.0), (Close(C1),)), (StartBackoff(0.0, At(540.0), 550.0, 1, C2), ())],
)
def test_t21_deadline_before_first_byte(
    s0: fsm.StartState, effects: tuple[fsm.Effect, ...]
) -> None:
    s, eff = step(ASK, s0, Tick(540.0))
    assert s == Done(Cut("deadline", 540.0, 0), NoIds())
    assert eff == effects


def test_t21_beats_backoff_elapsing_on_the_same_tick_start() -> None:
    """I16: a backoff whose `until` equals the deadline never reopens."""
    s, eff = step(ASK, StartBackoff(0.0, At(540.0), 540.0, 1, C2), Tick(540.0))
    assert s == Done(Cut("deadline", 540.0, 0), NoIds())
    assert not any(isinstance(x, Open) for x in eff)


def test_t21_beats_backoff_elapsing_on_the_same_tick_live() -> None:
    s, eff = step(ASK_RC, rc_backoff(live(ids=KNOWN, next_conn=3), until=540.0), Tick(540.0))
    assert s == Done(Cut("deadline", 540.0, 0), KNOWN)
    assert not any(isinstance(x, Open) for x in eff)


def test_t21_beats_open_due_on_the_same_tick() -> None:
    s, _ = step(ASK, starting(open_due_at=540.0), Tick(540.0))
    assert s == Done(Cut("deadline", 540.0, 0), NoIds())


def test_t21_beats_settle_on_the_same_tick() -> None:
    lv = live(phase=TextComplete(520.0, 525.0), ids=KNOWN, next_conn=3)
    s, _ = step(ASK, streaming(lv, last_byte_at=530.0), Tick(540.0))
    assert s == Done(SettledWithoutTerminal(0, "deadline"), KNOWN)


def test_t22_settle() -> None:
    lv = live(phase=TextComplete(20.0, 20.0), ids=KNOWN)
    s0 = streaming(lv, last_byte_at=30.0)
    assert step(ASK, s0, Tick(34.999)) == (s0, ())
    s, eff = step(ASK, s0, Tick(35.0))
    assert s == Done(SettledWithoutTerminal(0, "settle"), KNOWN)
    assert eff == (Close(C1),)


def test_t23_first_content() -> None:
    p = policy(first_content=FirstContentWithin(90.0))
    s0 = streaming(last_byte_at=85.0)
    assert fsm.due_class(p, s0, 89.999) == "none"
    s, _ = step(p, s0, Tick(90.0))
    assert s == Done(Cut("first_content", 90.0, 0), NoIds())


def test_t24_stall() -> None:
    s0 = streaming(live(phase=Producing(100.0), deadline=At(3600.0)), last_byte_at=570.0)
    p = policy(deadline=At(3600.0))
    assert step(p, s0, Tick(579.999)) == (s0, ())
    s, _ = step(p, s0, Tick(580.0))
    assert s == Done(Cut("stall", 480.0, 0), NoIds())


def test_t25_silence_is_a_stall_cut_not_lost() -> None:
    s0 = streaming(live(phase=Producing(100.0)), last_byte_at=100.0)
    assert fsm.due_class(ASK, s0, 124.999) == "none"
    s, _ = step(ASK, s0, Tick(125.0))
    assert s == Done(Cut("stall", 480.0, 0), NoIds())


def test_t26_nothing_due() -> None:
    s0 = streaming(live(phase=Producing(100.0)), last_byte_at=100.0)
    assert step(ASK, s0, Tick(110.0)) == (s0, ())


def test_grace_raises_stall_and_silence() -> None:
    s0 = streaming(live(phase=Producing(0.0)), last_byte_at=0.0, grace=GraceUntil(100.0))
    p = policy(deadline=At(3600.0), stall=StallAfter(50.0))
    assert fsm.due_class(p, s0, 99.999) == "none"
    assert fsm.due_class(p, s0, 100.0) == "stall"
    assert fsm.next_wake(p, s0) == 100.0


def test_settle_precedes_stall_and_silence_on_the_same_tick() -> None:
    s0 = streaming(live(phase=TextComplete(0.0, 10.0)), last_byte_at=0.0)
    p = policy(stall=StallAfter(25.0))
    assert fsm.due_class(p, s0, 25.0) == "settle"


# --- settle rows (§1 decision 12) ----------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.001, 7.5, 14.999])
def test_progress_after_text_complete_never_moves_settle(offset: float) -> None:
    at = 50.0
    s, _ = step(
        ASK, streaming(live(phase=Producing(40.0))), FrameIn(C1, at, frame("text_complete"))
    )
    assert isinstance(s, Streaming)
    for t in (at + 0.001, at + 7.5, at + offset):
        s, _ = step(ASK, s, FrameIn(C1, max(t, s.last_byte_at), frame("pending", "progress")))
        assert isinstance(s, Streaming)
    assert isinstance(s.live.phase, TextComplete)
    assert s.live.phase.at == at
    assert s.live.phase.last_progress_at == at + max(offset, 7.5)
    assert fsm.next_wake(ASK, s) == at + 15.0
    s2, _ = step(ASK, s, Tick(at + 15.0))
    assert isinstance(s2, Done)
    assert s2.outcome == SettledWithoutTerminal(0, "settle")


def test_repeated_text_complete_keeps_at() -> None:
    s = streaming(live(phase=TextComplete(20.0, 20.0)))
    for change in ("progress", "idle"):
        s2, _ = step(ASK, s, FrameIn(C1, 25.0, frame("text_complete", change)))  # pyright: ignore[reportArgumentType]
        assert isinstance(s2, Streaming)
        assert isinstance(s2.live.phase, TextComplete)
        assert s2.live.phase.at == 20.0


def test_i18_pending_progress_after_text_complete_stays_text_complete() -> None:
    s, _ = step(
        ASK,
        streaming(live(phase=TextComplete(20.0, 20.0))),
        FrameIn(C1, 25.0, frame("pending", "progress")),
    )
    assert isinstance(s, Streaming)
    assert s.live.phase == TextComplete(25.0, 20.0)


@given(now=st.floats(-1e6, 1e7), lp=st.floats(0, 1e5), last=st.floats(0, 1e5))
def test_settle_never_due_before_text_complete(now: float, lp: float, last: float) -> None:
    for phase in (AwaitingFirst(), Producing(lp)):
        s0 = streaming(live(phase=phase, deadline=Unbounded()), last_byte_at=last)
        assert fsm.due_class(ASK, s0, now) != "settle"


# --- R(reason) guard and fallback enumerations (§3.8) --------------------------------------------

GUARD_IDS = (NoIds(), UUID_ONLY, KNOWN)
GUARD_FLAGS: tuple[frames.Reconnectable, ...] = ("yes", "no", "absent")
GUARD_COUNTERS = ((0, 0), (3, 3), (1, 6))  # under both caps; consecutive at cap; total at cap
GUARD_REMAINING = (5.0, 4.999)


@pytest.mark.parametrize(
    ("rc", "ids", "flag", "counters", "left"),
    list(
        itertools.product(
            (Off(), Bounded(3, 6)), GUARD_IDS, GUARD_FLAGS, GUARD_COUNTERS, GUARD_REMAINING
        )
    ),
)
def test_reconnect_guard(
    rc: Off | Bounded,
    ids: fsm.Ids,
    flag: frames.Reconnectable,
    counters: tuple[int, int],
    left: float,
) -> None:
    p = policy(reconnect=rc)
    lv = live(
        phase=Producing(100.0),
        ids=ids,
        reconnectable=flag,
        rc_consecutive=counters[0],
        rc_total=counters[1],
    )
    now = 540.0 - left
    s, _ = step(p, streaming(lv, last_byte_at=now), StreamEnded(C1, now))
    expect = (
        isinstance(rc, Bounded)
        and not isinstance(ids, NoIds)
        and flag != "no"
        and counters == (0, 0)
        and left >= 5.0
    )
    assert isinstance(s, ReconnectBackoff) == expect
    if not expect:
        assert s == Done(EndedEarly(counters[1], "server"), ids)


def test_reconnect_guard_enumeration_is_108() -> None:
    assert 2 * len(GUARD_IDS) * len(GUARD_FLAGS) * len(GUARD_COUNTERS) * len(GUARD_REMAINING) == 108


@pytest.mark.parametrize("reason", REASONS)
@pytest.mark.parametrize("phase", PHASES)
def test_fallback_table(reason: ReconnectReason, phase: fsm.Phase) -> None:
    lv = live(phase=phase, rc_total=2)
    got = fsm.fallback(ASK_RC, reason, lv)
    table = {
        "drop": Lost("stream dropped"),
        "eof": EndedEarly(2, "server"),
        "silence": Cut("stall", 480.0, 2),
        "stall": Cut("stall", 480.0, 2),
        "first_content": Cut("first_content", 90.0, 2),
        "settle": Cut("stall", 480.0, 2),
    }
    by = {
        "settle": "settle",
        "stall": "stall",
        "silence": "stall",
        "first_content": "stall",
        "drop": "server",
        "eof": "server",
    }
    if isinstance(phase, TextComplete):
        assert got == SettledWithoutTerminal(2, by[reason])  # pyright: ignore[reportArgumentType]
    else:
        assert got == table[reason]


# --- §2.6: round 5's ends after text_done, and each `by` ----------------------------------------


def _run(p: Policy, events: list[fsm.Event]) -> fsm.State:
    s: fsm.State
    s, _ = fsm.initial(p, 0.0)
    for e in events:
        s, _ = step(p, s, e)
    return s


TEXT_DONE: list[fsm.Event] = [
    Opened(C1, 1.0),
    FrameIn(C1, 2.0, frame("pending", uuid=UUID, token=token(), context=CTX)),
    FrameIn(C1, 3.0, frame("text_complete")),
]


@pytest.mark.parametrize(
    ("tail", "outcome"),
    [
        ([Tick(18.0)], SettledWithoutTerminal(0, "settle")),
        ([HeartbeatIn(C1, 10.0), Tick(35.0)], SettledWithoutTerminal(0, "settle")),
        ([Tick(540.0)], SettledWithoutTerminal(0, "deadline")),
        ([StreamBroke(C1, 5.0, "transport", "reset")], SettledWithoutTerminal(0, "server")),
        ([StreamEnded(C1, 5.0)], SettledWithoutTerminal(0, "server")),
        ([FrameIn(C1, 4.0, frame("completed"))], Completed(0)),
        ([FrameIn(C1, 4.0, frame("failed", raw_status="FAILED"))], ServerFailed("FAILED")),
        ([StreamBroke(C1, 4.0, "oversize", "frame over 16 MiB")], "rejected-schema"),
    ],
)
def test_round5_ends_after_text_done(tail: list[fsm.Event], outcome: object) -> None:
    s = _run(ASK, [*TEXT_DONE, *tail])
    assert isinstance(s, Done)
    assert s.ids == KNOWN
    if outcome == "rejected-schema":
        assert isinstance(s.outcome, Rejected)
        assert type(s.outcome.error) is SchemaError
    else:
        assert s.outcome == outcome


def test_round5_stall_after_text_done() -> None:
    p = policy(stall=StallAfter(10.0))
    s = _run(p, [*TEXT_DONE, HeartbeatIn(C1, 12.0), Tick(13.0)])
    assert isinstance(s, Done)
    assert s.outcome == SettledWithoutTerminal(0, "stall")


def test_round5_silence_after_text_done() -> None:
    p = policy(completion=SettleAfterText(100.0), stall=StallAfter(400.0))
    s = _run(p, [*TEXT_DONE, Tick(28.0)])
    assert isinstance(s, Done)
    assert s.outcome == SettledWithoutTerminal(0, "stall")


def test_by_auth_only_from_t10_auth() -> None:
    """Every other ending that builds `by` gives a non-auth value (I23)."""
    for reason in REASONS:
        for phase in PHASES:
            o = fsm.fallback(ASK_RC, reason, live(phase=phase))
            assert getattr(o, "by", None) != "auth"
            o = fsm.deadline_outcome(live(phase=phase))
            assert getattr(o, "by", None) != "auth"


# --- next_wake -----------------------------------------------------------------------------------


def test_next_wake_is_the_earliest_timer() -> None:
    s0 = streaming(live(phase=Producing(100.0)), last_byte_at=110.0)
    assert fsm.next_wake(ASK, s0) == 135.0
    assert fsm.next_wake(ASK, Done(Completed(0), NoIds())) is None
    assert fsm.next_wake(ASK, starting()) == 30.0


# --- policy --------------------------------------------------------------------------------------


def test_verb_policies() -> None:
    ask, fetch, research = (for_verb(v) for v in ("ask", "fetch", "research"))  # pyright: ignore[reportArgumentType]
    assert isinstance(ask, Policy)
    assert isinstance(fetch, Policy)
    assert isinstance(research, Policy)
    assert (ask.deadline, ask.stall, ask.completion, ask.answer_paths) == (
        At(540.0),
        StallAfter(480.0),
        SettleAfterText(15.0),
        "ask_text_or_workflow",
    )
    assert (fetch.deadline, fetch.stall, fetch.completion) == (
        At(540.0),
        StallAfter(480.0),
        AtTextComplete(),
    )
    assert (research.deadline, research.stall, research.completion, research.answer_paths) == (
        At(3600.0),
        StallAfter(240.0),
        AtCompleted(),
        "ask_text_only",
    )
    for p in (ask, fetch, research):
        assert (p.silence_s, p.open_s, p.grace_s, p.rate_limit_attempts, p.min_useful_s) == (
            25.0,
            30.0,
            30.0,
            3,
            5.0,
        )
        assert p.low_speed_s == 35.0
        assert p.reconnect == Off()


@pytest.mark.parametrize(
    "kw",
    [
        {"stall": StallAfter(0.0)},
        {"stall": StallAfter(float("nan"))},
        {"deadline": At(-1.0)},
        {"completion": SettleAfterText(0.0)},
        {"first_content": FirstContentWithin(float("inf"))},
        {"reconnect": Bounded(0, 3)},
        {"reconnect": Bounded(4, 3)},
        {"rate_limit_attempts": 0},
        {"backoff_base_s": 10.0},
    ],
)
def test_policy_make_rejects(kw: dict[str, object]) -> None:
    args: dict[str, object] = {
        "deadline": At(540.0),
        "stall": StallAfter(480.0),
        "completion": SETTLE_,
        "answer_paths": "ask_text_or_workflow",
    }
    args.update(kw)
    assert isinstance(Policy.make(**args), PolicyError)  # pyright: ignore[reportArgumentType]


SETTLE_ = SettleAfterText(15.0)


def test_policy_unbounded_skips_the_stall_check() -> None:
    assert isinstance(
        Policy.make(
            deadline=Unbounded(),
            stall=StallAfter(1e6),
            completion=SETTLE_,
            answer_paths="ask_text_only",
        ),
        Policy,
    )


# --- the grace floor after a reconnect (I26) ------------------------------------------------------


def _settle_probe(snapshot_after: float | None) -> tuple[fsm.State, list[float]]:
    """text_completed at t=2, settle due at 17; a driver opens each reconnect
    0.1 s after its backoff and, when given, the snapshot with COMPLETED
    arrives `snapshot_after` seconds into the first reopened conn."""
    p = ASK_RC
    s: fsm.State
    s, _ = fsm.initial(p, 0.0)
    s, _ = step(p, s, Opened(C1, 0.5))
    s, _ = step(p, s, FrameIn(C1, 1.0, frame("pending", uuid=UUID, token=token(), context=CTX)))
    s, _ = step(p, s, FrameIn(C1, 2.0, frame("text_complete")))
    rcs: list[float] = []
    t = 2.0
    for _ in range(12):
        if isinstance(s, Done):
            break
        wake = fsm.next_wake(p, s)
        assert wake is not None
        t = max(t, wake)
        s, eff = step(p, s, Tick(t))
        if isinstance(s, ReconnectBackoff):
            rcs.append(t)
        opens = [x for x in eff if isinstance(x, Open)]
        if opens:
            t += 0.1
            s, _ = step(p, s, Opened(opens[0].conn, t))
            if snapshot_after is not None and len(rcs) == 1:
                t += snapshot_after
                wake = fsm.next_wake(p, s)
                assert wake is not None and wake > t
                s, _ = step(p, s, FrameIn(opens[0].conn, t, frame("completed", "idle")))
    return s, rcs


@pytest.mark.parametrize("snapshot_after", [0.0, 5.0, 29.9])
def test_settle_reconnect_gets_its_grace(snapshot_after: float) -> None:
    s, rcs = _settle_probe(snapshot_after)
    assert s == Done(Completed(1), KNOWN)
    assert rcs == [17.0]


def test_settle_reconnects_without_a_snapshot_are_a_grace_apart() -> None:
    s, rcs = _settle_probe(None)
    assert s == Done(SettledWithoutTerminal(3, "settle"), KNOWN)
    assert len(rcs) == 3
    for a, b in itertools.pairwise(rcs):
        assert b - a >= ASK_RC.grace_s


FLOOR_CASES: list[tuple[str, Policy, fsm.Phase]] = [
    ("settle", ASK_RC, TextComplete(10.0, 10.0)),
    ("first_content", ASK_RC, AwaitingFirst()),
    ("stall", policy(reconnect=Bounded(3, 6), stall=StallAfter(20.0)), Producing(10.0)),
    ("silence", ASK_RC, Producing(100.0)),
]


@pytest.mark.parametrize(("tag", "p", "phase"), FLOOR_CASES)
def test_every_reconnecting_timer_waits_for_the_grace(
    tag: str, p: Policy, phase: fsm.Phase
) -> None:
    """Each term is overdue when the reconnect opens at 105; none fires
    before 135."""
    s0 = streaming(live(phase=phase, ids=KNOWN, next_conn=3), last_byte_at=60.0)
    s = replace(s0, grace=GraceUntil(135.0))
    assert fsm.due_class(p, s, 134.999) == "none"
    assert fsm.due_class(p, s, 135.0) == tag
    assert fsm.next_wake(p, s) == 135.0


def test_progress_keeps_the_grace() -> None:
    """A snapshot with new blocks but no COMPLETED still leaves the settle
    reconnect its grace."""
    s0 = streaming(live(phase=TextComplete(10.0, 10.0), ids=KNOWN), grace=GraceUntil(135.0))
    s, _ = step(ASK_RC, replace(s0, last_byte_at=105.0), FrameIn(C1, 106.0, frame()))
    assert isinstance(s, Streaming)
    assert s.grace == GraceUntil(135.0)
    assert fsm.due_class(ASK_RC, s, 134.999) == "none"
    assert fsm.due_class(ASK_RC, s, 135.0) == "settle"


GRACE_POLICIES = [
    Policy.make(
        deadline=dl,
        stall=stall,
        completion=completion,
        answer_paths="ask_text_or_workflow",
        first_content=fc,
        reconnect=Bounded(3, 8),
        silence_s=silence,
    )
    for dl, stall, completion, fc, silence in [
        (At(3600.0), StallAfter(20.0), SettleAfterText(15.0), FirstContentWithin(4.0), 8.0),
        (Unbounded(), StallAfter(480.0), SettleAfterText(3.0), FirstContentWithin(90.0), 25.0),
        (At(900.0), StallOff(), AtCompleted(), FirstContentWithin(4.0), 8.0),
        (At(900.0), StallAfter(5.0), AtTextComplete(), FirstContentOff(), 25.0),
    ]
]
ACTIONS = st.sampled_from(["wake"] * 6 + ["idle", "progress", "text", "beat", "drop", "fail"])


@settings(max_examples=300, deadline=None)
@given(
    pi=st.integers(0, len(GRACE_POLICIES) - 1),
    acts=st.lists(st.tuples(ACTIONS, st.floats(0, 40)), min_size=1, max_size=60),
    open_after=st.floats(0, 5),
)
def test_no_timer_reconnects_within_the_grace(
    pi: int, acts: list[tuple[str, float]], open_after: float
) -> None:
    """I26 over random histories: once a reconnect opens, no settle,
    first-content, stall or silence reconnect comes before `grace_s` has
    passed, so no two timer reconnects are closer than the grace."""
    p = GRACE_POLICIES[pi]
    assert isinstance(p, Policy)
    s: fsm.State
    s, _ = fsm.initial(p, 0.0)
    s, _ = step(p, s, Opened(C1, 0.0))
    s, _ = step(p, s, FrameIn(C1, 0.0, frame("pending", "idle", uuid=UUID, token=token())))
    now, reopened_at = 0.0, None
    for act, dt in acts:
        if isinstance(s, Done):
            break
        if not isinstance(s, Streaming):
            wake = fsm.next_wake(p, s)
            assert wake is not None
            now = max(now, wake)
            s, eff = step(p, s, Tick(now))
            opens = [x for x in eff if isinstance(x, Open)]
            if opens and isinstance(s, Reconnecting):
                now += open_after
                e: fsm.Event = (
                    OpenFailed(s.conn, now, Transient("x"))
                    if act == "fail"
                    else Opened(s.conn, now)
                )
                s, _ = step(p, s, e)
                if isinstance(e, Opened):
                    reopened_at = now
            continue
        conn = s.conn
        if act == "wake":
            wake = fsm.next_wake(p, s)
            assert wake is not None
            now = max(now, wake)
            due = fsm.due_class(p, s, now)
            if due in ("settle", "first_content", "stall", "silence") and reopened_at is not None:
                assert now >= reopened_at + p.grace_s
            s, _ = step(p, s, Tick(now))
            continue
        # A body item never skips a timer that is already due.
        wake = fsm.next_wake(p, s)
        assert wake is not None
        now = min(now + dt, max(now, wake))
        body: fsm.Event = {
            "idle": FrameIn(conn, now, frame("pending", "idle")),
            "progress": FrameIn(conn, now, frame()),
            "text": FrameIn(conn, now, frame("text_complete", "idle")),
            "beat": HeartbeatIn(conn, now),
            "drop": StreamBroke(conn, now, "transport", "reset"),
            "fail": HeartbeatIn(conn, now),
        }[act]
        s, _ = step(p, s, body)


# --- a failed reconnect ends as reconnect Off would (§3.4) ----------------------------------------


def _trigger_case(reason: ReconnectReason) -> tuple[dict[str, object], float]:
    """Policy fields and last byte time under which `reason` is the first
    thing due, plus the time it is due."""
    kw: dict[str, object] = {"stall": StallAfter(120.0), "completion": SettleAfterText(1000.0)}
    last_byte = 200.0
    match reason:
        case "silence":
            last_byte = 10.0
        case "stall":
            kw["stall"] = StallAfter(50.0)
        case "settle":
            kw["completion"] = SettleAfterText(15.0)
        case "first_content":
            kw["first_content"] = FirstContentWithin(40.0)
        case "drop" | "eof":
            last_byte = 20.0
    return kw, last_byte


def _cut(reason: ReconnectReason, s: fsm.State, p: Policy) -> fsm.Event:
    match reason:
        case "drop":
            return StreamBroke(C1, 30.0, "transport", "reset")
        case "eof":
            return StreamEnded(C1, 30.0)
        case _:
            wake = fsm.next_wake(p, s)
            assert wake is not None
            assert fsm.due_class(p, s, wake) == reason
            return Tick(wake)


def _without_reconnects(o: object) -> object:
    return replace(o, reconnects=0) if hasattr(o, "reconnects") else o  # pyright: ignore[reportArgumentType]


REASON_PHASES = [
    (r, ph)
    for r in REASONS
    for ph in PHASES
    if (r != "settle" or isinstance(ph, TextComplete))
    and (r != "first_content" or isinstance(ph, AwaitingFirst))
]
REFUSALS: list[str] = ["gone", "fatal", "redirect", "transient", "open_timeout"]


@pytest.mark.parametrize(("reason", "phase"), REASON_PHASES)
@pytest.mark.parametrize("refusal", REFUSALS)
def test_failed_reconnect_ends_as_reconnect_off(
    reason: ReconnectReason, phase: fsm.Phase, refusal: str
) -> None:
    kw, last_byte = _trigger_case(reason)
    ph = TextComplete(10.0, 10.0) if isinstance(phase, TextComplete) else phase
    base = live(phase=ph, ids=KNOWN)
    s0 = streaming(base, last_byte_at=last_byte)

    off = policy(reconnect=Off(), **kw)  # pyright: ignore[reportArgumentType]
    want, _ = step(off, s0, _cut(reason, s0, off))
    assert isinstance(want, Done)

    # One reconnect allowed, so a retryable failure cannot try again.
    on = policy(reconnect=Bounded(1, 1), **kw)  # pyright: ignore[reportArgumentType]
    s, _ = step(on, s0, _cut(reason, s0, on))
    assert isinstance(s, ReconnectBackoff)
    s, _ = step(on, s, Tick(s.until))
    assert isinstance(s, Reconnecting)
    assert s.reason == reason
    failure: dict[str, fsm.Failure] = {
        "gone": Gone(403),
        "fatal": Fatal(SchemaError("x")),
        "redirect": Fatal(UnexpectedRedirect("302")),
        "transient": Transient("reset"),
    }
    if refusal == "open_timeout":
        got, _ = step(on, s, Tick(s.open_due_at))
    else:
        got, _ = step(on, s, OpenFailed(s.conn, s.open_due_at - 1.0, failure[refusal]))
    assert isinstance(got, Done)
    assert got.ids == want.ids
    assert _without_reconnects(got.outcome) == _without_reconnects(want.outcome)
