"""The sync driver over a scripted transport and a fake clock: fixture
replays, reconnects, bounds per open, error classification, Tick-first
ordering (oracle 3) and interrupts."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator

import pytest

from pplx_agent_tools.askstream.blocks import BlockStore
from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.driver import ConnBounds, Driver
from pplx_agent_tools.askstream.frames import AskFrame
from pplx_agent_tools.askstream.fsm import AUTH_NOTICE, Done, ReconnectBackoff
from pplx_agent_tools.askstream.outcome import Completed, Cut, EndedEarly, Rejected
from pplx_agent_tools.askstream.patch import Limits
from pplx_agent_tools.askstream.policy import (
    FIRST_CONTENT_OFF,
    At,
    Bounded,
    Deadline,
    FirstContent,
    Off,
    Policy,
    PolicyError,
    Reconnect,
    Stall,
    StallAfter,
    StallOff,
    for_verb,
)
from pplx_agent_tools.errors import (
    AuthError,
    NetworkError,
    PplxError,
    RateLimitError,
    ResourceLimitError,
    SchemaError,
    StreamDeadlineError,
    StreamSilenceError,
)
from tests._driver import (
    HEARTBEAT,
    FakeClient,
    FakeClock,
    Item,
    Script,
    fixture_items,
    message,
    paced,
)

UUID = "00000000-0000-4000-8000-000000000001"
AT_3600 = At(3600.0)
STALL_240 = StallAfter(240.0)
RC_3_8 = Bounded(3, 8)


def research_policy(
    *,
    deadline: Deadline = AT_3600,
    stall: Stall = STALL_240,
    reconnect: Reconnect = RC_3_8,
    first_content: FirstContent = FIRST_CONTENT_OFF,
    silence_s: float = 90.0,
) -> Policy:
    p = for_verb(
        "research",
        deadline=deadline,
        stall=stall,
        reconnect=reconnect,
        first_content=first_content,
        silence_s=silence_s,
    )
    assert not isinstance(p, PolicyError)
    return p


def drive(
    client: FakeClient,
    policy: Policy,
    *,
    store: BlockStore | None = None,
    progress: bool = False,
    err: io.StringIO | None = None,
    sleep: Callable[[float], None] | None = None,
    on_frame: Callable[[AskFrame], None] | None = None,
) -> tuple[Driver, Done]:
    d = Driver(
        policy,
        client,
        client.open_initial,
        store if store is not None else BlockStore("ask_text_only", policy.limits),
        label="pplx research",
        on_frame=on_frame,
        progress=progress,
        clock=client.clock,
        sleep=client.clock.sleep if sleep is None else sleep,
        rand=lambda: 0.5,
        err=io.StringIO() if err is None else err,
    )
    return d, d.run()


def p3_drop_then(reconnect: Script, *, drop_at: float = 26.0) -> FakeClient:
    clock = FakeClock()
    initial = [*paced(fixture_items("p3-research-initial")), (drop_at, NetworkError("reset"))]
    return FakeClient(clock, initials=[initial], reconnects=[reconnect])


P3_RECONNECT = fixture_items("p3-research-reconnect1")


# --- whole runs ----------------------------------------------------------------------------------


@pytest.mark.parametrize("stem", ["weather-nowcasting-apis", "ocio-fees-final-only"])
def test_legacy_fixture_completes_on_the_first_conn(stem: str) -> None:
    items = fixture_items(stem, legacy=True)
    client = FakeClient(FakeClock(), initials=[paced(items)])
    d, done = drive(client, research_policy())
    assert done.outcome == Completed(0)
    assert d.terminal_text == items[5]["data"]["text"]
    assert d.display_model == "pplx_alpha"
    assert d.trigger is None
    assert [o.kind for o in client.opens] == ["initial"]
    assert client.opens[0].bounds == ConnBounds(3600.0, 240.0, 90.0)


@pytest.mark.parametrize("stem", ["p1-A", "p1-B"])
def test_diff_fixture_completes_with_the_terminal_text(stem: str) -> None:
    items = fixture_items(stem)
    client = FakeClient(FakeClock(), initials=[paced(items)])
    d, done = drive(client, research_policy())
    assert done.outcome == Completed(0)
    assert d.terminal_text == items[-1]["data"]["text"]
    assert d.items == len(items)


def test_drop_then_reconnect_completes() -> None:
    client = p3_drop_then(paced(P3_RECONNECT, start=30.0))
    err = io.StringIO()
    d, done = drive(client, research_policy(), err=err)
    assert done.outcome == Completed(1)
    assert d.reconnect_opens == 1
    assert d.terminal_text == P3_RECONNECT[-1]["data"]["text"]
    assert [(o.kind, o.uuid) for o in client.opens] == [("initial", None), ("reconnect", UUID)]
    # jitter(0.5) is 1, so the first backoff is the 1 s base.
    assert client.opens[1].bounds == ConnBounds(3600.0 - 27.0, 240.0, 90.0)
    assert err.getvalue() == "pplx research: drop; reconnecting\n"
    assert isinstance(d.last_error, NetworkError)


def test_reconnect_bounds_shrink_toward_the_deadline() -> None:
    client = p3_drop_then(paced(P3_RECONNECT, start=62.0), drop_at=60.0)
    drive(client, research_policy(deadline=At(100.0)))
    # The stall window is capped at the deadline's 100 s.
    assert client.opens[1].bounds == ConnBounds(39.0, 100.0, 39.0)


def test_transport_silence_reconnects_as_silence() -> None:
    client = p3_drop_then(paced(P3_RECONNECT, start=30.0))
    initial = list(client.initials[0])
    client.initials[0] = [*initial[:-1], (26.0, StreamSilenceError("went silent", 90.0))]
    err = io.StringIO()
    _, done = drive(client, research_policy(), err=err)
    assert done.outcome == Completed(1)
    assert err.getvalue() == "pplx research: silence; reconnecting\n"


def test_transport_silence_without_reconnect_is_a_stall_cut() -> None:
    initial = [*paced(fixture_items("p3-research-initial")), (26.0, StreamSilenceError("x", 90.0))]
    client = FakeClient(FakeClock(), initials=[initial])
    d, done = drive(client, research_policy(reconnect=Off()))
    assert done.outcome == Cut("stall", 240.0, 0)
    assert d.trigger == "silence"


def test_reconnects_exhausted_is_lost_with_the_drop_trigger() -> None:
    client = p3_drop_then([(30.0, NetworkError("reset again"))])
    d, done = drive(client, research_policy(reconnect=Bounded(1, 1)))
    assert type(done.outcome).__name__ == "Lost"
    assert d.trigger == "drop"
    assert d.reconnect_opens == 1


# --- open errors ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "e",
    [
        NetworkError("reset"),
        AuthError("expired"),
        SchemaError("unexpected status 404"),
        StreamSilenceError("went silent", 90.0),
    ],
)
def test_initial_open_errors_reject_with_the_original_error(e: PplxError) -> None:
    client = FakeClient(FakeClock(), initials=[[(0.5, e)]])
    d, done = drive(client, research_policy())
    assert isinstance(done.outcome, Rejected)
    assert done.outcome.error is e
    assert d.last_error is e


def test_initial_429_backs_off_and_posts_again() -> None:
    clock = FakeClock()
    items = fixture_items("weather-nowcasting-apis", legacy=True)
    rate_limited: Script = [(0.0, RateLimitError("429", retry_after=2.0))]
    client = FakeClient(clock, initials=[rate_limited, paced(items, start=3.0)])
    err = io.StringIO()
    _, done = drive(client, research_policy(), err=err)
    assert done.outcome == Completed(0)
    assert [o.kind for o in client.opens] == ["initial", "initial"]
    assert clock.slept == [2.0]
    assert err.getvalue() == "pplx research: rate limited (attempt 1/3); retrying in 2.0s\n"


def test_reconnect_rate_limited_retries() -> None:
    client = p3_drop_then([(30.0, RateLimitError("429", retry_after=3.0))])
    client.reconnects.append(paced(P3_RECONNECT, start=40.0))
    d, done = drive(client, research_policy())
    assert done.outcome == Completed(2)
    assert d.reconnect_opens == 2


def test_reconnect_transport_error_retries() -> None:
    client = p3_drop_then([(30.0, NetworkError("connect failed"))])
    client.reconnects.append(paced(P3_RECONNECT, start=40.0))
    _, done = drive(client, research_policy())
    assert done.outcome == Completed(2)


def test_reconnect_auth_error_ends_early_keeping_the_partial() -> None:
    client = p3_drop_then([(30.0, AuthError("expired"))])
    err = io.StringIO()
    d, done = drive(client, research_policy(), err=err)
    assert done.outcome == EndedEarly(1, "auth")
    assert d.trigger == "drop"
    assert err.getvalue().endswith(f"pplx research: {AUTH_NOTICE}\n")


# --- mid-stream errors the FSM decides -----------------------------------------------------------


def test_trickle_without_events_ends_at_the_deadline() -> None:
    """The transport's deadline raises; the FSM's deadline Tick ends the run."""
    frame = message({"status": "PENDING", "text": "[]"})
    client = FakeClient(
        FakeClock(), initials=[[(1.0, frame), (100.0, StreamDeadlineError("deadline"))]]
    )
    d, done = drive(client, research_policy(deadline=At(100.0), stall=StallOff()))
    assert done.outcome == Cut("deadline", 100.0, 0)
    assert d.since_progress == 99.0


def test_transport_deadline_a_moment_early_waits_for_the_fsm() -> None:
    frame = message({"status": "PENDING", "text": "[]"})
    clock = FakeClock()
    client = FakeClient(clock, initials=[[(1.0, frame), (99.75, StreamDeadlineError("deadline"))]])
    # A silence window past the deadline leaves the deadline the only timer.
    policy = research_policy(deadline=At(100.0), stall=StallOff(), silence_s=200.0)
    _, done = drive(client, policy)
    assert done.outcome == Cut("deadline", 100.0, 0)
    assert clock.slept == [0.25]


def test_oversized_event_rejects() -> None:
    frame = message({"status": "PENDING", "text": "[]"})
    client = FakeClient(FakeClock(), initials=[[(1.0, frame), (2.0, SchemaError("16 MiB"))]])
    _, done = drive(client, research_policy())
    assert isinstance(done.outcome, Rejected)
    assert type(done.outcome.error) is SchemaError


def test_store_cap_rejects() -> None:
    client = FakeClient(FakeClock(), initials=[paced(fixture_items("p1-A"))])
    policy = research_policy()
    store = BlockStore("ask_text_only", Limits(field_weight=64, total_weight=64))
    _, done = drive(client, policy, store=store)
    assert isinstance(done.outcome, Rejected)
    assert type(done.outcome.error) is ResourceLimitError


# --- oracle 3: Tick first -------------------------------------------------------------------------


def _completed_at(t: float) -> Script:
    beats: Script = [(float(s), HEARTBEAT) for s in range(20, 100, 20)]
    return [*beats, (t, message({"status": "COMPLETED", "text": "[]"}))]


def test_late_completed_frame_past_the_deadline_is_cut() -> None:
    client = FakeClient(FakeClock(), initials=[_completed_at(100.001)])
    d, done = drive(client, research_policy(deadline=At(100.0), stall=StallOff()))
    assert done.outcome == Cut("deadline", 100.0, 0)
    assert d.terminal_text is None


def test_completed_frame_a_moment_before_the_deadline_completes() -> None:
    client = FakeClient(FakeClock(), initials=[_completed_at(99.999)])
    _, done = drive(client, research_policy(deadline=At(100.0), stall=StallOff()))
    assert done.outcome == Completed(0)


def _starving(first: Item, every: Callable[[int], Item]) -> Iterator[tuple[float, Item]]:
    yield 0.5, first
    i = 0
    while True:
        i += 1
        yield 0.5 + i / 1000, every(i)


def test_starved_by_heartbeats_the_stall_cut_lands_on_the_first_item_past_it() -> None:
    clock = FakeClock()
    first = message({"status": "PENDING", "text": "[]"})
    client = FakeClient(clock, initials=[_starving(first, lambda _: HEARTBEAT)])
    _, done = drive(client, research_policy(stall=StallAfter(5.0), reconnect=Off()))
    assert done.outcome == Cut("stall", 5.0, 0)
    assert 5.5 <= clock.now < 5.5 + 0.001 + 1e-9


def test_starved_by_progress_the_deadline_cut_lands_on_the_first_item_past_it() -> None:
    clock = FakeClock()
    first = message({"status": "PENDING", "text": "[]"})
    client = FakeClient(
        clock,
        initials=[_starving(first, lambda i: message({"status": "PENDING", "text": f"[{i}]"}))],
    )
    d, done = drive(client, research_policy(deadline=At(10.0), stall=StallOff()))
    assert done.outcome == Cut("deadline", 10.0, 0)
    assert 10.0 <= clock.now < 10.0 + 0.001 + 1e-9
    assert d.since_progress is not None
    assert 0.0 < d.since_progress <= 0.001 + 1e-9


# --- bookkeeping ---------------------------------------------------------------------------------


def test_progress_dots_break_for_a_notice_and_end_with_a_newline() -> None:
    client = p3_drop_then(paced(P3_RECONNECT, start=30.0))
    err = io.StringIO()
    d, _ = drive(client, research_policy(), progress=True, err=err)
    assert d.items == 25 + len(P3_RECONNECT)
    assert err.getvalue() == "..\npplx research: drop; reconnecting\n" + "." * 9 + "\n"


def test_drift_counts_unknown_events_and_unparseable_payloads() -> None:
    frame = message({"status": "COMPLETED", "text": "[]"})
    script: Script = [
        (1.0, {"event": "surprise", "data": {"status": "PENDING"}}),
        (2.0, {"event": "message", "data": "{not json"}),
        (3.0, frame),
    ]
    client = FakeClient(FakeClock(), initials=[script])
    d, done = drive(client, research_policy())
    assert done.outcome == Completed(0)
    assert d.drift[Drift("unknown_sse_event", name_of("surprise"))] == 1
    assert d.drift[Drift("unparseable_frame", name_of("syntax"))] == 1


def test_on_frame_sees_every_applied_frame() -> None:
    items = fixture_items("p1-B")
    seen: list[AskFrame] = []
    client = FakeClient(FakeClock(), initials=[paced(items)])
    drive(client, research_policy(), on_frame=seen.append)
    assert len(seen) == len(items)


def test_interrupt_during_a_backoff_leaves_the_live_state() -> None:
    client = p3_drop_then(paced(P3_RECONNECT, start=30.0))

    def interrupted(_: float) -> None:
        raise KeyboardInterrupt

    d = Driver(
        research_policy(),
        client,
        client.open_initial,
        BlockStore("ask_text_only", Limits()),
        label="pplx research",
        clock=client.clock,
        sleep=interrupted,
        rand=lambda: 0.5,
        err=io.StringIO(),
    )
    with pytest.raises(KeyboardInterrupt):
        d.run()
    assert isinstance(d.state, ReconnectBackoff)
    assert [o.kind for o in client.opens] == ["initial"]
