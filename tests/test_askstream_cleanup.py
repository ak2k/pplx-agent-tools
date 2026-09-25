"""cleanup_plan (plan §3.5, I9), stream_status/ask_status (§2.6, I12) and
the trace ring (§2.9, I19)."""

from __future__ import annotations

import typing
from typing import Literal, get_args, get_origin, get_type_hints

import pytest

from pplx_agent_tools.askstream import fsm, trace
from pplx_agent_tools.askstream.cleanup import (
    Delete,
    DeleteKept,
    DeleteNotNeeded,
    DeleteNoToken,
    Terminate,
    TerminateNotNeeded,
    TerminateRef,
    TerminateUnsupported,
    cleanup_plan,
)
from pplx_agent_tools.askstream.fsm import (
    Done,
    HeartbeatIn,
    Known,
    NoIds,
    Producing,
    ReconnectBackoff,
    Reconnecting,
    ReconnectTarget,
    StartBackoff,
    Starting,
    Tick,
    UuidOnly,
)
from pplx_agent_tools.askstream.ids import ConnId
from pplx_agent_tools.askstream.outcome import (
    Completed,
    Cut,
    EndedEarly,
    Lost,
    NotStreamed,
    Rejected,
    ServerFailed,
    SettledWithoutTerminal,
)
from pplx_agent_tools.askstream.policy import At
from pplx_agent_tools.errors import AuthError, ResourceLimitError
from pplx_agent_tools.render import ask_status, stream_status
from tests._fsm import ASK, CTX, KNOWN, TOKEN_RAW, UUID, live, streaming

MODEL = "claude48opusthinking"
OVER = (Completed(0), SettledWithoutTerminal(0, "settle"), ServerFailed("FAILED"))
MAY_RUN = (
    Cut("stall", 1.0, 0),
    Lost("x"),
    EndedEarly(0, "server"),
    Rejected(ResourceLimitError("cap")),
)
IDS = (NoIds(), UuidOnly(UUID, CTX), UuidOnly(UUID, None), KNOWN, Known(KNOWN.ref, None))


def _live_states(ids: fsm.Ids) -> list[fsm.State]:
    lv = live(ids=ids, phase=Producing(1.0), next_conn=3)
    return [
        streaming(lv),
        Reconnecting(lv, ConnId(2), ReconnectTarget(UUID), "drop", 50.0),
        ReconnectBackoff(lv, 50.0, ReconnectTarget(UUID), "drop"),
    ]


def _expected_delete(ids: fsm.Ids, keep: bool) -> object:
    if isinstance(ids, NoIds):
        return DeleteNotNeeded()
    if keep:
        return DeleteKept()
    return Delete(ids.ref) if isinstance(ids, Known) else DeleteNoToken()


def _expected_terminate(ids: fsm.Ids, model: str | None) -> object:
    if isinstance(ids, NoIds):
        return TerminateNotNeeded()
    if ids.context is None or model is None:
        return TerminateUnsupported()
    return Terminate(TerminateRef(UUID, ids.context, model))


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("model", [MODEL, None])
@pytest.mark.parametrize("ids", IDS)
def test_run_over_never_terminates(ids: fsm.Ids, keep: bool, model: str | None) -> None:
    for o in OVER:
        assert cleanup_plan(Done(o, ids), keep, model) == (
            TerminateNotNeeded(),
            _expected_delete(ids, keep),
        )


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("model", [MODEL, None])
@pytest.mark.parametrize("ids", IDS)
def test_run_that_may_still_spend_terminates(ids: fsm.Ids, keep: bool, model: str | None) -> None:
    """keep_thread gates only Delete; Terminate follows the ids."""
    for s in [*(Done(o, ids) for o in MAY_RUN), *_live_states(ids)]:
        assert cleanup_plan(s, keep, model) == (
            _expected_terminate(ids, model),
            _expected_delete(ids, keep),
        )


def test_rejected_auth_from_awaiting_first_with_ids() -> None:
    plan = cleanup_plan(Done(Rejected(AuthError("x")), KNOWN), False, MODEL)
    assert plan == (Terminate(TerminateRef(UUID, CTX, MODEL)), Delete(KNOWN.ref))


@pytest.mark.parametrize("keep", [False, True])
def test_before_first_byte_nothing(keep: bool) -> None:
    for s in (
        Starting(0.0, At(540.0), ConnId(1), 1, 30.0),
        StartBackoff(0.0, At(540.0), 9.0, 1, ConnId(2)),
        Done(Cut("deadline", 540.0, 0), NoIds()),
    ):
        assert cleanup_plan(s, keep, MODEL) == (TerminateNotNeeded(), DeleteNotNeeded())


def test_plan_never_prints_the_token() -> None:
    for s in [Done(Lost("x"), KNOWN), *_live_states(KNOWN)]:
        text = repr(cleanup_plan(s, False, MODEL))
        assert TOKEN_RAW not in text
        assert "<redacted>" in text


# --- I12 -----------------------------------------------------------------------------------------

ENDS = [
    Completed(0),
    SettledWithoutTerminal(1, "settle"),
    *(
        SettledWithoutTerminal(0, by)
        for by in get_args(get_type_hints(SettledWithoutTerminal)["by"])
    ),
    *(Cut(c, 1.0, 0) for c in get_args(get_type_hints(Cut)["cause"])),
    *(EndedEarly(0, by) for by in get_args(get_type_hints(EndedEarly)["by"])),
    NotStreamed(),
]


def test_stream_status_total_and_image() -> None:
    image = {stream_status(e) for e in ENDS}  # pyright: ignore[reportArgumentType]
    assert image <= {(True, None), (False, None), (False, "stall"), (False, "deadline")}
    assert stream_status(Cut("first_content", 90.0, 0)) == (False, "stall")
    assert stream_status(NotStreamed()) == (True, None)


@pytest.mark.parametrize(
    ("end", "row"),
    [
        (Completed(0), (True, None, True)),
        (SettledWithoutTerminal(0, "settle"), (True, None, False)),
        (SettledWithoutTerminal(0, "auth"), (True, None, False)),
        (Cut("stall", 1.0, 0), (False, "stall", False)),
        (Cut("first_content", 1.0, 0), (False, "stall", False)),
        (Cut("deadline", 1.0, 0), (False, "deadline", False)),
        (EndedEarly(0, "server"), (False, None, False)),
        (EndedEarly(0, "auth"), (False, None, False)),
    ],
)
def test_ask_status_matches_the_round5_table(
    end: object, row: tuple[bool, str | None, bool]
) -> None:
    assert ask_status(end) == row  # pyright: ignore[reportArgumentType]


def test_ask_status_sources_complete_only_for_completed() -> None:
    for e in ENDS:
        if not isinstance(e, NotStreamed):
            assert ask_status(e)[2] == isinstance(e, Completed)


# --- trace (I19) ---------------------------------------------------------------------------------


def _allowed(tp: object) -> bool:
    if tp is int:
        return True
    origin = get_origin(tp)
    if origin is Literal:
        return all(isinstance(a, str) for a in get_args(tp))
    if origin is tuple:
        args = get_args(tp)
        return len(args) == 2 and args[1] is Ellipsis and _allowed(args[0])
    if origin is typing.Union or type(tp).__name__ == "UnionType":
        return all(_allowed(a) for a in get_args(tp))
    return False


def test_trace_entry_fields_hold_no_server_strings() -> None:
    hints = get_type_hints(trace.TraceEntry)
    assert len(hints) == 12
    for name, tp in hints.items():
        assert _allowed(tp), (name, tp)


def test_trace_ring_merges_repeats_and_bounds() -> None:
    ring = trace.TraceRing()
    s = streaming(live(phase=Producing(1.0)))
    for i in range(1000):
        ring.add(trace.entry(ASK, 0.0, s, HeartbeatIn(ConnId(1), 2.0 + i), s, ()))
    assert len(ring.entries) == 1
    assert ring.entries[0].n == 1000
    assert ring.entries[0].t_ms == 2000
    for i in range(600):
        ring.add(trace.entry(ASK, 0.0, s, Tick(10.0 + i), s, ()))
        ring.add(trace.entry(ASK, 0.0, s, HeartbeatIn(ConnId(1 + i % 2), 10.5 + i), s, ()))
    assert len(ring.entries) == trace.RING
    assert ring.dropped == 1 + 1200 - trace.RING


def test_trace_entry_tags() -> None:
    s = streaming(live(phase=Producing(1.0), ids=KNOWN))
    s2, eff = fsm.step(ASK, s, Tick(600.0), 0.5)
    e = trace.entry(ASK, 0.0, s, Tick(600.0), s2, eff)
    assert (e.state, e.phase, e.event, e.sub, e.to_state, e.outcome, e.effects) == (
        "Streaming",
        "Producing",
        "Tick",
        ("deadline",),
        "Done",
        "Cut:deadline",
        ("Close",),
    )
    assert TOKEN_RAW not in repr(e)
