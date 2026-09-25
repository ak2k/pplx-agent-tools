"""Builders and restated rules shared by the lifecycle tests.

The `expected_*` functions restate plan §3.4 and §3.5 independently of
`fsm.py`, so a test that compares against them checks the implementation
rather than echoing it.
"""

from __future__ import annotations

from dataclasses import replace

from pplx_agent_tools.askstream import fsm
from pplx_agent_tools.askstream.fsm import (
    AwaitingFirst,
    FrameSummary,
    Known,
    Live,
    NoGrace,
    NoIds,
    Phase,
    Producing,
    ReconnectReason,
    Streaming,
    TextComplete,
    UuidOnly,
)
from pplx_agent_tools.askstream.ids import (
    BackendUuid,
    ConnId,
    ContextUuid,
    ReadWriteToken,
    ThreadRef,
)
from pplx_agent_tools.askstream.outcome import (
    Cut,
    EndedEarly,
    Lost,
    Outcome,
    SettledWithoutTerminal,
)
from pplx_agent_tools.askstream.policy import (
    At,
    AtCompleted,
    AtTextComplete,
    Bounded,
    Completion,
    FirstContent,
    FirstContentOff,
    FirstContentWithin,
    Off,
    Policy,
    PolicyError,
    Reconnect,
    SettleAfterText,
    Stall,
    StallAfter,
    Unbounded,
)
from pplx_agent_tools.errors import SchemaError

UUID = BackendUuid("0d2b1c3a-1111-2222-3333-444455556666")
UUID2 = BackendUuid("0d2b1c3a-1111-2222-3333-777788889999")
CTX = ContextUuid("5e6f7a8b-aaaa-bbbb-cccc-ddddeeeeffff")
TOKEN_RAW = "tok-RAW-value-0123456789abcdef"


def token(raw: str = TOKEN_RAW) -> ReadWriteToken:
    t = ReadWriteToken.parse(raw)
    assert t is not None
    return t


KNOWN = Known(ThreadRef(UUID, token()), CTX)
UUID_ONLY = UuidOnly(UUID, CTX)

SETTLE = SettleAfterText(15.0)
COMPLETIONS: tuple[Completion, ...] = (SETTLE, AtTextComplete(), AtCompleted())


class UnexpectedRedirect(SchemaError):
    """Stands in for the redirect error the transport unit adds."""


OFF = Off()
FC_OFF = FirstContentOff()
AT_540 = At(540.0)
AWAITING = AwaitingFirst()
NO_IDS = NoIds()
NO_GRACE = NoGrace()


STALL_480 = StallAfter(480.0)


def policy(
    completion: Completion = SETTLE,
    *,
    reconnect: Reconnect = OFF,
    first_content: FirstContent = FC_OFF,
    deadline: At | Unbounded = AT_540,
    stall: Stall = STALL_480,
) -> Policy:
    p = Policy.make(
        deadline=deadline,
        stall=stall,
        completion=completion,
        answer_paths="ask_text_or_workflow",
        first_content=first_content,
        reconnect=reconnect,
    )
    assert not isinstance(p, PolicyError), p
    return p


ASK = policy()
ASK_RC = policy(reconnect=Bounded(3, 6), first_content=FirstContentWithin(90.0))


def live(
    *,
    phase: Phase = AWAITING,
    ids: fsm.Ids = NO_IDS,
    started_at: float = 0.0,
    deadline: At | Unbounded = AT_540,
    reconnectable: fsm.Reconnectable = "absent",
    rc_consecutive: int = 0,
    rc_total: int = 0,
    next_conn: int = 2,
) -> Live:
    return Live(
        started_at=started_at,
        deadline=deadline,
        ids=ids,
        cursor=None,
        reconnectable=reconnectable,
        phase=phase,
        rc_consecutive=rc_consecutive,
        rc_total=rc_total,
        next_conn=ConnId(next_conn),
    )


def streaming(
    lv: Live | None = None, *, last_byte_at: float = 0.0, grace: fsm.Grace = NO_GRACE
) -> Streaming:
    lv = live() if lv is None else lv
    return Streaming(lv, ConnId(lv.next_conn - 1), last_byte_at, grace)


def frame(
    stage: fsm.Stage = "pending", change: fsm.Change = "progress", **kw: object
) -> FrameSummary:
    return replace(FrameSummary(stage, change), **kw)  # pyright: ignore[reportArgumentType]


def last_progress(lv: Live) -> float:
    match lv.phase:
        case AwaitingFirst():
            return lv.started_at
        case Producing(lp) | TextComplete(lp, _):
            return lp


def remaining(lv: Live, now: float) -> float:
    return lv.deadline.t - now if isinstance(lv.deadline, At) else float("inf")


def expected_reconnects(p: Policy, lv: Live, now: float) -> bool:
    """§3.4 R(reason): every guard must hold."""
    rc = p.reconnect
    return (
        isinstance(rc, Bounded)
        and not isinstance(lv.ids, NoIds)
        and lv.reconnectable != "no"
        and lv.rc_consecutive < rc.consecutive
        and lv.rc_total < rc.total
        and remaining(lv, now) >= p.min_useful_s
    )


def expected_fallback(p: Policy, reason: ReconnectReason, lv: Live) -> Outcome:
    """§3.4 fallback table, with §2.6's `by` for the TextComplete row."""
    n = lv.rc_total
    if isinstance(lv.phase, TextComplete):
        by = {
            "settle": "settle",
            "stall": "stall",
            "silence": "stall",
            "first_content": "stall",
            "drop": "server",
            "eof": "server",
        }[reason]
        return SettledWithoutTerminal(n, by)  # pyright: ignore[reportArgumentType]
    if reason == "drop":
        return Lost("stream dropped")
    if reason == "eof":
        return EndedEarly(n, "server")
    if reason == "first_content":
        fc = p.first_content
        return Cut("first_content", fc.s if isinstance(fc, FirstContentWithin) else 0.0, n)
    return Cut("stall", p.stall.s if isinstance(p.stall, StallAfter) else p.silence_s, n)


def expected_deadline_outcome(lv: Live) -> Outcome:
    if isinstance(lv.phase, TextComplete):
        return SettledWithoutTerminal(lv.rc_total, "deadline")
    assert isinstance(lv.deadline, At)
    return Cut("deadline", lv.deadline.t - lv.started_at, lv.rc_total)


def expected_phase(p: Policy, lv: Live, fs: FrameSummary, now: float) -> Phase:
    """§3.4 T15/T16 and I18."""
    ph = lv.phase
    progress = fs.change == "progress"
    if isinstance(ph, TextComplete):
        return TextComplete(now, ph.at) if progress else ph
    if fs.stage == "text_complete" and isinstance(p.completion, SettleAfterText):
        if progress:
            return TextComplete(now, now)
        return TextComplete(
            lv.started_at if isinstance(ph, AwaitingFirst) else ph.last_progress_at, now
        )
    return Producing(now) if progress else ph


def expected_ids(ids: fsm.Ids, fs: FrameSummary) -> fsm.Ids:
    """I15, restated: the first uuid; the first token and context at or after it."""
    uuid = (
        ids.uuid if isinstance(ids, UuidOnly) else ids.ref.uuid if isinstance(ids, Known) else None
    )
    tok = ids.ref.token if isinstance(ids, Known) else None
    ctx = None if isinstance(ids, NoIds) else ids.context
    if uuid is None:
        uuid = fs.uuid
    if uuid is None:
        return NoIds()
    tok = tok if tok is not None else fs.token
    ctx = ctx if ctx is not None else fs.context
    return Known(ThreadRef(uuid, tok), ctx) if tok is not None else UuidOnly(uuid, ctx)
