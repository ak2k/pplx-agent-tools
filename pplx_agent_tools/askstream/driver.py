"""The synchronous interpreter that runs the lifecycle FSM over the transport.

The driver owns the clock, the sleeps and the one open stream; `fsm.step`
decides everything else. Each item the transport yields is decoded, applied
to the `BlockStore` and summarized for the FSM; each transport error becomes
an FSM event by type. Every Open carries bounds sized from the FSM's own
deadline and windows, so the transport's aborts back the FSM's timers up
rather than race them.

Due timers are stepped before the item that revealed them (`Tick` first), so
an item that arrives after the deadline or a stall window cannot complete a
run the timer had already ended. The one timer never stepped is `open_due`:
an Open here blocks inside the first read until the headers and the first
item arrive or the transport's own timeouts fire, so time spent there says
nothing about missing headers.
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TextIO, TypeAlias, final

from typing_extensions import assert_never

from pplx_agent_tools.askstream import fsm
from pplx_agent_tools.askstream.blocks import BlockStore, CapExceeded
from pplx_agent_tools.askstream.drift import Drift
from pplx_agent_tools.askstream.frames import AskFrame, EndOfStream, Heartbeat, Unparseable
from pplx_agent_tools.askstream.fsm import (
    AwaitingFirst,
    Close,
    Done,
    Event,
    Failure,
    Fatal,
    FrameIn,
    FrameSummary,
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
    StartBackoff,
    Starting,
    State,
    StreamBroke,
    StreamEnded,
    Streaming,
    TextComplete,
    Tick,
    Transient,
)
from pplx_agent_tools.askstream.ids import ConnId, ThreadRef
from pplx_agent_tools.askstream.policy import At, Policy, StallAfter, StallOff, Unbounded
from pplx_agent_tools.askstream.sse import decode_parsed
from pplx_agent_tools.errors import (
    NetworkError,
    PplxError,
    RateLimitError,
    SchemaError,
    StreamDeadlineError,
    StreamSilenceError,
    ThreadGoneError,
)
from pplx_agent_tools.jsonval import as_object

PROGRESS_STRIDE = 10


@final
@dataclass(frozen=True, slots=True)
class ConnBounds:
    """The transport's bounds for one conn: the time left to the deadline,
    the stall window, and the silence window capped by the time left."""

    max_total_seconds: float | None
    stall_seconds: float | None
    silence_seconds: float


Item: TypeAlias = dict[str, Any]
Opener: TypeAlias = Callable[[ConnBounds], Iterator[Item]]


class StreamClient(Protocol):
    def sse_reconnect(
        self,
        backend_uuid: str,
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        silence_seconds: float | None = None,
    ) -> Iterator[Item]: ...

    def terminate(self, entry_uuid: str, context_uuid: str, model_preference: str) -> bool: ...

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool: ...


def delete(client: StreamClient, ref: ThreadRef) -> bool:
    """Delete the thread; the one place the raw token is read."""
    return client.delete_thread(ref.uuid, ref.token.reveal())


def reconnecting(client: StreamClient, backend_uuid: str) -> Opener:
    """The opener that reattaches to the thread `backend_uuid`."""

    def reopen(b: ConnBounds) -> Iterator[Item]:
        return client.sse_reconnect(
            backend_uuid,
            max_total_seconds=b.max_total_seconds,
            stall_seconds=b.stall_seconds,
            silence_seconds=b.silence_seconds,
        )

    return reopen


def _deadline(s: State) -> At | Unbounded:
    match s:
        case Starting() | StartBackoff():
            return s.deadline
        case Streaming() | Reconnecting() | ReconnectBackoff():
            return s.live.deadline
        case Done():
            return Unbounded()
        case _:
            assert_never(s)


def _since_progress(before: State, now: float) -> float | None:
    match before:
        case Streaming() | Reconnecting() | ReconnectBackoff():
            phase = before.live.phase
            match phase:
                case Producing(lp) | TextComplete(lp, _):
                    return now - lp
                case AwaitingFirst():
                    return None
                case _:
                    assert_never(phase)
        case Starting() | StartBackoff() | Done():
            return None
        case _:
            assert_never(before)


def _lazy(opener: Opener, bounds: ConnBounds) -> Generator[Item, None, None]:
    # Defers the call to the first read, so an opener that fails at once is
    # reported like one that fails on its first item.
    yield from opener(bounds)


class Driver:
    """One run: `run()` steps the FSM to `Done`. `state` is readable at any
    time, so a caller's `finally` sees the Live state an interrupt (Ctrl-C
    included, during a backoff sleep too) left behind."""

    def __init__(
        self,
        policy: Policy,
        client: StreamClient,
        opener: Opener,
        store: BlockStore,
        *,
        label: str,
        on_frame: Callable[[AskFrame], None] | None = None,
        on_data: Callable[[dict[str, object]], None] | None = None,
        progress: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rand: Callable[[], float] = random.random,
        err: TextIO | None = None,
    ) -> None:
        self._policy = policy
        self._client = client
        self._opener = opener
        self._store = store
        self._label = label
        self._on_frame = on_frame
        self._on_data = on_data
        self._progress = progress
        self._clock = clock
        self._sleep = sleep
        self._rand = rand
        self._err = err
        self._conn: ConnId | None = None
        self._gen: Generator[Item, None, None] | None = None
        self._opened = False
        self._dots_pending = False
        self.state: State | None = None
        # Set at the step that reaches Done.
        self.trigger: ReconnectReason | None = None
        self.since_progress: float | None = None
        self.gone = False
        self.display_model: str | None = None
        self.terminal_text: str | None = None
        self.last_error: PplxError | None = None
        self.drift: Counter[Drift] = Counter()
        self.items = 0
        self.reconnect_opens = 0

    # --- the loop ---------------------------------------------------------------------------

    def run(self) -> Done:
        now = self._clock()
        state, effects = fsm.initial(self._policy, now)
        self.state = state
        try:
            self._apply(effects, now)
            while True:
                s = self._live_state()
                if isinstance(s, Done):
                    return s
                if fsm.current_conn(s) is None:
                    self._wait()
                else:
                    self._read()
        finally:
            self._close()
            if self._dots_pending:
                print("", file=self._stderr(), flush=True)

    def _wait(self) -> None:
        """A backoff state: nothing to read until its timer."""
        wake = fsm.next_wake(self._policy, self._live_state())
        assert wake is not None
        self._sleep(max(0.0, wake - self._clock()))
        self._step(Tick(self._clock()))

    def _read(self) -> None:
        conn, gen = self._conn, self._gen
        assert conn is not None and gen is not None
        try:
            item = next(gen)
        except StopIteration:
            self._ended(conn)
            return
        except PplxError as e:
            self._failed(conn, e)
            return
        now = self._clock()
        self._first_result(conn, now)
        self._tick_due(now)
        if self._current() != conn:
            return
        self._item(conn, now, item)

    def _ended(self, conn: ConnId) -> None:
        now = self._clock()
        self._first_result(conn, now)
        self._tick_due(now)
        if self._current() == conn:
            self._step(StreamEnded(conn, now))

    def _failed(self, conn: ConnId, e: PplxError) -> None:
        now = self._clock()
        self.last_error = e
        self._tick_due(now)
        if self._current() != conn:
            return
        if not self._opened:
            self._step(OpenFailed(conn, now, self._open_failure(e)))
            return
        match e:
            case StreamSilenceError():
                self._step(StreamBroke(conn, now, "silence", str(e)))
            case StreamDeadlineError():
                # The deadline or the stall backstop: the FSM's own timer is
                # due by now, or soon, and decides what the cut means.
                self._drain(conn)
            case SchemaError():
                self._step(StreamBroke(conn, now, "oversize", str(e)))
            case _:
                self._step(StreamBroke(conn, now, "transport", str(e)))

    def _drain(self, conn: ConnId) -> None:
        """Tick a conn that can deliver nothing more until the FSM leaves it."""
        while self._current() == conn:
            wake = fsm.next_wake(self._policy, self._live_state())
            assert wake is not None
            self._sleep(max(0.0, wake - self._clock()))
            self._step(Tick(self._clock()))

    def _open_failure(self, e: PplxError) -> Failure:
        if isinstance(e, RateLimitError):
            return RateLimited(e.retry_after)
        if isinstance(e, ThreadGoneError):
            self.gone = True
        # A failed initial open rejects the run either way; Fatal keeps the
        # original error, where the FSM would rebuild one for a Transient or
        # a Gone.
        if isinstance(self.state, Starting):
            return Fatal(e)
        if isinstance(e, ThreadGoneError):
            return Gone(403)
        return Transient(str(e)) if isinstance(e, NetworkError) else Fatal(e)

    def _first_result(self, conn: ConnId, now: float) -> None:
        if not self._opened and self._current() == conn:
            self._opened = True
            self._step(Opened(conn, now))

    def _tick_due(self, now: float) -> None:
        while not isinstance(self.state, Done):
            before = self.state
            due = fsm.due_class(self._policy, self._live_state(), now)
            if due in ("none", "open_due"):
                return
            self._step(Tick(now))
            if self.state is before:
                return

    def _item(self, conn: ConnId, now: float, item: Item) -> None:
        self.items += 1
        if self._progress and self.items % PROGRESS_STRIDE == 0:
            print(".", end="", file=self._stderr(), flush=True)
            self._dots_pending = True
        data: object = item.get("data")
        event: object = item.get("event")
        frame, drift = decode_parsed(event if isinstance(event, str) else None, data)
        self.drift.update(drift)
        match frame:
            case AskFrame():
                obj = as_object(data)
                if self._on_data is not None and obj is not None:
                    self._on_data(obj)
                applied = self._store.apply_frame(frame)
                if frame.display_model is not None:
                    self.display_model = frame.display_model
                if isinstance(applied, CapExceeded):
                    detail = f"{applied.cap} {applied.observed} over {applied.limit}"
                    ids = FrameSummary(
                        stage=frame.stage,
                        change="idle",
                        uuid=frame.backend_uuid,
                        token=frame.token,
                        context=frame.context_uuid,
                    )
                    self._step(StreamBroke(conn, now, "cap", detail, ids))
                    return
                self.drift.update(applied.drift)
                if frame.stage == "completed":
                    self.terminal_text = frame.text
                if self._on_frame is not None:
                    self._on_frame(frame)
                summary = FrameSummary(
                    stage=frame.stage,
                    change=applied.change,
                    uuid=frame.backend_uuid,
                    token=frame.token,
                    cursor=frame.cursor,
                    context=frame.context_uuid,
                    reconnectable=frame.reconnectable,
                    raw_status=frame.raw_status,
                )
                self._step(FrameIn(conn, now, summary))
            case Heartbeat() | Unparseable():
                self._step(HeartbeatIn(conn, now))
            case EndOfStream():
                self._step(StreamEnded(conn, now))
            case _:
                assert_never(frame)

    # --- stepping and effects ---------------------------------------------------------------

    def _live_state(self) -> State:
        assert self.state is not None
        return self.state

    def _current(self) -> ConnId | None:
        return fsm.current_conn(self._live_state())

    def _step(self, e: Event) -> None:
        before = self._live_state()
        after, effects = fsm.step(self._policy, before, e, self._rand())
        self.state = after
        if isinstance(after, Done) and not isinstance(before, Done):
            self.trigger = fsm.ended_by(self._policy, before, e)
            self.since_progress = _since_progress(before, e.now)
        self._apply(effects, e.now)

    def _apply(self, effects: tuple[fsm.Effect, ...], now: float) -> None:
        for x in effects:
            match x:
                case Open(conn=conn, target=InitialPost()):
                    self._begin(conn, self._opener, now)
                case Open(conn=conn, target=ReconnectTarget(uuid=uuid)):
                    self._store.begin_reconnect()
                    self.reconnect_opens += 1
                    self._begin(conn, reconnecting(self._client, uuid), now)
                case Close(conn=conn):
                    if self._conn == conn:
                        self._close()
                case Notice(detail=detail):
                    prefix = "\n" if self._dots_pending else ""
                    self._dots_pending = False
                    print(f"{prefix}{self._label}: {detail}", file=self._stderr(), flush=True)
                case _:
                    assert_never(x)

    def _bounds(self, now: float) -> ConnBounds:
        left = self._deadline_left(now)
        silence = self._policy.silence_s if left is None else min(self._policy.silence_s, left)
        stall = self._policy.stall
        match stall:
            case StallAfter(s):
                stall_s: float | None = s
            case StallOff():
                stall_s = None
            case _:
                assert_never(stall)
        return ConnBounds(left, stall_s, silence)

    def _deadline_left(self, now: float) -> float | None:
        deadline = _deadline(self._live_state())
        match deadline:
            case At(t):
                return t - now
            case Unbounded():
                return None
            case _:
                assert_never(deadline)

    def _begin(self, conn: ConnId, opener: Opener, now: float) -> None:
        self._close()
        self._conn, self._gen, self._opened = conn, _lazy(opener, self._bounds(now)), False

    def _close(self) -> None:
        gen, self._gen, self._conn = self._gen, None, None
        if gen is not None:
            gen.close()

    def _stderr(self) -> TextIO:
        return sys.stderr if self._err is None else self._err
