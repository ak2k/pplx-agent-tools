"""Doubles for the stream driver: a clock that sleeps by advancing, a
client whose conns replay scripted items at scripted times, and a research
run over both."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pplx_agent_tools.askstream.driver import ConnBounds
from pplx_agent_tools.errors import PplxError
from pplx_agent_tools.verbs._research_stream import Decoder, ResearchRun, research_stream
from pplx_agent_tools.verbs.research import (
    ENDPOINT,
    _clarifying_warnings,
    _decode_parts,
    _join_answer,
)

DIFF = Path(__file__).parent / "fixtures" / "ask-diff"
LEGACY = Path(__file__).parent / "fixtures" / "research"

Item = dict[str, Any]
# (the clock time the item or error arrives at, the item or the error)
Script = Iterable[tuple[float, "Item | PplxError"]]

HEARTBEAT: Item = {"event": None, "data": None}


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def at_least(self, t: float) -> None:
        self.now = max(self.now, t)


def message(data: dict[str, Any]) -> Item:
    return {"event": "message", "data": data}


def fixture_items(stem: str, *, legacy: bool = False) -> list[Item]:
    path = (LEGACY if legacy else DIFF) / f"{stem}.events.jsonl"
    return [message(json.loads(line)) for line in path.read_text().splitlines()]


def paced(items: Iterable[Item | PplxError], *, start: float = 1.0, gap: float = 1.0) -> Script:
    return [(start + i * gap, x) for i, x in enumerate(items)]


def replay(clock: FakeClock, script: Script) -> Iterator[Item]:
    for t, x in script:
        clock.at_least(t)
        if isinstance(x, PplxError):
            raise x
        yield x


@dataclass
class Opened:
    kind: str
    bounds: ConnBounds
    uuid: str | None = None


@dataclass
class FakeClient:
    """Each initial POST replays the next of `initials`, each reconnect the
    next of `reconnects`; an open past the end replays nothing. Every open is
    logged with the bounds it was given."""

    clock: FakeClock
    initials: list[Script] = field(default_factory=list[Script])
    reconnects: list[Script] = field(default_factory=list[Script])
    opens: list[Opened] = field(default_factory=list[Opened])
    terminated: list[tuple[str, str, str]] = field(default_factory=list[tuple[str, str, str]])
    deleted: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    terminate_ok: bool = True

    def open_initial(self, bounds: ConnBounds) -> Iterator[Item]:
        self.opens.append(Opened("initial", bounds))
        return replay(self.clock, self.initials.pop(0) if self.initials else ())

    def sse_reconnect(
        self,
        backend_uuid: str,
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        silence_seconds: float | None = None,
    ) -> Iterator[Item]:
        assert silence_seconds is not None
        bounds = ConnBounds(max_total_seconds, stall_seconds, silence_seconds)
        self.opens.append(Opened("reconnect", bounds, backend_uuid))
        script = self.reconnects.pop(0) if self.reconnects else ()
        return replay(self.clock, script)

    def terminate(self, entry_uuid: str, context_uuid: str, model_preference: str) -> bool:
        self.terminated.append((entry_uuid, context_uuid, model_preference))
        return self.terminate_ok

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:
        self.deleted.append((entry_uuid, read_write_token))
        return True


DECODER = Decoder(_decode_parts, _join_answer, lambda q: _clarifying_warnings(q, no_answer=True))


def run_research(
    client: FakeClient,
    *,
    timeout: float | None = 3600.0,
    stall_seconds: float | None = 240.0,
    keep_thread: bool = False,
    err: io.StringIO | None = None,
    sleep: Callable[[float], None] | None = None,
    on_data: Callable[[dict[str, object]], None] | None = None,
    observe: Callable[[ResearchRun], None] | None = None,
) -> ResearchRun:
    """`research_stream` over the fake client, on its clock, with no jitter."""
    return research_stream(
        client,
        client.open_initial,
        DECODER,
        endpoint=ENDPOINT,
        keep_thread=keep_thread,
        timeout=timeout,
        stall_seconds=stall_seconds,
        on_data=on_data,
        observe=observe,
        clock=client.clock,
        sleep=client.clock.sleep if sleep is None else sleep,
        rand=lambda: 0.5,
        err=io.StringIO() if err is None else err,
    )
