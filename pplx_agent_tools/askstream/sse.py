"""One SSE event block (as split by the byte framer) into fields and a frame.

Field parsing follows the WHATWG event-stream rules: a line starting with `:`
is a comment, the field name runs to the first `:`, and one space after it is
dropped. Field names and event names outside the known sets are drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import EndOfStream, Frame, Heartbeat, decode_frame

__all__ = ["KNOWN_EVENTS", "SSE_FIELDS", "SseFields", "decode_event", "parse_fields"]

SSE_FIELDS = frozenset({"data", "event", "id", "retry"})
KNOWN_EVENTS = frozenset({"message", "end_of_stream"})


@final
@dataclass(frozen=True, slots=True)
class SseFields:
    event: str | None
    # The `data` lines joined by newlines; None when the block has none.
    data: str | None


def parse_fields(block: str) -> tuple[SseFields, tuple[Drift, ...]]:
    """The event name and data of one block. Never raises."""
    drift: list[Drift] = []
    event: str | None = None
    data: list[str] = []
    for line in block.split("\n"):
        if not line or line.startswith(":"):
            continue
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value
        elif name not in SSE_FIELDS:
            drift.append(Drift("unknown_sse_field", name_of(name)))
    if event is not None and event not in KNOWN_EVENTS:
        drift.append(Drift("unknown_sse_event", name_of(event)))
    return SseFields(event, "\n".join(data) if data else None), tuple(drift)


def decode_event(block: str) -> tuple[Frame, tuple[Drift, ...]]:
    """One block to a frame. A block without data is a heartbeat (comments
    and blank blocks alike) unless its event is `end_of_stream`. Never
    raises."""
    fields, drift = parse_fields(block)
    if fields.data is None:
        return (EndOfStream() if fields.event == "end_of_stream" else Heartbeat()), drift
    frame, more = decode_frame(fields.data)
    return frame, drift + more
