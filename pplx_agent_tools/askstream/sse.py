"""One SSE event block (as split by the byte framer) into fields and a frame.

Field parsing follows the WHATWG event-stream rules: a line starting with `:`
is a comment, the field name runs to the first `:`, and one space after it is
dropped. Field names and event names outside the known sets are drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import (
    EndOfStream,
    Frame,
    Heartbeat,
    decode_frame,
    decode_object,
    unparseable,
)
from pplx_agent_tools.askstream.jsonval import measure, weight

__all__ = [
    "KNOWN_EVENTS",
    "SSE_FIELDS",
    "SseFields",
    "decode_event",
    "decode_parsed",
    "parse_fields",
]

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


def decode_parsed(event: str | None, data: object) -> tuple[Frame, tuple[Drift, ...]]:
    """One event as the transport yields it, `data` already through its JSON
    parser (or left as text when that failed). An object or array payload
    decodes as `decode_event` would decode its text; None data is a
    heartbeat and str data is `syntax`, since the transport yields those
    for a JSON `null` or string too. Never raises."""
    drift: tuple[Drift, ...] = ()
    if event is not None and event not in KNOWN_EVENTS:
        drift = (Drift("unknown_sse_event", name_of(event)),)
    if data is None:
        return (EndOfStream() if event == "end_of_stream" else Heartbeat()), drift
    if isinstance(data, str):
        frame, more = unparseable("syntax", len(data))
        return frame, drift + more
    m = measure(data)
    if m is None:
        # The transport's parser accepts NaN and unbounded nesting; the text
        # decoder refuses the first as syntax and the second as depth.
        deep = measure(data, max_depth=2**62) is not None
        frame, more = unparseable("depth" if deep else "syntax", 0)
        return frame, drift + more
    if not isinstance(m.value, dict):
        frame, more = unparseable("not_object", weight(m.value))
        return frame, drift + more
    frame, more = decode_object(m.value, m.weight)
    return frame, drift + more
