"""SSE field parsing and event decoding (plan §2.3)."""

from __future__ import annotations

import dataclasses
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import (
    AskFrame,
    EndOfStream,
    Frame,
    Heartbeat,
    Unparseable,
)
from pplx_agent_tools.askstream.jsonval import JsonValue, weight
from pplx_agent_tools.askstream.sse import SseFields, decode_event, decode_parsed, parse_fields
from tests.test_askstream_frames import ENVELOPE, JSON


def test_data_lines_join_and_one_leading_space_is_dropped() -> None:
    fields, drift = parse_fields('event: message\ndata:{"a":\ndata:  1}')
    assert fields == SseFields("message", '{"a":\n 1}')
    assert drift == ()


def test_comments_id_and_retry_are_skipped() -> None:
    fields, drift = parse_fields(": ping\nid: 7\nretry: 1000\ndata: {}")
    assert fields == SseFields(None, "{}")
    assert drift == ()


def test_line_without_colon_is_a_field_with_empty_value() -> None:
    fields, drift = parse_fields("data\ndata")
    assert fields == SseFields(None, "\n")
    assert drift == ()


def test_unknown_field_and_event_are_drift() -> None:
    fields, drift = parse_fields("weird: 1\nevent: surprise\ndata: {}")
    assert fields.event == "surprise"
    assert drift == (
        Drift("unknown_sse_field", name_of("weird")),
        Drift("unknown_sse_event", name_of("surprise")),
    )


def test_event_name_is_last_one_given() -> None:
    fields, drift = parse_fields("event: surprise\nevent: message\ndata: {}")
    assert fields.event == "message"
    assert drift == ()


def test_comment_only_and_blank_blocks_are_heartbeats() -> None:
    assert decode_event(": keep-alive") == (Heartbeat(), ())
    assert decode_event("") == (Heartbeat(), ())
    assert decode_event("event: message") == (Heartbeat(), ())


def test_end_of_stream() -> None:
    assert decode_event("event: end_of_stream") == (EndOfStream(), ())
    assert decode_event("event: end_of_stream\ndata: {}") == (EndOfStream(), ())
    assert decode_event("data: {}") == (EndOfStream(), ())


def test_data_decodes_to_a_frame_and_drift_accumulates() -> None:
    frame, drift = decode_event('bogus: 1\ndata: {"status": "PENDING", "new_key": 1}')
    assert isinstance(frame, AskFrame)
    assert drift == (
        Drift("unknown_sse_field", name_of("bogus")),
        Drift("unknown_envelope_key", name_of("new_key")),
    )


def test_bad_data_is_unparseable() -> None:
    frame, drift = decode_event("data: {not json")
    assert frame == Unparseable("syntax", len("{not json"))
    assert drift == (Drift("unparseable_frame", name_of("syntax")),)


@given(st.text())
def test_decode_event_is_total(block: str) -> None:
    frame, drift = decode_event(block)
    assert isinstance(frame, (Heartbeat, EndOfStream, Unparseable, AskFrame))
    assert all(isinstance(d, Drift) for d in drift)


@given(
    st.lists(st.sampled_from(["data", "event", "id", "retry", ":c", "x"]), max_size=6),
    st.text(st.characters(exclude_characters="\n"), max_size=20),
)
def test_parse_fields_is_total_on_field_soup(names: list[str], value: str) -> None:
    block = "\n".join(f"{n}: {value}" for n in names)
    fields, _ = parse_fields(block)
    assert (fields.data is not None) == ("data" in names)


# --- decode_parsed: the transport's parsed events ------------------------------------------------


def _deep(n: int) -> dict[str, object]:
    d: dict[str, object] = {"status": "PENDING"}
    for _ in range(n - 1):
        d = {"status": "PENDING", "x": d}
    return d


@pytest.mark.parametrize("event", [None, "message"])
def test_parsed_without_data_is_a_heartbeat(event: str | None) -> None:
    assert decode_parsed(event, None) == (Heartbeat(), ())


def test_parsed_end_of_stream() -> None:
    assert decode_parsed("end_of_stream", None) == (EndOfStream(), ())
    assert decode_parsed("end_of_stream", {}) == (EndOfStream(), ())
    assert decode_parsed("message", {}) == (EndOfStream(), ())


def test_parsed_text_that_was_not_json_is_unparseable_syntax() -> None:
    assert decode_parsed("message", "{not json") == (
        Unparseable("syntax", len("{not json")),
        (Drift("unparseable_frame", name_of("syntax")),),
    )


@pytest.mark.parametrize("data", [[1, 2], 3, 1.5, True])
def test_parsed_non_object_is_unparseable(data: object) -> None:
    frame, drift = decode_parsed("message", data)
    assert isinstance(frame, Unparseable)
    assert frame.reason == "not_object"
    assert drift == (Drift("unparseable_frame", name_of("not_object")),)


def test_parsed_depth_bound_matches_the_text_decoder() -> None:
    frame, _ = decode_parsed("message", _deep(128))
    assert isinstance(frame, AskFrame)
    frame, drift = decode_parsed("message", _deep(129))
    assert isinstance(frame, Unparseable)
    assert frame.reason == "depth"
    assert drift == (Drift("unparseable_frame", name_of("depth")),)


def test_parsed_non_finite_number_is_unparseable_syntax() -> None:
    frame, _ = decode_parsed("message", {"status": float("nan")})
    assert isinstance(frame, Unparseable)
    assert frame.reason == "syntax"


def test_parsed_unknown_event_is_drift() -> None:
    frame, drift = decode_parsed("surprise", {"status": "PENDING"})
    assert isinstance(frame, AskFrame)
    assert drift == (Drift("unknown_sse_event", name_of("surprise")),)


def test_parsed_frame_size_is_the_payload_weight() -> None:
    payload: dict[str, JsonValue] = {"status": "PENDING", "text": "abc"}
    frame, _ = decode_parsed("message", payload)
    assert isinstance(frame, AskFrame)
    assert frame.size == weight(payload)


def _sizeless(frame: Frame) -> Frame:
    if isinstance(frame, (AskFrame, Unparseable)):
        return dataclasses.replace(frame, size=0)
    return frame


@settings(max_examples=300)
@given((ENVELOPE | JSON).filter(lambda v: v is not None and not isinstance(v, str)))
def test_parsed_decodes_as_the_text_decoder_does(value: object) -> None:
    """The one input the two paths share, a JSON value, decodes the same
    except for `size`: text length on one side, weight on the other. A
    top-level `null` or string is left out: the transport yields None data
    for `null` and for a block with no data alike, and a str both for a JSON
    string and for text its parser refused."""
    raw = json.dumps(value)
    frame, drift = decode_parsed("message", json.loads(raw))
    text_frame, text_drift = decode_event(f"event: message\ndata: {raw}")
    assert (_sizeless(frame), drift) == (_sizeless(text_frame), text_drift)


@given(st.none() | st.text(max_size=12), JSON)
def test_decode_parsed_is_total(event: str | None, data: object) -> None:
    frame, drift = decode_parsed(event, data)
    assert isinstance(frame, (Heartbeat, EndOfStream, Unparseable, AskFrame))
    assert all(isinstance(d, Drift) for d in drift)
