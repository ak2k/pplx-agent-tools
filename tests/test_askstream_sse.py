"""SSE field parsing and event decoding (plan §2.3)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import AskFrame, EndOfStream, Heartbeat, Unparseable
from pplx_agent_tools.askstream.sse import SseFields, decode_event, parse_fields


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
