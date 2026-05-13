"""Property-based tests for `_parse_sse_event`.

The SSE parser is the hottest piece of wire-level code (every event from
every streaming endpoint flows through it). The example tests in
`test_wire.py` cover the documented shapes; hypothesis here generates
adversarial inputs to flush out crashes and shape regressions.
"""

from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from pplx_agent_tools.wire import _parse_sse_event

# JSON values we expect Perplexity to ever embed in a `data:` line.
# `none` is excluded — `data: null` is technically valid but the parser
# treats it as "no data" in the dict path, which is a separate code path.
_json_values = st.recursive(
    st.one_of(
        st.booleans(),
        st.integers(min_value=-(2**31), max_value=2**31 - 1),
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        st.text(max_size=50),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
    ),
    max_leaves=20,
)

# event-type tokens: no newlines (would break framing), no leading colon
# (collides with comment-line prefix).
_event_types = st.text(
    alphabet=st.characters(blacklist_characters="\n\r:", min_codepoint=0x20, max_codepoint=0x7E),
    min_size=1,
    max_size=30,
).map(str.strip).filter(lambda s: len(s) > 0)


# ---------- shape invariants ----------


@given(_json_values)
def test_parse_json_data_roundtrips(value: object) -> None:
    raw = f"data: {json.dumps(value)}"
    out = _parse_sse_event(raw)
    assert out is not None
    assert out["event"] is None
    assert out["data"] == value


@given(_event_types, _json_values)
def test_parse_event_and_data_pair(event_type: str, value: object) -> None:
    raw = f"event: {event_type}\ndata: {json.dumps(value)}"
    out = _parse_sse_event(raw)
    assert out is not None
    assert out["event"] == event_type
    assert out["data"] == value


@given(_event_types)
def test_event_only_no_data_returns_data_none(event_type: str) -> None:
    raw = f"event: {event_type}"
    out = _parse_sse_event(raw)
    assert out is not None
    assert out["event"] == event_type
    assert out["data"] is None


@given(st.text(alphabet=st.characters(whitelist_categories=("Zs",)), max_size=20))
def test_whitespace_only_returns_none(ws: str) -> None:
    # Spaces, tabs and any unicode space-separator characters. Newline is
    # NOT in this set (would frame multiple empty events).
    assert _parse_sse_event(ws) is None


# ---------- comment-line invariance ----------

_comment_lines = st.text(
    alphabet=st.characters(blacklist_characters="\n\r", min_codepoint=0x20, max_codepoint=0x7E),
    max_size=40,
).map(lambda s: f":{s}")


@given(st.lists(_comment_lines, min_size=1, max_size=5), _json_values)
def test_comment_lines_are_ignored(comments: list[str], value: object) -> None:
    """Sprinkling SSE comment lines around a real data: payload must not
    change the parsed output (RFC 7-style "lines starting with `:` are
    comments")."""
    payload = f"data: {json.dumps(value)}"
    interleaved = "\n".join([*comments, payload, *comments])
    out = _parse_sse_event(interleaved)
    assert out is not None
    assert out["data"] == value


# ---------- multi-line data concatenation ----------


@given(
    st.lists(
        st.text(
            alphabet=st.characters(
                blacklist_characters="\n\r", min_codepoint=0x20, max_codepoint=0x7E
            ),
            min_size=1,
            max_size=20,
        ),
        min_size=2,
        max_size=5,
    )
)
def test_multiline_data_concatenates_with_newlines(parts: list[str]) -> None:
    """`data: a\\ndata: b` must parse as `"a\\nb"` (non-JSON path).

    We use payloads that are NOT valid JSON to stay on the string-return
    branch; the JSON branch is covered by `test_parse_json_data_roundtrips`.
    """
    # Force non-JSON by prefixing with a non-JSON-starter character.
    parts = [f"#{p}" for p in parts]
    raw = "\n".join(f"data: {p}" for p in parts)
    out = _parse_sse_event(raw)
    assert out is not None
    assert out["data"] == "\n".join(parts)


# ---------- robustness: parser never raises ----------


@given(
    st.text(
        alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E, whitelist_characters="\n"),
        max_size=500,
    )
)
def test_parser_never_raises_on_ascii_input(raw: str) -> None:
    # A streaming parser must NEVER raise on malformed input — it should
    # either return None or a {event, data} dict (with data falling back to
    # the raw string when JSON parsing fails).
    out = _parse_sse_event(raw)
    if out is None:
        return
    assert set(out.keys()) == {"event", "data"}
