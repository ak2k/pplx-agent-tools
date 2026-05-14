"""Hypothesis fuzz tests for input-robustness invariants.

Five surfaces, each with the same shape of invariant: against adversarial
input, the function MUST either return a sensible value or raise a typed
exception we own. Crashing with TypeError / KeyError / AttributeError is
a bug — it means the verb leaks an implementation detail.

  1. Search verb response parsing  — _keep, _to_hit, search_many
  2. Fetch verb SSE consumption    — _fetch_with_prompt
  3. FTS5 query escaping          — _fts5_escape (never crash SQLite)
  4. Cookie shape normalization   — auth._normalize (dict[str,str] or AuthError)
  5. Snippets retrieval pipeline  — _build_index + _hybrid_retrieve over
                                    adversarial corpora
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pplx_agent_tools.auth import _normalize
from pplx_agent_tools.errors import AuthError, SchemaError
from pplx_agent_tools.verbs.fetch import (
    _fetch_with_prompt,
    event_marks_completed,
    extract_chunks_from_event,
)
from pplx_agent_tools.verbs.search import (
    _keep,
    _to_hit,
    decode_search_response,
    search_many,
)
from pplx_agent_tools.verbs.snippets import (
    _build_index,
    _fts5_escape,
    _hybrid_retrieve,
    _vec_to_blob,
)
from tests._doubles import _TestClientBase

# ---------- shared strategies ----------

# JSON-like recursive value generator. Bounded leaves to keep individual
# inputs small (we want breadth across shapes, not enormous payloads).
_json_leaf = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=30),
)
_json_value = st.recursive(
    _json_leaf,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=15), children, max_size=5),
    ),
    max_leaves=15,
)
_json_dict = st.dictionaries(st.text(min_size=1, max_size=15), _json_value, max_size=8)


# ====================================================================
# 1. Search verb input robustness
# ====================================================================


@given(_json_value)
def test_keep_never_raises(payload: Any) -> None:
    """`_keep` is the inner filter — must return bool for ANY input."""
    result = _keep(payload)
    assert isinstance(result, bool)


@given(_json_dict)
def test_to_hit_either_succeeds_or_raises_schema(payload: dict[str, Any]) -> None:
    """`_to_hit` may raise SchemaError but no other exception type."""
    try:
        hit = _to_hit(payload)
    except SchemaError:
        return
    # On success: invariants
    assert isinstance(hit.url, str)
    assert isinstance(hit.title, str)
    assert isinstance(hit.images, list)


class _CannedClient(_TestClientBase):
    """Stand-in Client that returns a canned post_json payload."""

    def __init__(self, canned: Any) -> None:
        super().__init__()
        self._canned = canned

    def post_json(self, path: str, body: dict[str, Any]) -> Any:  # type: ignore[override]
        return self._canned


@given(_json_value)
def test_decode_search_response_returns_or_raises(payload: Any) -> None:
    """Pure decoder contract: SearchResult or SchemaError for ANY input.

    Tests the pure function directly (no wire setup) — faster than going
    through search_many + a canned client, and exercises exactly the
    parse logic. This is the trivially-fuzzable benefit of Move 3's
    pure-decoder extraction.
    """
    try:
        result = decode_search_response(payload, query="q", limit=10)
    except SchemaError:
        return
    assert isinstance(result.hits, list)
    assert result.total == len(result.hits)
    urls = [h.url for h in result.hits]
    assert len(urls) == len(set(urls))  # dedup invariant


@given(_json_value, st.integers(min_value=0, max_value=100))
def test_decode_search_response_respects_limit(payload: Any, limit: int) -> None:
    """`limit` is the hard cap on returned hits regardless of input size."""
    try:
        result = decode_search_response(payload, query="q", limit=limit)
    except SchemaError:
        return
    assert len(result.hits) <= limit


@given(_json_value)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_search_many_degrades_or_raises_schema(payload: Any) -> None:
    """Integration variant: `search_many` (orchestrator over the pure decoder)
    must also never leak unexpected exception types.
    """
    client = _CannedClient(payload)
    try:
        result = search_many(client, ["query"])
    except SchemaError:
        return
    assert isinstance(result.hits, list)
    assert result.total == len(result.hits)
    urls = [h.url for h in result.hits]
    assert len(urls) == len(set(urls))


# ====================================================================
# 2. Fetch verb SSE-event robustness
# ====================================================================


@given(_json_dict)
def test_extract_chunks_from_event_never_raises(event: dict[str, Any]) -> None:
    """Pure decoder: returns list[str] for ANY event dict, never raises.

    The chunk extractor runs once per SSE event in the fetch verb's hot
    path — a single KeyError here would crash mid-stream. Total function
    invariant: `list[str]` out, regardless of input.
    """
    chunks = extract_chunks_from_event(event)
    assert isinstance(chunks, list)
    for c in chunks:
        assert isinstance(c, str)


@given(_json_dict)
def test_event_marks_completed_returns_bool(event: dict[str, Any]) -> None:
    """Pure: returns bool for ANY event dict, never raises."""
    assert isinstance(event_marks_completed(event), bool)


def test_extract_chunks_known_shape() -> None:
    """Sanity: real SPA event shape produces the expected chunks."""
    event = {
        "data": {
            "blocks": [
                {
                    "intended_usage": "ask_text",
                    "markdown_block": {"chunks": ["Hello, ", "world."]},
                }
            ]
        }
    }
    assert extract_chunks_from_event(event) == ["Hello, ", "world."]


def test_extract_chunks_ignores_non_ask_text_blocks() -> None:
    """The parallel `ask_text_0_markdown` block carries the same chunks —
    consuming both would double-count. The filter must keep only ask_text.
    """
    event = {
        "data": {
            "blocks": [
                {
                    "intended_usage": "ask_text",
                    "markdown_block": {"chunks": ["A"]},
                },
                {
                    "intended_usage": "ask_text_0_markdown",
                    "markdown_block": {"chunks": ["A-dup"]},
                },
            ]
        }
    }
    assert extract_chunks_from_event(event) == ["A"]


def test_event_marks_completed_status_field() -> None:
    assert event_marks_completed({"data": {"status": "COMPLETED"}}) is True


def test_event_marks_completed_text_completed_field() -> None:
    assert event_marks_completed({"data": {"text_completed": True}}) is True


def test_event_marks_completed_neither_flag() -> None:
    assert event_marks_completed({"data": {"status": "PENDING"}}) is False


class _StreamClient(_TestClientBase):
    """Stand-in Client that yields canned SSE events from `sse_post`."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        super().__init__()
        self._events = events

    def sse_post(  # type: ignore[override]
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        yield from self._events

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        return True


# An adversarial SSE event has the {event, data} envelope but `data` is
# arbitrary (None, str, dict with weird types). The verb must not crash.
_event_envelope = st.fixed_dictionaries(
    {
        "event": st.one_of(st.none(), st.text(min_size=1, max_size=20)),
        "data": _json_value,
    }
)


@given(st.lists(_event_envelope, max_size=10))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_fetch_with_prompt_degrades_or_raises_schema(events: list[dict[str, Any]]) -> None:
    """`_fetch_with_prompt` consumes an arbitrary SSE stream. Allowed
    outcomes: FetchResult, or SchemaError. Nothing else.
    """
    client = _StreamClient(events)
    try:
        result = _fetch_with_prompt(
            client, "https://example.com", "p", "example.com", max_chars=None
        )
    except SchemaError:
        return
    # On success: invariants
    assert result.is_extracted is True
    assert isinstance(result.content, str)


# ====================================================================
# 3. _fts5_escape — never crash SQLite
# ====================================================================


@given(st.text(max_size=200))
def test_fts5_escape_never_crashes_sqlite(text: str) -> None:
    """For any input string, the escaped form must execute as a valid FTS5
    MATCH query without raising sqlite3.OperationalError.

    A fresh in-memory connection per call keeps the test self-contained;
    SQLite in-memory creation is microseconds.
    """
    expr = _fts5_escape(text)
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
        conn.execute("INSERT INTO t(body) VALUES ('hello world')")
        conn.execute("SELECT body FROM t WHERE t MATCH ?", (expr,)).fetchall()
    finally:
        conn.close()


# ====================================================================
# 4. Cookie shape normalization
# ====================================================================


@given(_json_value)
def test_normalize_returns_str_dict_or_raises_auth(payload: Any) -> None:
    """`_normalize` accepts dict or list-of-dicts. Anything else, or any
    invalid entry, must raise AuthError — never TypeError / KeyError.
    """
    try:
        out = _normalize(payload, source="fuzz")
    except AuthError:
        return
    # On success: must be flat str→str
    assert isinstance(out, dict)
    assert out, "empty result should have raised AuthError"
    for k, v in out.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


# Targeted strategy: well-formed Cookie-Editor entries with some adversarial
# extra fields. Hypothesis-shrunk failures here will pinpoint specific
# shapes that break the normalizer.
_cookie_entry = st.fixed_dictionaries(
    {
        "name": st.text(min_size=1, max_size=20),
        "value": st.one_of(_json_leaf, st.text(max_size=30)),
    },
    optional={
        "domain": st.text(max_size=30),
        "path": st.text(max_size=30),
        "expirationDate": st.floats(allow_nan=False, allow_infinity=False, width=32),
        "hostOnly": st.booleans(),
        "secure": st.booleans(),
    },
)


@given(st.lists(_cookie_entry, min_size=1, max_size=8))
def test_normalize_cookie_editor_array_well_formed(entries: list[dict[str, Any]]) -> None:
    """Well-formed Cookie-Editor arrays must always normalize cleanly.

    The `value` field may be any non-None JSON leaf (int, bool, float, str) —
    the normalizer must coerce to str.
    """
    # Drop entries where value is None (a documented hard failure mode)
    entries = [e for e in entries if e.get("value") is not None]
    if not entries:
        return
    out = _normalize(entries, source="fuzz")
    assert isinstance(out, dict)
    for k, v in out.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


# ====================================================================
# 5. Snippets _build_index + _hybrid_retrieve robustness
# ====================================================================


_url_strategy = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E, blacklist_characters="'\""),
    min_size=1,
    max_size=30,
).map(lambda s: f"https://x/{s}")

_paragraph_strategy = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0x7E,
        blacklist_categories=("Cs",),
    ),
    min_size=1,
    max_size=80,
)

# A single (url, paragraph_text, word_count) row matching _build_index's
# contract. word_count is generated independent of the actual word count
# because _build_index treats it as opaque (only retrieval uses it).
_row_strategy = st.tuples(
    _url_strategy, _paragraph_strategy, st.integers(min_value=1, max_value=50)
)


@given(rows=st.lists(_row_strategy, min_size=1, max_size=10))
@settings(
    suppress_health_check=[HealthCheck.too_slow],
    max_examples=30,
    # SQLite + sqlite-vec extension setup per example is naturally jittery
    # (~50-300ms on cold CI runners); the default 200ms deadline trips
    # spuriously. Robustness, not latency, is what this test is checking.
    deadline=None,
)
def test_hybrid_retrieve_never_crashes_on_arbitrary_corpus(
    rows: list[tuple[str, str, int]],
) -> None:
    """Build an index from adversarial-but-well-typed rows, run a hybrid
    retrieve scoped to one of those URLs. The invariant: the SQL pipeline
    (FTS5 MATCH + sqlite-vec KNN + RRF merge) must never raise — neither
    on the indexing side (unusual punctuation, very long strings) nor on
    retrieval (URL with no matching paragraphs).
    """
    dim = 3
    vecs = [[float(i % 3 == 0), float(i % 3 == 1), float(i % 3 == 2)] for i in range(len(rows))]
    conn = _build_index(rows, vecs, dim)
    try:
        query_blob = _vec_to_blob([1.0, 0.0, 0.0])
        target_url = rows[0][0]
        results = _hybrid_retrieve(conn, _fts5_escape("test query"), query_blob, target_url, k=5)
        # Whatever comes back must be a list of (text, score, words) where
        # text and score are reasonable and words is a positive int.
        for text, score, words in results:
            assert isinstance(text, str)
            assert isinstance(score, float)
            assert isinstance(words, int)
            assert words > 0
    finally:
        conn.close()
