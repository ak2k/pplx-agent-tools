"""In-process retrieval tests for verbs/snippets.py.

Exercises `_build_index` + `_hybrid_retrieve` with synthetic vectors (no
fastembed model required), so the SQL + RRF math is verified without paying
the ONNX-cold-start cost in CI.

The pure-helper coverage (split, escape, blob roundtrip) lives in
`test_verbs_snippets.py`; this file targets the retrieval pipeline.
"""

from __future__ import annotations

import sqlite3

import pytest

from pplx_agent_tools.verbs.snippets import (
    RRF_K,
    _build_index,
    _fts5_escape,
    _hybrid_retrieve,
    _vec_to_blob,
)

URL_A = "https://a.example/page"
URL_B = "https://b.example/page"


def _make_rows() -> tuple[list[tuple[str, str, int]], list[list[float]]]:
    """Return canned (rows, vectors) for a small 2-URL corpus.

    Vectors are 3-dim and crafted so that a query vector `[1.0, 0.0, 0.0]`
    is closest to "claude code is a CLI" (cosine ~1) and farthest from
    "completely unrelated banana bread recipe" (cosine 0).
    """
    rows = [
        (URL_A, "claude code is a CLI for talking to claude", 9),
        (URL_A, "the CLI supports plugins and skills", 6),
        (URL_A, "completely unrelated banana bread recipe", 5),
        (URL_B, "another page about claude code architecture", 6),
        (URL_B, "this paragraph has no relevant terms whatsoever", 7),
    ]
    vecs = [
        [1.0, 0.0, 0.0],  # most similar to query
        [0.9, 0.1, 0.0],  # second most
        [0.0, 0.0, 1.0],  # orthogonal
        [0.8, 0.2, 0.0],  # similar (other URL)
        [0.0, 1.0, 0.0],  # orthogonal
    ]
    return rows, vecs


@pytest.fixture
def conn() -> sqlite3.Connection:
    """Build the in-memory index once per test."""
    rows, vecs = _make_rows()
    return _build_index(rows, vecs, dim=3)


def test_retrieve_scopes_results_to_requested_url(conn: sqlite3.Connection) -> None:
    """A query for URL_A must never return paragraphs from URL_B."""
    query_blob = _vec_to_blob([1.0, 0.0, 0.0])
    results = _hybrid_retrieve(conn, _fts5_escape("claude code CLI"), query_blob, URL_A, k=5)
    assert results, "expected at least one hit for URL_A"
    # Every result text must be one of URL_A's rows
    rows, _ = _make_rows()
    url_a_texts = {text for url, text, _ in rows if url == URL_A}
    for text, _, _ in results:
        assert text in url_a_texts


def test_retrieve_orders_by_rrf_score_descending(conn: sqlite3.Connection) -> None:
    query_blob = _vec_to_blob([1.0, 0.0, 0.0])
    results = _hybrid_retrieve(conn, _fts5_escape("claude code CLI"), query_blob, URL_A, k=5)
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True), "RRF scores must be descending"


def test_retrieve_top_hit_is_most_relevant(conn: sqlite3.Connection) -> None:
    """BM25 (matches "claude code") + cosine (vec=[1,0,0]) both rank the
    first row highest, so it must come out on top under RRF."""
    query_blob = _vec_to_blob([1.0, 0.0, 0.0])
    results = _hybrid_retrieve(conn, _fts5_escape("claude code CLI"), query_blob, URL_A, k=5)
    top_text, _, _ = results[0]
    assert top_text == "claude code is a CLI for talking to claude"


def test_retrieve_returns_word_count_for_token_budgeting(conn: sqlite3.Connection) -> None:
    query_blob = _vec_to_blob([1.0, 0.0, 0.0])
    results = _hybrid_retrieve(conn, _fts5_escape("CLI"), query_blob, URL_A, k=5)
    rows, _ = _make_rows()
    expected = {text: words for url, text, words in rows if url == URL_A}
    for text, _, words in results:
        assert words == expected[text]


def test_rrf_score_in_expected_range_when_both_indices_hit_top(conn: sqlite3.Connection) -> None:
    """The first row is rank-1 in both BM25 and vector indices, so its RRF
    score must be exactly 2/(RRF_K + 1).
    """
    query_blob = _vec_to_blob([1.0, 0.0, 0.0])
    results = _hybrid_retrieve(conn, _fts5_escape("claude code"), query_blob, URL_A, k=5)
    top_text, top_score, _ = results[0]
    assert top_text == "claude code is a CLI for talking to claude"
    expected = 2.0 / (RRF_K + 1)
    # Float comparison: RRF math is exact rationals, but float32 in the
    # vec roundtrip may introduce rounding. A 1e-9 epsilon is well outside
    # anything that could shift ranks.
    assert abs(top_score - expected) < 1e-9


def test_retrieve_handles_query_with_no_bm25_match(conn: sqlite3.Connection) -> None:
    """If the FTS5 side returns zero rows, the vector side must still rank
    paragraphs (covering paraphrased / non-keyword queries).
    """
    query_blob = _vec_to_blob([1.0, 0.0, 0.0])
    results = _hybrid_retrieve(conn, _fts5_escape("xyzzy-not-a-word"), query_blob, URL_A, k=5)
    # vector index should still return URL_A's rows
    assert results, "vector index must contribute when BM25 misses"
    rows, _ = _make_rows()
    url_a_texts = {text for url, text, _ in rows if url == URL_A}
    for text, _, _ in results:
        assert text in url_a_texts
    # Vector-only ranking must still place the closest vector first
    # (query=[1,0,0], row 0 vec=[1,0,0] is exact match). Catches a sign or
    # ORDER BY distance direction flip in _hybrid_retrieve's vector branch.
    assert results[0][0] == "claude code is a CLI for talking to claude"


def test_retrieve_handles_url_with_no_paragraphs() -> None:
    """A URL that exists in the request but never made it into the index
    must yield an empty result, not crash.
    """
    rows, vecs = _make_rows()
    conn = _build_index(rows, vecs, dim=3)
    query_blob = _vec_to_blob([1.0, 0.0, 0.0])
    results = _hybrid_retrieve(
        conn, _fts5_escape("claude"), query_blob, "https://never-indexed/", k=5
    )
    assert results == []


def test_retrieve_k_limits_per_index_candidates() -> None:
    """k=1 should restrict each index to one candidate, so RRF merges at
    most 2 distinct rows (BM25-top union vector-top).
    """
    rows, vecs = _make_rows()
    conn = _build_index(rows, vecs, dim=3)
    query_blob = _vec_to_blob([1.0, 0.0, 0.0])
    results = _hybrid_retrieve(conn, _fts5_escape("claude code CLI"), query_blob, URL_A, k=1)
    assert 1 <= len(results) <= 2
