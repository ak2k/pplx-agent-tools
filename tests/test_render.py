"""Unit tests for render.py — pure functions over Result dataclasses."""

from __future__ import annotations

from pplx_agent_tools import __version__
from pplx_agent_tools.render import (
    render_fetch_json,
    render_fetch_text,
    render_search_json,
    render_search_text,
    render_snippets_json,
    render_snippets_text,
)
from pplx_agent_tools.verbs.fetch import FetchResult
from pplx_agent_tools.verbs.search import Hit, SearchResult
from pplx_agent_tools.verbs.snippets import Snippet, SnippetsResult, UrlSnippets


def _hit(**overrides: object) -> Hit:
    defaults: dict[str, object] = {
        "url": "https://example.com/foo",
        "title": "Foo",
        "domain": "example.com",
        "snippet": "A short snippet.",
        "summary": None,
        "published_date": None,
        "images": [],
    }
    defaults.update(overrides)
    return Hit(**defaults)  # type: ignore[arg-type]


# ---------- search ----------


def test_render_search_text_empty() -> None:
    result = SearchResult(query="x", hits=[], total=0)
    assert render_search_text(result) == "(no results)"


def test_render_search_text_three_lines_per_hit() -> None:
    result = SearchResult(
        query="q",
        hits=[_hit(title="Title One", url="https://a/", snippet="aaa")],
        total=1,
    )
    out = render_search_text(result)
    lines = out.splitlines()
    assert lines[0] == "1. Title One"
    assert lines[1].strip() == "https://a/"
    assert "aaa" in lines[2]


def test_render_search_text_collapses_snippet_newlines() -> None:
    # Reddit-style multi-line snippets must not break the 3-line-per-hit format
    result = SearchResult(
        query="q",
        hits=[_hit(snippet="line one\n\nline two\n  line three")],
        total=1,
    )
    out = render_search_text(result)
    # The third line should contain everything joined with spaces, no embedded \n
    assert "line one line two line three" in out
    # The hit block should still be a clean 3 lines (title / url / snippet)
    assert out.count("\n") == 2


def test_render_search_json_shape() -> None:
    result = SearchResult(
        query="q",
        hits=[_hit(summary="longer summary text", published_date="2026-05-12T00:00:00")],
        total=1,
        warnings=["test warning"],
    )
    js = render_search_json(result)
    assert js["_pplx_tools_version"] == __version__
    assert js["query"] == "q"
    assert "type" not in js
    assert js["total"] == 1
    assert js["warnings"] == ["test warning"]
    hit = js["hits"][0]
    assert hit["url"] == "https://example.com/foo"
    assert hit["title"] == "Foo"
    assert hit["summary"] == "longer summary text"
    assert hit["published_date"] == "2026-05-12T00:00:00"


def test_render_search_json_omits_optional_fields_when_none() -> None:
    result = SearchResult(query="q", hits=[_hit()], total=1)
    hit = render_search_json(result)["hits"][0]
    assert "summary" not in hit
    assert "published_date" not in hit
    assert "images" not in hit


def test_render_search_json_omits_warnings_when_empty() -> None:
    result = SearchResult(query="q", hits=[], total=0)
    assert "warnings" not in render_search_json(result)


# ---------- fetch ----------


def test_render_fetch_text_has_header_and_body() -> None:
    result = FetchResult(
        url="https://example.com/",
        title="Title",
        domain="example.com",
        content="page body",
        is_extracted=False,
    )
    out = render_fetch_text(result)
    assert "# Title" in out
    assert "https://example.com/" in out
    assert "domain: example.com" in out
    assert "page body" in out


def test_render_fetch_text_marks_extracted_mode() -> None:
    result = FetchResult(
        url="u",
        title=None,
        domain="d",
        content="LLM answer",
        is_extracted=True,
    )
    assert "extracted: yes (LLM)" in render_fetch_text(result)


def test_render_fetch_text_marks_incomplete_stream() -> None:
    # Partial answer from a deadline-clipped or server-cut stream must be
    # visually distinct from a complete one — a human eyeballing stdout
    # shouldn't mistake the two.
    result = FetchResult(
        url="u",
        title=None,
        domain="d",
        content="partial...",
        is_extracted=True,
        stream_complete=False,
    )
    out = render_fetch_text(result)
    assert "stream: incomplete" in out


def test_render_fetch_text_no_incomplete_marker_when_complete() -> None:
    result = FetchResult(
        url="u",
        title=None,
        domain="d",
        content="full answer",
        is_extracted=True,
        stream_complete=True,
    )
    assert "stream: incomplete" not in render_fetch_text(result)


def test_render_fetch_json_shape() -> None:
    result = FetchResult(
        url="https://example.com/",
        title="T",
        domain="example.com",
        content="C",
        is_extracted=False,
        published_date="2026-05-01",
        truncated=True,
    )
    js = render_fetch_json(result)
    assert js["_pplx_tools_version"] == __version__
    assert js["url"] == "https://example.com/"
    assert js["domain"] == "example.com"
    assert js["content"] == "C"
    assert js["is_extracted"] is False
    assert js["truncated"] is True
    assert js["title"] == "T"
    assert js["published_date"] == "2026-05-01"


def test_render_fetch_json_omits_none_optionals() -> None:
    result = FetchResult(url="u", title=None, domain="d", content="c", is_extracted=False)
    js = render_fetch_json(result)
    assert "title" not in js
    assert "published_date" not in js


# ---------- snippets ----------


def test_render_snippets_text_header_per_url() -> None:
    result = SnippetsResult(
        query="q",
        results=[
            UrlSnippets(url="https://a/", snippets=[Snippet(text="alpha", score=0.1, tokens=5)]),
            UrlSnippets(url="https://b/", error="fetch failed"),
        ],
    )
    out = render_snippets_text(result)
    assert "# https://a/" in out
    assert "alpha" in out
    assert "# https://b/" in out
    assert "error: fetch failed" in out


def test_render_snippets_text_marks_empty_results() -> None:
    result = SnippetsResult(
        query="q",
        results=[UrlSnippets(url="https://a/", snippets=[])],
    )
    assert "(no relevant snippets)" in render_snippets_text(result)


def test_render_snippets_json_shape() -> None:
    result = SnippetsResult(
        query="q",
        results=[
            UrlSnippets(
                url="https://a/",
                snippets=[Snippet(text="alpha", score=0.123456, tokens=5)],
            ),
            UrlSnippets(url="https://b/", error="fetch failed"),
        ],
    )
    js = render_snippets_json(result)
    assert js["_pplx_tools_version"] == __version__
    assert js["query"] == "q"
    assert len(js["results"]) == 2
    assert js["results"][0]["url"] == "https://a/"
    assert js["results"][0]["snippets"][0] == {
        "text": "alpha",
        "score": 0.12346,  # rounded to 5 decimals
        "tokens": 5,
    }
    assert js["results"][1]["error"] == "fetch failed"
