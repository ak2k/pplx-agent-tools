"""Snapshot tests locking the agent-contract JSON output per verb.

These intentionally use full-dict equality rather than partial-key checks.
The point is to catch silent breaking changes: if a render function adds,
removes, or renames a key, the test fails before consumers do.

When you intentionally change the JSON shape (pre-1.0 SemVer treats this
as a minor bump), update the expected dict here in the same commit so
the contract change is reviewable in a single diff.
"""

from __future__ import annotations

import pytest

from pplx_agent_tools import __version__
from pplx_agent_tools.render import (
    envelope,
    render_fetch_json,
    render_search_json,
    render_snippets_json,
)
from pplx_agent_tools.verbs.fetch import FetchResult
from pplx_agent_tools.verbs.search import Hit, SearchResult
from pplx_agent_tools.verbs.snippets import Snippet, SnippetsResult, UrlSnippets

# ---------- envelope() core ----------


def test_envelope_includes_version_and_verb() -> None:
    out = envelope("search", {"k": "v"})
    assert out["_pplx_tools_version"] == __version__
    assert out["_verb"] == "search"
    assert out["k"] == "v"


def test_envelope_omits_warnings_when_none() -> None:
    out = envelope("search", {"k": "v"})
    assert "warnings" not in out


def test_envelope_omits_warnings_when_empty_list() -> None:
    out = envelope("search", {"k": "v"}, warnings=[])
    assert "warnings" not in out


def test_envelope_includes_warnings_when_present() -> None:
    out = envelope("search", {"k": "v"}, warnings=["w1", "w2"])
    assert out["warnings"] == ["w1", "w2"]


def test_envelope_copies_warnings_list() -> None:
    src = ["w1"]
    out = envelope("search", {}, warnings=src)
    src.append("mutated")
    assert out["warnings"] == ["w1"]


@pytest.mark.parametrize("reserved", ["_pplx_tools_version", "_verb", "warnings"])
def test_envelope_rejects_reserved_keys_in_payload(reserved: str) -> None:
    with pytest.raises(ValueError, match="reserved keys"):
        envelope("search", {reserved: "no"})


def test_envelope_version_and_verb_appear_first() -> None:
    # Insertion order matters for human-readable JSON dumps: contract
    # metadata at the top of the object, payload below it.
    out = envelope("search", {"alpha": 1, "beta": 2})
    keys = list(out.keys())
    assert keys[0] == "_pplx_tools_version"
    assert keys[1] == "_verb"


# ---------- per-verb JSON snapshots ----------


def test_render_search_json_full_shape() -> None:
    result = SearchResult(
        query="claude code",
        hits=[
            Hit(
                url="https://anthropic.com/claude",
                title="Claude",
                domain="anthropic.com",
                snippet="A short snippet.",
                summary="A longer summary.",
                published_date="2026-01-01T00:00:00",
                images=["https://img/1"],
            ),
        ],
        total=1,
        warnings=["sample warning"],
    )
    assert render_search_json(result) == {
        "_pplx_tools_version": __version__,
        "_verb": "search",
        "query": "claude code",
        "hits": [
            {
                "url": "https://anthropic.com/claude",
                "title": "Claude",
                "domain": "anthropic.com",
                "snippet": "A short snippet.",
                "summary": "A longer summary.",
                "published_date": "2026-01-01T00:00:00",
                "images": ["https://img/1"],
            },
        ],
        "total": 1,
        "warnings": ["sample warning"],
    }


def test_render_search_json_minimal_hit() -> None:
    # Hit with everything optional set to None / empty list — verifies
    # the per-key omission discipline in _hit_to_json
    result = SearchResult(
        query="q",
        hits=[Hit(url="https://a/", title="A", domain=None, snippet=None)],
        total=1,
    )
    assert render_search_json(result) == {
        "_pplx_tools_version": __version__,
        "_verb": "search",
        "query": "q",
        "hits": [{"url": "https://a/", "title": "A"}],
        "total": 1,
    }


def test_render_fetch_json_full_shape() -> None:
    result = FetchResult(
        url="https://example.com/",
        title="Example",
        domain="example.com",
        content="Hello, world.",
        is_extracted=False,
        published_date="2026-01-01",
        truncated=False,
        stream_complete=True,
    )
    assert render_fetch_json(result) == {
        "_pplx_tools_version": __version__,
        "_verb": "fetch",
        "url": "https://example.com/",
        "domain": "example.com",
        "is_extracted": False,
        "truncated": False,
        "stream_complete": True,
        "content": "Hello, world.",
        "title": "Example",
        "published_date": "2026-01-01",
    }


def test_render_fetch_json_omits_optional_when_none() -> None:
    result = FetchResult(
        url="https://a/",
        title=None,
        domain="a",
        content="x",
        is_extracted=True,
        published_date=None,
        truncated=True,
        stream_complete=False,
    )
    assert render_fetch_json(result) == {
        "_pplx_tools_version": __version__,
        "_verb": "fetch",
        "url": "https://a/",
        "domain": "a",
        "is_extracted": True,
        "truncated": True,
        "stream_complete": False,
        "content": "x",
    }


def test_render_snippets_json_full_shape() -> None:
    result = SnippetsResult(
        query="async",
        results=[
            UrlSnippets(
                url="https://example.com",
                snippets=[Snippet(text="Async snippet.", score=0.5, tokens=3)],
            ),
        ],
        warnings=["fetched 1/1 urls"],
    )
    assert render_snippets_json(result) == {
        "_pplx_tools_version": __version__,
        "_verb": "snippets",
        "query": "async",
        "results": [
            {
                "url": "https://example.com",
                "snippets": [{"text": "Async snippet.", "score": 0.5, "tokens": 3}],
            },
        ],
        "warnings": ["fetched 1/1 urls"],
    }


def test_render_snippets_json_with_error_entry() -> None:
    result = SnippetsResult(
        query="q",
        results=[UrlSnippets(url="https://a/", error="fetch failed: 404")],
    )
    assert render_snippets_json(result) == {
        "_pplx_tools_version": __version__,
        "_verb": "snippets",
        "query": "q",
        "results": [
            {
                "url": "https://a/",
                "error": "fetch failed: 404",
                "snippets": [],
            },
        ],
    }
