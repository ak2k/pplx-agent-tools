"""Fixture-replay tests: feed real captured Perplexity responses through the
verb layer to catch schema regressions.

Fixtures under tests/fixtures/ are sanitized real responses (no cookie values,
no PII). When Perplexity changes a response shape, these tests fail loudly,
which is the whole point of this tier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pplx_agent_tools.verbs.search import decode_search_response, search_many
from tests._doubles import _TestClientBase

FIXTURES = Path(__file__).parent / "fixtures"


class FakeClient(_TestClientBase):
    """Test stand-in: skips the curl_cffi setup, returns a canned payload
    from `post_json`. Inherits Client so type checks downstream still hold.
    """

    def __init__(self, canned: dict[str, Any]) -> None:
        super().__init__()
        self._canned = canned
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post_json(self, path: str, body: dict[str, Any]) -> Any:  # type: ignore[override]
        self.calls.append((path, body))
        return self._canned


@pytest.fixture
def claude_code_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "search-web" / "claude-code-realtime.json").read_text())


def test_search_many_parses_real_response(claude_code_payload: dict[str, Any]) -> None:
    client = FakeClient(claude_code_payload)
    result = search_many(client, ["claude code"])

    # Endpoint and body shape we send
    assert client.calls[0][0] == "/rest/realtime/search-web"
    sent_body = client.calls[0][1]
    assert sent_body["queries"] == ["claude code"]
    assert isinstance(sent_body["session_id"], str)

    # We get hits back, each with the documented shape
    assert result.hits, "expected at least one hit from fixture"
    for h in result.hits:
        assert h.url.startswith("http")
        assert h.title  # non-empty
        # snippet/summary are commonly present but optional
        assert h.snippet is None or isinstance(h.snippet, str)
        assert h.summary is None or isinstance(h.summary, str)


def test_search_many_limit_applies_to_fixture(claude_code_payload: dict[str, Any]) -> None:
    client = FakeClient(claude_code_payload)
    result = search_many(client, ["claude code"], limit=2)
    assert len(result.hits) <= 2


def test_search_many_dedupes_by_url(claude_code_payload: dict[str, Any]) -> None:
    # Construct a payload with a deliberate duplicate URL
    payload = dict(claude_code_payload)
    if payload["web_results"]:
        payload["web_results"] = [*payload["web_results"], dict(payload["web_results"][0])]
    client = FakeClient(payload)
    result = search_many(client, ["q"])
    urls = [h.url for h in result.hits]
    assert len(urls) == len(set(urls))


def test_search_many_filters_widget_hits(claude_code_payload: dict[str, Any]) -> None:
    # Inject one widget-flagged hit; verify it gets filtered out
    payload = dict(claude_code_payload)
    payload["web_results"] = [
        *payload["web_results"],
        {"url": "https://widget.example/", "name": "Widget", "is_widget": True},
    ]
    n_before = len(payload["web_results"])
    client = FakeClient(payload)
    result = search_many(client, ["q"])
    # widget removed; everything else passes through (modulo dedupe)
    assert all(h.url != "https://widget.example/" for h in result.hits)
    assert len(result.hits) <= n_before - 1


def test_search_many_empty_query_list() -> None:
    client = FakeClient({"web_results": []})
    result = search_many(client, [])
    assert result.hits == []
    assert result.total == 0
    # Should NOT have called the endpoint at all
    assert client.calls == []


def test_search_many_passes_queries_through(claude_code_payload: dict[str, Any]) -> None:
    client = FakeClient(claude_code_payload)
    search_many(client, ["q1", "q2", "q3"])
    assert client.calls[0][1]["queries"] == ["q1", "q2", "q3"]


def test_search_many_documented_response_shape(claude_code_payload: dict[str, Any]) -> None:
    """Anti-drift: if Perplexity removes web_results or renames top-level keys,
    this fails immediately rather than silently returning zero hits.
    """
    assert "web_results" in claude_code_payload
    assert isinstance(claude_code_payload["web_results"], list)
    # Every hit must have the fields our verb relies on
    for raw in claude_code_payload["web_results"]:
        assert "url" in raw, "fixture broke: web_result missing 'url'"
        assert "name" in raw, "fixture broke: web_result missing 'name'"


def test_search_many_handles_missing_web_results_key() -> None:
    # Defensive: response with no web_results key at all
    client = FakeClient({"media_items": []})
    result = search_many(client, ["q"])
    assert result.hits == []
    assert result.total == 0


# ---------- decoder-direct fixture replays ----------
# These hit the pure `decode_search_response` rather than going through
# `search_many` + FakeClient. Faster and exercises exactly the parse logic.


def _load_search_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / "search-web" / name).read_text())


def test_decode_handles_empty_web_results() -> None:
    """Empty server response → empty hits + total=0, no SchemaError."""
    raw = _load_search_fixture("empty-web-results.json")
    result = decode_search_response(raw, query="q", limit=10)
    assert result.hits == []
    assert result.total == 0


def test_decode_filters_widget_image_nav_kc_hits() -> None:
    """Fixture has 4 hits, ALL flagged with one of the filter is_* booleans.
    Decoder must drop them all rather than surface widgets / images / nav
    suggestions as "web results."
    """
    raw = _load_search_fixture("all-filtered-hits.json")
    result = decode_search_response(raw, query="q", limit=10)
    assert result.hits == []
    assert result.total == 0


def test_decode_accepts_minimal_hit() -> None:
    """Only url + name is mandatory on a web_result; every other field is
    optional. Decoder must not fail when domain/snippet/summary/timestamp
    are absent — common for federated / private search sources.
    """
    raw = _load_search_fixture("minimal-hit.json")
    result = decode_search_response(raw, query="q", limit=10)
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.url == "https://example.com/page-1"
    assert hit.title == "Example Page"
    assert hit.domain is None
    assert hit.snippet is None
    assert hit.summary is None
    assert hit.images == []
