"""Unit tests for verbs/search.py — body shape + hit parsing + filter flags."""

from __future__ import annotations

import pytest

from pplx_agent_tools.errors import SchemaError
from pplx_agent_tools.verbs.search import _build_body, _keep, _to_hit


def test_keep_accepts_plain_hit() -> None:
    assert _keep({"url": "u", "name": "n"})


def test_keep_drops_navigational() -> None:
    assert not _keep({"url": "u", "name": "n", "is_navigational": True})


def test_keep_drops_widget() -> None:
    assert not _keep({"url": "u", "name": "n", "is_widget": True})


@pytest.mark.parametrize(
    "flag",
    [
        "is_navigational",
        "is_widget",
        "is_knowledge_card",
        "is_image",
        "is_video",
        "is_audio",
        "is_map",
        "is_memory",
        "is_conversation_history",
        "is_conversation_summary",
        "is_attachment",
        "is_extra_info",
        "is_pro_search_table",
    ],
)
def test_keep_drops_each_filtered_flag(flag: str) -> None:
    assert not _keep({"url": "u", "name": "n", flag: True})


def test_keep_rejects_non_dict() -> None:
    assert not _keep("not a dict")  # type: ignore[arg-type]


def test_to_hit_minimal() -> None:
    hit = _to_hit({"url": "https://a/", "name": "Title"})
    assert hit.url == "https://a/"
    assert hit.title == "Title"
    assert hit.domain is None
    assert hit.snippet is None
    assert hit.summary is None
    assert hit.published_date is None
    assert hit.images == []


def test_to_hit_full() -> None:
    raw = {
        "url": "https://a/foo",
        "name": "Foo Bar",
        "domain": "a.example.com",
        "snippet": "short.",
        "summary": "longer.",
        "timestamp": "2026-05-12T00:00:00",
        "meta_data": {"images": ["https://i/1", "https://i/2"]},
    }
    hit = _to_hit(raw)
    assert hit.url == "https://a/foo"
    assert hit.title == "Foo Bar"
    assert hit.domain == "a.example.com"
    assert hit.snippet == "short."
    assert hit.summary == "longer."
    assert hit.published_date == "2026-05-12T00:00:00"
    assert hit.images == ["https://i/1", "https://i/2"]


def test_to_hit_empty_strings_become_none() -> None:
    # Server commonly returns "" for missing fields — we normalize to None
    hit = _to_hit({"url": "u", "name": "n", "domain": "", "snippet": "", "summary": ""})
    assert hit.domain is None
    assert hit.snippet is None
    assert hit.summary is None


def test_to_hit_raises_schema_error_when_url_missing() -> None:
    with pytest.raises(SchemaError) as ei:
        _to_hit({"name": "n"})
    assert "url/name" in str(ei.value)


def test_to_hit_raises_schema_error_when_name_missing() -> None:
    with pytest.raises(SchemaError):
        _to_hit({"url": "u"})


def test_to_hit_meta_data_none_is_safe() -> None:
    hit = _to_hit({"url": "u", "name": "n", "meta_data": None})
    assert hit.images == []
    assert hit.domain is None


def test_build_body_required_fields() -> None:
    body = _build_body(["a query"], domains=None, excluded_domains=None, country="US")
    assert isinstance(body["session_id"], str)
    assert body["queries"] == ["a query"]
    # session_id is uuid-shaped
    assert len(body["session_id"]) == 36
    assert body["session_id"].count("-") == 4


def test_build_body_multi_query_native() -> None:
    body = _build_body(["q1", "q2", "q3"], domains=None, excluded_domains=None, country="US")
    assert body["queries"] == ["q1", "q2", "q3"]


def test_build_body_omits_default_country() -> None:
    body = _build_body(["q"], domains=None, excluded_domains=None, country="US")
    assert "country" not in body


def test_build_body_includes_non_default_country() -> None:
    body = _build_body(["q"], domains=None, excluded_domains=None, country="de")
    assert body["country"] == "DE"


def test_build_body_includes_domain_filter() -> None:
    body = _build_body(["q"], domains=["a.com", "b.org"], excluded_domains=None, country="US")
    assert body["domain_filter"] == ["a.com", "b.org"]


def test_build_body_includes_excluded_domains() -> None:
    body = _build_body(["q"], domains=None, excluded_domains=["spam.io"], country="US")
    assert body["excluded_domains"] == ["spam.io"]


def test_build_body_fresh_uuid_each_call() -> None:
    a = _build_body(["q"], domains=None, excluded_domains=None, country="US")
    b = _build_body(["q"], domains=None, excluded_domains=None, country="US")
    assert a["session_id"] != b["session_id"]
