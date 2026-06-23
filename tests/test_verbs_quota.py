"""Unit tests for verbs/quota.py — decode of /rest/rate-limit/status."""

from __future__ import annotations

import pytest

from pplx_agent_tools.errors import SchemaError
from pplx_agent_tools.verbs.quota import decode_quota

# Sanitized real shape (2026-06-22): not_provided for Pro modes, exact 0 for an
# un-set-up connector source.
RAW = {
    "free_queries": {"available": True, "remaining_detail": {"kind": "not_provided"}},
    "modes": {
        "research": {"available": True, "remaining_detail": {"kind": "not_provided"}},
        "agentic_research": {"available": True, "remaining_detail": {"kind": "not_provided"}},
    },
    "sources": {
        "bmj": {"available": True, "remaining_detail": {"kind": "not_provided"}},
        "box": {"available": False, "remaining_detail": {"kind": "exact", "remaining": 0}},
    },
}


def test_decode_basic() -> None:
    r = decode_quota(RAW)
    assert r.free_queries is not None
    assert r.free_queries.available is True
    assert r.free_queries.remaining is None
    # modes sorted by name
    assert [m.name for m in r.modes] == ["agentic_research", "research"]
    assert all(m.available for m in r.modes)


def test_decode_exact_remaining() -> None:
    r = decode_quota(RAW)
    box = next(s for s in r.sources if s.name == "box")
    assert box.available is False
    assert box.remaining == 0
    bmj = next(s for s in r.sources if s.name == "bmj")
    assert bmj.remaining is None


def test_decode_missing_groups_tolerated() -> None:
    r = decode_quota({})
    assert r.free_queries is None
    assert r.modes == []
    assert r.sources == []


def test_decode_non_dict_raises() -> None:
    with pytest.raises(SchemaError):
        decode_quota(["nope"])


def test_decode_skips_non_dict_entries() -> None:
    r = decode_quota({"modes": {"good": {"available": True}, "bad": "x"}})
    assert [m.name for m in r.modes] == ["good"]


def test_decode_non_int_remaining_ignored() -> None:
    r = decode_quota(
        {
            "modes": {
                "m": {"available": True, "remaining_detail": {"kind": "exact", "remaining": None}}
            }
        }
    )
    assert r.modes[0].remaining is None
