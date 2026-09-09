"""Unit tests for verbs/quota.py — decode of /rest/rate-limit/status, plus the
session pre-flight that keeps the anonymous zeroed payload from rendering as an
exhausted account."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pplx_agent_tools.errors import AuthError, SchemaError
from pplx_agent_tools.verbs.quota import ENDPOINT, decode_quota, quota
from tests._doubles import _TestClientBase

FIXTURES = Path(__file__).parent / "fixtures" / "rate-limit-status"

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


class FakeClient(_TestClientBase):
    """Canned `auth_session` + `get_json`; records the call order."""

    def __init__(self, payload: Any = None, *, authenticated: bool = True) -> None:
        super().__init__()
        self._payload = payload
        self._authenticated = authenticated
        self.calls: list[str] = []

    def auth_session(self) -> dict[str, Any]:
        self.calls.append("/api/auth/session")
        if not self._authenticated:
            raise AuthError("session expired or unauthenticated; re-import cookies")
        return {"user": {"email": "user@example.com"}}

    def get_json(self, path: str) -> Any:
        self.calls.append(path)
        return self._payload


@pytest.fixture
def anonymous_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "anonymous.json").read_text())


def test_quota_expired_session_raises_before_fetch() -> None:
    client = FakeClient(authenticated=False)
    with pytest.raises(AuthError, match="session expired or unauthenticated"):
        quota(client)
    assert client.calls == ["/api/auth/session"]


def test_quota_authenticated_fetches_and_decodes() -> None:
    client = FakeClient(RAW)
    r = quota(client)
    assert client.calls == ["/api/auth/session", ENDPOINT]
    assert [m.name for m in r.modes] == ["agentic_research", "research"]


def test_anonymous_fixture_decodes_as_fully_exhausted(anonymous_payload: dict[str, Any]) -> None:
    """The anonymous payload carries no marker: it decodes cleanly as an
    account with every mode and source at exact 0. That is why `quota()`
    must pre-flight the session rather than inspect the payload."""
    r = decode_quota(anonymous_payload)
    assert r.free_queries is not None and r.free_queries.available is True
    assert [m.name for m in r.modes] == ["agentic_research", "labs", "pro_search", "research"]
    assert all(not m.available and m.remaining == 0 for m in r.modes)
    assert r.sources
    assert all(not s.available and s.remaining == 0 for s in r.sources)


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
