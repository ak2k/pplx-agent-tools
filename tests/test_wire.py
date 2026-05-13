"""Unit tests for wire.py — SSE parser + status-code branching.

We don't mock curl_cffi.Session; instead we exercise the helpers and
status-branching method directly with a tiny FakeResponse stand-in.
"""

from __future__ import annotations

import pytest

from pplx_agent_tools.errors import (
    AntiBotError,
    AuthError,
    NetworkError,
    RateLimitError,
    SchemaError,
)
from pplx_agent_tools.wire import Client, _parse_sse_event

# ---------- _parse_sse_event ----------


def test_parse_sse_basic_event() -> None:
    raw = 'event: message\ndata: {"a": 1}'
    out = _parse_sse_event(raw)
    assert out == {"event": "message", "data": {"a": 1}}


def test_parse_sse_no_event_type() -> None:
    raw = 'data: {"hello": "world"}'
    out = _parse_sse_event(raw)
    assert out == {"event": None, "data": {"hello": "world"}}


def test_parse_sse_multiline_data_concatenated() -> None:
    raw = "event: x\ndata: line one\ndata: line two"
    out = _parse_sse_event(raw)
    assert out is not None
    assert out["data"] == "line one\nline two"


def test_parse_sse_non_json_data_returned_as_string() -> None:
    raw = "data: not a json string"
    out = _parse_sse_event(raw)
    assert out is not None
    assert out["data"] == "not a json string"


def test_parse_sse_comment_line_ignored() -> None:
    raw = ': this is an SSE comment\ndata: {"x": 1}'
    out = _parse_sse_event(raw)
    assert out is not None
    assert out["data"] == {"x": 1}


def test_parse_sse_empty_block_returns_none() -> None:
    assert _parse_sse_event("") is None
    assert _parse_sse_event("   \n  ") is None


def test_parse_sse_event_only_no_data() -> None:
    raw = "event: end_of_stream"
    out = _parse_sse_event(raw)
    assert out == {"event": "end_of_stream", "data": None}


# ---------- _check_status (status-code branching) ----------


class FakeResp:
    """Minimum response shape `_check_status` reads."""

    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content


@pytest.fixture
def client() -> Client:
    # Constructor only sets up state; no network calls. Cookies are unused
    # in the status-branching tests so an empty dict is fine.
    return Client({"any": "cookie"})


def test_check_status_2xx_passes(client: Client) -> None:
    client._check_status(FakeResp(200), "/whatever")
    client._check_status(FakeResp(204), "/whatever")


def test_check_status_401_is_auth(client: Client) -> None:
    with pytest.raises(AuthError):
        client._check_status(FakeResp(401), "/api/auth/session")


def test_check_status_403_is_auth(client: Client) -> None:
    with pytest.raises(AuthError):
        client._check_status(FakeResp(403), "/api/auth/session")


def test_check_status_403_with_cloudflare_html_is_anti_bot(client: Client) -> None:
    with pytest.raises(AntiBotError):
        client._check_status(
            FakeResp(
                403,
                headers={"content-type": "text/html"},
                content=b"<html>...Cloudflare Ray ID...</html>",
            ),
            "/anything",
        )


def test_check_status_200_with_cloudflare_html_is_anti_bot(client: Client) -> None:
    # CF interstitial returns 200 with HTML body
    with pytest.raises(AntiBotError):
        client._check_status(
            FakeResp(
                200,
                headers={"content-type": "text/html"},
                content=b"<title>Just a moment...</title>",
            ),
            "/anything",
        )


def test_check_status_429_rate_limit_with_retry_after(client: Client) -> None:
    with pytest.raises(RateLimitError) as ei:
        client._check_status(FakeResp(429, headers={"retry-after": "30"}), "/x")
    assert ei.value.retry_after == 30.0


def test_check_status_429_without_retry_after(client: Client) -> None:
    with pytest.raises(RateLimitError) as ei:
        client._check_status(FakeResp(429), "/x")
    assert ei.value.retry_after is None


def test_check_status_5xx_is_network(client: Client) -> None:
    with pytest.raises(NetworkError):
        client._check_status(FakeResp(500), "/x")
    with pytest.raises(NetworkError):
        client._check_status(FakeResp(502), "/x")


def test_check_status_unexpected_4xx_is_schema(client: Client) -> None:
    # 422 is a real-world case from Perplexity's FastAPI validation errors;
    # our agent-facing contract doesn't have a dedicated category for these.
    with pytest.raises(SchemaError):
        client._check_status(FakeResp(422), "/x")


def test_check_status_3xx_is_schema(client: Client) -> None:
    # Redirects shouldn't reach us (curl_cffi follows by default); if they
    # do, treat as unexpected.
    with pytest.raises(SchemaError):
        client._check_status(FakeResp(301), "/x")


# ---------- capture_rotated_cookies ----------


# For rotation tests we populate the real curl_cffi Cookies jar via its
# .set() API rather than monkeypatching .cookies entirely — curl_cffi's
# Cookies class has a custom __setattr__ that prevents wholesale replacement.


def test_capture_rotation_updates_only_existing_names() -> None:
    c = Client({"session-token": "OLD", "csrf": "C1"})
    c._session.cookies.set("session-token", "NEW")
    c._session.cookies.set("irrelevant", "X")
    changed = c.capture_rotated_cookies()
    assert changed is True
    assert c.cookies["session-token"] == "NEW"
    assert c.cookies["csrf"] == "C1"  # csrf wasn't in the jar; unchanged
    assert "irrelevant" not in c.cookies  # never pick up new third-party names


def test_capture_rotation_returns_false_when_unchanged() -> None:
    c = Client({"session-token": "SAME"})
    c._session.cookies.set("session-token", "SAME")
    assert c.capture_rotated_cookies() is False
    assert c.cookies["session-token"] == "SAME"


def test_capture_rotation_handles_empty_jar() -> None:
    # Jar has none of our cookies — nothing changes
    c = Client({"a": "1", "b": "2"})
    assert c.capture_rotated_cookies() is False
    assert c.cookies == {"a": "1", "b": "2"}


def test_cookies_property_returns_copy() -> None:
    c = Client({"a": "1"})
    snapshot = c.cookies
    snapshot["a"] = "MUTATED"
    # Caller's mutation must not affect Client state
    assert c.cookies["a"] == "1"
