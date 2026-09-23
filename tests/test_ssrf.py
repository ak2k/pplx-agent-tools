"""SSRF guard tests for verbs.fetch.

Private/loopback/metadata IPs are rejected, and a redirect into them is
re-validated and blocked. All cases use IP literals so socket.getaddrinfo
stays offline (no DNS).
"""

from __future__ import annotations

import pytest

from pplx_agent_tools.errors import BlockedUrlError
from pplx_agent_tools.netguard import check_url
from pplx_agent_tools.verbs.fetch import fetch_page


class _FakeResp:
    def __init__(self, status_code: int, *, location: str | None = None, text: str = "") -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {"location": location} if location else {}
        self.text = text


class _FakeSession:
    """Minimal curl_cffi-Session stand-in: returns queued responses, records URLs."""

    def __init__(self, responses: list[_FakeResp]) -> None:
        self._responses = responses
        self.requested: list[str] = []

    def get(
        self, url: str, timeout: float | None = None, allow_redirects: bool = False
    ) -> _FakeResp:
        self.requested.append(url)
        return self._responses.pop(0)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://[::1]/",
    ],
)
def test_check_url_rejects_internal(url: str) -> None:
    with pytest.raises(BlockedUrlError):
        check_url(url)


def test_check_url_allows_public_ip() -> None:
    assert check_url("http://8.8.8.8/").host == "8.8.8.8"


def test_fetch_page_blocks_private_url_without_requesting() -> None:
    sess = _FakeSession([_FakeResp(200, text="<html>secret</html>")])
    with pytest.raises(BlockedUrlError):
        fetch_page("http://127.0.0.1/admin", "127.0.0.1", max_chars=None, session=sess)  # type: ignore[arg-type]
    assert sess.requested == []  # never hit the wire


def test_redirect_into_internal_is_blocked() -> None:
    # Public first hop 302s to the metadata IP; the guard must block hop 2.
    sess = _FakeSession([_FakeResp(302, location="http://169.254.169.254/")])
    with pytest.raises(BlockedUrlError):
        fetch_page("http://8.8.8.8/", "8.8.8.8", max_chars=None, session=sess)  # type: ignore[arg-type]
    assert sess.requested == ["http://8.8.8.8/"]
