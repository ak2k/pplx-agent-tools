"""SSRF guard against the real resolver and real curl, on loopback.

The stubbed tables in test_fetch_refusals cannot see a differential between how
getaddrinfo and curl each read a host string (Darwin's getaddrinfo reads
`0177.0.0.1` as decimal 177.0.0.1; curl reads it as octal 127.0.0.1). Here the
listener counts every request it receives, so a refused URL must leave it at 0.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pplx_agent_tools import netguard
from pplx_agent_tools.errors import BlockedUrlError, PplxError, exit_code
from pplx_agent_tools.netguard import check_url
from pplx_agent_tools.verbs.fetch import fetch_page

# Each reads as a loopback address to curl (octal, hex, short or integer IPv4)
# or is a malformed numeric host.
LOOPBACK_FORMS = [
    "0177.0.0.1",
    "00177.0.0.1",
    "0177.0.0.01",
    "0x7f.0.0.1",
    "0x7f.0x0.0x0.0x1",
    "0x7f000001",
    "2130706433",
    "017700000001",
    "127.1",
    "0177.1",
    "127.000.000.001",
    "127.0.0.1.",
]

# Darwin's getaddrinfo reads these as public decimal addresses; curl reads them
# as octal: 10.0.0.1, 100.64.0.1 and 127.0.0.1.
DIFFERENTIAL_FORMS = ["012.0.0.1", "0144.0100.0.1", "0177.0.0.1"]


class _Counter(BaseHTTPRequestHandler):
    hits: list[str]
    location: str | None

    def do_GET(self) -> None:
        self.hits.append(self.path)
        if self.location is not None:
            self.send_response(302)
            self.send_header("Location", self.location)
        else:
            self.send_response(200)
        body = b"<html><body><p>INTERNAL</p></body></html>"
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class _Server:
    def __init__(self, family: socket.AddressFamily, host: str, location: str | None) -> None:
        handler = type("_H", (_Counter,), {"hits": [], "location": location})
        self.hits: list[str] = handler.hits
        server_cls = type("_S", (ThreadingHTTPServer,), {"address_family": family})
        self._httpd = server_cls((host, 0), handler)
        self.port: int = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def loopback() -> Iterator[_Server]:
    server = _Server(socket.AF_INET, "127.0.0.1", None)
    yield server
    server.close()


def test_listener_is_reachable(loopback: _Server) -> None:
    # Guards the tests below: a hit count of 0 must mean "refused", not "unreachable".
    from curl_cffi import requests as cf_requests

    resp = cf_requests.get(f"http://127.0.0.1:{loopback.port}/probe", timeout=5)
    assert resp.status_code == 200
    assert loopback.hits == ["/probe"]


@pytest.mark.parametrize("host", LOOPBACK_FORMS)
def test_numeric_host_forms_are_refused_before_connecting(loopback: _Server, host: str) -> None:
    url = f"http://{host}:{loopback.port}/"
    with pytest.raises(PplxError) as ei:
        fetch_page(url, host, max_chars=None)
    assert isinstance(ei.value, BlockedUrlError), repr(ei.value)
    assert exit_code(ei.value) == 1
    assert loopback.hits == []


@pytest.mark.parametrize("host", DIFFERENTIAL_FORMS)
def test_check_url_refuses_differential_forms(host: str) -> None:
    with pytest.raises(BlockedUrlError):
        check_url(f"http://{host}/")


@pytest.fixture
def ipv6_origin_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow [::1] only, so a real first hop can redirect to an IPv4 loopback form."""
    real = netguard._block_reason  # pyright: ignore[reportPrivateUsage]
    loop6 = ipaddress.IPv6Address("::1")

    def _allow_loop6(ip: netguard.IPAddress) -> str | None:
        return None if ip == loop6 else real(ip)

    monkeypatch.setattr(netguard, "_block_reason", _allow_loop6)


@pytest.mark.usefixtures("ipv6_origin_allowed")
@pytest.mark.parametrize("host", LOOPBACK_FORMS)
def test_numeric_redirect_targets_are_refused_before_connecting(
    loopback: _Server, host: str
) -> None:
    origin = _Server(socket.AF_INET6, "::1", f"http://{host}:{loopback.port}/hit")
    try:
        with pytest.raises(PplxError) as ei:
            fetch_page(f"http://[::1]:{origin.port}/start", "[::1]", max_chars=None)
        assert isinstance(ei.value, BlockedUrlError), repr(ei.value)
        assert origin.hits == ["/start"]
        assert loopback.hits == []
    finally:
        origin.close()
