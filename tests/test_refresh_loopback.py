"""`pplx auth refresh` over a real curl_cffi cookie jar on loopback.

curl_cffi unquotes cookie values when it rebuilds its jar after a request, so a
value the loader accepts (a quoted string holding an octal escape) comes back
decoded into one it refuses. A refresh must not persist that decoded value.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pplx_agent_tools import cli_auth
from pplx_agent_tools.auth import default_cookies_path, load_cookies
from pplx_agent_tools.wire import Client


class _SessionHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"user": {"email": "u@example.com"}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def session_server() -> Iterator[str]:
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _SessionHandler)
    except OSError as e:
        pytest.skip(f"loopback unavailable: {e}")
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_refresh_keeps_quoted_octal_cookie_loadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, session_server: str
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("PPLX_COOKIES_PATH", raising=False)
    monkeypatch.delenv("PPLX_COOKIES", raising=False)
    original = {"session": "tok", "preference": '"x\\073y"'}
    path = default_cookies_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(original))
    path.chmod(0o600)

    def from_default(cls: type[Client], profile: str | None = None) -> Client:
        return cls(load_cookies(profile), base_url=session_server)

    monkeypatch.setattr(cli_auth.Client, "from_default_cookies", classmethod(from_default))
    assert cli_auth.main(["refresh"]) == 0
    assert load_cookies() == original
