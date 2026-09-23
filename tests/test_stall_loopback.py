"""The stall guard over a real curl_cffi stream on loopback.

The scripted-transport tests fake curl's low-speed abort; these lock the real
thing, so a curl_cffi release that stops mapping a streaming (connect, read)
timeout onto LOW_SPEED_TIME fails here instead of hanging a silent stream.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pplx_agent_tools.errors import StreamStallError
from pplx_agent_tools.wire import Client

_STALL = 2.0
# curl checks the low-speed window about once a second and aborts several
# seconds past it; this bounds "promptly" without flaking on that granularity.
_CURL_SLACK = 12.0


class _Handler(BaseHTTPRequestHandler):
    heartbeat_every: float | None = None

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"text": "a"}\n\n')
        self.wfile.flush()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.heartbeat_every is None:
                time.sleep(0.5)
                continue
            time.sleep(self.heartbeat_every)
            try:
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
            except OSError:
                return

    def log_message(self, format: str, *args: object) -> None:
        pass


def _serve(heartbeat_every: float | None) -> Iterator[str]:
    handler = type("H", (_Handler,), {"heartbeat_every": heartbeat_every})
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except OSError as e:
        pytest.skip(f"loopback unavailable: {e}")
        return
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def silent_server() -> Iterator[str]:
    yield from _serve(None)


@pytest.fixture
def heartbeat_server() -> Iterator[str]:
    yield from _serve(0.3)


def _drain(base_url: str) -> float:
    client = Client({"x": "y"}, base_url=base_url)
    start = time.monotonic()
    with pytest.raises(StreamStallError):
        for _ in client.sse_post("/x", {}, max_total_seconds=120, stall_seconds=_STALL):
            pass
    return time.monotonic() - start


def test_total_silence_is_cut_by_curls_low_speed_abort(silent_server: str) -> None:
    assert _drain(silent_server) < _STALL + _CURL_SLACK


def test_heartbeats_alone_are_cut_by_the_stall_check(heartbeat_server: str) -> None:
    elapsed = _drain(heartbeat_server)
    assert _STALL <= elapsed < _STALL + 2.0
