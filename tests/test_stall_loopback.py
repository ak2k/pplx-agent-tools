"""The stall guard over a real curl_cffi stream on loopback.

The scripted-transport tests fake curl's low-speed abort; these lock the real
thing, so a curl_cffi release that stops mapping a streaming (connect, read)
timeout onto LOW_SPEED_TIME fails here instead of hanging a silent stream.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

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


def _drain(
    base_url: str,
    *,
    stall_seconds: float | None = None,
    silence_seconds: float | None = None,
    first_content_seconds: float | None = None,
    is_progress: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[float, StreamStallError]:
    client = Client({"x": "y"}, base_url=base_url)
    start = time.monotonic()
    with pytest.raises(StreamStallError) as exc:
        for _ in client.sse_post(
            "/x",
            {},
            max_total_seconds=120,
            stall_seconds=stall_seconds,
            silence_seconds=silence_seconds,
            first_content_seconds=first_content_seconds,
            is_progress=is_progress,
        ):
            pass
    return time.monotonic() - start, exc.value


def test_total_silence_is_cut_by_curls_low_speed_abort(silent_server: str) -> None:
    elapsed, _ = _drain(silent_server, stall_seconds=_STALL, silence_seconds=_STALL)
    assert elapsed < _STALL + _CURL_SLACK


def test_heartbeats_alone_are_cut_by_the_stall_check(heartbeat_server: str) -> None:
    elapsed, _ = _drain(heartbeat_server, stall_seconds=_STALL)
    assert _STALL <= elapsed < _STALL + 2.0


def test_silence_window_shorter_than_connect_leg_still_aborts(silent_server: str) -> None:
    """A silence window under the 30 s connect timeout leaves a read leg of 0;
    curl must still abort at the window, long before the stall window."""
    stall = 60.0
    elapsed, err = _drain(silent_server, stall_seconds=stall, silence_seconds=_STALL)
    assert type(err).__name__ == "StreamSilenceError"
    assert f"no bytes for {_STALL:.1f}s" in str(err)
    assert _STALL <= elapsed < stall


def test_heartbeats_without_content_are_cut_at_the_first_content_bound(
    heartbeat_server: str,
) -> None:
    first_content = 1.5
    silence = 30.0
    elapsed, err = _drain(
        heartbeat_server,
        stall_seconds=60.0,
        silence_seconds=silence,
        first_content_seconds=first_content,
        is_progress=lambda _event: False,
    )
    assert type(err).__name__ == "StreamFirstContentError"
    assert f"no content within {first_content:.1f}s" in str(err)
    assert first_content <= elapsed < silence
