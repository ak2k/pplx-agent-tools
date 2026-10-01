"""`Client.sse_reconnect`: the request it sends, the events it yields, and how
it reads a refusal.

A reconnect refused with 403 `application/json` `{}` means the thread is gone
(deleted, expired, or another account's), unless the session itself is dead,
which only `/api/auth/session` can tell apart. A 403 `text/html` is the
Cloudflare edge block, with or without the markers the challenge pages carry,
and never reads as gone.
"""

from __future__ import annotations

from typing import Any

import pytest
from curl_cffi.requests import Headers

from pplx_agent_tools.errors import (
    AntiBotError,
    AuthError,
    NetworkError,
    RateLimitError,
    SchemaError,
    SessionCheckError,
    StreamDeadlineError,
    ThreadGoneError,
    exit_code,
)
from pplx_agent_tools.wire import Client

UUID = "11111111-2222-4333-8444-555555555555"
LIVE_JSON = "application/json; charset=utf-8"


class _Resp:
    def __init__(
        self,
        status: int,
        *,
        headers: dict[str, str] | None = None,
        content: bytes = b"",
        chunks: list[bytes] | None = None,
        chunk_delay_s: float = 0.0,
        raise_after: BaseException | None = None,
        json_body: Any = None,
    ) -> None:
        self.status_code = status
        self.headers = Headers(headers)
        self.content = content
        self._chunks = chunks or []
        self._chunk_delay_s = chunk_delay_s
        self._raise_after = raise_after
        self._json = json_body
        self.closed = False

    def iter_content(self, chunk_size: int) -> Any:
        import time as _t

        def gen() -> Any:
            for c in self._chunks:
                if self._chunk_delay_s:
                    _t.sleep(self._chunk_delay_s)
                yield c
            if self._raise_after is not None:
                raise self._raise_after

        return gen()

    def json(self) -> Any:
        return self._json

    def close(self) -> None:
        self.closed = True


class _Session:
    """Answers the reconnect POST with `resp` and `/api/auth/session` with
    `session_body`, recording every request."""

    def __init__(self, resp: _Resp, *, session_body: Any = None) -> None:
        self._resp = resp
        self._session_body = session_body
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []
        self.cookies: dict[str, str] = {}

    def post(self, url: str, **kw: Any) -> _Resp:
        self.posts.append((url, kw))
        return self._resp

    def get(self, url: str, **_kw: Any) -> _Resp:
        self.gets.append(url)
        return _Resp(200, headers={"content-type": LIVE_JSON}, json_body=self._session_body)


def _client(resp: _Resp, *, session_body: Any = None) -> tuple[Client, _Session]:
    client = Client({"any": "cookie"})
    session = _Session(resp, session_body=session_body)
    client._session = session  # type: ignore[assignment]
    return client, session


LIVE_SESSION = {"user": {"id": "u"}}
DEAD_SESSION: dict[str, Any] = {}


def test_posts_the_snapshot_request_to_the_threads_reconnect_path() -> None:
    frame = b'data: {"status": "COMPLETED"}\n\n'
    client, session = _client(_Resp(200, chunks=[frame, b"event: end_of_stream\ndata: {}\n\n"]))
    events = list(client.sse_reconnect(UUID))
    assert events == [
        {"event": None, "data": {"status": "COMPLETED"}},
        {"event": "end_of_stream", "data": {}},
    ]
    [(url, kw)] = session.posts
    assert url == f"https://www.perplexity.ai/rest/sse/perplexity_ask/reconnect/{UUID}"
    assert kw["json"] == {"reconnectInitialSnapshot": True}
    assert kw["headers"] == {"accept": "text/event-stream"}
    assert kw["stream"] is True
    assert session.gets == []


def test_nothing_is_sent_until_the_stream_is_read() -> None:
    client, session = _client(_Resp(200))
    stream = client.sse_reconnect(UUID)
    assert session.posts == []
    assert list(stream) == []
    assert len(session.posts) == 1


def test_json_403_on_a_live_session_is_thread_gone() -> None:
    resp = _Resp(403, headers={"content-type": LIVE_JSON}, content=b"{}")
    client, session = _client(resp, session_body=LIVE_SESSION)
    with pytest.raises(ThreadGoneError) as exc:
        list(client.sse_reconnect(UUID))
    assert exit_code(exc.value) == 1
    assert session.gets == ["https://www.perplexity.ai/api/auth/session"]
    assert UUID not in str(exc.value)
    assert resp.closed


def test_bare_json_media_type_is_read_the_same() -> None:
    resp = _Resp(403, headers={"content-type": "Application/JSON"}, content=b"{}")
    client, _ = _client(resp, session_body=LIVE_SESSION)
    with pytest.raises(ThreadGoneError):
        list(client.sse_reconnect(UUID))


def test_json_403_on_a_dead_session_is_auth() -> None:
    resp = _Resp(403, headers={"content-type": LIVE_JSON}, content=b"{}")
    client, session = _client(resp, session_body=DEAD_SESSION)
    with pytest.raises(AuthError) as exc:
        list(client.sse_reconnect(UUID))
    assert not isinstance(exc.value, ThreadGoneError)
    assert exit_code(exc.value) == 2
    assert len(session.gets) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"<html><title>Just a moment...</title>Cloudflare Ray ID: 1</html>",
        b"<html><body>" + b"x" * 6000 + b"</body></html>",
    ],
    ids=["with-markers", "without-markers"],
)
def test_html_403_is_the_edge_block_never_gone(body: bytes) -> None:
    resp = _Resp(403, headers={"content-type": "text/html; charset=UTF-8"}, content=body)
    client, session = _client(resp, session_body=LIVE_SESSION)
    with pytest.raises(AntiBotError) as exc:
        list(client.sse_reconnect(UUID))
    assert exit_code(exc.value) == 5
    assert session.gets == []
    assert UUID not in str(exc.value)


@pytest.mark.parametrize(
    ("status", "headers", "error"),
    [
        (401, {}, AuthError),
        (403, {}, AuthError),
        (403, {"content-type": "text/plain"}, AuthError),
        (429, {"retry-after": "7"}, RateLimitError),
        (502, {}, NetworkError),
        (404, {}, SchemaError),
    ],
)
def test_other_statuses_map_as_on_every_other_path(
    status: int, headers: dict[str, str], error: type[Exception]
) -> None:
    client, session = _client(_Resp(status, headers=headers), session_body=LIVE_SESSION)
    with pytest.raises(error) as exc:
        list(client.sse_reconnect(UUID))
    assert not isinstance(exc.value, ThreadGoneError)
    assert session.gets == []
    assert UUID not in str(exc.value)


def test_a_401_json_is_auth_not_gone() -> None:
    resp = _Resp(401, headers={"content-type": LIVE_JSON}, content=b"{}")
    client, session = _client(resp, session_body=LIVE_SESSION)
    with pytest.raises(AuthError) as exc:
        list(client.sse_reconnect(UUID))
    assert not isinstance(exc.value, ThreadGoneError)
    assert session.gets == []


def test_the_deadline_bound_applies() -> None:
    framed = b'data: {"a": 1}\n\n'
    resp = _Resp(200, chunks=[framed, framed, framed], chunk_delay_s=0.05)
    client, _ = _client(resp)
    with pytest.raises(StreamDeadlineError):
        list(client.sse_reconnect(UUID, max_total_seconds=0.06))
    assert resp.closed


def test_a_midstream_failure_is_a_network_error_without_the_uuid() -> None:
    framed = b'data: {"a": 1}\n\n'
    resp = _Resp(200, chunks=[framed], raise_after=OSError("connection reset by peer"))
    client, _ = _client(resp)
    stream = client.sse_reconnect(UUID)
    assert next(stream) == {"event": None, "data": {"a": 1}}
    with pytest.raises(NetworkError, match="mid-stream") as exc:
        next(stream)
    assert not isinstance(exc.value, StreamDeadlineError)
    assert UUID not in str(exc.value)


def test_a_failed_post_is_a_network_error_without_the_uuid() -> None:
    client = Client({"any": "cookie"})

    class _Down:
        def post(self, url: str, **_kw: Any) -> Any:
            raise ConnectionError("connection refused")

    client._session = _Down()  # type: ignore[assignment]
    with pytest.raises(NetworkError) as exc:
        list(client.sse_reconnect(UUID))
    assert UUID not in str(exc.value)


class _ProbeFails(_Session):
    """Refuses the reconnect with a JSON 403, then fails the session probe."""

    def __init__(self, failure: str) -> None:
        super().__init__(_Resp(403, headers={"content-type": LIVE_JSON}, content=b"{}"))
        self._failure = failure

    def get(self, url: str, **_kw: Any) -> _Resp:
        self.gets.append(url)
        if self._failure == "network":
            raise ConnectionError("connection reset by peer")

        class _NotJson(_Resp):
            def json(self) -> Any:
                raise ValueError("Expecting value")

        return _NotJson(200, headers={"content-type": "text/html"})


@pytest.mark.parametrize("failure", ["network", "not json"])
def test_a_failed_session_check_on_a_json_403_is_its_own_error(failure: str) -> None:
    client = Client({"any": "cookie"})
    client._session = _ProbeFails(failure)  # type: ignore[assignment]
    with pytest.raises(SessionCheckError) as exc:
        list(client.sse_reconnect(UUID))
    assert not isinstance(exc.value, (NetworkError, ThreadGoneError, AuthError))
    assert exit_code(exc.value) == 4
    message = str(exc.value)
    assert "403" in message and "session" in message
    assert UUID not in message


def test_a_json_403_with_a_reason_is_gone_and_carries_the_reason() -> None:
    body = b'{"detail":"Pro subscription required for thread ' + UUID.encode() + b'"}'
    resp = _Resp(403, headers={"content-type": LIVE_JSON}, content=body)
    client, _ = _client(resp, session_body=LIVE_SESSION)
    with pytest.raises(ThreadGoneError) as exc:
        list(client.sse_reconnect(UUID))
    assert "Pro subscription required" in str(exc.value)
    assert UUID not in str(exc.value)
    assert resp.closed


def test_an_empty_json_403_adds_no_reason() -> None:
    resp = _Resp(403, headers={"content-type": LIVE_JSON}, content=b"{}")
    client, _ = _client(resp, session_body=LIVE_SESSION)
    with pytest.raises(ThreadGoneError) as exc:
        list(client.sse_reconnect(UUID))
    assert "{}" not in str(exc.value) and "said" not in str(exc.value)
