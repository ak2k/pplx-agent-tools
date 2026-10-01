"""Transport layer: curl_cffi chrome-impersonate session.

`curl_cffi` is required (not `requests`) so Cloudflare's TLS fingerprint check
accepts us as a real Chrome client. See balakumardev/perplexity-web-wrapper.
"""

from __future__ import annotations

import codecs
import contextlib
import hashlib
import json
import sys
import time
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

from curl_cffi import CurlECode, CurlError
from curl_cffi import requests as cf_requests

from .auth import cookie_pair_ok
from .errors import (
    AntiBotError,
    AuthError,
    NetworkError,
    PplxError,
    RateLimitError,
    SchemaError,
    SessionCheckError,
    StreamDeadlineError,
    StreamFirstContentError,
    StreamSilenceError,
    StreamStallError,
    ThreadGoneError,
)
from .jsonval import JsonValue, from_parser

BASE_URL = "https://www.perplexity.ai"
DEFAULT_TIMEOUT = 30.0
DEFAULT_IMPERSONATE = "chrome"
# SSE-only read leg when no silence window is given. curl_cffi turns a streaming
# (connect, read) timeout into a low-speed abort (< 1 B/s for connect + read
# seconds), so this only catches total silence: heartbeat comments keep the
# rate above 1 B/s. The progress-event stall check in `sse_post` covers that case.
DEFAULT_SSE_READ_TIMEOUT = 60.0
# Terminate and delete run in cleanup, often right after a Ctrl-C, so they get
# a bound of their own rather than the client's 30 s default. Terminate
# answered in 0.17 s when probed live.
CLEANUP_TIMEOUT_SECONDS = 5.0
# A thread's stream, reattached; the thread's backend uuid follows.
RECONNECT_PATH = "/rest/sse/perplexity_ask/reconnect/"
# Hard cap on un-dispatched SSE buffer (a single event with no `\n\n` terminator).
# Defends against a server that trickles bytes forever without a terminator.
_MAX_SSE_BUFFER_BYTES = 16 * 1024 * 1024


class Client:
    """Authenticated Perplexity session.

    Pass cookies (dict[name, value]) explicitly, or use `from_default_cookies`
    to pull them from the resolution chain in `auth.load_cookies`.
    """

    def __init__(
        self,
        cookies: dict[str, str],
        *,
        base_url: str = BASE_URL,
        impersonate: str = DEFAULT_IMPERSONATE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._cookies = dict(cookies)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # curl_cffi types `impersonate` as a closed Literal union; our default
        # is "chrome" (an alias the lib accepts at runtime). Cast rather than
        # mirror an internal Literal list that drifts on every curl_cffi release.
        self._session = cf_requests.Session(impersonate=impersonate)  # type: ignore[arg-type]

    @classmethod
    def from_default_cookies(
        cls,
        profile: str | None = None,
        *,
        base_url: str = BASE_URL,
        impersonate: str = DEFAULT_IMPERSONATE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Client:
        from .auth import load_cookies

        return cls(
            load_cookies(profile),
            base_url=base_url,
            impersonate=impersonate,
            timeout=timeout,
        )

    def auth_session(self) -> dict[str, Any]:
        """GET /api/auth/session. Returns parsed JSON.

        NextAuth returns `{}` for unauthenticated; a populated dict (with `user`)
        for an authenticated session. We treat empty/missing-user as AuthError.

        Captures rotated cookies into `self._cookies` (NextAuth's rolling-session
        pattern issues a fresh `__Secure-next-auth.session-token` on each call;
        without capture, a 30-day-old cookie that's been rotating silently still
        expires from our perspective on day 30).
        """
        resp = self._get("/api/auth/session")
        try:
            data = resp.json()
        except Exception as e:
            raise SchemaError("non-JSON response from /api/auth/session") from e
        if not isinstance(data, dict):
            raise SchemaError(f"/api/auth/session returned {type(data).__name__}, expected object")
        if not data or "user" not in data:
            raise AuthError("session expired or unauthenticated; re-import cookies")
        self._capture_rotated_cookies()
        return data

    def _capture_rotated_cookies(self) -> bool:
        """Update `self._cookies` with any rotated values from the underlying
        curl_cffi session jar. Returns True iff anything changed.

        Only updates names we already had (so we don't grow our cookie set
        unexpectedly with third-party cookies the server set). Empty-string
        rotations are captured (`is not None`, not truthiness) — a cookie
        rotated to empty is a real state change.
        """
        changed = False
        for name in list(self._cookies):
            try:
                new_val = self._session.cookies.get(name)
            except (KeyError, LookupError) as e:
                # Cookie jar lookup raised — log so silent loss is observable.
                print(f"warning: cookie jar lookup failed for {name!r}: {e}", file=sys.stderr)
                continue
            if new_val is None or new_val == self._cookies[name]:
                continue
            # curl_cffi unquotes values when it rebuilds its jar, so a quoted
            # value we sent can read back as one the loader refuses.
            if not cookie_pair_ok(name, new_val):
                print(
                    f"warning: keeping prior value of cookie {name!r}: "
                    "the jar's value would not load",
                    file=sys.stderr,
                )
                continue
            self._cookies[name] = new_val
            changed = True
        return changed

    @property
    def cookies(self) -> dict[str, str]:
        """Current in-memory cookies (may include rotated values from the
        latest authenticated call). Returns a copy so callers can't mutate
        internal state.
        """
        return dict(self._cookies)

    def get_json(self, path: str) -> JsonValue:
        """GET a path, return the parsed JSON response.

        Same error mapping as `_get` / `post_json`: auth/rate-limit/network/CF.
        Used by the read-only stateless verbs (quota, models).
        """
        return _json_body(self._get(path), path)

    def post_json(self, path: str, body: dict[str, Any]) -> JsonValue:
        """POST a JSON body, return the parsed JSON response.

        Same error mapping as `_get`: auth/rate-limit/network/CF.
        """
        url = self._base_url + path
        try:
            resp = self._session.post(
                url,
                cookies=self._cookies,
                json=body,
                timeout=self._timeout,
            )
        except Exception as e:
            raise NetworkError(f"POST {path} failed: {e!s}") from e
        self._check_status(resp, path)
        return _json_body(resp, path)

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:
        """Delete a thread by entry UUID. Best-effort: any failure is logged
        to stderr and returns False. Never raises, so callers can issue this
        as fire-and-forget cleanup.
        """
        url = self._base_url + "/rest/thread/delete_thread_by_entry_uuid"
        try:
            resp = self._session.request(
                "DELETE",
                url,
                cookies=self._cookies,
                json={"entry_uuid": entry_uuid, "read_write_token": read_write_token},
                timeout=CLEANUP_TIMEOUT_SECONDS,
            )
            if resp is None:
                print(
                    f"warning: thread cleanup failed: no response for thread {thread_ref(entry_uuid)}",
                    file=sys.stderr,
                )
                return False
            status = resp.status_code
            if status < 400:
                return True
            body = _body_excerpt(resp, redact=read_write_token)
        except Exception as e:
            print(f"warning: thread cleanup failed: {e}", file=sys.stderr)
            return False
        print(
            f"warning: thread cleanup failed: DELETE thread {thread_ref(entry_uuid)} "
            f"returned {status}: {body}",
            file=sys.stderr,
        )
        return False

    def terminate(self, entry_uuid: str, context_uuid: str, model_preference: str) -> bool:
        """Stop a run that may still be going on the server, as the web
        client's stop button does. `model_preference` is the `display_model`
        the stream carried. No read_write_token is needed.

        Best-effort like `delete_thread`: any failure is logged to stderr and
        returns False, so cleanup can send it while another exception is in
        flight.
        """
        url = self._base_url + "/rest/sse/perplexity_terminate"
        try:
            resp = self._session.post(
                url,
                cookies=self._cookies,
                json={
                    "entry_uuid": entry_uuid,
                    "context_uuid": context_uuid,
                    "model_preference": model_preference,
                    "terminate_requested_at_ms": int(time.time() * 1000),
                },
                headers={"X-Perplexity-Request-Reason": "thread-floating-footer"},
                timeout=CLEANUP_TIMEOUT_SECONDS,
            )
            status = resp.status_code
            if status < 400:
                return True
            body = _body_excerpt(resp)
        except Exception as e:
            print(f"warning: run terminate failed: {e}", file=sys.stderr)
            return False
        print(
            f"warning: run terminate failed: thread {thread_ref(entry_uuid)} "
            f"returned {status}: {body}",
            file=sys.stderr,
        )
        return False

    def sse_post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        is_progress: Callable[[dict[str, Any]], bool] | None = None,
        stall_window: Callable[[], float | None] | None = None,
        silence_seconds: float | None = None,
        first_content_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """POST a JSON body, stream the SSE response, yield parsed events.

        Each yielded value is `{"event": str | None, "data": parsed_json | str | None}`.
        Consumers may break the loop early — the underlying connection is closed
        when the iterator is no longer referenced (via curl_cffi's response context).

        `max_total_seconds` bounds the wall-clock duration of consuming the stream.
        When exceeded, raises `StreamDeadlineError` so the caller can decide
        whether to salvage partial results. None (default) preserves the legacy
        behavior of relying on `DEFAULT_SSE_READ_TIMEOUT` for per-chunk idle only
        — a stream that keeps trickling bytes more often than every 60 s but
        never reaches COMPLETED can otherwise run indefinitely.

        `stall_seconds` raises `StreamStallError` once no progress event has
        arrived for that long. `is_progress(event)` decides what counts as
        progress; None counts any event carrying data. Comment-only heartbeat
        frames never reset the clock, whatever the predicate.
        `first_content_seconds` raises `StreamFirstContentError` when no
        progress event has arrived that long after the response began.

        `silence_seconds` sizes the transport's low-speed abort (capped by
        `max_total_seconds`): no bytes at all, heartbeats included, for that
        long raises `StreamSilenceError`. The stall and first-content checks run
        only when bytes arrive, so this abort is what ends a stream gone fully
        silent. None keeps the default low-speed backstop. The abort counts
        from the last byte, so another bound may have come due before it; the
        error names whichever came due first.

        `stall_window()`, when given, is read at every stall check and replaces
        `stall_seconds` there, so a consumer can tighten the window mid-stream.
        The tightened window is enforced when the next frame or heartbeat arrives.

        Raises the same typed errors as the GET path (auth/rate-limit/etc.) on
        connection or status-code failure.
        """
        yield from self._sse(
            path,
            body,
            where=path,
            check_status=self._check_status,
            max_total_seconds=max_total_seconds,
            stall_seconds=stall_seconds,
            is_progress=is_progress,
            stall_window=stall_window,
            silence_seconds=silence_seconds,
            first_content_seconds=first_content_seconds,
        )

    def sse_reconnect(
        self,
        backend_uuid: str,
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        is_progress: Callable[[dict[str, Any]], bool] | None = None,
        stall_window: Callable[[], float | None] | None = None,
        silence_seconds: float | None = None,
        first_content_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Reattach to the stream of the thread `backend_uuid`, asking for a
        snapshot of its current state first. Creates no thread.

        Bounds, errors and the yielded event shape are `sse_post`'s, except
        how a 403 reads (`_check_reconnect_status`). Messages name the thread
        by `thread_ref` rather than its id.
        """
        yield from self._sse(
            RECONNECT_PATH + backend_uuid,
            {"reconnectInitialSnapshot": True},
            where=RECONNECT_PATH + thread_ref(backend_uuid),
            check_status=partial(self._check_reconnect_status, backend_uuid=backend_uuid),
            max_total_seconds=max_total_seconds,
            stall_seconds=stall_seconds,
            is_progress=is_progress,
            stall_window=stall_window,
            silence_seconds=silence_seconds,
            first_content_seconds=first_content_seconds,
        )

    def _sse(
        self,
        path: str,
        body: dict[str, Any],
        *,
        where: str,
        check_status: Callable[[Any, str], None],
        max_total_seconds: float | None,
        stall_seconds: float | None,
        is_progress: Callable[[dict[str, Any]], bool] | None,
        stall_window: Callable[[], float | None] | None,
        silence_seconds: float | None,
        first_content_seconds: float | None,
    ) -> Iterator[dict[str, Any]]:
        """The SSE read behind `sse_post` and `sse_reconnect`. `where` names
        the request in every message; `check_status` judges the response
        before its body is read."""
        url = self._base_url + path
        connect_timeout, read_timeout, silence_error = _silence_bounds(
            where, self._timeout, max_total_seconds, silence_seconds
        )
        try:
            resp = self._session.post(
                url,
                cookies=self._cookies,
                json=body,
                headers={"accept": "text/event-stream"},
                stream=True,
                timeout=(connect_timeout, read_timeout),
            )
        except Exception as e:
            raise NetworkError(f"POST {where} failed: {e!s}") from e
        try:
            # Headers / status validated before we start consuming the body.
            check_status(resp, where)
            bounds = _StreamBounds(
                where,
                max_total_seconds=max_total_seconds,
                stall_seconds=stall_seconds,
                stall_window=stall_window,
                first_content_seconds=first_content_seconds,
                silence_after=connect_timeout + read_timeout,
                silence_error=silence_error,
            )
            framer = _SSEFramer()
            try:
                for chunk in resp.iter_content(chunk_size=4096):
                    bounds.check_deadline()
                    if not chunk:
                        continue
                    bounds.saw_byte()
                    raw_events = framer.feed(chunk)
                    # Bound memory against a server that trickles bytes without ever
                    # emitting an event terminator (`\n\n`): the per-chunk idle timeout
                    # wouldn't fire on a continuous trickle, so cap the un-dispatched
                    # buffer. A single SSE event over 16 MiB is pathological.
                    if not raw_events and framer.pending_chars > _MAX_SSE_BUFFER_BYTES:
                        raise SchemaError(
                            f"SSE stream on {where} exceeded {_MAX_SSE_BUFFER_BYTES} bytes "
                            "without an event terminator"
                        )
                    for raw_event in raw_events:
                        parsed = _parse_sse_event(raw_event)
                        if parsed is not None:
                            if parsed["data"] is not None and (
                                is_progress is None or is_progress(parsed)
                            ):
                                bounds.saw_progress()
                            yield parsed
                            # Re-check between yields so a generator consumer
                            # that processes events slowly can't outrun the bounds.
                            bounds.check()
                    # A chunk of heartbeats alone yields no progress event, so the
                    # stall check has to run per chunk as well.
                    bounds.check()
            except PplxError:
                # Deadline and schema faults raised in the loop above already carry
                # their own exit-code contract; only transport faults are reclassified.
                raise
            except Exception as e:
                # curl_cffi queues a mid-stream failure as a base RequestException
                # carrying the curl code, not as its Timeout subclass, so the code
                # is the only reliable signal of the low-speed abort.
                if isinstance(e, CurlError) and e.code == CurlECode.OPERATION_TIMEDOUT:
                    raise bounds.abort_error() from e
                # A read that dies mid-stream is the same class of failure as a POST
                # that never connected, so it gets the same typed error and exit code;
                # whether the events already yielded are worth keeping is the
                # caller's call.
                raise NetworkError(f"SSE stream on {where} failed mid-stream: {e!s}") from e
        finally:
            with contextlib.suppress(Exception):
                resp.close()

    def _get(self, path: str, **kwargs: Any) -> Any:
        url = self._base_url + path
        try:
            resp = self._session.get(url, cookies=self._cookies, timeout=self._timeout, **kwargs)
        except Exception as e:
            raise NetworkError(f"request to {path} failed: {e!s}") from e
        self._check_status(resp, path)
        return resp

    def _check_status(self, resp: Any, path: str) -> None:
        status = resp.status_code
        if 200 <= status < 300:
            self._check_cloudflare_body(resp, path)
            return
        if status in (401, 403):
            if self._looks_like_cloudflare(resp):
                raise AntiBotError(f"Cloudflare block on {path} (status {status})")
            raise AuthError(f"auth rejected on {path} (status {status})")
        if status == 429:
            retry_after = self._parse_retry_after(resp)
            raise RateLimitError(f"rate limited on {path}", retry_after=retry_after)
        if status >= 500:
            raise NetworkError(f"server error {status} on {path}")
        raise SchemaError(f"unexpected status {status} on {path}")

    def _check_reconnect_status(self, resp: Any, where: str, *, backend_uuid: str) -> None:
        """`_check_status`, but a 403 reads by its media type: `text/html` is
        the Cloudflare edge block whether or not the page carries the markers
        `_looks_like_cloudflare` looks for, and `application/json` is a thread
        that is gone, unless the session itself is dead. A body other than
        `{}` ends the gone message, since only `{}` has been seen."""
        if resp.status_code == 403:
            media = _media_type(resp)
            if media == "text/html":
                raise AntiBotError(f"Cloudflare block on {where} (status 403)")
            if media == "application/json":
                said = _body_excerpt(resp, redact=backend_uuid).strip()
                with contextlib.suppress(Exception):
                    resp.close()
                # An expired session may be refused the same way; the session
                # endpoint is what tells the two apart, raising AuthError.
                try:
                    self.auth_session()
                except (NetworkError, SchemaError) as e:
                    raise SessionCheckError(
                        f"reconnect on {where} was refused (status 403 application/json), "
                        "and the session check that tells a gone thread from an expired "
                        f"session failed: {e}"
                    ) from e
                reason = f"; the server said: {said}" if said and said != "{}" else ""
                raise ThreadGoneError(
                    f"thread gone on {where} (status 403): deleted, expired "
                    f"(about 24 h after it started), or not this account's{reason}"
                )
        self._check_status(resp, where)

    @staticmethod
    def _looks_like_cloudflare(resp: Any) -> bool:
        # Authoritative header signals first — these are present on Cloudflare
        # interstitials regardless of body shape. `cf-ray` alone isn't enough
        # (legit Perplexity responses go through CF too), but combined with
        # the HTML-body fallback below it confirms an actual challenge page.
        ct = resp.headers.get("content-type", "")
        if "text/html" not in ct.lower():
            return False
        # 64KB scan window. A 2KB window misses challenge pages where the
        # CF marker is buried after CSS/inline-script preamble; in practice
        # the "cloudflare" / "ray id" / "just a moment" tokens land within
        # the first 64KB on every CF interstitial we've seen.
        body = (resp.content or b"")[:65_536].lower()
        return (
            b"cloudflare" in body
            or b"just a moment" in body
            or b"checking your browser" in body
            or b"cf-ray" in body
        )

    def _check_cloudflare_body(self, resp: Any, path: str) -> None:
        if self._looks_like_cloudflare(resp):
            raise AntiBotError(f"Cloudflare HTML body on {path}")

    @staticmethod
    def _parse_retry_after(resp: Any) -> float | None:
        ra = resp.headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra)
        except ValueError:
            return None


class _StreamBounds:
    """The deadline, stall and first-content bounds of one SSE read, on the
    monotonic clock so wall-clock jumps (NTP, sleep) don't trip them."""

    def __init__(
        self,
        path: str,
        *,
        max_total_seconds: float | None,
        stall_seconds: float | None,
        stall_window: Callable[[], float | None] | None,
        first_content_seconds: float | None,
        silence_after: float,
        silence_error: StreamDeadlineError,
    ) -> None:
        self._path = path
        self._max_total = max_total_seconds
        self._stall_seconds = stall_seconds
        self._stall_window = stall_window
        self._first_content = first_content_seconds
        self._silence_after = silence_after
        self._silence_error = silence_error
        now = time.monotonic()
        self._started = now
        self._deadline = now + max_total_seconds if max_total_seconds else None
        self._last_progress = now
        self._last_byte = now
        self._progressed = False

    def saw_byte(self) -> None:
        self._last_byte = time.monotonic()

    def saw_progress(self) -> None:
        self._last_progress = time.monotonic()
        self._progressed = True

    def _deadline_error(self) -> StreamDeadlineError:
        # Counted to when the cut is seen, not to the deadline: the next
        # heartbeat or curl's abort can land well past it, and the retry
        # advice has to judge the whole gap without progress.
        since = time.monotonic() - self._last_progress if self._progressed else None
        return StreamDeadlineError(
            f"SSE stream on {self._path} exceeded {self._max_total:.1f}s deadline", since
        )

    def check_deadline(self) -> None:
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise self._deadline_error()

    def _content_cut(self) -> tuple[float, StreamStallError] | None:
        """When the first-content or stall bound comes due, whichever is first,
        and the error it raises."""
        window = self._stall_window() if self._stall_window else self._stall_seconds
        stall_due = self._last_progress + window if window else None
        if self._first_content and not self._progressed:
            first_due = self._started + self._first_content
            if stall_due is None or first_due <= stall_due:
                return first_due, StreamFirstContentError(
                    f"SSE stream on {self._path} sent no content within {self._first_content:.1f}s",
                    self._first_content,
                )
        if stall_due is None or not window:
            return None
        return stall_due, StreamStallError(
            f"SSE stream on {self._path} stalled: no new content for {window:.1f}s", window
        )

    def check(self) -> None:
        self.check_deadline()
        cut = self._content_cut()
        if cut is not None and time.monotonic() > cut[0]:
            raise cut[1]

    def abort_error(self) -> StreamDeadlineError:
        """The error for curl's low-speed abort. The abort counts from the last
        byte, so the deadline or a content bound may have come due before it;
        name whichever came due first. On a tie the deadline wins, then the
        silence: a first-content bound as long as the silence window comes due
        with it on a socket that never sent a byte, and only "no bytes" tells
        that apart from a stream kept open by heartbeats."""
        now = time.monotonic()
        # `min` keeps the first of equal items, so this order is the tie order.
        due: list[tuple[float, StreamDeadlineError]] = []
        if self._deadline is not None and now >= self._deadline:
            due.append((self._deadline, self._deadline_error()))
        due.append((self._last_byte + self._silence_after, self._silence_error))
        cut = self._content_cut()
        if cut is not None and now >= cut[0]:
            due.append(cut)
        first = min(due, key=lambda d: d[0])[1]
        # A silence window capped by the deadline stands for the deadline; build
        # it here so it carries the progress timing.
        return self._deadline_error() if type(first) is StreamDeadlineError else first


class _SSEFramer:
    """Splits an SSE byte stream into raw event blocks in time linear in its size.

    Research snapshot events run to ~2.4 MB spread over hundreds of chunks, so
    each chunk's work must not depend on how much of the event is already
    buffered. A `\r` ending a chunk is held back until the next one shows whether
    it opens a CRLF, and a multibyte UTF-8 character split across chunks is
    decoded whole.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._parts: list[str] = []
        self._held_cr = False
        self.pending_chars = 0

    def feed(self, chunk: bytes) -> list[str]:
        """Consume one chunk; return the event blocks it completed, in order."""
        text = self._decoder.decode(chunk)
        if self._held_cr:
            text = "\r" + text
        self._held_cr = text.endswith("\r")
        if self._held_cr:
            text = text[:-1]
        text = text.replace("\r\n", "\n")
        if not text:
            return []
        straddles = text[0] == "\n" and bool(self._parts) and self._parts[-1].endswith("\n")
        if not straddles and "\n\n" not in text:
            self._parts.append(text)
            self.pending_chars += len(text)
            return []
        self._parts.append(text)
        events = "".join(self._parts).split("\n\n")
        rest = events.pop()
        self._parts = [rest] if rest else []
        self.pending_chars = len(rest)
        return events


def thread_ref(entry_uuid: str) -> str:
    """A short stable stand-in for a thread id in messages, so warnings and
    errors can be matched to a thread without printing the id itself."""
    return "#" + hashlib.sha256(entry_uuid.encode()).hexdigest()[:12]


def _media_type(resp: Any) -> str:
    """The response's media type, lowercased and without parameters."""
    return str(resp.headers.get("content-type", "")).split(";", 1)[0].strip().lower()


def _body_excerpt(resp: Any, *, redact: str | None = None) -> str:
    """The start of an error body, for a warning. Decoded here because
    `resp.text` raises when the declared charset is unknown and the body is
    not UTF-8, and a cleanup request must never raise.

    `redact` is removed before the cut, so a secret the body echoes cannot
    survive in part at the excerpt's end."""
    content = resp.content or b""
    if not redact:
        return content[:200].decode("utf-8", "replace")
    text = content.decode("utf-8", "replace").replace(redact, "<redacted>")
    return text[:200]


def _json_body(resp: Any, path: str) -> JsonValue:
    try:
        return from_parser(resp.json())
    except Exception as e:
        raise SchemaError(f"non-JSON response from {path}") from e


def _silence_bounds(
    path: str,
    connect_timeout: float,
    max_total_seconds: float | None,
    silence_seconds: float | None,
) -> tuple[float, float, StreamDeadlineError]:
    """The SSE (connect, read) legs, and the error their low-speed abort stands for.

    curl aborts after connect + read seconds below 1 B/s, so the two legs must
    sum to the silence window; a window shorter than the connect timeout
    shrinks the connect leg to the window and the read leg to 0, and curl
    still aborts at the sum. The window is `silence_seconds` capped by the
    overall deadline; when the deadline is the one that runs out first, the
    abort is reported as the deadline. The abort counts from the last byte,
    not the last progress event.
    """
    if silence_seconds:
        window = silence_seconds
        if max_total_seconds and max_total_seconds < silence_seconds:
            window = max_total_seconds
        connect_timeout = min(connect_timeout, window)
        read_timeout = window - connect_timeout
    else:
        read_timeout = DEFAULT_SSE_READ_TIMEOUT
    abort_after = connect_timeout + read_timeout
    if max_total_seconds and max_total_seconds <= abort_after:
        return (
            connect_timeout,
            read_timeout,
            StreamDeadlineError(f"SSE stream on {path} exceeded {max_total_seconds:.1f}s deadline"),
        )
    return (
        connect_timeout,
        read_timeout,
        StreamSilenceError(
            f"SSE stream on {path} went silent: no bytes for {abort_after:.1f}s", abort_after
        ),
    )


def _parse_sse_event(raw: str) -> dict[str, Any] | None:
    """Parse one SSE event block (without trailing blank line).

    Returns None for an empty block (multiple blank lines in a row). Otherwise
    returns {"event": <type or None>, "data": <parsed JSON, raw string, or None>}.
    """
    if not raw.strip():
        return None
    event_type: str | None = None
    data_lines: list[str] = []
    for line in raw.split("\n"):
        if line.startswith(":"):
            continue  # SSE comment line
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return {"event": event_type, "data": None}
    data_str = "\n".join(data_lines)
    try:
        data: Any = json.loads(data_str)
    except json.JSONDecodeError:
        data = data_str
    return {"event": event_type, "data": data}
