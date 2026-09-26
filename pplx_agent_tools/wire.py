"""Transport layer: curl_cffi chrome-impersonate session.

Minimal v1 surface: enough for /api/auth/session round-trips during Step 2.
Verb-specific methods (search, fetch, snippets) join in Step 4 once their
endpoints are reverse-engineered.

`curl_cffi` is required (not `requests`) so Cloudflare's TLS fingerprint check
accepts us as a real Chrome client. See balakumardev/perplexity-web-wrapper.
"""

from __future__ import annotations

import codecs
import contextlib
import json
import sys
import time
from collections.abc import Callable, Iterator
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
    StreamDeadlineError,
    StreamFirstContentError,
    StreamSilenceError,
    StreamStallError,
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
# The terminate request runs in cleanup, often right after a Ctrl-C, so it gets
# a bound of its own rather than the client's 30 s default. It answered in
# 0.17 s when probed live.
TERMINATE_TIMEOUT_SECONDS = 5.0
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
                timeout=self._timeout,
            )
        except Exception as e:
            print(f"warning: thread cleanup failed: {e}", file=sys.stderr)
            return False
        if resp is None:
            print(
                f"warning: thread cleanup failed: no response for {entry_uuid}",
                file=sys.stderr,
            )
            return False
        status = resp.status_code
        if status >= 400:
            body = (resp.text or "")[:200]
            print(
                f"warning: thread cleanup failed: DELETE {entry_uuid} returned {status}: {body}",
                file=sys.stderr,
            )
            return False
        return True

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
                timeout=TERMINATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            print(f"warning: run terminate failed: {e}", file=sys.stderr)
            return False
        status = resp.status_code
        if status >= 400:
            body = (resp.text or "")[:200]
            print(
                f"warning: run terminate failed: {entry_uuid} returned {status}: {body}",
                file=sys.stderr,
            )
            return False
        return True

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
        url = self._base_url + path
        connect_timeout, read_timeout, silence_error = _silence_bounds(
            path, self._timeout, max_total_seconds, silence_seconds
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
            raise NetworkError(f"POST {path} failed: {e!s}") from e
        # Headers / status validated before we start consuming the body.
        self._check_status(resp, path)

        # Use monotonic so wall-clock jumps (NTP, sleep) don't trip the deadline.
        started = time.monotonic()
        deadline = (started + max_total_seconds) if max_total_seconds else None
        last_progress = started
        last_byte = started
        progressed = False

        def _deadline_error() -> StreamDeadlineError:
            return StreamDeadlineError(
                f"SSE stream on {path} exceeded {max_total_seconds:.1f}s deadline"
            )

        def _content_cut() -> tuple[float, StreamStallError] | None:
            """When the first-content or stall bound comes due, whichever is
            first, and the error it raises."""
            window = stall_window() if stall_window else stall_seconds
            stall_due = last_progress + window if window else None
            if first_content_seconds and not progressed:
                first_due = started + first_content_seconds
                if stall_due is None or first_due <= stall_due:
                    return first_due, StreamFirstContentError(
                        f"SSE stream on {path} sent no content within {first_content_seconds:.1f}s",
                        first_content_seconds,
                    )
            if stall_due is None or not window:
                return None
            return stall_due, StreamStallError(
                f"SSE stream on {path} stalled: no new content for {window:.1f}s", window
            )

        def _check_bounds() -> None:
            now = time.monotonic()
            if deadline is not None and now > deadline:
                raise _deadline_error()
            cut = _content_cut()
            if cut is not None and now > cut[0]:
                raise cut[1]

        framer = _SSEFramer()
        try:
            try:
                for chunk in resp.iter_content(chunk_size=4096):
                    if deadline is not None and time.monotonic() > deadline:
                        raise StreamDeadlineError(
                            f"SSE stream on {path} exceeded {max_total_seconds:.1f}s deadline"
                        )
                    if not chunk:
                        continue
                    last_byte = time.monotonic()
                    raw_events = framer.feed(chunk)
                    # Bound memory against a server that trickles bytes without ever
                    # emitting an event terminator (`\n\n`): the per-chunk idle timeout
                    # wouldn't fire on a continuous trickle, so cap the un-dispatched
                    # buffer. A single SSE event over 16 MiB is pathological.
                    if not raw_events and framer.pending_chars > _MAX_SSE_BUFFER_BYTES:
                        raise SchemaError(
                            f"SSE stream on {path} exceeded {_MAX_SSE_BUFFER_BYTES} bytes "
                            "without an event terminator"
                        )
                    for raw_event in raw_events:
                        parsed = _parse_sse_event(raw_event)
                        if parsed is not None:
                            if parsed["data"] is not None and (
                                is_progress is None or is_progress(parsed)
                            ):
                                last_progress = time.monotonic()
                                progressed = True
                            yield parsed
                            # Re-check between yields so a generator consumer
                            # that processes events slowly can't outrun the bounds.
                            _check_bounds()
                    # A chunk of heartbeats alone yields no progress event, so the
                    # stall check has to run per chunk as well.
                    _check_bounds()
            except PplxError:
                # Deadline and schema faults raised in the loop above already carry
                # their own exit-code contract; only transport faults are reclassified.
                raise
            except Exception as e:
                # curl_cffi queues a mid-stream failure as a base RequestException
                # carrying the curl code, not as its Timeout subclass, so the code
                # is the only reliable signal of the low-speed abort.
                if isinstance(e, CurlError) and e.code == CurlECode.OPERATION_TIMEDOUT:
                    # The abort counts from the last byte, so it can land after
                    # the deadline or the content bounds came due; name whichever
                    # came due first (the deadline on a tie).
                    now = time.monotonic()
                    due: list[tuple[float, StreamDeadlineError]] = []
                    if deadline is not None and now >= deadline:
                        due.append((deadline, _deadline_error()))
                    cut = _content_cut()
                    if cut is not None and now >= cut[0]:
                        due.append(cut)
                    due.append((last_byte + connect_timeout + read_timeout, silence_error))
                    raise min(due, key=lambda d: d[0])[1] from e
                # A read that dies mid-stream is the same class of failure as a POST
                # that never connected, so it gets the same typed error and exit code.
                # Events already yielded are not salvaged: a truncated stream has no
                # completion signal, and salvage stays reserved for the deadline and
                # stall paths, where the stream was cut on our side of the wire.
                raise NetworkError(f"SSE stream on {path} failed mid-stream: {e!s}") from e
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
