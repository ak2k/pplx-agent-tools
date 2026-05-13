"""Transport layer: curl_cffi chrome-impersonate session.

Minimal v1 surface: enough for /api/auth/session round-trips during Step 2.
Verb-specific methods (search, fetch, snippets) join in Step 4 once their
endpoints are reverse-engineered.

`curl_cffi` is required (not `requests`) so Cloudflare's TLS fingerprint check
accepts us as a real Chrome client. See balakumardev/perplexity-web-wrapper.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from collections.abc import Iterator
from typing import Any

from curl_cffi import requests as cf_requests

from .errors import (
    AntiBotError,
    AuthError,
    NetworkError,
    RateLimitError,
    SchemaError,
    StreamDeadlineError,
)

BASE_URL = "https://www.perplexity.ai"
DEFAULT_TIMEOUT = 30.0
DEFAULT_IMPERSONATE = "chrome"
# SSE-only: idle deadline between consecutive chunks. A stalled server
# would otherwise hold our connection forever — DEFAULT_TIMEOUT only
# bounds the initial connect on streaming requests.
DEFAULT_SSE_READ_TIMEOUT = 60.0


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
            if new_val is not None and new_val != self._cookies[name]:
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

    def post_json(self, path: str, body: dict[str, Any]) -> Any:
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
        try:
            return resp.json()
        except Exception as e:
            raise SchemaError(f"non-JSON response from {path}") from e

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

    def sse_post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
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

        Raises the same typed errors as the GET path (auth/rate-limit/etc.) on
        connection or status-code failure.
        """
        url = self._base_url + path
        try:
            resp = self._session.post(
                url,
                cookies=self._cookies,
                json=body,
                headers={"accept": "text/event-stream"},
                stream=True,
                # Tuple form: (connect_timeout, read_timeout). The read leg
                # is the per-chunk idle deadline — a Perplexity stream that
                # stops emitting events should fail loudly rather than hang.
                timeout=(self._timeout, DEFAULT_SSE_READ_TIMEOUT),
            )
        except Exception as e:
            raise NetworkError(f"POST {path} failed: {e!s}") from e
        # Headers / status validated before we start consuming the body.
        self._check_status(resp, path)

        # Use monotonic so wall-clock jumps (NTP, sleep) don't trip the deadline.
        deadline = (time.monotonic() + max_total_seconds) if max_total_seconds else None
        buffer = ""
        try:
            for chunk in resp.iter_content(chunk_size=4096):
                if deadline is not None and time.monotonic() > deadline:
                    raise StreamDeadlineError(
                        f"SSE stream on {path} exceeded {max_total_seconds:.1f}s deadline"
                    )
                if not chunk:
                    continue
                buffer += chunk.decode("utf-8", errors="replace")
                # Normalize CRLF that SSE protocol uses.
                buffer = buffer.replace("\r\n", "\n")
                while "\n\n" in buffer:
                    raw_event, buffer = buffer.split("\n\n", 1)
                    parsed = _parse_sse_event(raw_event)
                    if parsed is not None:
                        yield parsed
                        # Re-check deadline between yields so a generator
                        # consumer that processes events slowly can't outrun
                        # the bound either.
                        if deadline is not None and time.monotonic() > deadline:
                            raise StreamDeadlineError(
                                f"SSE stream on {path} exceeded {max_total_seconds:.1f}s deadline"
                            )
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
