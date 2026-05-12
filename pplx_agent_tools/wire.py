"""Transport layer: curl_cffi chrome-impersonate session.

Minimal v1 surface: enough for /api/auth/session round-trips during Step 2.
Verb-specific methods (search, fetch, snippets) join in Step 4 once their
endpoints are reverse-engineered.

`curl_cffi` is required (not `requests`) so Cloudflare's TLS fingerprint check
accepts us as a real Chrome client. See balakumardev/perplexity-web-wrapper.
"""

from __future__ import annotations

from typing import Any

from curl_cffi import requests as cf_requests

from .errors import AntiBotError, AuthError, NetworkError, RateLimitError, SchemaError

BASE_URL = "https://www.perplexity.ai"
DEFAULT_TIMEOUT = 30.0
DEFAULT_IMPERSONATE = "chrome"


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
        self._session = cf_requests.Session(impersonate=impersonate)

    @classmethod
    def from_default_cookies(cls, profile: str | None = None, **kwargs: Any) -> "Client":
        from .auth import load_cookies

        return cls(load_cookies(profile), **kwargs)

    def auth_session(self) -> dict[str, Any]:
        """GET /api/auth/session. Returns parsed JSON.

        NextAuth returns `{}` for unauthenticated; a populated dict (with `user`)
        for an authenticated session. We treat empty/missing-user as AuthError.
        """
        resp = self._get("/api/auth/session")
        try:
            data = resp.json()
        except Exception as e:
            raise SchemaError("non-JSON response from /api/auth/session") from e
        if not isinstance(data, dict):
            raise SchemaError(
                f"/api/auth/session returned {type(data).__name__}, expected object"
            )
        if not data or "user" not in data:
            raise AuthError("session expired or unauthenticated; re-import cookies")
        return data

    def _get(self, path: str, **kwargs: Any) -> Any:
        url = self._base_url + path
        try:
            resp = self._session.get(
                url, cookies=self._cookies, timeout=self._timeout, **kwargs
            )
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
        ct = resp.headers.get("content-type", "")
        if "text/html" not in ct.lower():
            return False
        body = (resp.content or b"")[:2000].lower()
        return b"cloudflare" in body or b"just a moment" in body

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
