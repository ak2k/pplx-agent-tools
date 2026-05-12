"""Cookie loading + perms enforcement for pplx-agent-tools.

Resolution chain (first match wins):
  1. $PPLX_COOKIES_PATH  → JSON file at that path
  2. $PPLX_COOKIES       → inline JSON string
  3. $XDG_CONFIG_HOME/perplexity/<profile>/cookies.json   (profile = $PPLX_PROFILE or "default")

Accepts two on-disk shapes:
  - flat dict: {"name": "value", ...}
  - Cookie-Editor array: [{"name": "...", "value": "...", "domain": "...", ...}, ...]

Both are flattened to {name: value} for curl_cffi's session cookies kwarg.

Cookie files must be mode 0600. World-readable files are refused; group-readable
files are auto-chmodded with a warning to stderr.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from .errors import AuthError

DEFAULT_PROFILE = "default"

# rookiepy exposes a function per browser (rookiepy.brave, rookiepy.chrome, etc.).
# Our CLI flag is the lowercase function name. Each function takes a list of
# domain substrings and returns a list of cookie dicts in Cookie-Editor shape.
SUPPORTED_BROWSERS: tuple[str, ...] = (
    "brave",
    "chrome",
    "chromium",
    "edge",
    "firefox",
    "safari",
    "arc",
    "vivaldi",
    "opera",
    "librewolf",
    "zen",
)

# Domain filter passed to rookiepy. Substring match → covers perplexity.ai,
# www.perplexity.ai, and any other subdomain.
_COOKIE_DOMAIN = "perplexity.ai"


def resolve_profile(profile: str | None = None) -> str:
    """Resolve which profile to use: explicit arg → $PPLX_PROFILE → 'default'."""
    return profile or os.environ.get("PPLX_PROFILE") or DEFAULT_PROFILE


def default_cookies_path(profile: str | None = None) -> Path:
    """Where the XDG-default cookie file lives for a profile."""
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "perplexity" / resolve_profile(profile) / "cookies.json"


def load_cookies(profile: str | None = None) -> dict[str, str]:
    """Resolve and load cookies. Returns flat {name: value} dict.

    Raises AuthError if cookies cannot be found, parsed, or have unsafe perms.
    """
    if path_str := os.environ.get("PPLX_COOKIES_PATH"):
        path = Path(path_str)
        return _load_from_file(path)

    if inline := os.environ.get("PPLX_COOKIES"):
        try:
            data = json.loads(inline)
        except json.JSONDecodeError as e:
            raise AuthError(f"$PPLX_COOKIES is not valid JSON: {e.msg}") from e
        return _normalize(data, source="$PPLX_COOKIES")

    path = default_cookies_path(profile)
    if not path.exists():
        raise AuthError(
            f"no cookies found at {path}; run pplx-auth import --browser brave"
        )
    return _load_from_file(path)


def _load_from_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise AuthError(f"cookie file does not exist: {path}")
    _enforce_perms(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as e:
        raise AuthError(f"cookie file unreadable: {path}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise AuthError(f"cookie file invalid JSON: {path}: {e.msg}") from e
    return _normalize(data, source=str(path))


def _enforce_perms(path: Path) -> None:
    """Refuse world-readable; auto-chmod group-readable to 0600."""
    try:
        mode = path.stat().st_mode
    except OSError as e:
        raise AuthError(f"cannot stat cookie file: {path}: {e.strerror}") from e

    perms = stat.S_IMODE(mode)
    world_bits = perms & 0o007
    group_bits = perms & 0o070

    if world_bits:
        raise AuthError(
            f"cookie file is world-accessible (mode {perms:04o}): {path}; "
            f"run: chmod 600 {path}"
        )
    if group_bits:
        print(
            f"warning: cookie file is group-accessible (mode {perms:04o}); "
            f"chmodding to 0600: {path}",
            file=sys.stderr,
        )
        try:
            path.chmod(0o600)
        except OSError as e:
            raise AuthError(
                f"could not tighten perms on cookie file: {path}: {e.strerror}"
            ) from e


def _normalize(data: Any, *, source: str) -> dict[str, str]:
    """Coerce flat-dict or Cookie-Editor-array into {name: value}.

    `source` is used only in error messages to identify where the data came from.
    Never include cookie values in errors — only the source label.
    """
    if isinstance(data, dict):
        flat = {str(k): str(v) for k, v in data.items()}
    elif isinstance(data, list):
        flat = {}
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise AuthError(
                    f"cookie entry {i} in {source} is not an object"
                )
            name = entry.get("name")
            value = entry.get("value")
            if not isinstance(name, str) or value is None:
                raise AuthError(
                    f"cookie entry {i} in {source} missing 'name' or 'value'"
                )
            flat[name] = str(value)
    else:
        raise AuthError(
            f"cookie data in {source} must be an object or array, got {type(data).__name__}"
        )

    if not flat:
        raise AuthError(f"no cookies parsed from {source}")
    return flat


def import_from_browser(browser: str, profile: str | None = None) -> Path:
    """Read *.perplexity.ai cookies from a local browser via rookiepy,
    write them to the profile path. Atomic (tmp + rename), mode 0600.

    rookiepy handles platform details: keychain on macOS, GNOME-keyring /
    kwallet / plaintext on Linux, DPAPI on Windows, locked-DB copy-to-temp,
    v10/v11 prefix dispatch, host-key integrity-binding strip.

    Returned shape is Cookie-Editor array; we flatten name→value before write.
    """
    if browser not in SUPPORTED_BROWSERS:
        supported = ", ".join(SUPPORTED_BROWSERS)
        raise AuthError(f"unsupported browser: {browser!r} (supported: {supported})")

    try:
        import rookiepy
    except ImportError as e:
        raise AuthError(
            f"rookiepy is required for browser import: {e}"
        ) from e

    fn = getattr(rookiepy, browser, None)
    if fn is None:
        raise AuthError(
            f"rookiepy has no '{browser}' loader; "
            f"upgrade rookiepy or pick a different browser"
        )

    try:
        rows = fn([_COOKIE_DOMAIN])
    except Exception as e:
        raise AuthError(f"rookiepy.{browser} failed: {e}") from e

    cookies: dict[str, str] = {}
    for row in rows or []:
        name = row.get("name")
        value = row.get("value")
        if isinstance(name, str) and value is not None:
            cookies[name] = str(value)

    if not cookies:
        raise AuthError(
            f"no cookies for *.{_COOKIE_DOMAIN} in {browser}; "
            f"sign in at perplexity.ai in {browser} first"
        )

    dest = default_cookies_path(profile)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(cookies, indent=2, sort_keys=True))
    tmp.chmod(0o600)
    tmp.replace(dest)
    return dest
