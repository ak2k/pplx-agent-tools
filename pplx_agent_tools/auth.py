# pyright: strict
"""Cookie loading + perms enforcement for pplx-agent-tools.

Resolution chain (first match wins):
  1. $PPLX_COOKIES_PATH  → JSON file at that path
  2. $PPLX_COOKIES       → inline JSON string
  3. $XDG_CONFIG_HOME/perplexity/<profile>/cookies.json   (profile = $PPLX_PROFILE or "default")

Accepts two on-disk shapes:
  - flat dict: {"name": "value", ...}
  - Cookie-Editor array: [{"name": "...", "value": "...", "domain": "...", ...}, ...]

Both are flattened to {name: value} for curl_cffi's session cookies kwarg.
Every name and value must be a string that cannot split the Cookie header
(no control characters or ';'; names also no '='). Anything else is refused
with AuthError rather than coerced.

Cookie files must be mode 0600. World-readable files are refused; group-readable
files are auto-chmodded with a warning to stderr.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import cast

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
            data: object = json.loads(inline)
        except json.JSONDecodeError as e:
            raise AuthError(f"$PPLX_COOKIES is not valid JSON: {e.msg}") from e
        return _normalize(data, source="$PPLX_COOKIES")

    path = default_cookies_path(profile)
    if not path.exists():
        raise AuthError(f"no cookies found at {path}; run pplx auth import --browser brave")
    return _load_from_file(path)


def save_cookies(cookies: dict[str, str], profile: str | None = None) -> Path:
    """Persist cookies to the profile's on-disk file with mode 0600.

    Atomic via tmp + rename. Returns the path written.
    """
    dest = default_cookies_path(profile)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_0600(dest, json.dumps(cookies, indent=2, sort_keys=True))
    return dest


def _atomic_write_0600(dest: Path, content: str) -> None:
    """Write `content` to `dest` atomically with mode 0o600 from byte zero.

    `tempfile.NamedTemporaryFile` is backed by `mkstemp` on POSIX, which
    creates the file with mode 0o600 ignoring umask — so the tmp is never
    world-readable, even in the window before the rename. We place the tmp
    in dest.parent so the final replace stays on one filesystem (required
    for atomic rename) and unlink it on any failure to avoid leftovers.
    """
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dest.parent,
            prefix=f"{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(content)
        tmp_path.replace(dest)
        tmp_path = None  # ownership transferred; skip cleanup
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _load_from_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise AuthError(f"cookie file does not exist: {path}")
    _enforce_perms(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data: object = json.load(fh)
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
            f"cookie file is world-accessible (mode {perms:04o}): {path}; run: chmod 600 {path}"
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
            raise AuthError(f"could not tighten perms on cookie file: {path}: {e.strerror}") from e


def _header_safe(s: str, *, extra: str = "") -> bool:
    return not any(c in extra or c == ";" or ord(c) < 0x20 or ord(c) == 0x7F for c in s)


def _cookie_pair(name: object, value: object, *, where: str) -> tuple[str, str]:
    """The single gate every cookie passes before reaching the HTTP layer.

    Errors name the location only, never the value: these are session secrets.
    """
    if not isinstance(name, str) or not name or not _header_safe(name, extra="="):
        raise AuthError(
            f"{where}: cookie name must be a non-empty string without ';', '=' or control characters"
        )
    if not isinstance(value, str) or not _header_safe(value):
        raise AuthError(
            f"{where}: value of cookie {name!r} must be a string without ';' or control characters"
        )
    return name, value


def _normalize(data: object, *, source: str) -> dict[str, str]:
    """Parse flat-dict or Cookie-Editor-array into {name: value}.

    `source` is used only in error messages to identify where the data came from.
    """
    flat: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in cast("dict[object, object]", data).items():
            name, value = _cookie_pair(k, v, where=f"cookie data in {source}")
            flat[name] = value
    elif isinstance(data, list):
        for i, entry in enumerate(cast("list[object]", data)):
            if not isinstance(entry, dict):
                raise AuthError(f"cookie entry {i} in {source} is not an object")
            fields = cast("dict[object, object]", entry)
            if "name" not in fields or "value" not in fields:
                raise AuthError(f"cookie entry {i} in {source} missing 'name' or 'value'")
            name, value = _cookie_pair(
                fields["name"], fields["value"], where=f"cookie entry {i} in {source}"
            )
            flat[name] = value
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
        raise AuthError(f"rookiepy is required for browser import: {e}") from e

    fn = getattr(rookiepy, browser, None)
    if fn is None:
        raise AuthError(
            f"rookiepy has no '{browser}' loader; upgrade rookiepy or pick a different browser"
        )

    try:
        rows: object = fn([_COOKIE_DOMAIN])
    except Exception as e:
        raise AuthError(f"rookiepy.{browser} failed: {e}") from e

    if not rows:
        raise AuthError(
            f"no cookies for *.{_COOKIE_DOMAIN} in {browser}; "
            f"sign in at perplexity.ai in {browser} first"
        )
    cookies = _normalize(rows, source=f"rookiepy.{browser}")

    return save_cookies(cookies, profile=profile)
