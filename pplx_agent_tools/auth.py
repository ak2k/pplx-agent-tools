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
Every name and value must be a string the transport can send intact (see
`_cookie_text_ok`). Files and $PPLX_COOKIES are refused whole on any bad pair;
browser import and `save_cookies` skip bad pairs with a warning, so nothing
written here is refused on the next load.

Cookie files must be mode 0600. World-readable files are refused; group-readable
files are auto-chmodded with a warning to stderr.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EnvPathSource:
    path: Path


@dataclass(frozen=True)
class EnvInlineSource:
    text: str


@dataclass(frozen=True)
class ProfileSource:
    profile: str
    path: Path


CookieSource = EnvPathSource | EnvInlineSource | ProfileSource


def cookie_source(profile: str | None = None) -> CookieSource:
    """The one source `load_cookies` reads, by the resolution chain above."""
    if path_str := os.environ.get("PPLX_COOKIES_PATH"):
        return EnvPathSource(Path(path_str))
    if inline := os.environ.get("PPLX_COOKIES"):
        return EnvInlineSource(inline)
    return ProfileSource(resolve_profile(profile), default_cookies_path(profile))


def _describe(source: CookieSource) -> str:
    """Names the source for error messages; never includes cookie values."""
    match source:
        case EnvPathSource(path):
            return f"$PPLX_COOKIES_PATH file {path}"
        case EnvInlineSource():
            return "$PPLX_COOKIES"
        case ProfileSource(name, path):
            return f"profile {name!r} file {path}"
    # Unreachable while CookieSource has three members; never repr the source, it may hold cookies.
    raise AssertionError(f"unhandled cookie source {type(source).__name__}")


def load_cookies(profile: str | None = None) -> dict[str, str]:
    """Resolve and load cookies. Returns flat {name: value} dict.

    Raises AuthError if cookies cannot be found, parsed, or have unsafe perms.
    Every error names the source it read.
    """
    source = cookie_source(profile)
    label = _describe(source)
    match source:
        case EnvInlineSource(text):
            try:
                data: object = json.loads(text)
            except json.JSONDecodeError as e:
                raise AuthError(f"$PPLX_COOKIES is not valid JSON: {e.msg}") from e
            return _normalize(data, source=label)
        case ProfileSource(_, path) if not path.exists():
            raise AuthError(f"no cookies found at {label}; run pplx auth import --browser brave")
        case EnvPathSource(path) | ProfileSource(_, path):
            return _load_from_file(path, label=label)
    raise AssertionError(f"unhandled cookie source {type(source).__name__}")


def save_cookies(
    cookies: dict[str, str], profile: str | None = None, *, dest: Path | None = None
) -> Path:
    """Persist cookies to `dest` (default: the profile's file) with mode 0600.

    Pairs the loader would refuse are dropped with a warning naming the cookie,
    because one of them would make the whole file unloadable. Atomic via tmp +
    rename. Returns the path written.
    """
    loadable: dict[str, str] = {}
    for name, value in cookies.items():
        try:
            loadable[name] = _cookie_pair(name, value, where="not saving cookie")[1]
        except AuthError as e:
            print(f"warning: {e}", file=sys.stderr)
    if not loadable:
        raise AuthError("no loadable cookies to save; the cookie file was not changed")
    dest = dest or default_cookies_path(profile)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_0600(dest, json.dumps(loadable, indent=2, sort_keys=True))
    except OSError as e:
        raise AuthError(f"cannot write cookie file: {dest}: {e.strerror}") from e
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


def _load_from_file(path: Path, *, label: str) -> dict[str, str]:
    if not path.exists():
        raise AuthError(f"cookie file does not exist: {label}")
    _enforce_perms(path, label=label)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data: object = json.load(fh)
    except OSError as e:
        raise AuthError(f"cookie file unreadable: {label}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise AuthError(f"cookie file invalid JSON: {label}: {e.msg}") from e
    return _normalize(data, source=label)


def _enforce_perms(path: Path, *, label: str) -> None:
    """Refuse world-readable; auto-chmod group-readable to 0600."""
    try:
        mode = path.stat().st_mode
    except OSError as e:
        raise AuthError(f"cannot stat cookie file: {label}: {e.strerror}") from e

    perms = stat.S_IMODE(mode)
    world_bits = perms & 0o007
    group_bits = perms & 0o070

    if world_bits:
        raise AuthError(
            f"cookie file is world-accessible (mode {perms:04o}): {label}; run: chmod 600 {path}"
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
            raise AuthError(f"could not tighten perms on cookie file: {label}: {e.strerror}") from e


# RFC 6265bis 5.6 (the user-agent storage rule): a name or value holding a CTL
# other than HTAB is refused, and ';' ends the pair. HTAB is refused as well,
# because curl's Netscape cookie lines are tab-delimited and a tab silently
# drops the cookie. Lone surrogates cannot be UTF-8 encoded by curl_cffi.
_FORBIDDEN = frozenset([*map(chr, range(0x20)), "\x7f", ";"])


def _cookie_text_ok(s: str, *, extra: str = "") -> bool:
    if any(c in _FORBIDDEN or c in extra for c in s):
        return False
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _cookie_pair(name: object, value: object, *, where: str) -> tuple[str, str]:
    """The single gate every cookie passes before reaching the HTTP layer.

    Errors name the cookie at most, never the value: values are session secrets.
    """
    if not isinstance(name, str) or not name or not _cookie_text_ok(name, extra="="):
        raise AuthError(
            f"{where}: cookie name must be a non-empty string without ';', '=', "
            "control characters or unpaired surrogates"
        )
    if value is None:
        raise AuthError(f"{where}: cookie {name!r} has no value")
    if not isinstance(value, str) or not _cookie_text_ok(value):
        raise AuthError(
            f"{where}: value of cookie {name!r} must be a string without ';', "
            "control characters or unpaired surrogates"
        )
    return name, value


def cookie_pair_ok(name: str, value: str) -> bool:
    """Whether `load_cookies` would accept this pair."""
    try:
        _cookie_pair(name, value, where="")
    except AuthError:
        return False
    return True


def _cookie_entry(entry: object, *, where: str) -> tuple[str, str]:
    """One Cookie-Editor / rookiepy row: {"name": ..., "value": ..., ...}."""
    if not isinstance(entry, dict):
        raise AuthError(f"{where} is not an object")
    fields = cast("dict[object, object]", entry)
    if "name" not in fields or "value" not in fields:
        raise AuthError(f"{where} missing 'name' or 'value'")
    return _cookie_pair(fields["name"], fields["value"], where=where)


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
            name, value = _cookie_entry(entry, where=f"cookie entry {i} in {source}")
            flat[name] = value
    else:
        raise AuthError(
            f"cookie data in {source} must be an object or array, got {type(data).__name__}"
        )

    if not flat:
        raise AuthError(f"no cookies parsed from {source}")
    return flat


def import_from_browser(browser: str, profile: str | None = None) -> Path:
    """Read *.perplexity.ai cookies from a local browser via rookiepy and
    write them to the file `load_cookies` reads ($PPLX_COOKIES_PATH, else the
    profile file). Atomic (tmp + rename), mode 0600.

    rookiepy handles platform details: keychain on macOS, GNOME-keyring /
    kwallet / plaintext on Linux, DPAPI on Windows, locked-DB copy-to-temp,
    v10/v11 prefix dispatch, host-key integrity-binding strip.

    Returned shape is Cookie-Editor array; we flatten name→value before write.
    Unlike a cookie file, a bad row is skipped with a warning: the jar holds
    third-party cookies the user cannot edit, and one of them must not block
    the import.
    """
    if browser not in SUPPORTED_BROWSERS:
        supported = ", ".join(SUPPORTED_BROWSERS)
        raise AuthError(f"unsupported browser: {browser!r} (supported: {supported})")

    source = cookie_source(profile)
    match source:
        case EnvInlineSource():
            # A child process cannot change the parent's environment.
            raise AuthError(
                "$PPLX_COOKIES is set and overrides any cookie file, so an import "
                "would not be used; it was not changed. Replace or unset $PPLX_COOKIES"
            )
        case EnvPathSource(dest) | ProfileSource(_, dest):
            pass

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

    if not isinstance(rows, list):
        raise AuthError(f"rookiepy.{browser} returned {type(rows).__name__}, expected a list")
    cookies: dict[str, str] = {}
    for i, row in enumerate(cast("list[object]", rows)):
        try:
            name, value = _cookie_entry(row, where=f"cookie entry {i} from {browser}")
        except AuthError as e:
            print(f"warning: skipping {e}", file=sys.stderr)
            continue
        cookies[name] = value

    if not cookies:
        raise AuthError(
            f"no usable cookies for *.{_COOKIE_DOMAIN} in {browser}; "
            f"sign in at perplexity.ai in {browser} first"
        )

    return save_cookies(cookies, dest=dest)
