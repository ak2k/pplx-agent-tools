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
from collections.abc import Iterable
from dataclasses import dataclass
from http.cookiejar import Cookie
from pathlib import Path
from typing import cast

from typing_extensions import assert_never

from .errors import AuthError

DEFAULT_PROFILE = "default"

SUPPORTED_BROWSERS: tuple[str, ...] = (
    "brave",
    "chrome",
    "chromium",
    "edge",
    "firefox",
    "safari",
    "vivaldi",
    "opera",
    "librewolf",
    "zen",
)

# Firefox forks whose cookie store yt-dlp reads as Firefox's, given the
# fork's profile root; the value is the fork's Flatpak app id.
_FIREFOX_FORKS = {
    "librewolf": "io.gitlab.librewolf-community",
    "zen": "app.zen_browser.zen",
}

_COOKIE_DOMAIN = "perplexity.ai"
# The host of wire.BASE_URL; wire imports this module, so it cannot import wire.
_REQUEST_HOST = "www.perplexity.ai"


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


def describe_cookie_source(source: CookieSource) -> str:
    """Names the source for messages; never includes cookie values."""
    match source:
        case EnvPathSource(path):
            return f"$PPLX_COOKIES_PATH file {path}"
        case EnvInlineSource():
            return "$PPLX_COOKIES"
        case ProfileSource(name, path):
            return f"profile {name!r} file {path}"
    assert_never(source)


def load_cookies(profile: str | None = None) -> dict[str, str]:
    """Resolve and load cookies. Returns flat {name: value} dict.

    Raises AuthError if cookies cannot be found, parsed, or have unsafe perms.
    Every error names the source it read.
    """
    source = cookie_source(profile)
    label = describe_cookie_source(source)
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
    assert_never(source)


def cookie_write_path(profile: str | None, *, inline_refusal: str) -> Path:
    """The file `load_cookies` reads, so a write there is what the next load sees.

    Raises AuthError with `inline_refusal` when $PPLX_COOKIES is set and
    $PPLX_COOKIES_PATH is not: a child process cannot change the parent's
    environment, so no write would be read.
    """
    source = cookie_source(profile)
    match source:
        case EnvInlineSource():
            raise AuthError(inline_refusal)
        case EnvPathSource(path) | ProfileSource(_, path):
            return path
    assert_never(source)


def save_cookies(cookies: dict[str, str], *, dest: Path, expected: Path | None = None) -> Path:
    """Persist cookies to `dest` with mode 0600.

    Pairs the loader would refuse are dropped with a warning naming the cookie,
    because one of them would make the whole file unloadable. Atomic via tmp +
    rename. Returns the path written: `dest` with symlinks resolved. Raises
    AuthError unless `dest` as given then leads to that file, which is what the
    next load opens; when that is found only after the write, the file is left
    where it was written and the error names it. Never deletes a cookie file.

    `expected` is the file the cookies were read from, as `cookie_file_target`
    named it before the load; if `dest` now resolves elsewhere, nothing is
    written.
    """
    loadable: dict[str, str] = {}
    for name, value in cookies.items():
        try:
            loadable[name] = _cookie_pair(name, value, where="not saving cookie")[1]
        except AuthError as e:
            print(f"warning: {e}", file=sys.stderr)
    if not loadable:
        raise AuthError("no loadable cookies to save; the cookie file was not changed")
    # The rename would replace a symlink itself and leave the file it points
    # to (synced or managed elsewhere) stale; write beside the target instead.
    target = cookie_file_target(dest)
    if expected is not None and target != expected:
        raise AuthError(
            f"cannot write cookie file: {dest}: it now leads to {target}, "
            f"not {expected}, which the cookies were read from"
        )
    try:
        # resolve() collapses "missing/.." even when "missing" is absent, but
        # opening `dest` cannot walk it until the directory exists. The target's
        # directory goes first: mkdir refuses a symlink in `dest` that dangles.
        target.parent.mkdir(parents=True, exist_ok=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # A symlink in `dest` can still name such a path, so check that `dest`
        # leads to the file, before overwriting one and after creating one.
        if target.exists() and not _leads_to(dest, target):
            raise AuthError(f"cannot write cookie file: {dest}: the path does not lead to {target}")
        atomic_write_0600(target, json.dumps(loadable, indent=2, sort_keys=True))
        # The file stays: a concurrent save can rename its own file in between
        # the check's two stats, and deleting `target` would delete that one.
        if not _leads_to(dest, target):
            raise AuthError(
                f"cannot write cookie file: {dest}: wrote {target}, "
                "but the path does not lead to it"
            )
    except OSError as e:
        raise AuthError(f"cannot write cookie file: {target}: {e.strerror}") from e
    return target


def cookie_file_target(dest: Path) -> Path:
    """The file a save to `dest` writes: `dest` with symlinks resolved.

    Raises AuthError on a symlink loop.
    """
    loop = f"cannot write cookie file: {dest}: symlink loop"
    target = dest
    try:
        target = dest.resolve()
        # From Python 3.13 resolve() returns a loop instead of raising; any other
        # resolved path is free of links.
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise AuthError(loop)
    except OSError as e:
        raise AuthError(f"cannot write cookie file: {target}: {e.strerror}") from e
    except RuntimeError as e:  # a symlink loop, before Python 3.13
        raise AuthError(loop) from e
    return target


def _leads_to(path: Path, target: Path) -> bool:
    """Whether opening `path` as given reaches the file at `target`."""
    try:
        return path.samefile(target)
    except OSError:
        return False


def atomic_write_0600(dest: Path, content: str) -> None:
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
    """One Cookie-Editor row: {"name": ..., "value": ..., ...}."""
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
    """Read *.perplexity.ai cookies from a local browser's cookie store and
    write them to the file `load_cookies` reads ($PPLX_COOKIES_PATH, else the
    profile file). Atomic (tmp + rename), mode 0600.

    yt-dlp reads the store: Keychain on macOS, GNOME keyring or KWallet on
    Linux, DPAPI on Windows, a copy of a locked database. Any failure there is
    an AuthError naming the browser and the cause.

    Unlike a cookie file, a bad row is skipped with a warning: the jar holds
    third-party cookies the user cannot edit, and one of them must not block
    the import.
    """
    if browser not in SUPPORTED_BROWSERS:
        supported = ", ".join(SUPPORTED_BROWSERS)
        raise AuthError(f"unsupported browser: {browser!r} (supported: {supported})")

    dest = cookie_write_path(
        profile,
        inline_refusal=(
            "$PPLX_COOKIES is set and overrides any cookie file, so an import "
            "would not be used; it was not changed. Replace or unset $PPLX_COOKIES"
        ),
    )

    if browser in _FIREFOX_FORKS:
        source, browser_profile = "firefox", _firefox_fork_root(browser)
    else:
        source, browser_profile = browser, None
    problems: list[str] = []
    try:
        rows = [c for c in _extract_jar(source, browser_profile, problems) if _for_site(c.domain)]
    except Exception as e:
        cause = "; ".join([str(e) or type(e).__name__, *problems])
        if browser == "safari" and isinstance(e, OSError):
            cause += " (reading Safari's cookies needs Full Disk Access for this terminal)"
        raise AuthError(f"cannot read {browser} cookies: {cause}") from e

    chosen: dict[str, tuple[tuple[bool, str, str], str]] = {}
    for row in rows:
        try:
            name, value = _cookie_pair(
                row.name, row.value, where=f"cookie for {row.domain} from {browser}"
            )
        except AuthError as e:
            print(f"warning: skipping {e}", file=sys.stderr)
            continue
        # A flat file holds one value per name. Prefer the cookie a browser
        # sends to the root of BASE_URL; among equals keep the last in
        # (domain, path) order, as `_normalize` keeps the last repeat, so
        # www.perplexity.ai beats .perplexity.ai. The jar's own order differs
        # across Python versions. yt-dlp has already merged container and
        # partition copies of one (domain, path, name).
        rank = (_sent_to_request_root(row), row.domain, row.path)
        if name not in chosen or rank >= chosen[name][0]:
            chosen[name] = (rank, value)
    cookies = {name: value for name, (_, value) in chosen.items()}

    if not cookies:
        if problems:
            raise AuthError(
                f"no usable cookies for *.{_COOKIE_DOMAIN} in {browser}: " + "; ".join(problems)
            )
        raise AuthError(
            f"no usable cookies for *.{_COOKIE_DOMAIN} in {browser}; "
            f"sign in at perplexity.ai in {browser} first"
        )
    for problem in problems:
        print(f"warning: {browser}: {problem}", file=sys.stderr)

    return save_cookies(cookies, dest=dest)


def _extract_jar(browser: str, profile: str | None, problems: list[str]) -> Iterable[Cookie]:
    """Every cookie in the browser's store; the one call into yt-dlp.

    yt-dlp's diagnostics never reach stdout. The ones that explain missing
    cookies are appended to `problems`, each once.
    """
    # Imported here so that no other verb pays for loading yt-dlp.
    from yt_dlp.cookies import YDLLogger, extract_cookies_from_browser

    def note(message: str) -> None:
        if message not in problems:
            problems.append(message)

    class Log(YDLLogger):
        def info(self, message: str) -> None:
            # The only count of cookies that failed to decrypt.
            if "could not be decrypted" in message:
                note(message)

        def warning(self, message: str, only_once: bool = False) -> None:  # noqa: ARG002 - note() keeps each once
            note(message)

        def error(self, message: str) -> None:
            note(message)

    return extract_cookies_from_browser(browser, profile, Log())


def _firefox_fork_root(browser: str) -> str:
    """The directory holding a Firefox fork's profiles.ini.

    A directory without profiles.ini does not count: LibreWolf users create
    ~/.librewolf for librewolf.overrides.cfg alone, and LibreWolf then uses
    its XDG directory. The order is the browser's: the legacy dot directory
    wins over the XDG one.
    """
    home = Path.home()
    if sys.platform == "darwin":
        candidates = [home / "Library" / "Application Support" / browser]
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        candidates = [Path(appdata) / browser] if appdata else []
    else:
        config = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
        flatpak = home / ".var" / "app" / _FIREFOX_FORKS[browser]
        candidates = [
            home / f".{browser}",
            config / browser / browser,
            flatpak / f".{browser}",
            flatpak / "config" / browser / browser,
        ]
    for root in candidates:
        if (root / "profiles.ini").is_file():
            return str(root)
    looked = f"looked in {', '.join(map(str, candidates))}" if candidates else "$APPDATA is not set"
    raise AuthError(f"{browser} profile directory not found ({looked})")


def _for_site(domain: str) -> bool:
    """Whether a cookie domain is perplexity.ai or a subdomain of it."""
    host = domain.lstrip(".").lower()
    return host == _COOKIE_DOMAIN or host.endswith(f".{_COOKIE_DOMAIN}")


def _sent_to_request_root(cookie: Cookie) -> bool:
    """Whether a browser sends `cookie` with a request for https://www.perplexity.ai/.

    A stored domain with a leading dot came from a Domain attribute and
    matches subdomains; one without is host-only (RFC 6265 5.3 and 5.4).
    """
    domain = cookie.domain.lower()
    if domain.startswith("."):
        host_ok = domain[1:] == _REQUEST_HOST or _REQUEST_HOST.endswith(domain)
    else:
        host_ok = domain == _REQUEST_HOST
    return host_ok and cookie.path in ("", "/") and not cookie.is_expired()
