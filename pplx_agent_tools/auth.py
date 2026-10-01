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
import time
from collections.abc import Iterable
from dataclasses import dataclass
from http.cookiejar import Cookie
from pathlib import Path
from typing import NamedTuple, cast

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


class _Fork(NamedTuple):
    flatpak_id: str
    # Linux profile root under $XDG_CONFIG_HOME.
    xdg_root: tuple[str, ...]


# Firefox forks whose cookie store yt-dlp reads as Firefox's, given the
# fork's profile root. Zen's XDG root drops Firefox's vendor level.
_FIREFOX_FORKS = {
    "librewolf": _Fork("io.gitlab.librewolf-community", ("librewolf", "librewolf")),
    "zen": _Fork("app.zen_browser.zen", ("zen",)),
}

# yt-dlp's warnings that reading the browser's key failed. Like its errors they
# refuse the import even when cookies decrypted: after a KWallet failure yt-dlp
# decrypts with an empty password, which can be the wrong key.
_NO_KEY = (
    "find-generic-password failed",
    "exception running find-generic-password",
    "exception running kwallet-query",
)
# Logged at debug level, then yt-dlp decrypts with an empty password.
_KWALLET_EMPTY_PASSWORD = "failed to read password from kwallet. Using empty string instead"
# In yt-dlp's count of cookies it skipped.
_UNDECRYPTED = "could not be decrypted"

# NextAuth's session cookie. A token too large for one cookie is split into
# <name>.0, <name>.1, ..., which the server joins in order.
_SESSION_COOKIE = "__Secure-next-auth.session-token"

# Seconds from 1601-01-01, Chromium's time origin, to the Unix epoch.
_CHROMIUM_EPOCH_OFFSET = 11_644_473_600

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


def import_from_browser(
    browser: str, profile: str | None = None, browser_profile: str | None = None
) -> Path:
    """Read *.perplexity.ai cookies from a local browser's cookie store and
    write them to the file `load_cookies` reads ($PPLX_COOKIES_PATH, else the
    profile file). Atomic (tmp + rename), mode 0600.

    `browser_profile` is the browser profile to read, by directory name or
    path, or for Safari the path of a Cookies.binarycookies file; without it
    yt-dlp reads the profile whose cookie database changed last.

    yt-dlp reads the store: Keychain on macOS, GNOME keyring or KWallet on
    Linux, DPAPI on Windows, a copy of a locked database. Any failure there is
    an AuthError naming the browser and the cause, and so is a store whose key
    yt-dlp could not get (see `_extract_jar`) and an import without a session
    token (see `_has_session`); nothing is written then.

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

    if browser_profile is not None and browser_profile.startswith("~"):
        browser_profile = str(Path(browser_profile).expanduser())
    if browser in _FIREFOX_FORKS:
        source, where = "firefox", _fork_profile(browser, browser_profile)
    else:
        source, where = browser, browser_profile

    def named(message: str) -> str:
        # yt-dlp reads a fork's store as Firefox's and says "firefox".
        return message.replace(f"{source} cookies", f"{browser} cookies").replace(
            f"from {source}", f"from {browser}"
        )

    problems: list[str] = []
    try:
        rows = [c for c in _extract_jar(source, where, problems) if _for_site(c.domain)]
    except Exception as e:
        # yt-dlp often logs the message it then raises.
        causes = dict.fromkeys(named(m) for m in (str(e) or type(e).__name__, *problems))
        cause = "; ".join(causes)
        # A Safari profile is a file path, so not finding it means a wrong path.
        if (
            browser == "safari"
            and isinstance(e, OSError)
            and not (browser_profile and isinstance(e, FileNotFoundError))
        ):
            cause += " (reading Safari's cookies needs Full Disk Access for this terminal)"
        raise AuthError(f"cannot read {browser} cookies: {cause}") from e
    problems = [named(p) for p in problems]

    # yt-dlp gives Firefox's and Safari's expiry in Unix time, Chromium's as stored.
    chromium = source not in ("firefox", "safari")
    chosen: dict[str, tuple[tuple[bool, bool, bool, str, str], str]] = {}
    for row in rows:
        try:
            name, value = _cookie_pair(
                row.name, row.value, where=f"cookie for {row.domain} from {browser}"
            )
        except AuthError as e:
            print(f"warning: skipping {e}", file=sys.stderr)
            continue
        expires = _chromium_expiry(row.expires) if chromium else row.expires
        # A flat file holds one value per name. Prefer a non-empty value: an
        # empty one authenticates nothing, and is what a wrong key most often
        # decrypts to. Then an unexpired one, whatever its path. Then the
        # cookie a browser sends to the root of BASE_URL; among equals keep the
        # last in (domain, path) order, as `_normalize` keeps the last repeat,
        # so www.perplexity.ai beats .perplexity.ai. The jar's own order
        # differs across Python versions. yt-dlp has already merged partition
        # copies of one (domain, path, name).
        unexpired = expires is None or expires > time.time()
        rank = (value != "", unexpired, _sent_to_request_root(row), row.domain, row.path)
        if name not in chosen or rank >= chosen[name][0]:
            chosen[name] = (rank, value)
    cookies = {name: value for name, (_, value) in chosen.items()}

    # Without it the other cookies authenticate nothing, and must not replace
    # a working file. A wrong key all but never decrypts it to a value that loads.
    if not _has_session(cookies):
        cause = "; ".join(problems) or f"sign in at perplexity.ai in {browser} first"
        raise AuthError(
            f"no usable {_SESSION_COOKIE} cookie for *.{_COOKIE_DOMAIN} in {browser}: {cause}"
        )
    for problem in problems:
        print(f"warning: {browser}: {problem}", file=sys.stderr)

    return save_cookies(cookies, dest=dest)


def _has_session(cookies: dict[str, str]) -> bool:
    """Whether `cookies` hold a non-empty, loadable session token: whole, or
    as every chunk from .0 up with none missing."""

    def present(name: str) -> bool:
        value = cookies.get(name, "")
        return value != "" and cookie_pair_ok(name, value)

    if present(_SESSION_COOKIE):
        return True
    prefix = f"{_SESSION_COOKIE}."
    count = sum(1 for n in cookies if n.startswith(prefix) and n[len(prefix) :].isdigit())
    return count > 0 and all(present(f"{prefix}{i}") for i in range(count))


class _YtDlpFailure(Exception):
    """yt-dlp logged an error, or that reading the key failed, and returned anyway."""


def _extract_jar(browser: str, profile: str | None, problems: list[str]) -> Iterable[Cookie]:
    """Every cookie in the browser's store, Firefox's outside any container;
    the one call into yt-dlp.

    yt-dlp's diagnostics never reach stdout. The ones that explain missing or
    doubtful cookies are appended to `problems`, each once.

    Raises if yt-dlp logs an error or one of `_NO_KEY`, though it returns
    cookies then. Cookies it skipped, for want of a key or otherwise, are not
    a failure: whether a session token remains decides.
    """
    # Imported here so that no other verb pays for loading yt-dlp.
    from yt_dlp.cookies import YDLLogger, extract_cookies_from_browser

    failures: list[str] = []

    def note(message: str, *, failed: bool = False) -> None:
        if message not in problems:
            problems.append(message)
        if failed:
            failures.append(message)

    class Log(YDLLogger):
        def debug(self, message: str) -> None:
            # Not a failure: yt-dlp's comment there says Chrome on KDE "does not
            # check hasEntry and instead just tries to read the value (which
            # kwallet returns "")", so Chrome encrypted with the same empty
            # password. A wrong key leaves no session token.
            if message == _KWALLET_EMPTY_PASSWORD:
                note(message)

        def info(self, message: str) -> None:
            # The only count of cookies that failed to decrypt.
            if _UNDECRYPTED in message:
                note(message)

        def warning(self, message: str, only_once: bool = False) -> None:  # noqa: ARG002 - note() keeps each once
            note(message, failed=message.startswith(_NO_KEY))

        def error(self, message: str) -> None:
            note(message, failed=True)

    # Firefox only reads `container`; "none" leaves out container copies,
    # which yt-dlp would otherwise merge, keeping the last row.
    jar: Iterable[Cookie] = extract_cookies_from_browser(browser, profile, Log(), container="none")
    if failures:
        raise _YtDlpFailure(failures[0])
    return jar


def _is_path(value: str) -> bool:
    """Whether yt-dlp takes a profile argument as a path rather than a name."""
    return any(sep in value for sep in (os.sep, os.altsep) if sep)


def _fork_profile(browser: str, browser_profile: str | None) -> str:
    """The profile path to give yt-dlp for a Firefox fork it reads as Firefox.

    yt-dlp looks a profile name up under Firefox's directories, not the fork's.
    """
    if browser_profile is not None and _is_path(browser_profile):
        return browser_profile
    root = Path(_firefox_fork_root(browser))
    if browser_profile is None:
        return str(root)
    # Profiles sit under Profiles/ on macOS and Windows, beside profiles.ini on Linux.
    if sys.platform in ("darwin", "win32"):
        root /= "Profiles"
    return str(root / browser_profile)


def _firefox_fork_root(browser: str) -> str:
    """The directory holding a Firefox fork's profiles.ini.

    The order is the browser's: it uses the legacy dot directory whenever that
    exists (LibreWolf's FAQ: ~/.librewolf "always takes precedence"), else the
    XDG one. A directory without profiles.ini is passed over: the browser has
    not run since it was created (LibreWolf users create ~/.librewolf for
    librewolf.overrides.cfg alone), so its last profile is in the next one.
    """
    home = Path.home()
    if sys.platform == "darwin":
        candidates = [home / "Library" / "Application Support" / browser]
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        candidates = [Path(appdata) / browser] if appdata else []
    else:
        config = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
        fork = _FIREFOX_FORKS[browser]
        flatpak = home / ".var" / "app" / fork.flatpak_id
        candidates = [
            home / f".{browser}",
            config.joinpath(*fork.xdg_root),
            flatpak / f".{browser}",
            flatpak.joinpath("config", *fork.xdg_root),
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


def _chromium_expiry(expires: float | None) -> float | None:
    """Chromium's expires_utc, microseconds since 1601-01-01 or 0 for a session
    cookie, in Unix time; None for a session cookie."""
    return expires / 1_000_000 - _CHROMIUM_EPOCH_OFFSET if expires else None


def _sent_to_request_root(cookie: Cookie) -> bool:
    """Whether a browser sends `cookie`, unless expired, with a request for
    https://www.perplexity.ai/.

    A stored domain with a leading dot came from a Domain attribute and
    matches subdomains; one without is host-only (RFC 6265 5.3 and 5.4).
    """
    domain = cookie.domain.lower()
    if domain.startswith("."):
        host_ok = domain[1:] == _REQUEST_HOST or _REQUEST_HOST.endswith(domain)
    else:
        host_ok = domain == _REQUEST_HOST
    return host_ok and cookie.path in ("", "/")
