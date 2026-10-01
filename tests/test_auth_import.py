"""Browser import: `pplx auth import` and `auth.read_browser_cookies`.

Most tests replace `auth._extract_jar`, the one call into yt-dlp. The ones
that run yt-dlp point it at tmp_path, with a stand-in keyring where it needs
one, so no test reads a real browser's cookie store or key. A stand-in for
perplexity.ai answers the session check, so none reaches the network.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from http.cookiejar import Cookie
from pathlib import Path
from typing import Any

import pytest

from pplx_agent_tools import auth, cli_auth
from pplx_agent_tools.auth import (
    SUPPORTED_BROWSERS,
    default_cookies_path,
    load_cookies,
    read_browser_cookies,
)
from pplx_agent_tools.errors import EXIT_AUTH, EXIT_GENERIC, EXIT_NETWORK, AuthError
from tests._doubles import _TestClientBase

FAR_FUTURE = 4102444800  # 2100-01-01
PAST = 1577836800  # 2020-01-01
CHROMIUM_BROWSERS = ("brave", "chrome", "chromium", "edge", "opera", "vivaldi")


def _chromium_time(unix: int) -> int:
    """Chromium's expires_utc for a Unix time: microseconds since 1601-01-01."""
    return (unix + 11_644_473_600) * 1_000_000


def _unix_time(unix: int) -> int:
    return unix


@pytest.fixture(autouse=True)
def _own_cookie_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cookie file per test, so a missing one proves nothing was written."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


SESSION_URL = "https://www.perplexity.ai/api/auth/session"
SIGNED_IN = {
    "user": {"id": "0a0a0a0a-1111-4222-8333-444444444444", "email": "me@example.com"},
    "expires": "2099-01-01T00:00:00.000Z",
}


class _Answer:
    def __init__(self, status: int, body: object) -> None:
        self.status_code = status
        self.headers = {"content-type": "application/json; charset=utf-8"}
        self.content = json.dumps(body).encode()
        self._body = body

    def json(self) -> object:
        return self._body


class _Server:
    """curl_cffi's session under a real `Client`, standing in for perplexity.ai:
    every GET gets `status` and `body`, or raises `error`; a 200 also sets the
    `rotate` cookies, as Set-Cookie would."""

    def __init__(self) -> None:
        self.status = 200
        self.body: object = SIGNED_IN
        self.error: Exception | None = None
        self.rotate: dict[str, str] = {}
        self.cookies: dict[str, str] = {}
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.on_request: Callable[[], None] = lambda: None

    def get(self, url: str, *, cookies: dict[str, str], **_: object) -> _Answer:
        self.requests.append((url, dict(cookies)))
        self.on_request()
        if self.error is not None:
            raise self.error
        if self.status == 200:
            self.cookies.update(self.rotate)
        return _Answer(self.status, self.body)


class _CheckedClient(_TestClientBase):
    """A real `Client` over `_Server`: its `auth_session` judges the answer and
    captures rotated cookies as it would against perplexity.ai."""

    def __init__(self, cookies: dict[str, str], server: _Server) -> None:
        super().__init__(cookies)
        self._session = server  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def server(monkeypatch: pytest.MonkeyPatch) -> _Server:
    """perplexity.ai as `pplx auth import` reaches it: by default it accepts the session."""
    srv = _Server()
    monkeypatch.setattr(cli_auth, "Client", lambda cookies: _CheckedClient(cookies, srv))
    return srv


def _cookie(
    name: str,
    value: str | None,
    domain: str = ".perplexity.ai",
    path: str = "/",
    # None, a session cookie, means the same in every browser's expiry format.
    expires: int | None = None,
) -> Cookie:
    return Cookie(
        version=0,
        name=name,
        value=value,
        port=None,
        port_specified=False,
        domain=domain,
        domain_specified=True,
        domain_initial_dot=domain.startswith("."),
        path=path,
        path_specified=True,
        secure=True,
        expires=expires,
        discard=False,
        comment=None,
        comment_url=None,
        rest={},
    )


TOKEN = "__Secure-next-auth.session-token"
SESSION = _cookie(TOKEN, "tok", "www.perplexity.ai")


def _fake_extractor(
    monkeypatch: pytest.MonkeyPatch,
    rows: Sequence[Cookie] = (),
    *,
    problems: Sequence[str] = (),
    raises: Exception | None = None,
) -> list[tuple[str, str | None]]:
    """Replace the yt-dlp call; returns the (browser, profile) of each call."""
    calls: list[tuple[str, str | None]] = []

    def extract(browser: str, profile: str | None, noted: list[str]) -> Sequence[Cookie]:
        calls.append((browser, profile))
        noted.extend(problems)
        if raises is not None:
            raise raises
        return rows

    monkeypatch.setattr(auth, "_extract_jar", extract)
    return calls


def _saved() -> dict[str, str]:
    return json.loads(default_cookies_path().read_text())


def _working_file() -> str:
    """A cookie file from an earlier import; returns its text."""
    dest = default_cookies_path()
    dest.parent.mkdir(parents=True)
    dest.write_text('{"__Secure-next-auth.session-token": "PREVIOUS-GOOD"}')
    dest.chmod(0o600)
    return dest.read_text()


def _scripted_yt_dlp(
    monkeypatch: pytest.MonkeyPatch, log: Callable[[Any], None], rows: Sequence[Cookie]
) -> list[tuple[str, str | None]]:
    """Replace yt-dlp's extractor under the real `_extract_jar`: it runs `log`
    on the logger it is given, then returns `rows`."""
    import yt_dlp.cookies

    calls: list[tuple[str, str | None]] = []

    def extract(browser: str, profile: str | None, logger: object, **_: object) -> list[Cookie]:
        calls.append((browser, profile))
        log(logger)
        return list(rows)

    monkeypatch.setattr(yt_dlp.cookies, "extract_cookies_from_browser", extract)
    return calls


# ---------- supported browsers ----------


def test_supported_browsers() -> None:
    assert set(SUPPORTED_BROWSERS) == {
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
    }


def test_arc_is_a_usage_error_naming_the_choices(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        cli_auth.main(["import", "--browser", "arc"])
    assert ei.value.code == EXIT_GENERIC
    err = capsys.readouterr().err
    assert "'arc'" in err
    for name in SUPPORTED_BROWSERS:
        assert name in err


def test_import_help_lists_the_browsers(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli_auth.main(["import", "--help"])
    out = capsys.readouterr().out
    assert "{" + ",".join(SUPPORTED_BROWSERS) + "}" in out
    assert "--browser-profile NAME_OR_PATH" in out
    assert "--no-verify" in out


def test_unsupported_browser_from_python_is_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_extractor(monkeypatch)
    with pytest.raises(AuthError, match="unsupported browser: 'arc'"):
        read_browser_cookies("arc")
    assert calls == []


@pytest.mark.parametrize(
    "browser", [b for b in SUPPORTED_BROWSERS if b not in ("librewolf", "zen")]
)
def test_browser_is_passed_to_the_extractor_by_name(
    monkeypatch: pytest.MonkeyPatch, browser: str
) -> None:
    calls = _fake_extractor(monkeypatch, [SESSION])
    read_browser_cookies(browser)
    assert calls == [(browser, None)]


@pytest.mark.parametrize("browser", ["librewolf", "zen"])
def test_firefox_fork_is_read_as_firefox_from_its_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, browser: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    root = tmp_path / "Library" / "Application Support" / browser
    root.mkdir(parents=True)
    (root / "profiles.ini").touch()
    calls = _fake_extractor(monkeypatch, [SESSION])
    read_browser_cookies(browser)
    assert calls == [("firefox", str(root))]


@pytest.mark.parametrize(
    ("browser", "given"),
    [
        ("chrome", "Profile 1"),
        ("chrome", "/somewhere/Profile 1"),
        ("firefox", "abc.default-release"),
        ("firefox", "/somewhere/abc.default-release"),
        # yt-dlp reads Safari's as the path of a cookie file.
        ("safari", "/somewhere/Cookies.binarycookies"),
    ],
)
def test_browser_profile_is_passed_to_the_extractor(
    monkeypatch: pytest.MonkeyPatch, browser: str, given: str
) -> None:
    calls = _fake_extractor(monkeypatch, [SESSION])
    read_browser_cookies(browser, browser_profile=given)
    assert calls == [(browser, given)]


def test_browser_profile_path_expands_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = _fake_extractor(monkeypatch, [SESSION])
    read_browser_cookies("chrome", browser_profile="~/chrome/Default")
    assert calls == [("chrome", str(tmp_path / "chrome" / "Default"))]


def test_browser_profile_reaches_import_from_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_extractor(monkeypatch, [SESSION])
    assert cli_auth.main(["import", "--browser", "edge", "--browser-profile", "Work"]) == 0
    assert calls == [("edge", "Work")]


@pytest.mark.parametrize(
    ("platform", "nested"), [("darwin", ("Profiles", "abc.work")), ("linux", ("abc.work",))]
)
def test_fork_profile_name_is_found_under_the_fork_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, nested: tuple[str, ...]
) -> None:
    # yt-dlp would look a bare name up under Firefox's own directories.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", platform)
    root = (
        tmp_path / "Library" / "Application Support" / "zen"
        if platform == "darwin"
        else tmp_path / ".zen"
    )
    root.joinpath(*nested).mkdir(parents=True)
    (root / "profiles.ini").touch()
    calls = _fake_extractor(monkeypatch, [SESSION])
    read_browser_cookies("zen", browser_profile="abc.work")
    assert calls == [("firefox", str(root.joinpath(*nested)))]


def test_fork_profile_path_needs_no_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = _fake_extractor(monkeypatch, [SESSION])
    read_browser_cookies("librewolf", browser_profile="/elsewhere/abc.default")
    assert calls == [("firefox", "/elsewhere/abc.default")]


@pytest.mark.skipif(sys.platform == "win32", reason="profile roots laid out for macOS and Linux")
def test_missing_browser_profile_is_auth_error_naming_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    (tmp_path / "xdg" / "chromium" / "Default").mkdir(parents=True)
    rc = cli_auth.main(["import", "--browser", "chromium", "--browser-profile", "Nope"])
    assert rc == EXIT_AUTH
    err = capsys.readouterr().err
    assert "cannot read chromium cookies: could not find chromium cookies database" in err
    assert str(tmp_path / "xdg" / "chromium" / "Nope") in err
    assert not default_cookies_path().exists()


@pytest.mark.skipif(sys.platform == "win32", reason="profile roots laid out for macOS and Linux")
@pytest.mark.parametrize("given", [None, "nope"])
def test_fork_store_errors_name_the_fork(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, given: str | None
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    root = tmp_path / ".zen"
    root.mkdir()
    (root / "profiles.ini").touch()
    with pytest.raises(AuthError) as ei:
        read_browser_cookies("zen", browser_profile=given)
    assert str(ei.value).startswith(
        "cannot read zen cookies: could not find zen cookies database in "
    )
    assert "firefox" not in str(ei.value)


@pytest.mark.skipif(sys.platform == "win32", reason="profile roots laid out for macOS and Linux")
@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_missing_fork_profile_names_the_platforms_profile_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setattr(sys, "platform", platform)
    root = {
        "darwin": tmp_path / "Library" / "Application Support" / "zen",
        "linux": tmp_path / ".zen",
        "win32": tmp_path / "AppData" / "zen",
    }[platform]
    root.mkdir(parents=True)
    (root / "profiles.ini").touch()
    looked = root / "nope" if platform == "linux" else root / "Profiles" / "nope"
    with pytest.raises(AuthError) as ei:
        read_browser_cookies("zen", browser_profile="nope")
    assert str(ei.value) == (
        f"cannot read zen cookies: could not find zen cookies database in {str(looked)!r}"
    )


# ---------- Firefox fork profile roots ----------


def _linux_roots(home: Path, config: Path, browser: str) -> list[Path]:
    """Where a fork keeps profiles.ini on Linux, in the order it looks."""
    if browser == "librewolf":
        flatpak = home / ".var" / "app" / "io.gitlab.librewolf-community"
        return [
            home / ".librewolf",
            config / "librewolf" / "librewolf",
            flatpak / ".librewolf",
            flatpak / "config" / "librewolf" / "librewolf",
        ]
    # Zen's XDG root has no vendor level.
    flatpak = home / ".var" / "app" / "app.zen_browser.zen"
    return [home / ".zen", config / "zen", flatpak / ".zen", flatpak / "config" / "zen"]


@pytest.mark.parametrize("browser", ["librewolf", "zen"])
@pytest.mark.parametrize("index", [0, 1, 2, 3], ids=["legacy", "xdg", "flatpak", "flatpak-xdg"])
def test_fork_root_on_linux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, browser: str, index: int
) -> None:
    home, config = tmp_path / "home", tmp_path / "config"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setattr(sys, "platform", "linux")
    root = _linux_roots(home, config, browser)[index]
    root.mkdir(parents=True)
    (root / "profiles.ini").touch()
    assert auth._firefox_fork_root(browser) == str(root)


@pytest.mark.parametrize(
    ("browser", "under"), [("zen", ("zen",)), ("librewolf", ("librewolf", "librewolf"))]
)
def test_fork_root_on_linux_defaults_xdg_to_dot_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, browser: str, under: tuple[str, ...]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setattr(sys, "platform", "linux")
    root = tmp_path.joinpath(".config", *under)
    root.mkdir(parents=True)
    (root / "profiles.ini").touch()
    assert auth._firefox_fork_root(browser) == str(root)


def test_fork_root_legacy_wins_but_needs_profiles_ini(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setattr(sys, "platform", "linux")
    legacy, xdg = _linux_roots(tmp_path, tmp_path / ".config", "librewolf")[:2]
    xdg.mkdir(parents=True)
    (xdg / "profiles.ini").touch()
    # LibreWolf's docs have users create ~/.librewolf for the overrides file alone.
    legacy.mkdir()
    (legacy / "librewolf.overrides.cfg").touch()
    assert auth._firefox_fork_root("librewolf") == str(xdg)
    (legacy / "profiles.ini").touch()
    assert auth._firefox_fork_root("librewolf") == str(legacy)


def test_fork_root_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    root = tmp_path / "zen"
    root.mkdir()
    (root / "profiles.ini").touch()
    assert auth._firefox_fork_root("zen") == str(root)


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
@pytest.mark.parametrize("browser", ["librewolf", "zen"])
def test_missing_fork_root_is_auth_error_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    platform: str,
    browser: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    calls = _fake_extractor(monkeypatch, [SESSION])
    assert cli_auth.main(["import", "--browser", browser]) == EXIT_AUTH
    assert f"{browser} profile directory not found" in capsys.readouterr().err
    assert calls == []
    assert not default_cookies_path().exists()


# ---------- which cookies are written ----------


def test_import_saves_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extractor(monkeypatch, [_cookie("a", "1"), SESSION])
    assert cli_auth.main(["import", "--browser", "brave"]) == 0
    dest = default_cookies_path()
    assert json.loads(dest.read_text()) == {"a": "1", TOKEN: "tok"}
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "domain",
    [
        "perplexity.ai",
        ".perplexity.ai",
        "www.perplexity.ai",
        ".www.perplexity.ai",
        "api.www.perplexity.ai",
        "WWW.Perplexity.AI",
    ],
)
def test_domain_on_label_boundary_matches(monkeypatch: pytest.MonkeyPatch, domain: str) -> None:
    _fake_extractor(monkeypatch, [_cookie(TOKEN, "1", domain)])
    assert read_browser_cookies("chrome") == {TOKEN: "1"}


@pytest.mark.parametrize(
    "domain",
    [
        "notperplexity.ai",
        ".notperplexity.ai",
        "perplexity.ai.evil.com",
        ".perplexity.ai.evil.com",
        "perplexity.aix",
        "perplexity",
        "",
    ],
)
def test_domain_off_label_boundary_is_ignored(monkeypatch: pytest.MonkeyPatch, domain: str) -> None:
    _fake_extractor(monkeypatch, [SESSION, _cookie("other", "SECRET", domain)])
    assert read_browser_cookies("chrome") == {TOKEN: "tok"}


def _pick(monkeypatch: pytest.MonkeyPatch, rows: list[Cookie], browser: str = "firefox") -> str:
    """The value written for name "n", whichever order the rows arrive in."""
    picked: set[str] = set()
    for order in (rows, rows[::-1]):
        _fake_extractor(monkeypatch, [*order, SESSION])
        picked.add(read_browser_cookies(browser)["n"])
    assert len(picked) == 1, f"choice depends on row order: {picked}"
    return picked.pop()


def test_duplicate_prefers_cookie_sent_to_www_over_host_only_apex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_cookie("n", "apex", "perplexity.ai"), _cookie("n", "domain", ".perplexity.ai")]
    assert _pick(monkeypatch, rows) == "domain"


def test_duplicate_prefers_root_path(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _cookie("n", "rest", "www.perplexity.ai", "/rest"),
        _cookie("n", "root", ".perplexity.ai", "/"),
    ]
    assert _pick(monkeypatch, rows) == "root"


# yt-dlp gives Firefox's and Safari's expiry in Unix time and Chromium's as stored.
_STORED_TIME = [
    *((b, _chromium_time) for b in CHROMIUM_BROWSERS),
    ("firefox", _unix_time),
    ("safari", _unix_time),
]


@pytest.mark.parametrize(("browser", "stored"), _STORED_TIME)
def test_duplicate_prefers_unexpired(
    monkeypatch: pytest.MonkeyPatch, browser: str, stored: Callable[[int], int]
) -> None:
    rows = [
        _cookie("n", "stale", "www.perplexity.ai", expires=stored(PAST)),
        _cookie("n", "live", ".perplexity.ai", expires=stored(FAR_FUTURE)),
    ]
    assert _pick(monkeypatch, rows, browser) == "live"


@pytest.mark.parametrize(("browser", "stored"), _STORED_TIME)
@pytest.mark.parametrize(
    ("stale_at", "live_at"),
    [
        (("www.perplexity.ai", "/api"), (".perplexity.ai", "/api")),
        (("www.perplexity.ai", "/"), (".perplexity.ai", "/api")),
    ],
    ids=["neither-at-root", "expired-at-root"],
)
def test_expired_never_beats_unexpired_whatever_the_paths(
    monkeypatch: pytest.MonkeyPatch,
    browser: str,
    stored: Callable[[int], int],
    stale_at: tuple[str, str],
    live_at: tuple[str, str],
) -> None:
    rows = [
        _cookie("n", "stale", *stale_at, expires=stored(PAST)),
        _cookie("n", "live", *live_at, expires=stored(FAR_FUTURE)),
    ]
    assert _pick(monkeypatch, rows, browser) == "live"


@pytest.mark.parametrize(
    ("browser", "expires"), [("chrome", None), ("chrome", 0), ("firefox", None), ("safari", None)]
)
def test_duplicate_session_cookie_counts_as_unexpired(
    monkeypatch: pytest.MonkeyPatch, browser: str, expires: int | None
) -> None:
    rows = [
        _cookie("n", "session", ".perplexity.ai", expires=expires),
        _cookie("n", "apex", "perplexity.ai"),
    ]
    assert _pick(monkeypatch, rows, browser) == "session"


def test_duplicate_prefers_a_value_over_an_empty_one(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_cookie("n", "", "www.perplexity.ai"), _cookie("n", "v", "perplexity.ai", "/api")]
    assert _pick(monkeypatch, rows) == "v"


_NO_SESSION = re.escape(f"no usable {TOKEN} cookie for *.perplexity.ai in ") + "{browser}: "


def test_only_empty_values_write_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _working_file()
    _fake_extractor(monkeypatch, [_cookie("a", ""), _cookie(TOKEN, "", "www.perplexity.ai")])
    assert cli_auth.main(["import", "--browser", "chrome"]) == EXIT_AUTH
    assert re.search(_NO_SESSION.format(browser="chrome"), capsys.readouterr().err)
    assert default_cookies_path().read_text() == before


@pytest.mark.parametrize(
    "rows",
    [
        [_cookie("pplx.visitor-id", "v"), _cookie("__cf_bm", "cf")],
        [_cookie("pplx.visitor-id", "v"), _cookie(TOKEN, "")],
        [_cookie("pplx.visitor-id", "v"), _cookie(TOKEN, "x;y")],
    ],
    ids=["missing", "empty", "unloadable"],
)
def test_no_session_token_leaves_the_file_untouched(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], rows: list[Cookie]
) -> None:
    # Cookies that authenticate nothing must not replace a working file.
    before = _working_file()
    _fake_extractor(monkeypatch, rows)
    assert cli_auth.main(["import", "--browser", "brave"]) == EXIT_AUTH
    out, err = capsys.readouterr()
    assert out == ""
    assert err.endswith(
        f"pplx auth import: no usable {TOKEN} cookie for *.perplexity.ai in brave: "
        "sign in at perplexity.ai in brave first\n"
    )
    assert default_cookies_path().read_text() == before


@pytest.mark.parametrize("count", [1, 3])
def test_chunked_session_token_is_accepted(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    # NextAuth splits a large token into cookies named <name>.0, <name>.1, ...
    chunks = {f"{TOKEN}.{i}": f"part{i}" for i in range(count)}
    _fake_extractor(monkeypatch, [_cookie(n, v) for n, v in chunks.items()])
    assert read_browser_cookies("chrome") == chunks


@pytest.mark.parametrize(
    "chunks",
    [{".1": "b"}, {".1": "b", ".2": "c"}, {".0": "", ".1": "b"}],
    ids=["one", "two", "empty-first"],
)
def test_chunked_session_token_without_its_first_chunk_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], chunks: dict[str, str]
) -> None:
    before = _working_file()
    _fake_extractor(monkeypatch, [_cookie(TOKEN + n, v) for n, v in chunks.items()])
    assert cli_auth.main(["import", "--browser", "chrome"]) == EXIT_AUTH
    assert re.search(_NO_SESSION.format(browser="chrome"), capsys.readouterr().err)
    assert default_cookies_path().read_text() == before


def test_duplicate_both_sent_keeps_last_in_domain_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_cookie("n", "domain", ".perplexity.ai"), _cookie("n", "www", "www.perplexity.ai")]
    assert _pick(monkeypatch, rows) == "www"


def test_duplicate_neither_sent_keeps_last_in_domain_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _cookie("n", "api", ".perplexity.ai", "/api"),
        _cookie("n", "apex", "perplexity.ai", "/api"),
    ]
    assert _pick(monkeypatch, rows) == "apex"


def test_duplicate_with_unloadable_value_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [_cookie("n", "x;SECRET", "www.perplexity.ai"), _cookie("n", "ok", "perplexity.ai")]
    assert _pick(monkeypatch, rows) == "ok"
    assert "SECRET" not in capsys.readouterr().err


def test_import_skips_bad_rows_with_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        _cookie(TOKEN, "good"),
        _cookie("nullish", None),
        _cookie("crlf", "SECRET1\r\nX: y"),
        _cookie("tabbed", "SECRET2\tz"),
        _cookie("surr", "SECRET3\ud800"),
        _cookie("a;b", "SECRET4"),
    ]
    _fake_extractor(monkeypatch, rows)
    assert read_browser_cookies("brave") == {TOKEN: "good"}
    err = capsys.readouterr().err
    assert err.count("warning: skipping") == 5
    for name in ("nullish", "crlf", "tabbed", "surr"):
        assert repr(name) in err
    assert "has no value" in err
    assert "SECRET" not in err


def test_import_all_rows_bad_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_extractor(monkeypatch, [_cookie("a", "x\r\ny")])
    assert cli_auth.main(["import", "--browser", "brave"]) == EXIT_AUTH
    assert "sign in" in capsys.readouterr().err
    assert not default_cookies_path().exists()


def test_import_empty_says_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extractor(monkeypatch, [_cookie("other", "1", "example.com")])
    with pytest.raises(AuthError, match=r"sign in at perplexity\.ai in brave first"):
        read_browser_cookies("brave")


# ---------- destination: the file load_cookies reads ----------


def test_import_writes_to_cookies_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "custom" / "jar.json"
    monkeypatch.setenv("PPLX_COOKIES_PATH", str(target))
    _fake_extractor(monkeypatch, [SESSION])
    assert cli_auth.main(["import", "--browser", "brave"]) == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert load_cookies() == {TOKEN: "tok"}
    assert not default_cookies_path().exists()


def test_import_refuses_when_inline_env_overrides(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], server: _Server
) -> None:
    monkeypatch.setenv("PPLX_COOKIES", '{"a": "SECRET"}')
    calls = _fake_extractor(monkeypatch, [SESSION])
    assert cli_auth.main(["import", "--browser", "brave"]) == EXIT_AUTH
    msg = capsys.readouterr().err
    assert "$PPLX_COOKIES" in msg and "not changed" in msg
    assert "SECRET" not in msg
    assert (calls, server.requests) == ([], [])
    assert not default_cookies_path().exists()


# ---------- extractor failures ----------


@pytest.mark.parametrize(
    ("browser", "error", "cause"),
    [
        (
            "chrome",
            FileNotFoundError("could not find chrome cookies database in '/x'"),
            "could not find chrome cookies database",
        ),
        ("firefox", sqlite3.OperationalError("database is locked"), "database is locked"),
        ("brave", sqlite3.DatabaseError("file is not a database"), "file is not a database"),
        ("safari", ValueError("unsupported platform: linux"), "unsupported platform: linux"),
        ("edge", RuntimeError(), "RuntimeError"),
        ("vivaldi", ModuleNotFoundError("No module named 'yt_dlp'"), "No module named 'yt_dlp'"),
    ],
)
def test_extractor_exception_exits_two_naming_browser_and_cause(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    browser: str,
    error: Exception,
    cause: str,
) -> None:
    _fake_extractor(monkeypatch, raises=error)
    rc = cli_auth.main(["import", "--browser", browser])
    assert rc == EXIT_AUTH
    out, err = capsys.readouterr()
    assert out == ""
    assert f"cannot read {browser} cookies: {cause}" in err
    assert "Traceback" not in err
    assert not default_cookies_path().exists()


def test_extractor_exception_includes_logged_problems(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extractor(
        monkeypatch,
        problems=["secretstorage not available"],
        raises=OSError("no D-Bus session"),
    )
    with pytest.raises(AuthError) as ei:
        read_browser_cookies("chromium")
    assert str(ei.value) == (
        "cannot read chromium cookies: no D-Bus session; secretstorage not available"
    )


def test_cause_logged_and_raised_is_named_once(monkeypatch: pytest.MonkeyPatch) -> None:
    message = "Failed to decrypt with DPAPI"
    _fake_extractor(monkeypatch, problems=[message, "other"], raises=RuntimeError(message))
    with pytest.raises(AuthError) as ei:
        read_browser_cookies("edge")
    assert str(ei.value) == f"cannot read edge cookies: {message}; other"


def test_safari_os_error_names_full_disk_access(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extractor(monkeypatch, raises=PermissionError(1, "Operation not permitted"))
    with pytest.raises(AuthError, match="Full Disk Access"):
        read_browser_cookies("safari")


def test_safari_profile_not_found_is_a_wrong_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(AuthError) as ei:
        read_browser_cookies("safari", browser_profile=str(tmp_path / "Cookies.binarycookies"))
    assert str(ei.value) == "cannot read safari cookies: custom safari cookies database not found"


def test_keychain_refusal_exits_two_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # What yt-dlp logs when the Keychain prompt is refused: every encrypted
    # cookie fails to decrypt and the jar holds only the plaintext ones.
    def log(logger: Any) -> None:
        logger.warning("find-generic-password failed")
        logger.warning("cannot decrypt v10 cookies: no key found", only_once=True)
        logger.info("Extracted 1 cookies from chrome (40 could not be decrypted)")

    _scripted_yt_dlp(monkeypatch, log, [_cookie("plain", "1")])
    rc = cli_auth.main(["import", "--browser", "chrome"])
    assert rc == EXIT_AUTH
    out, err = capsys.readouterr()
    assert out == ""
    assert err == (
        "pplx auth import: cannot read chrome cookies: find-generic-password failed; "
        "cannot decrypt v10 cookies: no key found; "
        "Extracted 1 cookies from chrome (40 could not be decrypted)\n"
    )
    assert not default_cookies_path().exists()


# yt-dlp's messages when reading the store's key failed, by the logger method
# that carries them. On Linux it then decrypts with an empty password.
_KEY_FAILURES = [
    ("error", "failed to read from keyring"),
    ("error", "secretstorage not available No module named 'secretstorage'"),
    ("error", "kwallet-query command not found. KWallet and kwallet-query must be installed"),
    ("error", "kwallet-query failed with return code 1. Please consult the man page"),
    ("warning", "exception running kwallet-query: [Errno 2] No such file or directory"),
    ("warning", "find-generic-password failed"),
    ("warning", "exception running find-generic-password: timed out"),
    ("error", "could not find local state file"),
    ("error", "opera does not support profiles"),
]


@pytest.mark.parametrize(("level", "message"), _KEY_FAILURES)
def test_key_failure_refuses_even_a_session_token_and_keeps_the_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], level: str, message: str
) -> None:
    before = _working_file()
    _scripted_yt_dlp(
        monkeypatch,
        lambda logger: getattr(logger, level)(message),
        [SESSION, _cookie("pplx.visitor-id", "Zq3")],
    )
    rc = cli_auth.main(["import", "--browser", "chrome"])
    assert rc == EXIT_AUTH
    out, err = capsys.readouterr()
    assert out == ""
    assert err == f"pplx auth import: cannot read chrome cookies: {message}\n"
    assert default_cookies_path().read_text() == before


@pytest.mark.parametrize("version", ["v10", "v11"])
@pytest.mark.parametrize("has_session", [True, False], ids=["session", "no-session"])
def test_cookies_skipped_for_want_of_a_key_leave_it_to_the_session_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version: str,
    has_session: bool,
) -> None:
    message = f"cannot decrypt {version} cookies: no key found"
    before = _working_file()
    rows = [_cookie("pplx.visitor-id", "Zq3"), *([SESSION] if has_session else [])]
    _scripted_yt_dlp(monkeypatch, lambda logger: logger.warning(message, only_once=True), rows)
    rc = cli_auth.main(["import", "--browser", "chrome"])
    err = capsys.readouterr().err
    if has_session:
        assert (rc, err) == (0, f"warning: chrome: {message}\n")
        assert _saved() == {"pplx.visitor-id": "Zq3", TOKEN: "tok"}
    else:
        assert rc == EXIT_AUTH
        assert err == (
            f"pplx auth import: no usable {TOKEN} cookie for *.perplexity.ai in chrome: {message}\n"
        )
        assert default_cookies_path().read_text() == before


def test_undecryptable_cookies_warn_but_the_rest_import(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # With the key in hand, a cookie that still fails to decrypt is skipped;
    # it must not block the others.
    def log(logger: Any) -> None:
        logger.warning(
            "failed to decrypt cookie (AES-CBC) because UTF-8 decoding failed. "
            "Possibly the key is wrong?",
            only_once=True,
        )
        logger.warning("unknown cookie version: \"b'v20'\"", only_once=True)
        logger.info("Extracted 9 cookies from chrome (3 could not be decrypted)")

    _scripted_yt_dlp(monkeypatch, log, [SESSION])
    rc = cli_auth.main(["import", "--browser", "chrome"])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == (
        f"imported chrome cookies to {default_cookies_path()}\nsession valid: me@example.com\n"
    )
    assert err.splitlines() == [
        "warning: chrome: failed to decrypt cookie (AES-CBC) because UTF-8 decoding failed. "
        "Possibly the key is wrong?",
        "warning: chrome: unknown cookie version: \"b'v20'\"",
        "warning: chrome: Extracted 9 cookies from chrome (3 could not be decrypted)",
    ]
    assert _saved() == {TOKEN: "tok"}


def test_yt_dlp_diagnostics_reach_problems_not_output(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    def log(logger: Any) -> None:
        logger.debug("noise")
        logger.info("Extracting cookies from chrome")
        logger.info("Extracted 9 cookies from chrome (3 could not be decrypted)")
        logger.warning("unknown cookie version", only_once=True)
        logger.warning("unknown cookie version", only_once=True)

    _scripted_yt_dlp(monkeypatch, log, [_cookie("a", "1")])
    problems: list[str] = []
    assert [c.name for c in auth._extract_jar("chrome", None, problems)] == ["a"]
    assert problems == [
        "Extracted 9 cookies from chrome (3 could not be decrypted)",
        "unknown cookie version",
    ]
    assert capfd.readouterr() == ("", "")


def test_yt_dlp_error_raises_with_every_problem_noted(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    def log(logger: Any) -> None:
        logger.warning("unknown cookie version", only_once=True)
        logger.error("failed to read from keyring")
        logger.error("could not find local state file")

    _scripted_yt_dlp(monkeypatch, log, [_cookie("a", "")])
    problems: list[str] = []
    with pytest.raises(auth._YtDlpFailure, match=r"^failed to read from keyring$"):
        auth._extract_jar("chrome", None, problems)
    assert problems == [
        "unknown cookie version",
        "failed to read from keyring",
        "could not find local state file",
    ]
    assert capfd.readouterr() == ("", "")


# ---------- yt-dlp itself ----------


def test_other_verbs_do_not_import_yt_dlp(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "from pplx_agent_tools import cli\n"
        "cli.main(['skill-path'])\n"
        "cli.main(['auth', 'check'])\n"
        "print('yt_dlp loaded:', 'yt_dlp' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path
    )
    assert proc.stdout.splitlines()[-1] == "yt_dlp loaded: False"


def _firefox_profile(root: Path, profile: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    """A profile root as Firefox lays it out, with a schema-17 cookie database.

    Each row is (host, path, name, value, originAttributes).
    """
    profile.mkdir(parents=True)
    (root / "profiles.ini").write_text(f"[Profile0]\nPath={profile.relative_to(root)}\n")
    con = sqlite3.connect(profile / "cookies.sqlite")
    with con:
        con.execute(
            "CREATE TABLE moz_cookies (id INTEGER PRIMARY KEY, originAttributes TEXT NOT NULL "
            "DEFAULT '', name TEXT, value TEXT, host TEXT, path TEXT, expiry INTEGER, "
            "isSecure INTEGER)"
        )
        con.executemany(
            "INSERT INTO moz_cookies (host, path, name, value, originAttributes, expiry, isSecure) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            [(*row, FAR_FUTURE * 1000) for row in rows],
        )
        con.execute("PRAGMA user_version = 17")
    con.close()


@pytest.mark.skipif(sys.platform == "win32", reason="profile roots laid out for macOS and Linux")
def test_yt_dlp_round_trip_through_a_fork_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    rows = [
        (".perplexity.ai", "/", "pplx.visitor-id", "v1", ""),
        ("www.perplexity.ai", "/", TOKEN, "tok", ""),
        (".perplexity.ai", "/", TOKEN, "older", ""),
        ("notperplexity.ai", "/", "other", "SECRET", ""),
        (".perplexity.ai.evil.com", "/", "evil", "SECRET", ""),
    ]
    if sys.platform == "darwin":
        root = tmp_path / "Library" / "Application Support" / "zen"
        profile = root / "Profiles" / "abc.default-release"
    else:
        root = tmp_path / ".zen"
        profile = root / "abc.default-release"
    _firefox_profile(root, profile, rows)
    assert cli_auth.main(["import", "--browser", "zen"]) == 0
    out, err = capsys.readouterr()
    assert out == (
        f"imported zen cookies to {default_cookies_path()}\nsession valid: me@example.com\n"
    )
    assert err == ""
    assert _saved() == {"pplx.visitor-id": "v1", TOKEN: "tok"}


def test_firefox_container_cookies_are_not_imported(tmp_path: Path) -> None:
    # yt-dlp would merge each container's copy of a cookie, keeping the last row.
    rows = [
        ("www.perplexity.ai", "/", TOKEN, "default-context", ""),
        ("www.perplexity.ai", "/", TOKEN, "work-container", "^userContextId=2"),
        (".perplexity.ai", "/", "pplx.visitor-id", "container-only", "^userContextId=2"),
    ]
    profile = tmp_path / "firefox" / "abc.default"
    _firefox_profile(tmp_path / "firefox", profile, rows)
    assert read_browser_cookies("firefox", str(profile)) == {TOKEN: "default-context"}


def _gnome_keyring(monkeypatch: pytest.MonkeyPatch, items: dict[str, bytes]) -> None:
    """Stand in for the GNOME keyring that yt-dlp reads through secretstorage."""
    import yt_dlp.cookies

    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")

    class Item:
        def __init__(self, label: str, secret: bytes) -> None:
            self.label, self.secret = label, secret

        def get_label(self) -> str:
            return self.label

        def get_secret(self) -> bytes:
            return self.secret

    class Collection:
        def get_all_items(self) -> list[Item]:
            return [Item(label, secret) for label, secret in items.items()]

    class Connection:
        def close(self) -> None:
            pass

    class SecretStorage:
        @staticmethod
        def dbus_init() -> Connection:
            return Connection()

        @staticmethod
        def get_default_collection(_con: Connection) -> Collection:
            return Collection()

    monkeypatch.setattr(yt_dlp.cookies, "secretstorage", SecretStorage)


def _kwallet(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[], tuple[bytes, bytes, int]]
) -> None:
    """A KDE 5 session whose kwallet-query returns `answer()` as (stdout, stderr, status)."""
    import shutil

    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setenv("KDE_SESSION_VERSION", "5")
    which = shutil.which
    monkeypatch.setattr(
        "yt_dlp.cookies.shutil.which",
        lambda cmd, *a, **kw: (
            "/usr/bin/kwallet-query" if cmd == "kwallet-query" else which(cmd, *a, **kw)
        ),
    )

    def run(args: list[str], **_: object) -> tuple[object, object, int]:
        if args[0] == "dbus-send":
            return "kdewallet\n", "", 0
        assert args[0] == "kwallet-query"
        return answer()

    monkeypatch.setattr("yt_dlp.cookies.Popen.run", staticmethod(run))


def _basic_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A desktop yt-dlp maps to Chromium's basic-text store, which has no keyring key."""
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "sway")
    for var in ("DESKTOP_SESSION", "GNOME_DESKTOP_SESSION_ID", "KDE_FULL_SESSION"):
        monkeypatch.delenv(var, raising=False)


_ChromeRow = tuple[str, str, str, bytes, bytes]


def _encrypted(
    version: bytes, secret: bytes, rows: Sequence[tuple[str, str, str]]
) -> list[_ChromeRow]:
    """(host, name, value) rows to store under `version`, keyed from `secret`."""
    return [(host, name, value, version, secret) for host, name, value in rows]


def _linux_chrome_store(config: Path, rows: Sequence[_ChromeRow]) -> None:
    """A Linux Chrome profile with each value encrypted as its row says."""
    from yt_dlp.aes import aes_cbc_encrypt_bytes
    from yt_dlp.cookies import pbkdf2_sha1

    profile = config / "google-chrome" / "Default"
    profile.mkdir(parents=True)
    con = sqlite3.connect(profile / "Cookies")
    with con:
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO meta VALUES ('version', '24')")
        con.execute(
            "CREATE TABLE cookies (host_key TEXT, name TEXT, value TEXT, encrypted_value BLOB, "
            "path TEXT, expires_utc INTEGER, is_secure INTEGER)"
        )
        for host, name, value, version, secret in rows:
            key = pbkdf2_sha1(secret, b"saltysalt", 1, 16)
            # From database version 24 the plaintext starts with the host's SHA-256.
            plaintext = hashlib.sha256(host.encode()).digest() + value.encode()
            encrypted = version + aes_cbc_encrypt_bytes(
                plaintext, key, b" " * 16, padding_mode="pkcs7"
            )
            con.execute(
                "INSERT INTO cookies VALUES (?, ?, '', ?, '/', ?, 1)",
                (host, name, encrypted, _chromium_time(FAR_FUTURE)),
            )
    con.close()


_CHROME_ROWS = [
    (".perplexity.ai", "pplx.visitor-id", "a3f1c2d4-1111-2222-3333-444455556666"),
    ("www.perplexity.ai", TOKEN, "eyJ" + "x" * 800),
    (".perplexity.ai", "pplx.edge-sid", "abcdefgh"),
    (".perplexity.ai", "__cf_bm", "cfbm-value-0123456789"),
]
_CHROME_SAVED = {name: value for _, name, value in _CHROME_ROWS}


@pytest.fixture
def linux_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Linux with its Chrome profile under the test's XDG config; returns that config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    return tmp_path / "xdg"


@pytest.mark.parametrize("has_key", [True, False], ids=["key", "no-key"])
def test_yt_dlp_round_trip_through_a_linux_chrome_keyring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    linux_config: Path,
    has_key: bool,
) -> None:
    secret = b"the-real-keyring-secret"
    _linux_chrome_store(linux_config, _encrypted(b"v11", secret, _CHROME_ROWS))
    # Without the item yt-dlp decrypts with an empty password, the wrong key.
    _gnome_keyring(monkeypatch, {"Chrome Safe Storage": secret} if has_key else {})
    before = _working_file()
    rc = cli_auth.main(["import", "--browser", "chrome"])
    out, err = capsys.readouterr()
    if has_key:
        assert (rc, err) == (0, "")
        assert _saved() == _CHROME_SAVED
    else:
        assert rc == EXIT_AUTH
        assert out == ""
        assert err.startswith(
            "pplx auth import: cannot read chrome cookies: failed to read from keyring"
        )
        assert default_cookies_path().read_text() == before


def test_session_token_yt_dlp_skips_leaves_the_file_untouched(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], linux_config: Path
) -> None:
    # yt-dlp skips a value it cannot decrypt, such as Windows' App-Bound v20, and
    # returns the rest, which authenticate nothing.
    secret = b"the-real-keyring-secret"
    session = [r for r in _CHROME_ROWS if r[1] == TOKEN]
    rest = [r for r in _CHROME_ROWS if r[1] != TOKEN]
    rows = [*_encrypted(b"v20", secret, session), *_encrypted(b"v11", secret, rest)]
    _linux_chrome_store(linux_config, rows)
    _gnome_keyring(monkeypatch, {"Chrome Safe Storage": secret})
    before = _working_file()
    rc = cli_auth.main(["import", "--browser", "chrome"])
    out, err = capsys.readouterr()
    assert (rc, out) == (EXIT_AUTH, "")
    assert err == (
        f"pplx auth import: no usable {TOKEN} cookie for *.perplexity.ai in chrome: "
        "unknown cookie version: \"b'v20'\"; "
        "Extracted 3 cookies from chrome (1 could not be decrypted)\n"
    )
    assert default_cookies_path().read_text() == before


def test_kwallet_empty_password_decrypts_as_chrome_did(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], linux_config: Path
) -> None:
    # kwallet-query reports a missing entry where Chrome read "" and encrypted with it.
    _linux_chrome_store(linux_config, _encrypted(b"v11", b"", _CHROME_ROWS))
    _kwallet(monkeypatch, lambda: (b"Failed to read entry Chrome Safe Storage\n", b"", 0))
    rc = cli_auth.main(["import", "--browser", "chrome"])
    err = capsys.readouterr().err
    assert (rc, err) == (
        0,
        "warning: chrome: failed to read password from kwallet. Using empty string instead\n",
    )
    assert _saved() == _CHROME_SAVED


def test_kwallet_query_exception_refuses_even_what_decrypted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], linux_config: Path
) -> None:
    def crash() -> tuple[bytes, bytes, int]:
        raise OSError("kwallet-query crashed")

    # The empty password yt-dlp falls back to happens to be right here.
    _linux_chrome_store(linux_config, _encrypted(b"v11", b"", _CHROME_ROWS))
    _kwallet(monkeypatch, crash)
    before = _working_file()
    rc = cli_auth.main(["import", "--browser", "chrome"])
    out, err = capsys.readouterr()
    assert (rc, out) == (EXIT_AUTH, "")
    assert err == (
        "pplx auth import: cannot read chrome cookies: "
        "exception running kwallet-query: OSError: kwallet-query crashed\n"
    )
    assert default_cookies_path().read_text() == before


def test_stale_v11_cookie_of_another_site_does_not_block_the_import(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], linux_config: Path
) -> None:
    # A basic-text store has no key for a v11 cookie left from a keyring setup.
    _basic_text(monkeypatch)
    stale = _encrypted(b"v11", b"old-keyring-secret", [(".example.com", "old", "stale")])
    _linux_chrome_store(linux_config, [*_encrypted(b"v10", b"peanuts", _CHROME_ROWS), *stale])
    rc = cli_auth.main(["import", "--browser", "chrome"])
    err = capsys.readouterr().err
    assert rc == 0
    assert err.splitlines() == [
        "warning: chrome: cannot decrypt v11 cookies: no key found",
        "warning: chrome: Extracted 4 cookies from chrome (1 could not be decrypted)",
    ]
    assert _saved() == _CHROME_SAVED


def test_yt_dlp_messages_relied_on_are_in_its_source() -> None:
    import inspect

    import yt_dlp.cookies

    source = inspect.getsource(yt_dlp.cookies)
    for message in (*auth._NO_KEY, auth._KWALLET_EMPTY_PASSWORD, auth._UNDECRYPTED):
        assert message in source, message


# ---------- the server decides ----------


def test_import_saves_the_cookies_as_the_session_check_rotated_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], server: _Server
) -> None:
    _fake_extractor(monkeypatch, [_cookie("pplx.visitor-id", "v"), SESSION])
    server.rotate = {TOKEN: "rotated", "__cf_bm": "set-by-the-server"}
    assert cli_auth.main(["import", "--browser", "firefox"]) == 0
    assert server.requests == [(SESSION_URL, {"pplx.visitor-id": "v", TOKEN: "tok"})]
    assert _saved() == {"pplx.visitor-id": "v", TOKEN: "rotated"}
    assert capsys.readouterr() == (
        f"imported firefox cookies to {default_cookies_path()}\nsession valid: me@example.com\n",
        "",
    )


@pytest.mark.parametrize(
    ("user", "shown"), [({"id": "a1", "username": "me"}, "me"), ({"id": "a1"}, "a1")]
)
def test_import_names_the_account_without_an_email(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    server: _Server,
    user: dict[str, str],
    shown: str,
) -> None:
    _fake_extractor(monkeypatch, [SESSION])
    server.body = {"user": user}
    assert cli_auth.main(["import", "--browser", "firefox"]) == 0
    assert capsys.readouterr().out.endswith(f"\nsession valid: {shown}\n")


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (200, {}),
        (200, {"user": None}),
        (200, {"user": {}}),
        (200, {"user": {"email": ""}}),
        (401, {}),
        (403, {"error": "forbidden"}),
    ],
    ids=["empty", "null-user", "empty-user", "blank-email", "401", "403"],
)
def test_session_the_server_rejects_exits_two_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    server: _Server,
    status: int,
    body: object,
) -> None:
    before = _working_file()
    _fake_extractor(monkeypatch, [SESSION])
    server.status, server.body = status, body
    # NextAuth clears a session cookie it rejects.
    server.rotate = {TOKEN: ""}
    assert cli_auth.main(["import", "--browser", "firefox"]) == EXIT_AUTH
    assert capsys.readouterr() == (
        "",
        "pplx auth import: perplexity.ai did not accept the firefox session: it is missing "
        "or expired; sign in at perplexity.ai in firefox, then import again\n",
    )
    assert len(server.requests) == 1
    assert default_cookies_path().read_text() == before


def test_unreachable_server_exits_four_naming_no_verify(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], server: _Server
) -> None:
    before = _working_file()
    _fake_extractor(monkeypatch, [SESSION])
    server.error = ConnectionError("Could not resolve host: www.perplexity.ai")
    assert cli_auth.main(["import", "--browser", "firefox"]) == EXIT_NETWORK
    out, err = capsys.readouterr()
    assert out == ""
    assert err == (
        "pplx auth import: cannot verify the firefox session: request to /api/auth/session "
        "failed: Could not resolve host: www.perplexity.ai; nothing was saved. Retry, or pass "
        "--no-verify to save the cookies unverified\n"
    )
    assert default_cookies_path().read_text() == before


def test_no_verify_saves_without_a_request_and_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], server: _Server
) -> None:
    _fake_extractor(monkeypatch, [_cookie("a", "1"), SESSION])
    assert cli_auth.main(["import", "--browser", "firefox", "--no-verify"]) == 0
    assert server.requests == []
    assert _saved() == {"a": "1", TOKEN: "tok"}
    assert capsys.readouterr() == (
        f"imported firefox cookies to {default_cookies_path()}\n",
        "warning: the firefox session was not verified with perplexity.ai (--no-verify)\n",
    )


@pytest.mark.parametrize("no_verify", [False, True], ids=["verify", "no-verify"])
def test_no_session_token_fails_without_a_request(
    monkeypatch: pytest.MonkeyPatch, server: _Server, no_verify: bool
) -> None:
    before = _working_file()
    _fake_extractor(monkeypatch, [_cookie("pplx.visitor-id", "v"), _cookie(TOKEN + ".1", "b")])
    argv = ["import", "--browser", "firefox", *(["--no-verify"] if no_verify else [])]
    assert cli_auth.main(argv) == EXIT_AUTH
    assert server.requests == []
    assert default_cookies_path().read_text() == before


def test_symlink_repointed_during_the_session_check_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    server: _Server,
    tmp_path: Path,
) -> None:
    account_a, account_b = tmp_path / "account-a.json", tmp_path / "account-b.json"
    for path, value in ((account_a, "a-value"), (account_b, "b-value")):
        path.write_text(json.dumps({TOKEN: value}))
        path.chmod(0o600)
    current = tmp_path / "current.json"
    current.symlink_to(account_a)
    monkeypatch.setenv("PPLX_COOKIES_PATH", str(current))

    def repoint() -> None:
        current.unlink()
        current.symlink_to(account_b)

    server.on_request = repoint
    _fake_extractor(monkeypatch, [SESSION])
    assert cli_auth.main(["import", "--browser", "firefox"]) == EXIT_AUTH
    assert capsys.readouterr().err == (
        f"pplx auth import: cannot write cookie file: {current}: it now leads to "
        f"{account_b}, not to {account_a} as it did before the session check\n"
    )
    assert json.loads(account_a.read_text()) == {TOKEN: "a-value"}
    assert json.loads(account_b.read_text()) == {TOKEN: "b-value"}


# ---------- what is sent and saved ----------


@pytest.mark.parametrize(("browser", "stored"), _STORED_TIME)
def test_expired_and_empty_cookies_are_neither_sent_nor_saved(
    monkeypatch: pytest.MonkeyPatch, server: _Server, browser: str, stored: Callable[[int], int]
) -> None:
    rows = [
        _cookie(TOKEN, "tok", "www.perplexity.ai", expires=stored(FAR_FUTURE)),
        _cookie("expired", "old", expires=stored(PAST)),
        _cookie("empty", "", expires=stored(FAR_FUTURE)),
        _cookie("live", "v", expires=stored(FAR_FUTURE)),
        # Both Chromium and Firefox store 0 for a cookie that ends with the session.
        _cookie("session-only", "s", expires=0),
    ]
    _fake_extractor(monkeypatch, rows)
    assert cli_auth.main(["import", "--browser", browser]) == 0
    kept = {TOKEN: "tok", "live": "v", "session-only": "s"}
    assert _saved() == kept
    assert server.requests == [(SESSION_URL, kept)]


@pytest.mark.parametrize(
    ("stored", "kept"),
    [
        ({".0": "a", ".1": "b", ".3": "d"}, {".0": "a", ".1": "b"}),
        ({".0": "a", ".2": "c"}, {".0": "a"}),
        ({".0": "a", ".1": "", ".2": "c"}, {".0": "a"}),
        ({"": "whole", ".0": "a", ".1": "b"}, {"": "whole"}),
        ({".0": "a", ".01": "b", ".x": "c"}, {".0": "a"}),
    ],
    ids=["leftover-past-a-gap", "gap", "empty-chunk", "whole-beside-chunks", "not-chunk-names"],
)
def test_only_the_session_token_the_server_would_read_whole_is_kept(
    monkeypatch: pytest.MonkeyPatch, server: _Server, stored: dict[str, str], kept: dict[str, str]
) -> None:
    # NextAuth joins every cookie whose name starts with the session cookie's.
    rows = [_cookie(TOKEN + suffix, value) for suffix, value in stored.items()]
    _fake_extractor(monkeypatch, [_cookie("a", "1"), *rows])
    assert cli_auth.main(["import", "--browser", "firefox"]) == 0
    expected = {"a": "1", **{TOKEN + suffix: value for suffix, value in kept.items()}}
    assert _saved() == expected
    assert server.requests == [(SESSION_URL, expected)]


def test_profile_under_an_unknown_users_home_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], server: _Server
) -> None:
    calls = _fake_extractor(monkeypatch, [SESSION])
    given = "~no-such-user-pplx/Default"
    assert cli_auth.main(["import", "--browser", "chrome", "--browser-profile", given]) == EXIT_AUTH
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith(f"pplx auth import: cannot find the chrome profile {given!r}: ")
    assert "home directory" in err
    assert (calls, server.requests) == ([], [])
    assert not default_cookies_path().exists()


def test_fork_root_without_a_home_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_home(cls: type[Path]) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", classmethod(no_home))
    calls = _fake_extractor(monkeypatch, [SESSION])
    assert cli_auth.main(["import", "--browser", "zen"]) == EXIT_AUTH
    assert capsys.readouterr().err == (
        "pplx auth import: cannot find the zen profile: Could not determine home directory.\n"
    )
    assert calls == []
