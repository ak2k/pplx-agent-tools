"""Browser import: `pplx auth import` and `auth.import_from_browser`.

Every test but the yt-dlp round trip replaces `auth._extract_jar`, the one
call into yt-dlp, so no test reads a real browser's cookie store.
"""

from __future__ import annotations

import json
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Sequence
from http.cookiejar import Cookie
from pathlib import Path

import pytest

from pplx_agent_tools import auth, cli_auth
from pplx_agent_tools.auth import (
    SUPPORTED_BROWSERS,
    default_cookies_path,
    import_from_browser,
    load_cookies,
)
from pplx_agent_tools.errors import EXIT_AUTH, EXIT_GENERIC, AuthError

FAR_FUTURE = 4102444800  # 2100-01-01


@pytest.fixture(autouse=True)
def _own_cookie_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cookie file per test, so a missing one proves nothing was written."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _cookie(
    name: str,
    value: str | None,
    domain: str = ".perplexity.ai",
    path: str = "/",
    expires: int | None = FAR_FUTURE,
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
    assert "{" + ",".join(SUPPORTED_BROWSERS) + "}" in capsys.readouterr().out


def test_unsupported_browser_from_python_is_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_extractor(monkeypatch)
    with pytest.raises(AuthError, match="unsupported browser: 'arc'"):
        import_from_browser("arc")
    assert calls == []


@pytest.mark.parametrize(
    "browser", [b for b in SUPPORTED_BROWSERS if b not in ("librewolf", "zen")]
)
def test_browser_is_passed_to_the_extractor_by_name(
    monkeypatch: pytest.MonkeyPatch, browser: str
) -> None:
    calls = _fake_extractor(monkeypatch, [_cookie("a", "1")])
    import_from_browser(browser)
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
    calls = _fake_extractor(monkeypatch, [_cookie("a", "1")])
    import_from_browser(browser)
    assert calls == [("firefox", str(root))]


# ---------- Firefox fork profile roots ----------


def _linux_roots(home: Path, config: Path, browser: str) -> list[Path]:
    flatpak = home / ".var" / "app" / auth._FIREFOX_FORKS[browser]
    return [
        home / f".{browser}",
        config / browser / browser,
        flatpak / f".{browser}",
        flatpak / "config" / browser / browser,
    ]


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


def test_fork_root_on_linux_defaults_xdg_to_dot_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setattr(sys, "platform", "linux")
    root = tmp_path / ".config" / "zen" / "zen"
    root.mkdir(parents=True)
    (root / "profiles.ini").touch()
    assert auth._firefox_fork_root("zen") == str(root)


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, browser: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    calls = _fake_extractor(monkeypatch, [_cookie("a", "1")])
    with pytest.raises(AuthError) as ei:
        import_from_browser(browser)
    assert f"{browser} profile directory not found" in str(ei.value)
    assert calls == []
    assert not default_cookies_path().exists()


# ---------- which cookies are written ----------


def test_import_saves_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extractor(monkeypatch, [_cookie("a", "1"), _cookie("b", "2", "www.perplexity.ai")])
    dest = import_from_browser("brave")
    assert dest == default_cookies_path()
    assert json.loads(dest.read_text()) == {"a": "1", "b": "2"}
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
    _fake_extractor(monkeypatch, [_cookie("a", "1", domain)])
    import_from_browser("chrome")
    assert _saved() == {"a": "1"}


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
    _fake_extractor(monkeypatch, [_cookie("a", "1"), _cookie("other", "SECRET", domain)])
    import_from_browser("chrome")
    assert _saved() == {"a": "1"}


def _pick(monkeypatch: pytest.MonkeyPatch, rows: list[Cookie]) -> str:
    """The value written for name "n", whichever order the rows arrive in."""
    picked: set[str] = set()
    for order in (rows, rows[::-1]):
        _fake_extractor(monkeypatch, order)
        import_from_browser("chrome")
        picked.add(_saved()["n"])
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


def test_duplicate_prefers_unexpired(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _cookie("n", "stale", "www.perplexity.ai", expires=1),
        _cookie("n", "live", ".perplexity.ai"),
    ]
    assert _pick(monkeypatch, rows) == "live"


def test_duplicate_session_cookie_counts_as_unexpired(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _cookie("n", "session", ".perplexity.ai", expires=None),
        _cookie("n", "apex", "perplexity.ai"),
    ]
    assert _pick(monkeypatch, rows) == "session"


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
        _cookie("session", "good"),
        _cookie("nullish", None),
        _cookie("crlf", "SECRET1\r\nX: y"),
        _cookie("tabbed", "SECRET2\tz"),
        _cookie("surr", "SECRET3\ud800"),
        _cookie("a;b", "SECRET4"),
    ]
    _fake_extractor(monkeypatch, rows)
    import_from_browser("brave")
    assert _saved() == {"session": "good"}
    err = capsys.readouterr().err
    assert err.count("warning: skipping") == 5
    for name in ("nullish", "crlf", "tabbed", "surr"):
        assert repr(name) in err
    assert "has no value" in err
    assert "SECRET" not in err


def test_import_all_rows_bad_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extractor(monkeypatch, [_cookie("a", "x\r\ny")])
    with pytest.raises(AuthError, match="sign in"):
        import_from_browser("brave")
    assert not default_cookies_path().exists()


def test_import_empty_says_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extractor(monkeypatch, [_cookie("other", "1", "example.com")])
    with pytest.raises(AuthError, match=r"sign in at perplexity\.ai in brave first"):
        import_from_browser("brave")
    assert not default_cookies_path().exists()


# ---------- destination: the file load_cookies reads ----------


def test_import_writes_to_cookies_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "custom" / "jar.json"
    monkeypatch.setenv("PPLX_COOKIES_PATH", str(target))
    _fake_extractor(monkeypatch, [_cookie("a", "1")])
    assert import_from_browser("brave") == target
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert load_cookies() == {"a": "1"}
    assert not default_cookies_path().exists()


def test_import_refuses_when_inline_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPLX_COOKIES", '{"a": "SECRET"}')
    calls = _fake_extractor(monkeypatch, [_cookie("a", "1")])
    with pytest.raises(AuthError) as ei:
        import_from_browser("brave")
    msg = str(ei.value)
    assert "$PPLX_COOKIES" in msg and "not changed" in msg
    assert "SECRET" not in msg
    assert calls == []
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
        import_from_browser("chromium")
    assert str(ei.value) == (
        "cannot read chromium cookies: no D-Bus session; secretstorage not available"
    )


def test_safari_os_error_names_full_disk_access(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_extractor(monkeypatch, raises=PermissionError(1, "Operation not permitted"))
    with pytest.raises(AuthError, match="Full Disk Access"):
        import_from_browser("safari")


def test_keychain_refusal_exits_two_with_the_cause(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A refused Keychain prompt raises nothing: every encrypted cookie fails
    # to decrypt and the jar holds only the plaintext ones.
    _fake_extractor(
        monkeypatch,
        [_cookie("plain", "1", "example.com")],
        problems=[
            "find-generic-password failed",
            "cannot decrypt v10 cookies: no key found",
            "Extracted 1 cookies from chrome (40 could not be decrypted)",
        ],
    )
    rc = cli_auth.main(["import", "--browser", "chrome"])
    assert rc == EXIT_AUTH
    out, err = capsys.readouterr()
    assert out == ""
    assert "no usable cookies for *.perplexity.ai in chrome" in err
    assert "find-generic-password failed" in err
    assert "cannot decrypt v10 cookies: no key found" in err
    assert "40 could not be decrypted" in err
    assert not default_cookies_path().exists()


def test_partial_decryption_writes_and_warns_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_extractor(
        monkeypatch,
        [_cookie("a", "1")],
        problems=[
            "cannot decrypt v11 cookies: no key found",
            "Extracted 9 cookies from chrome (3 could not be decrypted)",
        ],
    )
    rc = cli_auth.main(["import", "--browser", "chrome"])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == f"imported chrome cookies to {default_cookies_path()}\n"
    assert "warning: chrome: cannot decrypt v11 cookies: no key found" in err
    assert "warning: chrome: Extracted 9 cookies from chrome (3 could not be decrypted)" in err
    assert _saved() == {"a": "1"}


def test_yt_dlp_diagnostics_reach_problems_not_output(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    import yt_dlp.cookies

    def extract(
        browser: str, profile: str | None, logger: yt_dlp.cookies.YDLLogger
    ) -> list[Cookie]:
        logger.debug("noise")
        logger.info("Extracting cookies from chrome")
        logger.info("Extracted 9 cookies from chrome (3 could not be decrypted)")
        logger.warning("cannot decrypt v10 cookies: no key found", only_once=True)
        logger.warning("cannot decrypt v10 cookies: no key found", only_once=True)
        logger.error("failed to read from keyring")
        return [_cookie("a", "1")]

    monkeypatch.setattr(yt_dlp.cookies, "extract_cookies_from_browser", extract)
    problems: list[str] = []
    assert [c.name for c in auth._extract_jar("chrome", None, problems)] == ["a"]
    assert problems == [
        "Extracted 9 cookies from chrome (3 could not be decrypted)",
        "cannot decrypt v10 cookies: no key found",
        "failed to read from keyring",
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


def _firefox_profile(root: Path, profile: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """A profile root as Firefox lays it out, with a schema-17 cookie database."""
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
            "INSERT INTO moz_cookies (host, path, name, value, expiry, isSecure) "
            "VALUES (?, ?, ?, ?, ?, 1)",
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
        (".perplexity.ai", "/", "pplx.visitor-id", "v1"),
        ("www.perplexity.ai", "/", "__Secure-next-auth.session-token", "tok"),
        (".perplexity.ai", "/", "__Secure-next-auth.session-token", "older"),
        ("notperplexity.ai", "/", "other", "SECRET"),
        (".perplexity.ai.evil.com", "/", "evil", "SECRET"),
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
    assert out == f"imported zen cookies to {default_cookies_path()}\n"
    assert err == ""
    assert _saved() == {"pplx.visitor-id": "v1", "__Secure-next-auth.session-token": "tok"}
