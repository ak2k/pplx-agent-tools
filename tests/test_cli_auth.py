"""Unit tests for cli_auth dispatch surface (Move 6/6: coverage bump).

`pplx auth` is the user's first interaction with the toolkit — cookie
import, validation, refresh. Coverage was 18% before this move; everything
beyond the bare argparse setup was unexercised. These tests close that
gap by mocking Client / save_cookies / import_from_browser at the cli_auth
seam and exercising each subcommand's success and PplxError paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pplx_agent_tools import cli_auth
from pplx_agent_tools.errors import (
    EXIT_AUTH,
    EXIT_NETWORK,
    AuthError,
    NetworkError,
)

# ---------- shared stubs ----------


class _StubClient:
    """Test stand-in for wire.Client: drop-in for cli_auth's monkey-patched
    `Client.from_default_cookies` factory. Configurable to either return a
    canned session dict or raise on auth_session().
    """

    def __init__(self, session_or_error: dict[str, Any] | Exception) -> None:
        self._target = session_or_error
        self.cookies = {"session-token": "rotated-value"}
        self.session_calls = 0

    def auth_session(self) -> dict[str, Any]:
        self.session_calls += 1
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def _install_client_factory(
    monkeypatch: pytest.MonkeyPatch, client: _StubClient | Exception
) -> None:
    """Patch `cli_auth.Client.from_default_cookies` to yield `client` (or raise)."""

    def factory(cls: Any, **_: Any) -> Any:
        if isinstance(client, Exception):
            raise client
        return client

    monkeypatch.setattr(cli_auth.Client, "from_default_cookies", classmethod(factory))


@pytest.fixture(autouse=True)
def _xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point cookie storage at a tmp dir for every test."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("PPLX_PROFILE", raising=False)


# ---------- main() routing ----------


def test_main_dispatches_to_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    client = _StubClient({"user": {"email": "u@example.com"}, "expires": "2026-12-31"})
    _install_client_factory(monkeypatch, client)
    rc = cli_auth.main(["check"])
    assert rc == 0
    assert "u@example.com" in capsys.readouterr().out


def test_main_dispatches_to_refresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _StubClient({"user": {"email": "u@example.com"}})
    _install_client_factory(monkeypatch, client)
    saved: list[Any] = []
    monkeypatch.setattr(cli_auth, "save_cookies", lambda c, profile=None: saved.append(c))
    rc = cli_auth.main(["refresh"])
    assert rc == 0
    assert client.session_calls == 1
    assert saved == [client.cookies]


def test_main_dispatches_to_import(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    dest = tmp_path / "cookies.json"
    monkeypatch.setattr(cli_auth, "import_from_browser", lambda browser, profile=None: dest)
    rc = cli_auth.main(["import", "--browser", "chrome"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "chrome" in out
    assert str(dest) in out


def test_main_requires_subcommand(capsys: pytest.CaptureFixture) -> None:
    # argparse exits with code 2 (SystemExit) when a required subcommand is missing
    with pytest.raises(SystemExit) as ei:
        cli_auth.main([])
    assert ei.value.code == 2


# ---------- check ----------


def test_check_prints_email_expires_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    client = _StubClient({"user": {"email": "dev@example.com"}, "expires": "2026-12-31T00:00:00Z"})
    _install_client_factory(monkeypatch, client)
    rc = cli_auth.main(["check"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dev@example.com" in out
    assert "2026-12-31T00:00:00Z" in out
    assert "profile: default" in out


def test_check_tolerates_missing_user_email(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # session may have user={} or no user at all — print "(no email)" sentinel
    _install_client_factory(monkeypatch, _StubClient({"user": {}}))
    rc = cli_auth.main(["check"])
    assert rc == 0
    assert "(no email)" in capsys.readouterr().out


def test_check_tolerates_missing_expires(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_client_factory(monkeypatch, _StubClient({"user": {"email": "x@y"}}))
    rc = cli_auth.main(["check"])
    assert rc == 0
    assert "(no expiry)" in capsys.readouterr().out


def test_check_auth_error_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_client_factory(monkeypatch, AuthError("cookies expired"))
    rc = cli_auth.main(["check"])
    cap = capsys.readouterr()
    assert rc == EXIT_AUTH
    assert "pplx auth check: cookies expired" in cap.err
    assert cap.out == ""


def test_check_network_error_maps_to_exit_four(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # Error raised by auth_session, not by the factory, exercises the
    # inner try/except path distinct from client-construction failure.
    client = _StubClient(NetworkError("DNS failed"))
    _install_client_factory(monkeypatch, client)
    rc = cli_auth.main(["check"])
    assert rc == EXIT_NETWORK
    assert "DNS failed" in capsys.readouterr().err


def test_check_respects_profile_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, Any] = {}

    def factory(cls: Any, **kwargs: Any) -> Any:
        received.update(kwargs)
        return _StubClient({"user": {"email": "x@y"}})

    monkeypatch.setattr(cli_auth.Client, "from_default_cookies", classmethod(factory))
    cli_auth.main(["check", "--profile", "kanerai"])
    assert received["profile"] == "kanerai"


# ---------- refresh ----------


def test_refresh_persists_rotated_cookies(monkeypatch: pytest.MonkeyPatch) -> None:
    """`refresh` MUST call save_cookies after auth_session — that's the whole
    point. Without persistence the rotated session-token is dropped on the
    floor and the next pplx call reads the stale one.
    """
    client = _StubClient({"user": {"email": "x@y"}})
    _install_client_factory(monkeypatch, client)
    saved: list[Any] = []
    monkeypatch.setattr(
        cli_auth, "save_cookies", lambda c, profile=None: saved.append((c, profile))
    )

    rc = cli_auth.main(["refresh"])
    assert rc == 0
    assert client.session_calls == 1
    assert len(saved) == 1
    cookies, profile = saved[0]
    assert cookies == client.cookies  # rotated value
    assert profile is None


def test_refresh_silent_on_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Designed for cron/launchd: exit 0 with no stdout/stderr noise."""
    _install_client_factory(monkeypatch, _StubClient({"user": {"email": "x@y"}}))
    monkeypatch.setattr(cli_auth, "save_cookies", lambda c, profile=None: None)
    cli_auth.main(["refresh"])
    cap = capsys.readouterr()
    assert cap.out == ""
    assert cap.err == ""


def test_refresh_auth_error_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _install_client_factory(monkeypatch, AuthError("invalid session"))
    monkeypatch.setattr(cli_auth, "save_cookies", lambda *a, **kw: None)
    rc = cli_auth.main(["refresh"])
    assert rc == EXIT_AUTH
    assert "pplx auth refresh: invalid session" in capsys.readouterr().err


def test_refresh_respects_profile_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    received_factory: dict[str, Any] = {}
    saved: list[tuple[Any, Any]] = []

    def factory(cls: Any, **kwargs: Any) -> Any:
        received_factory.update(kwargs)
        return _StubClient({"user": {"email": "x@y"}})

    monkeypatch.setattr(cli_auth.Client, "from_default_cookies", classmethod(factory))
    monkeypatch.setattr(
        cli_auth, "save_cookies", lambda c, profile=None: saved.append((c, profile))
    )

    cli_auth.main(["refresh", "--profile", "work"])
    assert received_factory["profile"] == "work"
    assert saved[0][1] == "work"


# ---------- import ----------


def test_import_prints_destination(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    dest = tmp_path / "out" / "cookies.json"
    monkeypatch.setattr(cli_auth, "import_from_browser", lambda b, profile=None: dest)
    rc = cli_auth.main(["import", "--browser", "firefox"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "firefox" in out
    assert str(dest) in out


def test_import_auth_error_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def boom(browser: str, profile: str | None = None) -> Path:
        raise AuthError("rookiepy couldn't read browser DB")

    monkeypatch.setattr(cli_auth, "import_from_browser", boom)
    rc = cli_auth.main(["import", "--browser", "chrome"])
    assert rc == EXIT_AUTH
    assert "rookiepy couldn't read" in capsys.readouterr().err


def test_import_requires_browser_flag() -> None:
    # argparse enforces required=True on --browser
    with pytest.raises(SystemExit) as ei:
        cli_auth.main(["import"])
    assert ei.value.code == 2


def test_import_rejects_unsupported_browser() -> None:
    # argparse's choices=SUPPORTED_BROWSERS catches typos at parse time
    with pytest.raises(SystemExit):
        cli_auth.main(["import", "--browser", "not-a-real-browser"])


def test_import_passes_profile_to_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, Any] = {}

    def fake_import(browser: str, profile: str | None = None) -> Path:
        received["browser"] = browser
        received["profile"] = profile
        return Path("/tmp/x")

    monkeypatch.setattr(cli_auth, "import_from_browser", fake_import)
    cli_auth.main(["import", "--browser", "safari", "--profile", "personal"])
    assert received == {"browser": "safari", "profile": "personal"}
