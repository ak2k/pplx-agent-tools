"""Unit tests for the top-level `pplx` dispatcher."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from pplx_agent_tools import __version__, cli


def _stub_verb_main(name: str, captured: list[tuple[str, list[str]]]) -> object:
    def _impl(argv: Sequence[str] | None) -> int:
        captured.append((name, list(argv or [])))
        return 0

    return _impl


@pytest.fixture
def stub_verbs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    captured: list[tuple[str, list[str]]] = []
    stub_table = {
        name: (_stub_verb_main(name, captured), desc) for name, (_, desc) in cli.VERBS.items()
    }
    monkeypatch.setattr(cli, "VERBS", stub_table)
    return captured


def test_top_level_help_no_args(capsys: pytest.CaptureFixture, stub_verbs: list) -> None:
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage: pplx <verb>" in out
    assert "search" in out and "fetch" in out and "snippets" in out and "auth" in out


def test_top_level_help_dash_h(capsys: pytest.CaptureFixture, stub_verbs: list) -> None:
    assert cli.main(["-h"]) == 0
    out = capsys.readouterr().out
    assert "search" in out


def test_top_level_help_double_dash_help(capsys: pytest.CaptureFixture, stub_verbs: list) -> None:
    assert cli.main(["--help"]) == 0


def test_dispatch_search(stub_verbs: list[tuple[str, list[str]]]) -> None:
    cli.main(["search", "claude code", "-n", "3"])
    assert stub_verbs == [("search", ["claude code", "-n", "3"])]


def test_dispatch_fetch(stub_verbs: list[tuple[str, list[str]]]) -> None:
    cli.main(["fetch", "https://example.com", "--prompt", "tldr"])
    assert stub_verbs == [("fetch", ["https://example.com", "--prompt", "tldr"])]


def test_dispatch_snippets(stub_verbs: list[tuple[str, list[str]]]) -> None:
    cli.main(["snippets", "q", "https://a", "https://b"])
    assert stub_verbs == [("snippets", ["q", "https://a", "https://b"])]


def test_dispatch_auth(stub_verbs: list[tuple[str, list[str]]]) -> None:
    cli.main(["auth", "check"])
    assert stub_verbs == [("auth", ["check"])]


def test_unknown_verb_exits_1(capsys: pytest.CaptureFixture, stub_verbs: list) -> None:
    rc = cli.main(["bogus", "args"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "unknown verb 'bogus'" in err
    # The top-level help also goes to stderr so the user sees the verb list
    assert "search" in err


def test_dispatch_propagates_verb_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def returns_42(argv: Sequence[str] | None) -> int:
        return 42

    monkeypatch.setattr(cli, "VERBS", {"search": (returns_42, "test")})
    assert cli.main(["search"]) == 42


def test_dispatch_empty_args_to_verb(stub_verbs: list[tuple[str, list[str]]]) -> None:
    # `pplx search` (no further args) should dispatch with empty argv
    cli.main(["search"])
    assert stub_verbs == [("search", [])]


def test_version_flag_long(capsys: pytest.CaptureFixture, stub_verbs: list) -> None:
    rc = cli.main(["--version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == f"pplx {__version__}"
    # `--version` must not dispatch to any verb
    assert stub_verbs == []


def test_version_flag_short(capsys: pytest.CaptureFixture, stub_verbs: list) -> None:
    rc = cli.main(["-V"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == f"pplx {__version__}"


def test_version_flag_takes_precedence_over_unknown_verb(
    capsys: pytest.CaptureFixture, stub_verbs: list
) -> None:
    # `pplx --version not-a-verb` prints version, doesn't error
    rc = cli.main(["--version", "not-a-verb"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == f"pplx {__version__}"


def test_help_mentions_version_flag(capsys: pytest.CaptureFixture, stub_verbs: list) -> None:
    cli.main([])
    out = capsys.readouterr().out
    assert "--version" in out
