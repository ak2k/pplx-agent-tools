"""Unit tests for pplx skill-path."""

from __future__ import annotations

from pathlib import Path

import pytest

from pplx_agent_tools import cli_skill


def test_find_skill_path_in_repo_root() -> None:
    """The function must return *some* path in the local checkout (the
    editable-mode fallback finds SKILL.md at the repo root).
    """
    p = cli_skill.find_skill_path()
    assert p is not None
    assert p.is_file()
    assert p.name == "SKILL.md"


def test_main_prints_path(capsys: pytest.CaptureFixture) -> None:
    rc = cli_skill.main(None)
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("SKILL.md")
    assert Path(out).is_file()


def test_main_argv_ignored(capsys: pytest.CaptureFixture) -> None:
    # Accepts argv but doesn't parse it
    cli_skill.main(["unused", "args"])
    out = capsys.readouterr().out.strip()
    assert out.endswith("SKILL.md")


def test_main_returns_1_when_not_found(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(cli_skill, "find_skill_path", lambda: None)
    rc = cli_skill.main(None)
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err
