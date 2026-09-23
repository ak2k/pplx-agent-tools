"""CLI tests for `pplx ask` — exit-code contract incl. partial→exit 6."""

from __future__ import annotations

from typing import Any

import pytest

from pplx_agent_tools import cli_ask, cli_runner
from pplx_agent_tools.errors import EXIT_OK, EXIT_PARTIAL
from pplx_agent_tools.grounding import Unchecked, check_grounding
from pplx_agent_tools.verbs._ask_common import Source
from pplx_agent_tools.verbs.ask import AskResult


@pytest.fixture(autouse=True)
def _stub_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Dummy:
        pass

    monkeypatch.setattr(
        cli_runner.Client, "from_default_cookies", classmethod(lambda cls, **_: _Dummy())
    )


def _stub(monkeypatch: pytest.MonkeyPatch, result: AskResult) -> None:
    def _fake(*_a: Any, **_k: Any) -> AskResult:
        return result

    monkeypatch.setattr(cli_ask, "ask", _fake)


def test_complete_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _stub(monkeypatch, AskResult("q", "the answer", "turbo", True))
    rc = cli_ask.main(["what is quic", "--timeout", "0"])
    cap = capsys.readouterr()
    assert rc == EXIT_OK
    assert "the answer" in cap.out
    assert "did not reach COMPLETED" not in cap.err


def test_partial_exits_six(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    _stub(monkeypatch, AskResult("q", "partial", "turbo", False))
    rc = cli_ask.main(["q"])
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    assert "partial" in cap.out
    assert "did not reach COMPLETED" in cap.err


def test_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    _stub(monkeypatch, AskResult("q", "A", "claude48opusthinking", True))
    rc = cli_ask.main(["q", "--json", "--model", "claude48opusthinking"])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert '"_verb": "ask"' in out
    assert "claude48opusthinking" in out


_UNGROUNDED = check_grounding("It is 4.0.", "q", [Source("https://a.test/", "t", "s")])


@pytest.mark.parametrize("json_flag", [[], ["--json"]])
def test_ungrounded_warns_and_keeps_exit_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, json_flag: list[str]
) -> None:
    _stub(monkeypatch, AskResult("q", "It is 4.0.", "turbo", True, grounding=_UNGROUNDED))
    rc = cli_ask.main(["q", *json_flag])
    cap = capsys.readouterr()
    assert rc == EXIT_OK
    warnings = [ln for ln in cap.err.splitlines() if "not grounded" in ln]
    assert warnings == [
        "warning: ask answer not grounded in its sources: "
        "every cited URL is a site root; 0 of 1 figures/names appear in a cited "
        "source's title or snippet; unsupported: 4.0"
    ]


def test_ungrounded_partial_keeps_exit_six(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _stub(monkeypatch, AskResult("q", "It is 4.0.", "turbo", False, grounding=_UNGROUNDED))
    assert cli_ask.main(["q"]) == EXIT_PARTIAL
    assert "not grounded" in capsys.readouterr().err


def test_grounded_is_silent(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    grounded = check_grounding("It costs $45.", "q", [Source("https://a.test/p", "t", "$45")])
    _stub(monkeypatch, AskResult("q", "A", "turbo", True, grounding=grounded))
    assert cli_ask.main(["q"]) == EXIT_OK
    assert "not grounded" not in capsys.readouterr().err


def test_nothing_to_check_is_silent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    unchecked = Unchecked("no_checkable_terms")
    _stub(monkeypatch, AskResult("q", "Yes.", "turbo", True, grounding=unchecked))
    assert cli_ask.main(["q"]) == EXIT_OK
    cap = capsys.readouterr()
    assert "not grounded" not in cap.err
    assert "grounded: no" not in cap.out


def test_no_grounded_check_flag_reaches_the_verb(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def _fake(*_a: Any, **k: Any) -> AskResult:
        seen.update(k)
        return AskResult("q", "A", "turbo", True)

    monkeypatch.setattr(cli_ask, "ask", _fake)
    cli_ask.main(["q"])
    assert seen["grounded_check"] is True
    cli_ask.main(["q", "--no-grounded-check"])
    assert seen["grounded_check"] is False
