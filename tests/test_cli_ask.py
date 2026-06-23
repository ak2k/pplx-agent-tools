"""CLI tests for `pplx ask` — exit-code contract incl. partial→exit 6."""

from __future__ import annotations

from typing import Any

import pytest

from pplx_agent_tools import cli_ask, cli_runner
from pplx_agent_tools.errors import EXIT_OK, EXIT_PARTIAL
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
