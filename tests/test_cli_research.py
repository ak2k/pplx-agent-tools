"""CLI tests for `pplx research` — exit-code contract incl. partial→exit 6."""

from __future__ import annotations

from typing import Any

import pytest

from pplx_agent_tools import cli_research, cli_runner
from pplx_agent_tools.errors import EXIT_OK, EXIT_PARTIAL
from pplx_agent_tools.verbs.research import ResearchResult, ResearchSource


@pytest.fixture(autouse=True)
def _stub_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Dummy:
        pass

    monkeypatch.setattr(
        cli_runner.Client,
        "from_default_cookies",
        classmethod(lambda cls, **_: _Dummy()),
    )


def _stub(monkeypatch: pytest.MonkeyPatch, result: ResearchResult) -> None:
    def _fake(*_a: Any, **_k: Any) -> ResearchResult:
        return result

    monkeypatch.setattr(cli_research, "research", _fake)


def test_complete_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _stub(
        monkeypatch,
        ResearchResult(
            "q", "the report", [ResearchSource("https://a", "A", None)], "research", True
        ),
    )
    rc = cli_research.main(["what is quic", "--timeout", "0"])
    cap = capsys.readouterr()
    assert rc == EXIT_OK
    assert "the report" in cap.out
    assert "stream did not reach COMPLETED" not in cap.err


def test_partial_exits_six_with_content(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _stub(monkeypatch, ResearchResult("q", "partial report", [], "research", False))
    rc = cli_research.main(["what is quic"])
    cap = capsys.readouterr()
    assert rc == EXIT_PARTIAL
    assert "partial report" in cap.out
    assert "did not reach COMPLETED" in cap.err


def test_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    _stub(monkeypatch, ResearchResult("q", "A", [], "research", True))
    rc = cli_research.main(["q", "--json"])
    assert rc == EXIT_OK
    assert '"_verb": "research"' in capsys.readouterr().out
