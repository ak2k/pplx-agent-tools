"""CLI exit-code/contract tests for `pplx quota` and `pplx models`.

The verb functions are monkeypatched to canned Results so we exercise the CLI
runner + rendering without touching curl_cffi or real cookies.
"""

from __future__ import annotations

from typing import Any

import pytest

from pplx_agent_tools import cli_models, cli_quota, cli_runner
from pplx_agent_tools.errors import EXIT_OK
from pplx_agent_tools.verbs.models import ModeInfo, ModelInfo, ModelsResult
from pplx_agent_tools.verbs.quota import QuotaItem, QuotaResult


@pytest.fixture(autouse=True)
def _stub_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Dummy:
        pass

    monkeypatch.setattr(
        cli_runner.Client,
        "from_default_cookies",
        classmethod(lambda cls, **_: _Dummy()),
    )


def test_quota_text_exit_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        cli_quota,
        "quota",
        lambda *_a, **_k: QuotaResult(QuotaItem("free_queries", True, None), [], []),
    )
    rc = cli_quota.main([])
    assert rc == EXIT_OK
    assert "free queries" in capsys.readouterr().out


def test_quota_json_exit_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        cli_quota,
        "quota",
        lambda *_a, **_k: QuotaResult(None, [QuotaItem("research", True, None)], []),
    )
    rc = cli_quota.main(["--json"])
    assert rc == EXIT_OK
    assert '"_verb": "quota"' in capsys.readouterr().out


def test_models_exit_zero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def _fake(*_a: Any, **_k: Any) -> ModelsResult:
        return ModelsResult(
            models=[ModelInfo("turbo", "Best", None, "search", "PERPLEXITY")],
            modes=[ModeInfo("search", "Search", None)],
            default_models={"search": "pplx_pro"},
        )

    monkeypatch.setattr(cli_models, "models", _fake)
    rc = cli_models.main([])
    assert rc == EXIT_OK
    assert "turbo" in capsys.readouterr().out
