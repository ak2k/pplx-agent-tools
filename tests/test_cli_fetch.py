"""Unit tests for cli_fetch.main — exit-code contract on partial vs. complete.

`fetch()` is monkeypatched to return a canned FetchResult so we exercise the
CLI's rendering + exit-code logic without touching curl_cffi.
"""

from __future__ import annotations

from typing import Any

import pytest

from pplx_agent_tools import cli_fetch, cli_runner
from pplx_agent_tools.errors import EXIT_OK, EXIT_PARTIAL
from pplx_agent_tools.verbs.fetch import FetchResult


@pytest.fixture(autouse=True)
def _stub_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid loading real cookies — CLI tests should not depend on $HOME state.

    Client is imported by cli_runner (the generic verb runner that handles
    auth setup), so the patch target is cli_runner.Client rather than the
    individual cli_X module that no longer references Client directly.
    """

    class _DummyClient:
        pass

    monkeypatch.setattr(
        cli_runner.Client,
        "from_default_cookies",
        classmethod(lambda cls, **_: _DummyClient()),
    )


def _stub_fetch(monkeypatch: pytest.MonkeyPatch, result: FetchResult) -> None:
    def _fake(*_args: Any, **_kwargs: Any) -> FetchResult:
        return result

    monkeypatch.setattr(cli_fetch, "fetch", _fake)


def test_complete_stream_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _stub_fetch(
        monkeypatch,
        FetchResult(
            url="https://example.com",
            title=None,
            domain="example.com",
            content="full answer",
            is_extracted=True,
            stream_complete=True,
        ),
    )
    rc = cli_fetch.main(["https://example.com", "--prompt", "tldr"])
    cap = capsys.readouterr()
    assert rc == EXIT_OK
    assert "full answer" in cap.out
    # Nothing on stderr when complete — no false warnings.
    assert "stream did not reach COMPLETED" not in cap.err


def test_partial_stream_exits_six_but_emits_content(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _stub_fetch(
        monkeypatch,
        FetchResult(
            url="https://example.com",
            title=None,
            domain="example.com",
            content="partial content here",
            is_extracted=True,
            stream_complete=False,
        ),
    )
    rc = cli_fetch.main(["https://example.com", "--prompt", "tldr"])
    cap = capsys.readouterr()
    # Caller's exit-code contract: 6 = partial, even though stdout is usable.
    assert rc == EXIT_PARTIAL
    # Partial content still emitted so callers that want to salvage can.
    assert "partial content here" in cap.out
    assert "stream: incomplete" in cap.out  # header marker from render
    assert "stream did not reach COMPLETED" in cap.err
    # exit 6 hint surfaced in the stderr warning so a casual grep finds it.
    assert "exit 6" in cap.err
