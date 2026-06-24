"""cli_runner error-path tests: under --json, failures emit a parseable error
envelope to stdout; without --json, stdout stays empty and the reason goes to
stderr."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from pplx_agent_tools.cli_runner import run_verb
from pplx_agent_tools.errors import EXIT_AUTH, AuthError


def _boom(_client: object) -> object:
    raise AuthError("cookies expired")


def test_json_mode_emits_error_envelope(capsys: pytest.CaptureFixture[str]) -> None:
    rc = run_verb(
        "fetch",
        Namespace(json=True),
        requires_auth=False,
        run=_boom,
        render_text=lambda _r: "",
        render_json=lambda _r: {},
    )
    assert rc == EXIT_AUTH
    out = json.loads(capsys.readouterr().out)
    assert out["_verb"] == "fetch"
    assert out["error"]["type"] == "AuthError"
    assert out["error"]["message"] == "cookies expired"
    assert out["error"]["exit_code"] == EXIT_AUTH


def test_text_mode_keeps_stdout_empty_on_error(capsys: pytest.CaptureFixture[str]) -> None:
    rc = run_verb(
        "fetch",
        Namespace(json=False),
        requires_auth=False,
        run=_boom,
        render_text=lambda _r: "",
        render_json=lambda _r: {},
    )
    assert rc == EXIT_AUTH
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cookies expired" in captured.err
