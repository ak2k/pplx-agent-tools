"""Unit tests for cli_runner.run_verb — the generic verb-execution shell.

These lock the agent contract enforced by the runner: error-to-exit-code
mapping, auth-vs-no-auth dispatch, JSON-vs-text rendering, warnings
propagation, and finalize-overrides-exit-code.

A stubbed `run` callable returns canned results so the runner is tested
in isolation from any real verb.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field

import pytest

from pplx_agent_tools import cli_runner
from pplx_agent_tools.cli_runner import run_verb
from pplx_agent_tools.errors import (
    EXIT_AUTH,
    EXIT_GENERIC,
    EXIT_NETWORK,
    EXIT_OK,
    EXIT_PARTIAL,
    EXIT_RATE_LIMIT,
    AuthError,
    NetworkError,
    RateLimitError,
    SchemaError,
)


@dataclass
class _StubResult:
    payload: str
    warnings: list[str] = field(default_factory=list)


def _text(r: _StubResult) -> str:
    return f"TEXT:{r.payload}"


def _json(r: _StubResult) -> dict[str, str]:
    return {"payload": r.payload}


@pytest.fixture(autouse=True)
def _stub_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid loading real cookies for any requires_auth=True path."""

    class _DummyClient:
        pass

    monkeypatch.setattr(
        cli_runner.Client,
        "from_default_cookies",
        classmethod(lambda cls, **_: _DummyClient()),
    )


# ---------- happy path ----------


def test_returns_exit_ok_on_success_text(capsys: pytest.CaptureFixture) -> None:
    args = Namespace(json=False, profile=None)
    rc = run_verb(
        "v",
        args,
        requires_auth=False,
        run=lambda _c: _StubResult("hello"),
        render_text=_text,
        render_json=_json,
    )
    cap = capsys.readouterr()
    assert rc == EXIT_OK
    assert cap.out.strip() == "TEXT:hello"
    assert cap.err == ""


def test_json_flag_switches_render(capsys: pytest.CaptureFixture) -> None:
    args = Namespace(json=True, profile=None)
    rc = run_verb(
        "v",
        args,
        requires_auth=False,
        run=lambda _c: _StubResult("hi"),
        render_text=_text,
        render_json=_json,
    )
    cap = capsys.readouterr()
    assert rc == EXIT_OK
    # json.dumps wraps in braces; payload key visible
    assert '"payload": "hi"' in cap.out


# ---------- auth dispatch ----------


def test_requires_auth_invokes_client_factory(capsys: pytest.CaptureFixture) -> None:
    seen: list[object] = []

    def runner(client: object) -> _StubResult:
        seen.append(client)
        return _StubResult("ok")

    args = Namespace(json=False, profile=None)
    rc = run_verb("v", args, requires_auth=True, run=runner, render_text=_text, render_json=_json)
    assert rc == EXIT_OK
    assert len(seen) == 1
    assert seen[0] is not None  # got the stub client


def test_no_auth_passes_none_client() -> None:
    seen: list[object] = []

    def runner(client: object) -> _StubResult:
        seen.append(client)
        return _StubResult("ok")

    args = Namespace(json=False)
    run_verb("v", args, requires_auth=False, run=runner, render_text=_text, render_json=_json)
    assert seen == [None]


def test_auth_error_during_client_setup_skips_run(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_runner.Client,
        "from_default_cookies",
        classmethod(lambda cls, **_: (_ for _ in ()).throw(AuthError("no cookies"))),
    )
    ran = False

    def runner(_c: object) -> _StubResult:
        nonlocal ran
        ran = True
        return _StubResult("x")

    rc = run_verb(
        "search",
        Namespace(json=False, profile=None),
        requires_auth=True,
        run=runner,
        render_text=_text,
        render_json=_json,
    )
    assert rc == EXIT_AUTH
    assert ran is False  # never called the verb
    cap = capsys.readouterr()
    assert "pplx search: no cookies" in cap.err


# ---------- error mapping ----------


@pytest.mark.parametrize(
    "err,code",
    [
        (RateLimitError("limited"), EXIT_RATE_LIMIT),
        (NetworkError("timeout"), EXIT_NETWORK),
        (SchemaError("missing field"), EXIT_GENERIC),
    ],
)
def test_verb_error_maps_to_exit_code(
    err: Exception, code: int, capsys: pytest.CaptureFixture
) -> None:
    def runner(_c: object) -> _StubResult:
        raise err

    rc = run_verb(
        "v",
        Namespace(json=False),
        requires_auth=False,
        run=runner,
        render_text=_text,
        render_json=_json,
    )
    assert rc == code
    cap = capsys.readouterr()
    assert "pplx v:" in cap.err


# ---------- warnings ----------


def test_warnings_emit_to_stderr(capsys: pytest.CaptureFixture) -> None:
    args = Namespace(json=False)
    rc = run_verb(
        "v",
        args,
        requires_auth=False,
        run=lambda _c: _StubResult("ok", warnings=["a", "b"]),
        render_text=_text,
        render_json=_json,
    )
    cap = capsys.readouterr()
    assert rc == EXIT_OK
    assert "warning: a" in cap.err
    assert "warning: b" in cap.err


def test_no_warnings_means_clean_stderr(capsys: pytest.CaptureFixture) -> None:
    args = Namespace(json=False)
    run_verb(
        "v",
        args,
        requires_auth=False,
        run=lambda _c: _StubResult("ok"),
        render_text=_text,
        render_json=_json,
    )
    cap = capsys.readouterr()
    assert cap.err == ""


def test_result_without_warnings_attribute_is_tolerated(
    capsys: pytest.CaptureFixture,
) -> None:
    # A verb whose Result doesn't have a .warnings field must still work —
    # the runner uses getattr(..., default=[]) so it never crashes.
    @dataclass
    class _Bare:
        payload: str

    args = Namespace(json=False)
    rc = run_verb(
        "v",
        args,
        requires_auth=False,
        run=lambda _c: _Bare("ok"),
        render_text=lambda r: f"T:{r.payload}",
        render_json=lambda r: {"p": r.payload},
    )
    assert rc == EXIT_OK
    cap = capsys.readouterr()
    assert cap.err == ""


# ---------- finalize ----------


def test_finalize_overrides_exit_code(capsys: pytest.CaptureFixture) -> None:
    rc = run_verb(
        "v",
        Namespace(json=False),
        requires_auth=False,
        run=lambda _c: _StubResult("ok"),
        render_text=_text,
        render_json=_json,
        finalize=lambda _r: EXIT_PARTIAL,
    )
    assert rc == EXIT_PARTIAL


def test_finalize_receives_result() -> None:
    seen: list[_StubResult] = []
    run_verb(
        "v",
        Namespace(json=False),
        requires_auth=False,
        run=lambda _c: _StubResult("payload-value"),
        render_text=_text,
        render_json=_json,
        finalize=lambda r: (seen.append(r), EXIT_OK)[1],
    )
    assert len(seen) == 1
    assert seen[0].payload == "payload-value"


def test_finalize_runs_after_render(capsys: pytest.CaptureFixture) -> None:
    # finalize should see stdout already written — verifies ordering
    def fin(_r: _StubResult) -> int:
        print("from-finalize")  # goes to stdout
        return EXIT_OK

    run_verb(
        "v",
        Namespace(json=False),
        requires_auth=False,
        run=lambda _c: _StubResult("rendered"),
        render_text=_text,
        render_json=_json,
        finalize=fin,
    )
    cap = capsys.readouterr()
    # Both lines on stdout, render before finalize
    assert cap.out.index("TEXT:rendered") < cap.out.index("from-finalize")
