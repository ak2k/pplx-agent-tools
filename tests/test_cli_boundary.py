"""CLI boundary contract: usage errors, parsed arg ranges, plain fetch without
cookies, and the --json envelope on unexpected exceptions."""

from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pplx_agent_tools import cli, cli_auth, cli_fetch, cli_runner, cli_types
from pplx_agent_tools.cli_runner import resolve_timeout, run_verb
from pplx_agent_tools.errors import EXIT_AUTH, EXIT_GENERIC, EXIT_OK, AuthError
from pplx_agent_tools.verbs import fetch as fetch_verb
from pplx_agent_tools.verbs.fetch import FetchResult

# ---------- usage errors exit 1, never the auth code ----------


@pytest.mark.parametrize(
    "argv",
    [[verb, "--no-such-flag"] for verb in cli.VERBS] + [["auth", "check", "--no-such-flag"]],
    ids=lambda argv: " ".join(argv[:-1]),
)
def test_unknown_flag_exits_generic(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        cli.main(argv)
    assert ei.value.code == EXIT_GENERIC


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "--limit", "x", "q"],
        ["search", "--limit", "0", "q"],
        ["fetch", "--max-chars", "-10", "https://example.com"],
        ["ask"],
        ["ask", "--timeout", "nan", "q"],
        ["research", "--stall-timeout", "nan", "q"],
        ["fetch", "--max-chars", "0", "https://example.com"],
        ["snippets", "--max-tokens", "-1", "q", "https://example.com"],
    ],
    ids=" ".join,
)
def test_bad_value_or_missing_positional_exits_generic(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        cli.main(argv)
    assert ei.value.code == EXIT_GENERIC


def test_help_still_exits_zero() -> None:
    with pytest.raises(SystemExit) as ei:
        cli.main(["search", "--help"])
    assert ei.value.code == 0


# ---------- parsers: reject, or return a value meeting the invariant ----------

_numeric_text = st.one_of(
    st.integers().map(str),
    st.floats(allow_nan=True, allow_infinity=True).map(repr),
    st.sampled_from(["nan", "-nan", "inf", "-inf", "Infinity", "0", "-0", "1e999", " 3 ", ""]),
    st.text(max_size=8),
)


@given(_numeric_text)
def test_positive_int_rejects_or_is_positive(text: str) -> None:
    try:
        n = cli_types.positive_int(text)
    except argparse.ArgumentTypeError:
        return
    assert isinstance(n, int)
    assert n >= 1


@given(_numeric_text)
def test_duration_rejects_or_is_finite_positive(text: str) -> None:
    try:
        d = cli_types.duration(text)
    except argparse.ArgumentTypeError:
        return
    if d is not cli_types.DISABLED:
        assert isinstance(d, float)
        assert math.isfinite(d)
        assert d > 0


@pytest.mark.parametrize("text", ["nan", "-nan", "NaN"])
def test_duration_rejects_nan(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli_types.duration(text)


@pytest.mark.parametrize("text", ["0", "-5", "0.0", "inf", "-inf", "Infinity", "1e999"])
def test_duration_non_positive_or_infinite_disables(text: str) -> None:
    assert cli_types.duration(text) is cli_types.DISABLED


@pytest.mark.parametrize("env", ["nan", "abc"])
def test_resolve_timeout_bad_env_warns_and_defaults(
    env: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PPLX_TEST_TIMEOUT", env)
    assert resolve_timeout(None, "PPLX_TEST_TIMEOUT", 42.0, "ask") == 42.0
    assert "PPLX_TEST_TIMEOUT" in capsys.readouterr().err


@pytest.mark.parametrize("env", ["0", "-1", "inf", "-inf"])
def test_resolve_timeout_env_disables(env: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPLX_TEST_TIMEOUT", env)
    assert resolve_timeout(None, "PPLX_TEST_TIMEOUT", 42.0, "ask") is None


def test_resolve_timeout_flag_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPLX_TEST_TIMEOUT", "7")
    assert resolve_timeout(cli_types.duration("3"), "PPLX_TEST_TIMEOUT", 42.0, "ask") == 3.0
    assert resolve_timeout(cli_types.duration("inf"), "PPLX_TEST_TIMEOUT", 42.0, "ask") is None


# ---------- usage errors under --json print exactly one envelope ----------


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "--json", "--limit", "0", "q"],
        ["search", "-j", "--no-such-flag", "q"],
        ["ask", "--js"],
        ["fetch", "--timeout", "nan", "-j", "https://example.com"],
    ],
    ids=" ".join,
)
def test_json_usage_error_prints_one_envelope(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as ei:
        cli.main(argv)
    assert ei.value.code == EXIT_GENERIC
    cap = capsys.readouterr()
    out = json.loads(cap.out)  # raises on zero or two documents
    assert out["_verb"] == argv[0]
    assert out["error"]["type"] == "UsageError"
    assert out["error"]["exit_code"] == EXIT_GENERIC
    assert "usage:" in cap.err


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "--limit", "0", "q"],
        ["search", "--limit", "0", "--", "--json"],
        ["auth", "--json"],
    ],
    ids=" ".join,
)
def test_usage_error_without_json_keeps_stdout_empty(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as ei:
        cli.main(argv)
    assert ei.value.code == EXIT_GENERIC
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "error:" in cap.err


# ---------- plain fetch needs no cookies ----------


@pytest.fixture
def _no_cookies(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(cls: object, **_: object) -> object:
        raise AuthError("no cookies found")

    monkeypatch.setattr(cli_runner.Client, "from_default_cookies", classmethod(_raise))


@pytest.mark.usefixtures("_no_cookies")
def test_plain_fetch_runs_without_cookies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _page(url: str, domain: str, **_: Any) -> FetchResult:
        return FetchResult(url=url, title=None, domain=domain, content="body", is_extracted=False)

    monkeypatch.setattr(fetch_verb, "fetch_page", _page)
    assert cli_fetch.main(["https://example.com/a"]) == EXIT_OK
    assert "body" in capsys.readouterr().out


@pytest.mark.usefixtures("_no_cookies")
def test_prompt_fetch_still_requires_cookies() -> None:
    assert cli_fetch.main(["https://example.com/a", "--prompt", "tldr"]) == EXIT_AUTH


# ---------- unexpected exceptions still yield a --json envelope ----------


def _render_boom(_r: object) -> dict[str, Any]:
    raise RuntimeError("renderer bug")


def test_json_internal_error_envelope(capsys: pytest.CaptureFixture[str]) -> None:
    rc = run_verb(
        "search",
        Namespace(json=True),
        requires_auth=False,
        run=lambda _c: object(),
        render_text=lambda _r: "",
        render_json=_render_boom,
    )
    assert rc == EXIT_GENERIC
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    assert out["_verb"] == "search"
    assert out["error"]["type"] == "InternalError"
    assert "renderer bug" in out["error"]["message"]
    assert out["error"]["exit_code"] == EXIT_GENERIC
    assert "Traceback" in cap.err


def test_text_internal_error_keeps_stdout_empty(capsys: pytest.CaptureFixture[str]) -> None:
    def _boom(_c: object) -> object:
        raise RuntimeError("verb bug")

    rc = run_verb(
        "search",
        Namespace(json=False),
        requires_auth=False,
        run=_boom,
        render_text=lambda _r: "",
        render_json=lambda _r: {},
    )
    assert rc == EXIT_GENERIC
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "verb bug" in cap.err


def test_failure_after_json_output_does_not_emit_second_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _finalize(_r: object) -> int:
        raise RuntimeError("finalize bug")

    rc = run_verb(
        "search",
        Namespace(json=True),
        requires_auth=False,
        run=lambda _c: object(),
        render_text=lambda _r: "",
        render_json=lambda _r: {"ok": True},
        finalize=_finalize,
    )
    assert rc == EXIT_GENERIC
    assert json.loads(capsys.readouterr().out) == {"ok": True}


@pytest.mark.parametrize("json_mode", [True, False])
def test_typed_error_from_renderer_keeps_its_exit_code(
    json_mode: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    def _render(_r: object) -> Any:
        raise AuthError("session expired mid-render")

    rc = run_verb(
        "search",
        Namespace(json=json_mode),
        requires_auth=False,
        run=lambda _c: object(),
        render_text=_render,
        render_json=_render,
    )
    assert rc == EXIT_AUTH
    out = capsys.readouterr().out
    if json_mode:
        assert json.loads(out)["error"]["type"] == "AuthError"
    else:
        assert out == ""


def test_typed_error_from_finalize_after_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    def _finalize(_r: object) -> int:
        raise AuthError("late")

    rc = run_verb(
        "search",
        Namespace(json=True),
        requires_auth=False,
        run=lambda _c: object(),
        render_text=lambda _r: "",
        render_json=lambda _r: {"ok": True},
        finalize=_finalize,
    )
    assert rc == EXIT_AUTH
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_keyboard_interrupt_is_not_caught() -> None:
    def _interrupt(_c: object) -> object:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_verb(
            "search",
            Namespace(json=True),
            requires_auth=False,
            run=_interrupt,
            render_text=lambda _r: "",
            render_json=lambda _r: {},
        )


def test_auth_subparser_inherits_parser_class() -> None:
    with pytest.raises(SystemExit) as ei:
        cli_auth.main(["import"])
    assert ei.value.code == EXIT_GENERIC
