"""`for_verb` takes every `--timeout` / `--stall-timeout` value, flag or env,
that the ask-family CLIs accept today, with the same meaning."""

from __future__ import annotations

import pytest

from pplx_agent_tools import cli_ask, cli_fetch, cli_research
from pplx_agent_tools.askstream.policy import (
    At,
    Deadline,
    Policy,
    PolicyError,
    SettleAfterText,
    Stall,
    StallAfter,
    StallOff,
    Unbounded,
    Verb,
    deadline_of,
    for_verb,
    stall_of,
)
from pplx_agent_tools.cli_runner import resolve_timeout
from pplx_agent_tools.verbs._ask_common import COPILOT_STALL_SECONDS, DEFAULT_STALL_SECONDS

# Parser argv prefix, deadline env var, default deadline, default stall, per verb,
# as each CLI's main() resolves them.
CLI = {
    "ask": (
        cli_ask.build_parser,
        ["q"],
        "PPLX_ASK_TIMEOUT",
        cli_ask._DEFAULT_TIMEOUT_SECONDS,  # pyright: ignore[reportPrivateUsage]
        COPILOT_STALL_SECONDS,
    ),
    "fetch": (
        cli_fetch.build_parser,
        ["https://example.com", "--prompt", "p"],
        "PPLX_FETCH_TIMEOUT",
        cli_fetch._DEFAULT_PROMPT_TIMEOUT_SECONDS,  # pyright: ignore[reportPrivateUsage]
        COPILOT_STALL_SECONDS,
    ),
    "research": (
        cli_research.build_parser,
        ["q"],
        "PPLX_RESEARCH_TIMEOUT",
        cli_research._DEFAULT_TIMEOUT_SECONDS,  # pyright: ignore[reportPrivateUsage]
        DEFAULT_STALL_SECONDS,
    ),
}
ENV_VARS = ("PPLX_ASK_TIMEOUT", "PPLX_FETCH_TIMEOUT", "PPLX_RESEARCH_TIMEOUT", "PPLX_STALL_TIMEOUT")


def _cli_policy(
    verb: Verb, argv: list[str], env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Policy | PolicyError:
    parser, prefix, env_var, default, stall_default = CLI[verb]
    for k in ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    args = parser().parse_args([*prefix, *argv])
    timeout = resolve_timeout(args.timeout, env_var, default, verb)
    stall = resolve_timeout(args.stall_timeout, "PPLX_STALL_TIMEOUT", stall_default, verb)
    return for_verb(verb, deadline=deadline_of(timeout), stall=stall_of(stall))


ROWS: list[tuple[Verb, list[str], dict[str, str], Deadline, Stall]] = [
    ("ask", [], {}, At(540.0), StallAfter(480.0)),
    # A deadline shorter than the stall window caps the window.
    ("ask", ["--timeout", "60"], {}, At(60.0), StallAfter(60.0)),
    ("ask", ["--timeout", "0"], {}, Unbounded(), StallAfter(480.0)),
    ("ask", ["--timeout", "-5"], {}, Unbounded(), StallAfter(480.0)),
    ("ask", ["--timeout", "inf"], {}, Unbounded(), StallAfter(480.0)),
    ("ask", ["--stall-timeout", "0"], {}, At(540.0), StallOff()),
    ("ask", ["--stall-timeout", "600"], {}, At(540.0), StallAfter(540.0)),
    ("ask", ["--timeout", "0", "--stall-timeout", "0"], {}, Unbounded(), StallOff()),
    ("ask", ["--timeout", "0", "--stall-timeout", "9000"], {}, Unbounded(), StallAfter(9000.0)),
    ("ask", [], {"PPLX_ASK_TIMEOUT": "0"}, Unbounded(), StallAfter(480.0)),
    ("ask", ["--timeout", "60"], {"PPLX_ASK_TIMEOUT": "0"}, At(60.0), StallAfter(60.0)),
    ("ask", [], {"PPLX_STALL_TIMEOUT": "-1"}, At(540.0), StallOff()),
    ("fetch", [], {}, At(540.0), StallAfter(480.0)),
    ("fetch", [], {"PPLX_FETCH_TIMEOUT": "30"}, At(30.0), StallAfter(30.0)),
    ("fetch", [], {"PPLX_FETCH_TIMEOUT": "nan"}, At(540.0), StallAfter(480.0)),
    ("fetch", ["--stall-timeout", "0"], {}, At(540.0), StallOff()),
    ("research", [], {}, At(3600.0), StallAfter(240.0)),
    ("research", ["--timeout", "120"], {}, At(120.0), StallAfter(120.0)),
    ("research", [], {"PPLX_RESEARCH_TIMEOUT": "inf"}, Unbounded(), StallAfter(240.0)),
    ("research", [], {"PPLX_STALL_TIMEOUT": "0"}, At(3600.0), StallOff()),
    ("research", ["--timeout", "0", "--stall-timeout", "inf"], {}, Unbounded(), StallOff()),
]


@pytest.mark.parametrize(("verb", "argv", "env", "deadline", "stall"), ROWS)
def test_for_verb_accepts_every_cli_value(
    verb: Verb,
    argv: list[str],
    env: dict[str, str],
    deadline: Deadline,
    stall: Stall,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _cli_policy(verb, argv, env, monkeypatch)
    assert isinstance(p, Policy), p
    assert (p.deadline, p.stall) == (deadline, stall)


@pytest.mark.parametrize("verb", ["ask", "fetch", "research"])
def test_for_verb_defaults_match_the_cli(verb: Verb, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _cli_policy(verb, [], {}, monkeypatch)
    q = for_verb(verb)
    assert isinstance(p, Policy)
    assert isinstance(q, Policy)
    assert (q.deadline, q.stall) == (p.deadline, p.stall)


@pytest.mark.parametrize("v", [None, 0.0, -1.0, float("inf")])
def test_disabled_values(v: float | None) -> None:
    assert deadline_of(v) == Unbounded()
    assert stall_of(v) == StallOff()


def test_make_caps_the_stall_window_at_the_deadline() -> None:
    p = Policy.make(
        deadline=At(60.0),
        stall=StallAfter(480.0),
        completion=SettleAfterText(15.0),
        answer_paths="ask_text_or_workflow",
    )
    assert isinstance(p, Policy)
    assert p.stall == StallAfter(60.0)
