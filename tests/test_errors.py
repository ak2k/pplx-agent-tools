"""Unit tests for errors.py — exit-code mapping is part of our public contract."""

from __future__ import annotations

import pytest

from pplx_agent_tools.errors import (
    EXIT_ANTI_BOT,
    EXIT_AUTH,
    EXIT_GENERIC,
    EXIT_NETWORK,
    EXIT_RATE_LIMIT,
    AntiBotError,
    AuthError,
    NetworkError,
    PplxError,
    RateLimitError,
    SchemaError,
    exit_code,
)


def test_auth_error_maps_to_2() -> None:
    assert exit_code(AuthError("missing")) == EXIT_AUTH == 2


def test_rate_limit_error_maps_to_3() -> None:
    err = RateLimitError("429", retry_after=12.5)
    assert exit_code(err) == EXIT_RATE_LIMIT == 3
    assert err.retry_after == 12.5


def test_network_error_maps_to_4() -> None:
    assert exit_code(NetworkError("dns")) == EXIT_NETWORK == 4


def test_anti_bot_error_maps_to_5() -> None:
    assert exit_code(AntiBotError("cf challenge")) == EXIT_ANTI_BOT == 5


def test_schema_error_maps_to_generic_1() -> None:
    # SchemaError is a bug / unexpected response — exit 1, not retryable
    assert exit_code(SchemaError("bad shape")) == EXIT_GENERIC == 1


def test_base_pplxerror_maps_to_generic() -> None:
    assert exit_code(PplxError("?")) == EXIT_GENERIC


def test_unknown_exception_maps_to_generic() -> None:
    # Non-Pplx exceptions get the generic exit code as well — the harness
    # converts them to exit 1 at the CLI boundary.
    assert exit_code(ValueError("boom")) == EXIT_GENERIC


@pytest.mark.parametrize(
    ("err_cls", "expected_code"),
    [
        (AuthError, 2),
        (RateLimitError, 3),
        (NetworkError, 4),
        (AntiBotError, 5),
        (SchemaError, 1),
    ],
)
def test_exit_code_table(err_cls: type[PplxError], expected_code: int) -> None:
    assert exit_code(err_cls("x")) == expected_code


def test_rate_limit_retry_after_optional() -> None:
    assert RateLimitError("no header").retry_after is None
