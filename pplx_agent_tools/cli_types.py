"""Parse-time types for CLI arguments.

Each parser either rejects its input with a usage error or returns a value
that already meets the verb's invariant, so verb code never re-checks ranges.
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import sys
from collections.abc import Iterable
from typing import Final, Literal, NewType, NoReturn, TypeAlias, TypeVar, overload

from .errors import EXIT_GENERIC
from .render import envelope

_N = TypeVar("_N")

PositiveInt = NewType("PositiveInt", int)
# Finite and > 0.
Seconds = NewType("Seconds", float)


class _Disabled(enum.Enum):
    DISABLED = "disabled"


# An explicit "no bound", distinct from None ("not given, use env/default").
DISABLED: Final = _Disabled.DISABLED
Duration: TypeAlias = Seconds | Literal[_Disabled.DISABLED]


class PplxArgumentParser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors exit EXIT_GENERIC.

    argparse's default of 2 collides with EXIT_AUTH, which tells an agent to
    refresh cookies. Subparsers inherit this class via `add_subparsers`.
    When the command line asks for --json, a usage error also prints one
    error envelope to stdout, so a pure-JSON consumer still gets a document.
    """

    _argv: tuple[str, ...] = ()

    @overload
    def parse_known_args(
        self, args: Iterable[str] | None = None, namespace: None = None
    ) -> tuple[argparse.Namespace, list[str]]: ...
    @overload
    def parse_known_args(
        self, args: Iterable[str] | None, namespace: _N
    ) -> tuple[_N, list[str]]: ...
    @overload
    def parse_known_args(self, *, namespace: _N) -> tuple[_N, list[str]]: ...
    def parse_known_args(
        self, args: Iterable[str] | None = None, namespace: object = None
    ) -> tuple[object, list[str]]:
        # error() receives only the message; keep the tokens it needs.
        self._argv = tuple(sys.argv[1:] if args is None else args)
        return super().parse_known_args(self._argv, namespace)

    def _wants_json(self) -> bool:
        if "--json" not in self._option_string_actions:
            return False
        for tok in self._argv:
            if tok == "--":
                break
            # argparse accepts unambiguous long-option prefixes and short clusters.
            if tok.startswith("--") and len(tok) > 2 and "--json".startswith(tok):
                return True
            if tok.startswith("-j"):
                return True
        return False

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        if self._wants_json():
            verb = self.prog.removeprefix("pplx ").split(" ")[0]
            error_obj = {"type": "UsageError", "message": message, "exit_code": EXIT_GENERIC}
            print(json.dumps(envelope(verb, {"error": error_obj}), indent=2))
        self.exit(EXIT_GENERIC, f"{self.prog}: error: {message}\n")


def positive_int(text: str) -> PositiveInt:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {n}")
    return PositiveInt(n)


def duration(text: str) -> Duration:
    """Seconds, or DISABLED for 0, a negative value, or infinity. NaN is
    rejected: a NaN deadline never fires because every comparison with it is
    False, and it has no sensible meaning to map to."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of seconds, got {text!r}") from None
    if math.isnan(v):
        raise argparse.ArgumentTypeError(f"expected a number of seconds, got {text!r}")
    if v <= 0 or math.isinf(v):
        return DISABLED
    return Seconds(v)
