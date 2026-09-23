"""Parse-time types for CLI arguments.

Each parser either rejects its input with a usage error or returns a value
that already meets the verb's invariant, so verb code never re-checks ranges.
"""

from __future__ import annotations

import argparse
import enum
import math
import sys
from typing import Final, Literal, NewType, NoReturn, TypeAlias

from .errors import EXIT_GENERIC

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
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
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
    """Seconds, or DISABLED for 0 or a negative value. NaN and infinities are
    rejected: a NaN deadline never fires because every comparison with it is
    False."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of seconds, got {text!r}") from None
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"expected a finite number of seconds, got {text!r}")
    if v <= 0:
        return DISABLED
    return Seconds(v)
