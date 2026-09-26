"""Wire drift: everything the decoder and the store saw but did not expect.

Drift never changes a run's outcome; it is counted so a run can report it and
a canary can fail on it. Every name that reaches a report is a `DriftName`,
built only by `name_of`, which scrubs characters and id-like values, so a
server that keys a map by ids cannot leak one through a drift item.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, NewType, final

DriftName = NewType("DriftName", str)

DriftKind = Literal[
    "unparseable_frame",
    "unknown_sse_field",
    "unknown_sse_event",
    "unknown_envelope_key",
    "unknown_block_key",
    "unknown_diff_key",
    "unexpected_type",
    "unknown_status",
    "unknown_usage",
    "unknown_block_field",
    "malformed_block",
    "patch_rejected",
    "stage_regression",
    "projection_missing",
    "projection_ambiguous",
    "projection_mismatch",
]

MAX_NAME = 64
MAX_KEYS = 64

NO_LOCATION = DriftName("<none>")

_UNSAFE = re.compile(r"[^A-Za-z0-9_.:/-]")
_UUID = re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}")
_RUN = re.compile(r"[A-Za-z0-9_-]{16,}")


def _hide(m: re.Match[str]) -> str:
    run = m.group(0)
    return f"<id-like:{len(run)}>" if any(c.isdigit() for c in run) else run


def name_of(raw: object) -> DriftName:
    """A report-safe name for `raw`: every character outside
    `[A-Za-z0-9_.:/-]` becomes `_`, UUIDs and runs of 16 or more
    `[A-Za-z0-9_-]` characters holding a digit become `<id-like:N>` (N their
    length), and the result is cut to 64 characters. Scrubbing runs before
    the cut, so a cut never exposes part of an id."""
    text = raw if isinstance(raw, str) else type(raw).__name__
    text = _UNSAFE.sub("_", text)
    text = _UUID.sub(lambda m: f"<id-like:{len(m.group(0))}>", text)
    text = _RUN.sub(_hide, text)
    return DriftName(text[:MAX_NAME])


@final
@dataclass(frozen=True, slots=True)
class Drift:
    kind: DriftKind
    name: DriftName


@final
class DriftLedger:
    """Counts drift items: at most `MAX_KEYS` distinct items, after which
    new items only increment `overflow`. Single owner, like the store."""

    __slots__ = ("_counts", "overflow")

    def __init__(self) -> None:
        self._counts: dict[Drift, int] = {}
        self.overflow = 0

    def add(self, items: Iterable[Drift]) -> None:
        for d in items:
            if d in self._counts:
                self._counts[d] += 1
            elif len(self._counts) < MAX_KEYS:
                self._counts[d] = 1
            else:
                self.overflow += 1

    @property
    def total(self) -> int:
        return sum(self._counts.values()) + self.overflow

    def items(self) -> tuple[tuple[Drift, int], ...]:
        return tuple(self._counts.items())

    def __len__(self) -> int:
        return self.total
