"""CPU bounds expressed against a bare node walk timed on the same machine.

CI runners are several times slower than a workstation and shared hosts run
under load, so a fixed number of seconds is either too tight there or too
loose here; a ratio to this walk holds on both.
"""

from __future__ import annotations

import time
from typing import Any


def walk_seconds(nodes: int) -> float:
    """Process time of the fastest of three stack walks over `nodes` list items."""
    data: list[Any] = [0] * nodes
    best = float("inf")
    for _ in range(3):
        start = time.process_time()
        stack: list[Any] = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
        best = min(best, time.process_time() - start)
    return best
