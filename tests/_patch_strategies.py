"""Hypothesis strategies for JSON documents and RFC 6902 ops aimed at them."""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator
from typing import Any, cast

import coverage
from hypothesis import strategies as st

# Scalars on which JSON equality and Python equality agree (no 0/1 next to
# false/true, no floats), so `test` means the same thing in both.
SCALARS = st.none() | st.booleans() | st.integers(2, 99) | st.text("ab~/-0", max_size=3)
KEYS = st.text("ab~/-01", max_size=3)
DOCS = st.recursive(
    SCALARS,
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(KEYS, inner, max_size=4),
    max_leaves=12,
)


def escape(seg: str) -> str:
    return seg.replace("~", "~0").replace("/", "~1")


def ptr(parts: tuple[str, ...]) -> str:
    return "".join("/" + escape(p) for p in parts)


def paths(doc: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    out = [prefix]
    if isinstance(doc, dict):
        for k, v in cast("dict[str, Any]", doc).items():
            out += paths(v, (*prefix, k))
    elif isinstance(doc, list):
        for i, v in enumerate(cast("list[Any]", doc)):
            out += paths(v, (*prefix, str(i)))
    return out


def get(doc: Any, parts: tuple[str, ...]) -> Any:
    for p in parts:
        doc = doc[p] if isinstance(doc, dict) else doc[int(p)]
    return doc


def targets(doc: Any, draw: st.DrawFn) -> tuple[str, ...]:
    """An add/move/copy target: an existing path, a new member, an index
    (possibly one past the end or out of range), `-`, or a child of a scalar."""
    base = draw(st.sampled_from(paths(doc)))
    node = get(doc, base)
    if isinstance(node, dict):
        return (*base, draw(KEYS))
    if isinstance(node, list):
        n = len(cast("list[Any]", node))
        return (*base, draw(st.sampled_from(["-", str(n), str(n + 1), *map(str, range(n))])))
    return (*base, "zz") if draw(st.booleans()) else base


@st.composite
def op_for(draw: st.DrawFn, doc: Any, values: st.SearchStrategy[Any] = DOCS) -> dict[str, Any]:
    kind = draw(st.sampled_from(["add", "remove", "replace", "move", "copy", "test"]))
    existing = draw(st.sampled_from(paths(doc)))
    if kind == "remove":
        return {"op": kind, "path": ptr(existing)}
    if kind in ("replace", "test"):
        value = copy.deepcopy(get(doc, existing)) if draw(st.booleans()) else draw(values)
        return {"op": kind, "path": ptr(existing), "value": value}
    if kind == "add":
        return {"op": kind, "path": ptr(targets(doc, draw)), "value": draw(values)}
    src = existing if existing else ("zz",)  # jsonpatch refuses a root `from`
    return {"op": kind, "from": ptr(src), "path": ptr(targets(doc, draw))}


@contextlib.contextmanager
def untraced() -> Iterator[None]:
    """Pause coverage tracing, which would otherwise dominate a timed region."""
    cov = coverage.Coverage.current()
    if cov is not None:
        cov.stop()
    try:
        yield
    finally:
        if cov is not None:
            cov.start()
