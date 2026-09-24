"""JSON values for the stream decoder: parsing, validation and weight.

`JsonValue` and the one parser cast come from `pplx_agent_tools.jsonval`.
Every walk here is iterative, so no input depth can exhaust the C stack.

Weight is an additive size measure, within one byte per container of compact
`json.dumps`: a scalar weighs its JSON text length, a list
`2 + sum(weight(e) + 1)`, an object `2 + sum(len(json(k)) + weight(v) + 2)`.
Being additive, it can be kept exact by crediting and charging subtrees.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from json.encoder import encode_basestring_ascii
from typing import Literal, NoReturn, cast, final

from pplx_agent_tools.jsonval import JsonValue, from_parser

__all__ = [
    "LIST_OVERHEAD",
    "MAX_DEPTH",
    "JsonError",
    "JsonValue",
    "Measured",
    "key_overhead",
    "loads",
    "measure",
    "scalar_weight",
    "weight",
]

MAX_DEPTH = 128
LIST_OVERHEAD = 1


@final
@dataclass(frozen=True, slots=True)
class JsonError:
    reason: Literal["syntax", "depth"]


@final
@dataclass(frozen=True, slots=True)
class Measured:
    value: JsonValue
    weight: int
    nodes: int


def scalar_weight(v: JsonValue) -> int:
    """`len(json.dumps(v))` for a scalar; containers are not scalars."""
    if isinstance(v, str):
        return len(encode_basestring_ascii(v))
    if v is None or isinstance(v, bool):
        return 5 if v is False else 4
    if isinstance(v, int):
        try:
            return len(int.__repr__(v))
        except ValueError:
            # Past the interpreter's int-to-str digit limit; a JSON parser under
            # that limit never produces one. Upper bound on the digit count.
            return v.bit_length() * 30103 // 100000 + 2
    if isinstance(v, float):
        # A finite float's JSON text is its repr, which skips the encoder setup.
        return len(float.__repr__(v)) if math.isfinite(v) else len(json.dumps(v))
    return 0


def key_overhead(key: str) -> int:
    """What an object member adds to its object's weight besides its value."""
    return len(encode_basestring_ascii(key)) + 2


def _children(v: object) -> Iterator[tuple[int, object]] | None:
    """(member overhead, child) pairs of a container, else None."""
    if isinstance(v, dict):
        items = cast("dict[object, object]", v).items()
        return ((key_overhead(k) if isinstance(k, str) else -1, c) for k, c in items)
    if isinstance(v, list):
        return ((LIST_OVERHEAD, c) for c in cast("list[object]", v))
    return None


def _is_scalar(v: object) -> bool:
    """A JSON scalar; NaN and the infinities have no JSON form."""
    if isinstance(v, float):
        return math.isfinite(v)
    return v is None or isinstance(v, (bool, int, str))


def measure(value: object, max_depth: int = MAX_DEPTH) -> Measured | None:
    """Weight and node count of `value`, or None when it is not JSON-shaped
    (a non-str key, a non-finite float, a type JSON has no form for) or nests
    deeper than `max_depth` containers. The depth bound also ends the walk on
    a cycle."""
    if _is_scalar(value):
        return Measured(from_parser(value), scalar_weight(from_parser(value)), 1)
    root = _children(value)
    if root is None:
        return None
    total, nodes = 2, 1
    stack = [root]
    while stack:
        pair = next(stack[-1], None)
        if pair is None:
            stack.pop()
            continue
        overhead, child = pair
        if overhead < 0:
            return None
        nodes += 1
        total += overhead
        if _is_scalar(child):
            total += scalar_weight(from_parser(child))
            continue
        sub = _children(child)
        if sub is None or len(stack) >= max_depth:
            return None
        total += 2
        stack.append(sub)
    return Measured(from_parser(value), total, nodes)


def weight(value: JsonValue) -> int:
    """Weight of a JSON value at any depth."""
    m = measure(value, max_depth=2**62)
    return 0 if m is None else m.weight


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(name)


def _finite_float(text: str) -> float:
    f = float(text)
    if not math.isfinite(f):
        raise ValueError(text)
    return f


def loads(raw: str) -> JsonValue | JsonError:
    """Parse JSON text; nesting deeper than `MAX_DEPTH` is an error too, so
    every later walker may recurse safely. `NaN`, `Infinity`, `-Infinity`
    and numbers too large for a float are not JSON: `syntax`. Never raises."""
    try:
        value = cast(
            "object",
            json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float),
        )
    except RecursionError:
        return JsonError("depth")
    except ValueError:
        return JsonError("syntax")
    m = measure(value)
    return JsonError("depth") if m is None else m.value
