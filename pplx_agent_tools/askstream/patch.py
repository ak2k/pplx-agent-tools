"""RFC 6902 JSON Patch over untrusted ops, with bounded cost.

`apply_ops` never raises. Every op is charged work units computed from the
document before the op, and its weight change is known, before anything is
mutated; a charge that would pass a cap ends the call as `CapExceeded` with
the op not applied. Work units:

- for each segment of each pointer the op uses, one plus `len // 64`;
- for each node read while measuring a subtree (the old value of a
  non-root `replace` or `remove`, an overwritten object member, a `copy`
  source, a `move` value when the target is the root, an ancestor object
  member, or more pointer segments deep than the source) or comparing
  (`test`, and the changed/noop check on `replace`, which stops at the
  first difference): one, plus `len // 64` for its member key and for a
  string value, `(digits // 64) ** 2` for an int (the decimal conversion is
  quadratic; digits is an upper bound from the bit length), and 3 for a
  float. A compared pair is charged by the document
  side's node. Each node's units are checked against the caps before it is
  read, so a long string costs nothing until it is paid for;
- one per node deep-copied: a `copy` source, and the value of an `add` or
  `replace`, which is copied so that the document owns every node it holds
  (strings and keys are shared, not copied);
- one per array element shifted: an insert at i on length n shifts n - i,
  a removal at i shifts n - i - 1, including a root `move`'s removal of
  the value from its parent.

No op leaves a container nested deeper than `MAX_DEPTH`, the depth `loads`
accepts: a value placed under a pointer of n segments may be at most
`MAX_DEPTH - n` containers deep, else the op is `Rejected("too_deep")`.

A unit costs about 0.43 us of CPU on tiny scalars and up to about 0.65 us on
the worst measured inputs; the timing tests bound it at 30 times a bare node
walk on the same machine. 64
characters of the costliest string to encode (non-BMP, 12 output bytes each)
take about 0.2 us.

A root `remove` is rejected, as in RFC 6902 there is nothing left to hold.
A root `move` or `copy` target replaces the document; a root `move` detaches
the value from its old parent, so the old tree holds no node of the new one.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast, final

from typing_extensions import assert_never

from pplx_agent_tools.askstream.jsonval import (
    LIST_OVERHEAD,
    MAX_DEPTH,
    JsonValue,
    Measured,
    key_overhead,
    measure,
    scalar_weight,
)

Pointer: TypeAlias = tuple[str, ...]
MAX_POINTER_SEGMENTS = 128
_BAD_ESCAPE = re.compile(r"~(?![01])")

Changed = Literal["changed", "noop"]
CapName = Literal["field_weight", "total_weight", "ops_per_frame", "work_per_frame", "work_per_run"]
RejectReason = Literal[
    "missing_target",
    "not_a_container",
    "bad_index",
    "index_out_of_range",
    "end_index",
    "test_failed",
    "move_into_child",
    "remove_root",
    "too_deep",
]

_Container: TypeAlias = "dict[str, JsonValue] | list[JsonValue]"

KIB = 1024
MIB = 1024 * KIB

TEXT_CHUNK = 64
FLOAT_UNITS = 3


def _text_units(s: str) -> int:
    return len(s) // TEXT_CHUNK


def _scalar_units(v: JsonValue) -> int:
    """Units, beyond the node's one, to read a scalar's JSON text."""
    if isinstance(v, str):
        return len(v) // TEXT_CHUNK
    if isinstance(v, bool) or v is None:
        return 0
    if isinstance(v, int):
        digits = v.bit_length() * 30103 // 100000 + 1
        return (digits // TEXT_CHUNK) ** 2
    return FLOAT_UNITS if isinstance(v, float) else 0


def node_units(key: str | None, v: JsonValue) -> int:
    """Work units to read one node (and its member key) in a walk."""
    return 1 + (0 if key is None else _text_units(key)) + _scalar_units(v)


def pointer_units(p: Pointer) -> int:
    """Work units to resolve a pointer: each segment is read like a key."""
    return len(p) + sum(len(seg) // TEXT_CHUNK for seg in p)


def parse_pointer(raw: object) -> Pointer | None:
    """RFC 6901 pointer, or None: not a string, no leading `/`, a `~` not
    followed by `0` or `1`, or more than 128 segments (counted before split)."""
    if not isinstance(raw, str):
        return None
    if raw == "":
        return ()
    if raw[0] != "/" or raw.count("/") > MAX_POINTER_SEGMENTS or _BAD_ESCAPE.search(raw):
        return None
    return tuple(s.replace("~1", "/").replace("~0", "~") for s in raw[1:].split("/"))


@final
@dataclass(frozen=True, slots=True)
class Add:
    path: Pointer
    value: JsonValue
    value_weight: int
    value_nodes: int
    value_depth: int
    op: Literal["add"] = "add"


@final
@dataclass(frozen=True, slots=True)
class Remove:
    path: Pointer
    op: Literal["remove"] = "remove"


@final
@dataclass(frozen=True, slots=True)
class Replace:
    path: Pointer
    value: JsonValue
    value_weight: int
    value_nodes: int
    value_depth: int
    op: Literal["replace"] = "replace"


@final
@dataclass(frozen=True, slots=True)
class Move:
    from_: Pointer
    path: Pointer
    op: Literal["move"] = "move"


@final
@dataclass(frozen=True, slots=True)
class Copy:
    from_: Pointer
    path: Pointer
    op: Literal["copy"] = "copy"


@final
@dataclass(frozen=True, slots=True)
class Test:
    path: Pointer
    value: JsonValue
    value_weight: int
    op: Literal["test"] = "test"


PatchOp: TypeAlias = Add | Remove | Replace | Move | Copy | Test


def parse_patch_op(raw: object) -> PatchOp | None:
    """One op from a decoded JSON object, or None when it is not a valid op.
    Members RFC 6902 does not define are ignored. A `value` must be a JSON
    tree nesting at most 128 deep; its weight, node count and depth are
    computed here. The op borrows `value`; the applier inserts copies."""
    if not isinstance(raw, dict):
        return None
    obj = cast("dict[object, object]", raw)
    op = obj.get("op")
    path = parse_pointer(obj.get("path"))
    if path is None:
        return None
    if op == "remove":
        return Remove(path)
    if op in ("move", "copy"):
        src = parse_pointer(obj.get("from"))
        if src is None:
            return None
        return Move(src, path) if op == "move" else Copy(src, path)
    m = measure(obj["value"]) if op in ("add", "replace", "test") and "value" in obj else None
    return None if m is None else _value_op(cast("str", op), path, m)


def _value_op(op: str, path: Pointer, m: Measured) -> PatchOp:
    if op == "test":
        return Test(path, m.value, m.weight)
    make = Add if op == "add" else Replace
    return make(path, m.value, m.weight, m.nodes, m.depth)


@final
@dataclass(frozen=True, slots=True)
class Limits:
    field_weight: int = 16 * MIB
    total_weight: int = 24 * MIB
    ops_per_frame: int = 4096
    work_per_frame: int = 1 * MIB
    # The run cap is run_work_factor times the bytes received, plus run_work_base.
    run_work_factor: int = 8
    run_work_base: int = 16 * MIB


@final
class Budget:
    """Per-run accounting shared by every field of one store; single owner.

    `total_weight` is the sum of the tracked fields' weights: the owner adds
    a field's weight when it starts tracking one and credits it when it drops
    one; `apply_ops` moves it by each op's weight change.
    """

    __slots__ = ("bytes_received", "frame_ops", "frame_work", "limits", "run_work", "total_weight")

    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.total_weight = 0
        self.bytes_received = 0
        self.run_work = 0
        self.frame_work = 0
        self.frame_ops = 0

    def start_frame(self, bytes_in: int) -> None:
        self.bytes_received += bytes_in
        self.frame_work = 0
        self.frame_ops = 0

    @property
    def run_work_limit(self) -> int:
        return self.limits.run_work_factor * self.bytes_received + self.limits.run_work_base


class Probe(Protocol):
    """Test instrumentation: every node visit, every key or scalar whose
    text is read, every array shift, and the state after every applied op."""

    def visit(self) -> None: ...
    def read(self, text: JsonValue) -> None: ...
    def shift(self, n: int) -> None: ...
    def op_done(self, index: int, charge: int, doc: JsonValue, weight: int) -> None: ...


@final
@dataclass(frozen=True, slots=True)
class Applied:
    doc: JsonValue
    weight: int
    changed: Changed


@final
@dataclass(frozen=True, slots=True)
class Rejected:
    """Op `index` is invalid for the document; `weight` is as on every result
    (see `apply_ops`)."""

    index: int
    reason: RejectReason
    weight: int


@final
@dataclass(frozen=True, slots=True)
class CapExceeded:
    """Op `index` would pass `cap`; it was not applied. `observed` is a lower
    bound on the value the cap would have reached. `weight` is as on every
    result (see `apply_ops`)."""

    cap: CapName
    limit: int
    observed: int
    index: int
    weight: int


class _Halt(Exception):
    def __init__(self, result: Rejected | CapExceeded) -> None:
        super().__init__()
        self.result = result


def _json_scalar_eq(a: JsonValue, b: JsonValue) -> bool:
    # JSON equality: true is not 1, and 1 equals 1.0.
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    return a == b


def _is_container(v: JsonValue) -> bool:
    return isinstance(v, (dict, list))


def _canonical_index(seg: str, n: int, allow_end: bool) -> int | RejectReason:
    """Index into an array of length n. `-` and `n` itself are allowed only
    for an insert. Checked on the string before `int()`, so an index of any
    length costs no allocation."""
    if seg == "-":
        return n if allow_end else "end_index"
    if not seg or not seg.isascii() or not seg.isdigit() or (seg[0] == "0" and seg != "0"):
        return "bad_index"
    if len(seg) > len(str(n)):
        return "index_out_of_range"
    i = int(seg)
    if i > n or (i == n and not allow_end):
        return "index_out_of_range"
    return i


class _Run:
    __slots__ = ("budget", "doc", "index", "probe", "weight")

    def __init__(self, doc: JsonValue, weight: int, budget: Budget, probe: Probe | None) -> None:
        self.doc = doc
        self.weight = weight
        self.budget = budget
        self.probe = probe
        self.index = 0

    def _reject(self, reason: RejectReason) -> _Halt:
        return _Halt(Rejected(self.index, reason, self.weight))

    def _cap(self, cap: CapName, limit: int, observed: int) -> _Halt:
        return _Halt(CapExceeded(cap, limit, observed, self.index, self.weight))

    # --- work ---------------------------------------------------------------

    def _room(self) -> int:
        b = self.budget
        return max(0, min(b.limits.work_per_frame - b.frame_work, b.run_work_limit - b.run_work))

    def _spend(self, n: int) -> None:
        b = self.budget
        if b.frame_work + n > b.limits.work_per_frame:
            raise self._cap("work_per_frame", b.limits.work_per_frame, b.frame_work + n)
        limit = b.run_work_limit
        if b.run_work + n > limit:
            raise self._cap("work_per_run", limit, b.run_work + n)
        b.frame_work += n
        b.run_work += n

    def _spend_pointers(self, *ptrs: Pointer) -> None:
        self._spend(sum(pointer_units(p) for p in ptrs))
        if self.probe is not None:
            for p in ptrs:
                for seg in p:
                    self.probe.read(seg)

    def _overrun(self, done: int, need: int) -> _Halt:
        """A bounded walk spent `done` units and its next node, costing
        `need`, does not fit the room left."""
        self._spend(done)
        b = self.budget
        if b.frame_work + need > b.limits.work_per_frame:
            return self._cap("work_per_frame", b.limits.work_per_frame, b.frame_work + need)
        return self._cap("work_per_run", b.run_work_limit, b.run_work + need)

    # --- walks (each bounded by the remaining work, each visit recorded) ----

    def _read(self, key: str | None, v: JsonValue) -> None:
        probe = self.probe
        if probe is not None:
            probe.visit()
            if key is not None:
                probe.read(key)
            if not _is_container(v):
                probe.read(v)

    def _measure(self, v: JsonValue) -> tuple[int, int, int]:
        """(nodes, weight, depth in containers) of a subtree, charged
        `node_units` per node."""
        room = self._room()
        units = nodes = total = depth = 0
        stack: list[Iterator[tuple[str | None, JsonValue]]] = [iter(((None, v),))]
        while stack:
            item = next(stack[-1], None)
            if item is None:
                stack.pop()
                continue
            key, child = item
            cost = node_units(key, child)
            if units + cost > room:
                raise self._overrun(units, cost)
            units += cost
            nodes += 1
            self._read(key, child)
            if len(stack) > 1:
                total += LIST_OVERHEAD if key is None else key_overhead(key)
            if isinstance(child, (dict, list)):
                total += 2
                stack.append(_keyed(child))
                depth = max(depth, len(stack) - 1)
            else:
                total += scalar_weight(child)
        self._spend(units)
        return nodes, total, depth

    def _equal(self, a: JsonValue, b: JsonValue) -> bool:
        """JSON equality, pre-order, stopping at the first difference. Each
        pair is charged `node_units` of its `a` side; a string compare or key
        lookup costs at most the length of that side's text."""
        room = self._room()
        units = 0
        stack: list[Iterator[_Pair]] = [iter(((None, a, b),))]
        while stack:
            item = next(stack[-1], None)
            if item is None:
                stack.pop()
                continue
            key, x, y = item
            cost = node_units(key, x)
            if units + cost > room:
                raise self._overrun(units, cost)
            units += cost
            self._read(key, x)
            if key is not None:
                y = cast("dict[str, JsonValue]", y).get(key, _MISSING)
            if isinstance(y, _Missing):
                same = False
            elif isinstance(x, dict):
                same = isinstance(y, dict) and len(x) == len(y)
                if same:
                    stack.append(_dict_pairs(x, cast("dict[str, JsonValue]", y)))
            elif isinstance(x, list):
                same = isinstance(y, list) and len(x) == len(y)
                if same:
                    stack.append(_list_pairs(x, cast("list[JsonValue]", y)))
            else:
                same = not _is_container(y) and _json_scalar_eq(x, y)
            if not same:
                self._spend(units)
                return False
        self._spend(units)
        return True

    def _deep_copy(self, v: JsonValue) -> JsonValue:
        """A deep copy; its node count is charged by `_put` before the call."""
        probe = self.probe
        if probe is not None:
            probe.visit()
        if not isinstance(v, (dict, list)):
            return v
        root: _Container = {} if isinstance(v, dict) else []
        stack: list[tuple[Iterator[tuple[str | None, JsonValue]], _Container]] = [(_keyed(v), root)]
        while stack:
            src, dst = stack[-1]
            item = next(src, None)
            if item is None:
                stack.pop()
                continue
            if probe is not None:
                probe.visit()
            key, child = item
            new: JsonValue = child
            if isinstance(child, (dict, list)):
                fresh: _Container = {} if isinstance(child, dict) else []
                stack.append((_keyed(child), fresh))
                new = fresh
            if isinstance(dst, dict):
                dst[cast("str", key)] = new
            else:
                dst.append(new)
        return root

    # --- pointers -------------------------------------------------------------

    def _index(self, seg: str, n: int, allow_end: bool) -> int:
        i = _canonical_index(seg, n, allow_end)
        if isinstance(i, str):
            raise self._reject(i)
        return i

    def _parent(
        self, path: Pointer, skip: tuple[list[JsonValue], int] | None = None
    ) -> tuple[_Container, str]:
        """The container holding `path`'s last segment. With `skip`, resolve
        as if element `skip[1]` of array `skip[0]` were already removed."""
        cur = self.doc
        for seg in path[:-1]:
            if self.probe is not None:
                self.probe.visit()
            if isinstance(cur, dict):
                if seg not in cur:
                    raise self._reject("missing_target")
                cur = cur[seg]
            elif isinstance(cur, list):
                if skip is not None and cur is skip[0]:
                    i = self._index(seg, len(cur) - 1, allow_end=False)
                    cur = cur[i if i < skip[1] else i + 1]
                else:
                    cur = cur[self._index(seg, len(cur), allow_end=False)]
            else:
                raise self._reject("not_a_container")
        if isinstance(cur, (dict, list)):
            return cur, path[-1]
        raise self._reject("not_a_container")

    def _locate(self, path: Pointer) -> tuple[_Container, str | int, JsonValue]:
        """(parent, key or index, value) of an existing non-root target."""
        parent, seg = self._parent(path)
        if self.probe is not None:
            self.probe.visit()
        if isinstance(parent, dict):
            if seg not in parent:
                raise self._reject("missing_target")
            return parent, seg, parent[seg]
        i = self._index(seg, len(parent), allow_end=False)
        return parent, i, parent[i]

    def _get(self, path: Pointer) -> JsonValue:
        return self.doc if not path else self._locate(path)[2]

    # --- weight ---------------------------------------------------------------

    def _admit(self, delta: int) -> None:
        lim = self.budget.limits
        nw = self.weight + delta
        if nw > lim.field_weight:
            raise self._cap("field_weight", lim.field_weight, nw)
        nt = self.budget.total_weight + delta
        if nt > lim.total_weight:
            raise self._cap("total_weight", lim.total_weight, nt)

    def _fits(self, path: Pointer, depth: int) -> None:
        # Each segment of `path` passes through one container above the value.
        if len(path) + depth > MAX_DEPTH:
            raise self._reject("too_deep")

    def _commit(self, delta: int) -> None:
        self.weight += delta
        self.budget.total_weight += delta

    def _shift(self, n: int) -> None:
        if self.probe is not None:
            self.probe.shift(n)

    # --- ops --------------------------------------------------------------------

    def _put(
        self, path: Pointer, src: JsonValue, value_weight: int, nodes: int, depth: int
    ) -> None:
        """RFC 6902 `add` of a deep copy of `src`, which has `nodes` nodes and
        is `depth` containers deep. The copy is charged, and made, only after
        every other charge and check has passed."""
        if not path:
            self._fits(path, depth)
            self._admit(value_weight - self.weight)
            self._spend(nodes)
            self.doc = self._deep_copy(src)
            self._commit(value_weight - self.weight)
            return
        parent, seg = self._parent(path)
        self._fits(path, depth)
        if isinstance(parent, dict):
            if seg in parent:
                delta = value_weight - self._measure(parent[seg])[1]
            else:
                delta = key_overhead(seg) + value_weight
            self._admit(delta)
            self._spend(nodes)
            parent[seg] = self._deep_copy(src)
        else:
            n = len(parent)
            i = self._index(seg, n, allow_end=True)
            self._spend(n - i)
            delta = value_weight + LIST_OVERHEAD
            self._admit(delta)
            self._spend(nodes)
            new = self._deep_copy(src)
            self._shift(n - i)
            parent.insert(i, new)
        self._commit(delta)

    def add(self, op: Add) -> Changed:
        self._spend_pointers(op.path)
        self._put(op.path, op.value, op.value_weight, op.value_nodes, op.value_depth)
        return "changed"

    def remove(self, op: Remove) -> Changed:
        self._spend_pointers(op.path)
        if not op.path:
            raise self._reject("remove_root")
        parent, key, old = self._locate(op.path)
        w_old = self._measure(old)[1]
        if isinstance(parent, dict):
            delta = -(key_overhead(cast("str", key)) + w_old)
            self._commit(delta)
            del parent[cast("str", key)]
        else:
            i = cast("int", key)
            self._spend(len(parent) - i - 1)
            self._shift(len(parent) - i - 1)
            self._commit(-(LIST_OVERHEAD + w_old))
            parent.pop(i)
        return "changed"

    def replace(self, op: Replace) -> Changed:
        self._spend_pointers(op.path)
        if not op.path:
            self._fits(op.path, op.value_depth)
            same = self._equal(self.doc, op.value)
            delta = op.value_weight - self.weight
            self._admit(delta)
            self._spend(op.value_nodes)
            self.doc = self._deep_copy(op.value)
            self._commit(delta)
            return "noop" if same else "changed"
        parent, key, old = self._locate(op.path)
        self._fits(op.path, op.value_depth)
        w_old = self._measure(old)[1]
        same = self._equal(old, op.value)
        delta = op.value_weight - w_old
        self._admit(delta)
        self._spend(op.value_nodes)
        new = self._deep_copy(op.value)
        if isinstance(parent, dict):
            parent[cast("str", key)] = new
        else:
            parent[cast("int", key)] = new
        self._commit(delta)
        return "noop" if same else "changed"

    def test(self, op: Test) -> Changed:
        self._spend_pointers(op.path)
        if not self._equal(self._get(op.path), op.value):
            raise self._reject("test_failed")
        return "noop"

    def copy(self, op: Copy) -> Changed:
        self._spend_pointers(op.from_, op.path)
        src = self._get(op.from_)
        nodes, w_src, depth = self._measure(src)
        self._put(op.path, src, w_src, nodes, depth)
        return "changed"

    def _move_target(
        self,
        op: Move,
        value: JsonValue,
        fparent: _Container,
        from_overhead: int,
        skip: tuple[list[JsonValue], int] | None,
    ) -> tuple[_Container, str, int, int, int]:
        """(parent, segment, insert index, insert shift, weight change) of a
        non-root move target, resolved as if the value were already removed."""
        tparent, tseg = self._parent(op.path, skip)
        if isinstance(tparent, dict):
            if tseg not in tparent:
                return tparent, tseg, -1, 0, key_overhead(tseg) - from_overhead
            w_t = self._measure(tparent[tseg])[1]
            if op.from_[: len(op.path)] == op.path:
                # The overwritten member holds the moved value.
                return tparent, tseg, -1, 0, self._measure(value)[1] - w_t
            return tparent, tseg, -1, 0, -from_overhead - w_t
        n = len(tparent) - (1 if tparent is fparent else 0)
        tindex = self._index(tseg, n, allow_end=True)
        return tparent, tseg, tindex, n - tindex, LIST_OVERHEAD - from_overhead

    def _detach(self, parent: _Container, key: str | int, shift: int) -> None:
        if isinstance(parent, dict):
            del parent[cast("str", key)]
        else:
            self._shift(shift)
            parent.pop(cast("int", key))

    def move(self, op: Move) -> Changed:
        self._spend_pointers(op.from_, op.path)
        if not op.from_:
            if op.path:
                raise self._reject("move_into_child")
            return "noop"
        fparent, fkey, value = self._locate(op.from_)
        if op.path == op.from_:
            return "noop"
        if op.path[: len(op.from_)] == op.from_:
            raise self._reject("move_into_child")
        if isinstance(fparent, dict):
            from_overhead = key_overhead(cast("str", fkey))
            skip = None
        else:
            from_overhead = LIST_OVERHEAD
            skip = (fparent, cast("int", fkey))
        remove_shift = len(fparent) - cast("int", fkey) - 1 if skip is not None else 0
        if not op.path:
            w_v = self._measure(value)[1]
            self._spend(remove_shift)
            self._admit(w_v - self.weight)
            # Detached, so the old tree holds no node of the new document.
            self._detach(fparent, fkey, remove_shift)
            self._commit(w_v - self.weight)
            self.doc = value
            return "changed"
        if len(op.path) > len(op.from_):
            self._fits(op.path, self._measure(value)[2])
        tparent, tseg, tindex, add_shift, delta = self._move_target(
            op, value, fparent, from_overhead, skip
        )
        self._spend(remove_shift + add_shift)
        self._admit(delta)
        self._detach(fparent, fkey, remove_shift)
        if isinstance(tparent, dict):
            tparent[tseg] = value
        else:
            self._shift(add_shift)
            tparent.insert(tindex, value)
        self._commit(delta)
        return "changed"


@final
class _Missing:
    __slots__ = ()


_MISSING = _Missing()


def _keyed(c: _Container) -> Iterator[tuple[str | None, JsonValue]]:
    if isinstance(c, dict):
        return iter(c.items())
    return ((None, v) for v in c)


# (key of x's member or None, x's node, y's node). For a member, the third
# item is y's object: the key is looked up only after the pair is charged.
_Pair: TypeAlias = "tuple[str | None, JsonValue, JsonValue | _Missing]"


def _dict_pairs(x: dict[str, JsonValue], y: dict[str, JsonValue]) -> Iterator[_Pair]:
    # Equal sizes were checked; a key of x absent from y is a difference.
    return ((k, v, y) for k, v in x.items())


def _list_pairs(x: list[JsonValue], y: list[JsonValue]) -> Iterator[_Pair]:
    return ((None, u, w) for u, w in zip(x, y, strict=True))


def apply_ops(
    doc: JsonValue,
    weight: int,
    ops: Sequence[PatchOp],
    budget: Budget,
    probe: Probe | None = None,
) -> Applied | Rejected | CapExceeded:
    """Apply `ops` to `doc`, a field whose weight `weight` is counted in
    `budget.total_weight`.

    `doc` is consumed: ops may mutate it, and a root op replaces it, so the
    caller never uses it again. Values are copied in, so the document shares
    no node with the ops or with any other document.

    Every result's `weight` is the field's weight as `budget.total_weight`
    counts it when the call returns. The caller's one rule: on `Applied`,
    keep `result.doc` as the field, at `result.weight`; on `Rejected` or
    `CapExceeded`, drop the field and do `budget.total_weight -=
    result.weight`. Either way the budget then counts exactly the documents
    the caller holds.
    """
    lim = budget.limits
    if budget.frame_ops + len(ops) > lim.ops_per_frame:
        return CapExceeded(
            "ops_per_frame", lim.ops_per_frame, budget.frame_ops + len(ops), 0, weight
        )
    budget.frame_ops += len(ops)
    run = _Run(doc, weight, budget, probe)
    changed: Changed = "noop"
    try:
        for i, op in enumerate(ops):
            run.index = i
            before = budget.run_work
            match op:
                case Add():
                    c = run.add(op)
                case Remove():
                    c = run.remove(op)
                case Replace():
                    c = run.replace(op)
                case Move():
                    c = run.move(op)
                case Copy():
                    c = run.copy(op)
                case Test():
                    c = run.test(op)
                case _:
                    assert_never(op)
            if c == "changed":
                changed = "changed"
            if probe is not None:
                probe.op_done(i, budget.run_work - before, run.doc, run.weight)
    except _Halt as h:
        return h.result
    return Applied(run.doc, run.weight, changed)
