"""RFC 6902 JSON Patch over untrusted ops, in place, with bounded cost.

`apply_ops` never raises. Every op is charged work units computed from the
document before the op, and its weight change is known, before anything is
mutated; a charge that would pass a cap ends the call as `CapExceeded` with
the op not applied. Work units:

- one per pointer segment of each pointer the op uses;
- one per node visited while measuring a subtree (the old value of a
  non-root `replace` or `remove`, an overwritten object member, a `copy`
  source, a `move` value when the target is the root or an ancestor object
  member), deep-copying (`copy`), or comparing (`test`, and the
  changed/noop check on `replace`, which stops at the first difference);
- one per array element shifted: an insert at i on length n shifts n - i,
  a removal at i shifts n - i - 1.

A root `remove` is rejected, as in RFC 6902 there is nothing left to hold.
A root `move` or `copy` target replaces the document without a removal.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast, final

from typing_extensions import assert_never

from pplx_agent_tools.askstream.jsonval import (
    LIST_OVERHEAD,
    JsonValue,
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
]

_Container: TypeAlias = "dict[str, JsonValue] | list[JsonValue]"

KIB = 1024
MIB = 1024 * KIB


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


_VALUE_OPS: dict[str, Callable[[Pointer, JsonValue, int], PatchOp]] = {
    "add": Add,
    "replace": Replace,
    "test": Test,
}


def parse_patch_op(raw: object) -> PatchOp | None:
    """One op from a decoded JSON object, or None when it is not a valid op.
    Members RFC 6902 does not define are ignored. A `value` must be
    JSON-shaped and nest at most 128 deep; its weight is computed here."""
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
    make = _VALUE_OPS.get(op) if isinstance(op, str) else None
    m = measure(obj["value"]) if make is not None and "value" in obj else None
    return None if make is None or m is None else make(path, m.value, m.weight)


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
    """Test instrumentation: every node visit, every array shift, and the
    state after every applied op."""

    def visit(self) -> None: ...
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
    """Op `index` is invalid for the document. Earlier ops stay applied (the
    caller drops the field); `weight` is the field's weight at that point,
    already counted in `Budget.total_weight`."""

    index: int
    reason: RejectReason
    weight: int


@final
@dataclass(frozen=True, slots=True)
class CapExceeded:
    """Op `index` would pass `cap`; it was not applied. `observed` is a lower
    bound on the value the cap would have reached."""

    cap: CapName
    limit: int
    observed: int
    index: int


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
        return _Halt(CapExceeded(cap, limit, observed, self.index))

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

    def _overrun(self, done: int) -> _Halt:
        """A bounded walk used all `done` units of room and needs one more."""
        self._spend(done)
        b = self.budget
        if b.frame_work >= b.limits.work_per_frame:
            return self._cap("work_per_frame", b.limits.work_per_frame, b.frame_work + 1)
        return self._cap("work_per_run", b.run_work_limit, b.run_work + 1)

    # --- walks (each bounded by the remaining work, each visit recorded) ----

    def _measure(self, v: JsonValue) -> tuple[int, int]:
        """(nodes, weight) of a subtree, charged one unit per node."""
        room = self._room()
        probe = self.probe
        if room < 1:
            raise self._overrun(0)
        if probe is not None:
            probe.visit()
        if not _is_container(v):
            self._spend(1)
            return 1, scalar_weight(v)
        nodes, total = 1, 2
        stack: list[Iterator[tuple[int, JsonValue]]] = [_members(cast("_Container", v))]
        while stack:
            pair = next(stack[-1], None)
            if pair is None:
                stack.pop()
                continue
            if nodes >= room:
                raise self._overrun(nodes)
            nodes += 1
            if probe is not None:
                probe.visit()
            overhead, child = pair
            total += overhead
            if isinstance(child, (dict, list)):
                total += 2
                stack.append(_members(child))
            else:
                total += scalar_weight(child)
        self._spend(nodes)
        return nodes, total

    def _equal(self, a: JsonValue, b: JsonValue) -> bool:
        """JSON equality, charged one unit per node pair compared, pre-order,
        stopping at the first difference."""
        room = self._room()
        probe = self.probe
        steps = 0
        stack: list[Iterator[tuple[JsonValue, JsonValue | _Missing]]] = [iter(((a, b),))]
        while stack:
            pair = next(stack[-1], None)
            if pair is None:
                stack.pop()
                continue
            if steps >= room:
                raise self._overrun(steps)
            steps += 1
            if probe is not None:
                probe.visit()
            x, y = pair
            if isinstance(y, _Missing):
                self._spend(steps)
                return False
            if isinstance(x, dict):
                if not isinstance(y, dict) or len(x) != len(y):
                    self._spend(steps)
                    return False
                stack.append(_dict_pairs(x, y))
            elif isinstance(x, list):
                if not isinstance(y, list) or len(x) != len(y):
                    self._spend(steps)
                    return False
                stack.append(zip(x, y, strict=False))
            elif _is_container(y) or not _json_scalar_eq(x, y):
                self._spend(steps)
                return False
        self._spend(steps)
        return True

    def _deep_copy(self, v: JsonValue) -> JsonValue:
        """A deep copy; its node count was charged by the caller."""
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

    def _commit(self, delta: int) -> None:
        self.weight += delta
        self.budget.total_weight += delta

    def _shift(self, n: int) -> None:
        if self.probe is not None:
            self.probe.shift(n)

    # --- ops --------------------------------------------------------------------

    def _put(self, path: Pointer, make: Callable[[], JsonValue], value_weight: int) -> None:
        """RFC 6902 `add`. `make` builds the value, and runs only after every
        charge and cap check has passed."""
        if not path:
            self._admit(value_weight - self.weight)
            self.doc = make()
            self._commit(value_weight - self.weight)
            return
        parent, seg = self._parent(path)
        if isinstance(parent, dict):
            if seg in parent:
                delta = value_weight - self._measure(parent[seg])[1]
            else:
                delta = key_overhead(seg) + value_weight
            self._admit(delta)
            parent[seg] = make()
        else:
            n = len(parent)
            i = self._index(seg, n, allow_end=True)
            self._spend(n - i)
            delta = value_weight + LIST_OVERHEAD
            self._admit(delta)
            new = make()
            self._shift(n - i)
            parent.insert(i, new)
        self._commit(delta)

    def add(self, op: Add) -> Changed:
        self._spend(len(op.path))
        value = op.value
        self._put(op.path, lambda: value, op.value_weight)
        return "changed"

    def remove(self, op: Remove) -> Changed:
        self._spend(len(op.path))
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
        self._spend(len(op.path))
        if not op.path:
            same = self._equal(self.doc, op.value)
            delta = op.value_weight - self.weight
            self._admit(delta)
            self._commit(delta)
            self.doc = op.value
            return "noop" if same else "changed"
        parent, key, old = self._locate(op.path)
        w_old = self._measure(old)[1]
        same = self._equal(old, op.value)
        delta = op.value_weight - w_old
        self._admit(delta)
        if isinstance(parent, dict):
            parent[cast("str", key)] = op.value
        else:
            parent[cast("int", key)] = op.value
        self._commit(delta)
        return "noop" if same else "changed"

    def test(self, op: Test) -> Changed:
        self._spend(len(op.path))
        if not self._equal(self._get(op.path), op.value):
            raise self._reject("test_failed")
        return "noop"

    def copy(self, op: Copy) -> Changed:
        self._spend(len(op.from_) + len(op.path))
        src = self._get(op.from_)
        nodes, w_src = self._measure(src)
        self._spend(nodes)
        self._put(op.path, lambda: self._deep_copy(src), w_src)
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

    def move(self, op: Move) -> Changed:
        self._spend(len(op.from_) + len(op.path))
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
        if not op.path:
            w_v = self._measure(value)[1]
            self._admit(w_v - self.weight)
            self._commit(w_v - self.weight)
            self.doc = value
            return "changed"
        remove_shift = len(fparent) - cast("int", fkey) - 1 if skip is not None else 0
        tparent, tseg, tindex, add_shift, delta = self._move_target(
            op, value, fparent, from_overhead, skip
        )
        self._spend(remove_shift + add_shift)
        self._admit(delta)
        if isinstance(fparent, dict):
            del fparent[cast("str", fkey)]
        else:
            self._shift(remove_shift)
            fparent.pop(cast("int", fkey))
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


def _members(c: _Container) -> Iterator[tuple[int, JsonValue]]:
    if isinstance(c, dict):
        return ((key_overhead(k), v) for k, v in c.items())
    return ((LIST_OVERHEAD, v) for v in c)


def _keyed(c: _Container) -> Iterator[tuple[str | None, JsonValue]]:
    if isinstance(c, dict):
        return iter(c.items())
    return ((None, v) for v in c)


def _dict_pairs(
    x: dict[str, JsonValue], y: dict[str, JsonValue]
) -> Iterator[tuple[JsonValue, JsonValue | _Missing]]:
    # Equal sizes were checked; a key of x absent from y is a difference.
    return ((v, y.get(k, _MISSING)) for k, v in x.items())


def apply_ops(
    doc: JsonValue,
    weight: int,
    ops: Sequence[PatchOp],
    budget: Budget,
    probe: Probe | None = None,
) -> Applied | Rejected | CapExceeded:
    """Apply `ops` to `doc` (whose weight is `weight`) in place.

    The document takes ownership of the op values it inserts. On `Rejected`
    and `CapExceeded` the document may hold the earlier ops' changes and must
    not be used as a synced copy again.
    """
    lim = budget.limits
    if budget.frame_ops + len(ops) > lim.ops_per_frame:
        return CapExceeded("ops_per_frame", lim.ops_per_frame, budget.frame_ops + len(ops), 0)
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
