"""Weight and work bounds of the patch applier (plan §2.4.1, I20, I21).

The applier runs with an instrumented probe that records every node visit and
every array shift. Each applied op's charge is checked against a reference
cost model computed here, independently, from the document before the op.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, cast

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from pplx_agent_tools.askstream.jsonval import measure, weight
from pplx_agent_tools.askstream.patch import (
    MIB,
    Add,
    Applied,
    Budget,
    CapExceeded,
    Copy,
    Limits,
    PatchOp,
    Remove,
    Replace,
    apply_ops,
    parse_patch_op,
)
from pplx_agent_tools.askstream.patch import (
    Test as _TestOp,
)
from pplx_agent_tools.jsonval import JsonValue
from tests._patch_strategies import KEYS, SCALARS, get, op_for, paths, ptr, untraced
from tests._timing import walk_seconds

SMALL = Limits(
    field_weight=4 * 1024,
    total_weight=8 * 1024,
    ops_per_frame=16,
    work_per_frame=4096,
    run_work_factor=1,
    run_work_base=16_384,
)

# --- reference model (recursive; independent of the applier's walkers) --------------------------


def ref_weight(v: Any) -> int:
    if isinstance(v, dict):
        items = cast("dict[str, Any]", v).items()
        return 2 + sum(len(json.dumps(k)) + ref_weight(c) + 2 for k, c in items)
    if isinstance(v, list):
        return 2 + sum(ref_weight(c) + 1 for c in cast("list[Any]", v))
    return len(json.dumps(v))


def ref_extra(v: Any) -> int:
    """Units beyond one to read a key's or scalar's text (§2.4.1)."""
    if isinstance(v, str):
        return len(v) // 64
    if isinstance(v, bool) or v is None:
        return 0
    if isinstance(v, int):
        return ((v.bit_length() * 30103 // 100000 + 1) // 64) ** 2
    return 3 if isinstance(v, float) else 0


def ref_units(v: Any, key: str | None = None) -> int:
    """Units to measure a subtree: per node one, plus its key's and its
    scalar text's extra."""
    own = 1 + (0 if key is None else ref_extra(key))
    if isinstance(v, dict):
        return own + sum(ref_units(c, k) for k, c in cast("dict[str, Any]", v).items())
    if isinstance(v, list):
        return own + sum(ref_units(c) for c in cast("list[Any]", v))
    return own + ref_extra(v)


def ref_pointer(p: tuple[str, ...]) -> int:
    """Units to resolve a pointer: each segment is read like a key."""
    return sum(1 + ref_extra(seg) for seg in p)


def ref_nodes(v: Any) -> int:
    if isinstance(v, dict):
        return 1 + sum(ref_nodes(c) for c in cast("dict[str, Any]", v).values())
    if isinstance(v, list):
        return 1 + sum(ref_nodes(c) for c in cast("list[Any]", v))
    return 1


def _scalar_eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b
    if isinstance(a, str) != isinstance(b, str):
        return False
    return bool(a == b)


def ref_compare(a: Any, b: Any, key: str | None = None) -> tuple[int, bool]:  # noqa: PLR0911
    """(units of the pairs visited in pre-order up to the first difference,
    equal). A pair costs its `a` node's units, key included."""
    own = 1 + (0 if key is None else ref_extra(key))
    if isinstance(a, dict):
        a_d = cast("dict[str, Any]", a)
        if not isinstance(b, dict) or len(a_d) != len(cast("dict[str, Any]", b)):
            return own, False
        b_d = cast("dict[str, Any]", b)
        steps = own
        for k, v in a_d.items():
            if k not in b_d:
                return steps + 1 + ref_extra(k) + ref_extra(v), False
            s, eq = ref_compare(v, b_d[k], k)
            steps += s
            if not eq:
                return steps, False
        return steps, True
    if isinstance(a, list):
        a_l = cast("list[Any]", a)
        if not isinstance(b, list) or len(a_l) != len(cast("list[Any]", b)):
            return own, False
        steps = own
        for x, y in zip(a_l, cast("list[Any]", b), strict=True):
            s, eq = ref_compare(x, y)
            steps += s
            if not eq:
                return steps, False
        return steps, True
    own += ref_extra(a)
    if isinstance(b, (dict, list)):
        return own, False
    return own, _scalar_eq(a, b)


def _slot(doc: Any, path: tuple[str, ...]) -> tuple[Any, str]:
    return get(doc, path[:-1]), path[-1]


def _index(parent: list[Any], seg: str) -> int:
    return len(parent) if seg == "-" else int(seg)


def _put_cost(doc: Any, path: tuple[str, ...]) -> int:
    if not path:
        return 0
    parent, seg = _slot(doc, path)
    if isinstance(parent, dict):
        p = cast("dict[str, Any]", parent)
        return ref_units(p[seg]) if seg in p else 0
    p_l = cast("list[Any]", parent)
    return len(p_l) - _index(p_l, seg)


def ref_charge(doc: Any, op: PatchOp) -> int:  # noqa: PLR0911
    """Work units of an op that applied, from the document before it. An
    inserted value is deep-copied: one unit per node."""
    if isinstance(op, Add):
        return ref_pointer(op.path) + _put_cost(doc, op.path) + ref_nodes(op.value)
    if isinstance(op, Remove):
        parent, seg = _slot(doc, op.path)
        if isinstance(parent, dict):
            return ref_pointer(op.path) + ref_units(parent[seg])
        p_l = cast("list[Any]", parent)
        i = int(seg)
        return ref_pointer(op.path) + ref_units(p_l[i]) + len(p_l) - i - 1
    if isinstance(op, Replace):
        old = get(doc, op.path)
        measured = ref_units(old) if op.path else 0
        copied = ref_nodes(op.value)
        return ref_pointer(op.path) + measured + ref_compare(old, op.value)[0] + copied
    if isinstance(op, _TestOp):
        return ref_pointer(op.path) + ref_compare(get(doc, op.path), op.value)[0]
    base = ref_pointer(op.from_) + ref_pointer(op.path)
    if isinstance(op, Copy):
        src = get(doc, op.from_)
        return base + ref_units(src) + ref_nodes(src) + _put_cost(doc, op.path)
    if not op.from_ or op.path == op.from_:
        return base
    value = get(doc, op.from_)
    if not op.path:
        # The value is also removed from its old parent.
        fparent, fseg = _slot(doc, op.from_)
        shift = len(fparent) - int(fseg) - 1 if isinstance(fparent, list) else 0
        return base + ref_units(value) + shift
    if len(op.path) > len(op.from_):
        # A deeper target: the value's depth is measured.
        base += ref_units(value)
    post = copy.deepcopy(doc)
    fparent, fseg = _slot(post, op.from_)
    shift = 0
    if isinstance(fparent, dict):
        del cast("dict[str, Any]", fparent)[fseg]
    else:
        f_l = cast("list[Any]", fparent)
        shift = len(f_l) - int(fseg) - 1
        f_l.pop(int(fseg))
    tparent, tseg = _slot(post, op.path)
    if isinstance(tparent, dict):
        t_d = cast("dict[str, Any]", tparent)
        if tseg not in t_d:
            return base + shift
        if op.from_[: len(op.path)] == op.path:
            # Measured before the removal (so it still holds the value), then
            # the value itself.
            return base + shift + ref_units(get(doc, op.path)) + ref_units(value)
        return base + shift + ref_units(t_d[tseg])
    t_l = cast("list[Any]", tparent)
    return base + shift + len(t_l) - _index(t_l, tseg)


# --- instrumented probe -----------------------------------------------------------------------------


class Recorder:
    """Checks I20 and I21 after every op, and after a frame that stops early."""

    def __init__(self, doc: JsonValue, ops: list[PatchOp], budget: Budget) -> None:
        self.before: Any = copy.deepcopy(doc)
        self.root: JsonValue = doc
        self.ops = ops
        self.budget = budget
        self.steps = 0
        self.shifts: list[int] = []
        self.work_mark = budget.run_work
        self.charges: list[int] = []

    def visit(self) -> None:
        self.steps += 1

    def read(self, text: JsonValue) -> None:
        self.steps += ref_extra(text)

    def shift(self, n: int) -> None:
        self.steps += n
        self.shifts.append(n)

    def _bounds(self) -> None:
        b, lim = self.budget, self.budget.limits
        assert b.frame_work <= lim.work_per_frame
        assert b.run_work <= b.run_work_limit
        assert b.total_weight <= lim.total_weight

    def op_done(self, index: int, charge: int, doc: JsonValue, weight: int) -> None:
        assert self.steps <= charge, (self.ops[index], self.steps, charge)
        assert charge == ref_charge(self.before, self.ops[index]), (self.before, self.ops[index])
        assert weight == ref_weight(doc)
        assert weight <= self.budget.limits.field_weight
        self._bounds()
        self.charges.append(charge)
        self.before = copy.deepcopy(doc)
        self.root = doc
        self.steps = 0
        self.work_mark = self.budget.run_work

    def stopped(self) -> None:
        """The op that ended the frame early did at most what it paid for,
        and the document is as the previous op left it."""
        assert self.steps <= self.budget.run_work - self.work_mark
        self._bounds()
        assert self.root == self.before


def _parse_all(raws: list[Any]) -> list[PatchOp]:
    return [op for op in (parse_patch_op(r) for r in copy.deepcopy(raws)) if op is not None]


# --- the property -----------------------------------------------------------------------------------------

# Long keys, long strings, many-digit ints and floats: text whose reading is
# charged by its length.
_LONG_KEYS = KEYS | st.text("ab", min_size=64, max_size=150)
_BIG = st.recursive(
    SCALARS
    | st.text(max_size=3000)
    | st.integers(10**60, 10**200)
    | st.floats(allow_nan=False, allow_infinity=False),
    lambda inner: st.lists(inner, max_size=6) | st.dictionaries(_LONG_KEYS, inner, max_size=6),
    max_leaves=40,
)


def _nest(depth: int) -> Any:
    doc: Any = [1, 2]
    for i in range(depth):
        doc = {"k": doc, str(i): i + 2}
    return doc


_DEEP = st.integers(1, 60).map(_nest)
# Many tiny nodes: cheap in weight, expensive in work.
_WIDE = st.dictionaries(
    KEYS, st.lists(st.integers(2, 9), min_size=300, max_size=900), min_size=1, max_size=2
)


def _containers(doc: Any) -> list[tuple[str, ...]]:
    return [p for p in paths(doc) if isinstance(get(doc, p), (dict, list))] or [()]


@st.composite
def _bounds_op(draw: st.DrawFn, doc: Any) -> dict[str, Any]:  # noqa: PLR0911
    """Mostly valid, costly ops (whole-subtree compares, copies, front
    inserts, copy-doubling), mixed with arbitrary ones."""
    src = draw(st.sampled_from(_containers(doc)))
    node = get(doc, src)
    kind = draw(st.integers(0, 14))
    if kind == 0:
        # Copy-doubling: a container copied into itself or beside itself.
        dst = (*src, "-") if isinstance(node, list) else (*src[:-1], draw(KEYS))
        return {"op": "copy", "from": ptr(src), "path": ptr(dst if src else ("d",))}
    if kind == 1 and src:
        return {"op": "copy", "from": ptr(src), "path": ptr((*src[:-1], "c" + draw(KEYS)))}
    if kind == 2:
        return {"op": "test", "path": ptr(src), "value": copy.deepcopy(node)}
    if kind == 3:
        return {"op": "replace", "path": ptr(src), "value": copy.deepcopy(node)}
    if kind == 4 and isinstance(node, list):
        return {"op": "add", "path": ptr((*src, "0")), "value": draw(SCALARS)}
    if kind == 5 and src:
        return {"op": "move", "from": ptr(src), "path": ptr((*src[:-1], "m" + draw(KEYS)))}
    if kind == 6:
        return draw(op_for(doc, values=_BIG))
    ups = [q for q in paths(doc) if len(q) >= 2 and isinstance(get(doc, q[:-2]), dict)]
    if kind == 7 and ups:
        # A value moved over the object member that holds it.
        q = draw(st.sampled_from(ups))
        return {"op": "move", "from": ptr(q), "path": ptr(q[:-1])}
    if kind == 8 and src:
        return {"op": "move", "from": ptr(src), "path": ""}
    if kind == 9 and isinstance(node, dict):
        # A pointer whose text is charged by its length.
        key = draw(st.text("ab", min_size=64, max_size=300))
        return {"op": "add", "path": ptr((*src, key)), "value": draw(_BIG)}
    dst = draw(st.sampled_from(_containers(doc)))
    if kind == 10 and src and isinstance(get(doc, dst), dict) and dst[: len(src)] != src:
        # A move to a deeper place: the value's depth is measured.
        return {"op": "move", "from": ptr(src), "path": ptr((*dst, "d"))}
    return {"op": "test", "path": ptr(src), "value": copy.deepcopy(node)}


@settings(max_examples=300)
@given(st.data())
def test_weight_and_work_bounds(data: st.DataObject) -> None:
    budget = Budget(SMALL)
    fields: list[Any] = [data.draw(_BIG | _DEEP | _WIDE) for _ in range(3)]
    weights = [weight(cast("JsonValue", f)) for f in fields]
    budget.total_weight = sum(weights)
    if max(weights) > SMALL.field_weight or budget.total_weight > SMALL.total_weight:
        return
    for _ in range(data.draw(st.integers(1, 20))):
        k = data.draw(st.integers(0, 2))
        raws = [data.draw(_bounds_op(fields[k])) for _ in range(data.draw(st.integers(0, 17)))]
        ops = _parse_all(raws)
        budget.start_frame(len(json.dumps(raws)))
        rec = Recorder(fields[k], ops, budget)
        result = apply_ops(fields[k], weights[k], ops, budget, rec)
        event(f"{type(result).__name__}:{getattr(result, 'cap', getattr(result, 'reason', ''))}")
        others = sum(ref_weight(f) for j, f in enumerate(fields) if j != k)
        if isinstance(result, Applied):
            assert result.weight == ref_weight(result.doc)
            assert budget.total_weight == others + result.weight
            assert measure(result.doc) is not None  # a tree that `loads` would accept
            fields[k], weights[k] = result.doc, result.weight
            continue
        rec.stopped()
        # The weight after the ops that applied before the failing one.
        assert result.weight == ref_weight(rec.root)
        # The failed field is dropped (the `apply_ops` rule).
        budget.total_weight -= result.weight
        assert budget.total_weight == others
        if isinstance(result, CapExceeded):
            return
        # A later snapshot resyncs it.
        fields[k], weights[k] = {}, 2
        fresh: Any = data.draw(_WIDE | _BIG)
        w_fresh = weight(fresh)
        if w_fresh <= SMALL.field_weight and budget.total_weight + w_fresh <= SMALL.total_weight:
            fields[k], weights[k] = fresh, w_fresh
        budget.total_weight += weights[k]


# --- rows -------------------------------------------------------------------------------------------------------


def _adds_at_front(count: int) -> list[PatchOp]:
    return _parse_all([{"op": "add", "path": "/a/0", "value": 1}] * count)


def test_4096_front_inserts_on_2_pow_20_array_stop_before_first_crossing_op() -> None:
    n = 2**20
    arr: list[JsonValue] = [0] * n
    doc: JsonValue = {"a": arr}
    budget = Budget(Limits())
    budget.total_weight = w = weight(doc)
    result = apply_ops(doc, w, _adds_at_front(4096), budget)
    # The first op alone is charged 2 + 2**20 > 2**20.
    assert result == CapExceeded("work_per_frame", 2**20, 2 + n, 0, w)
    assert len(arr) == n
    assert arr[0] == 0
    assert budget.frame_work <= 2**20


def test_front_inserts_stop_exactly_at_the_crossing_op() -> None:
    n = 1000
    arr: list[JsonValue] = [0] * n
    doc: JsonValue = {"a": arr}
    budget = Budget(Limits())
    budget.total_weight = w = weight(doc)
    ops = _adds_at_front(4096)
    rec = Recorder(doc, ops, budget)
    result = apply_ops(doc, w, ops, budget, rec)
    spent, expected = 0, 0
    # Each op: two pointer segments, the shifts, one node copied.
    while spent + 3 + n + expected <= 2**20:
        spent += 3 + n + expected
        expected += 1
    assert isinstance(result, CapExceeded)
    assert (result.cap, result.index) == ("work_per_frame", expected)
    assert len(arr) == n + expected
    # The crossing op's two pointer segments were charged before its shifts.
    assert budget.frame_work == spent + 2
    rec.stopped()


def test_copy_doubling_stops_before_crossing_op() -> None:
    doc: JsonValue = {"a": [0]}
    limits = Limits(work_per_frame=4096)
    budget = Budget(limits)
    budget.total_weight = w = weight(doc)
    raws = [{"op": "copy", "from": "/a", "path": "/a/-"}] * 32
    ops = _parse_all(raws)
    rec = Recorder(doc, ops, budget)
    result = apply_ops(doc, w, ops, budget, rec)
    assert isinstance(result, CapExceeded)
    assert result.cap == "work_per_frame"
    assert 0 < result.index < 32
    assert len(rec.charges) == result.index
    assert sum(rec.charges) + ref_charge(rec.before, ops[result.index]) > limits.work_per_frame
    rec.stopped()


def test_small_copy_frames_stop_at_work_per_run() -> None:
    doc: JsonValue = {"src": list(range(200))}
    budget = Budget(SMALL)
    budget.total_weight = w = weight(doc)
    raws = [{"op": "copy", "from": "/src", "path": "/dst"}, {"op": "remove", "path": "/dst"}]
    frame_bytes = len(json.dumps(raws))
    for frame in range(10_000):
        ops = _parse_all(raws)
        budget.start_frame(frame_bytes)
        rec = Recorder(doc, ops, budget)
        result = apply_ops(doc, w, ops, budget, rec)
        if isinstance(result, CapExceeded):
            assert result.cap == "work_per_run"
            assert frame > 1
            rec.stopped()
            return
        assert isinstance(result, Applied)
        assert result.weight == w  # weight stays flat; only work accrues
        assert budget.frame_work < SMALL.work_per_frame
    pytest.fail("work_per_run never reached")


# CPU per charged unit is bounded relative to a bare node walk timed on the same
# machine, so the bound holds on slower CI runners; the applier measures about
# 11x the walk, and the uncharged long-string defect this guards against was
# about 50x over its charge.
_MAX_UNIT_COST_VS_WALK = 30


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("\u00e9" * 1_300_000, id="1.3M-char-string"),
        pytest.param("\U0001f600" * 650_000, id="650k-non-bmp-string"),
        pytest.param([int("9" * 4300)] * 1900, id="1900-ints-of-4300-digits"),
        pytest.param({f"{i:06d}" + "k" * 4000: 0 for i in range(1000)}, id="1000-4k-char-keys"),
    ],
)
def test_copies_of_long_text_stop_at_work_cap_in_bounded_time(value: JsonValue) -> None:
    """Measuring a copy source reads all its text; that is charged by length,
    so a frame of copies stops at the work cap within the CPU the cap buys."""
    doc: JsonValue = {"s": value}
    budget = Budget(Limits())
    budget.total_weight = w = weight(doc)
    ops = _parse_all([{"op": "copy", "from": "/s", "path": "/t"}] * 4096)
    first = ref_charge(doc, ops[0])
    later = first + ref_units(value)  # every later copy also measures the old /t
    expected = 0 if first > MIB else 1 + (MIB - first) // later
    with untraced():
        start = time.process_time()
        result = apply_ops(doc, w, ops, budget)
        elapsed = time.process_time() - start
    assert isinstance(result, CapExceeded)
    assert (result.cap, result.index) == ("work_per_frame", expected)
    assert budget.frame_work <= MIB
    assert elapsed <= _MAX_UNIT_COST_VS_WALK * walk_seconds(MIB), elapsed


def test_end_appends_cost_zero_shifts() -> None:
    doc: Any = {"chunks": ["x"] * 50}
    budget = Budget(Limits())
    budget.total_weight = w = weight(doc)
    ops = _parse_all([{"op": "add", "path": "/chunks/-", "value": "y"}] * 100)
    rec = Recorder(doc, ops, budget)
    assert isinstance(apply_ops(doc, w, ops, budget, rec), Applied)
    assert rec.shifts == [0] * 100
    assert rec.charges == [3] * 100


@pytest.mark.parametrize("cap", ["field_weight", "total_weight"])
def test_weight_caps_reject_before_mutation(cap: str) -> None:
    doc: JsonValue = {"a": "x"}
    w = weight(doc)
    budget = Budget(Limits(field_weight=100, total_weight=1000))
    budget.total_weight = w if cap == "field_weight" else 950
    value = "y" * (200 if cap == "field_weight" else 60)
    ops = _parse_all([{"op": "add", "path": "/b", "value": value}])
    result = apply_ops(doc, w, ops, budget)
    assert isinstance(result, CapExceeded)
    assert (result.cap, result.index) == (cap, 0)
    assert doc == {"a": "x"}


def test_ops_per_frame_cap_applies_nothing() -> None:
    doc: JsonValue = {"a": []}
    budget = Budget(SMALL)
    budget.total_weight = w = weight(doc)
    ops = _parse_all([{"op": "add", "path": "/a/-", "value": 1}] * 17)
    assert apply_ops(doc, w, ops, budget) == CapExceeded("ops_per_frame", 16, 17, 0, w)
    assert doc == {"a": []}


@pytest.mark.slow
def test_frame_at_work_cap_in_node_visits_takes_bounded_cpu() -> None:
    # replace /a: 1 segment + (K + 1) nodes measured + 1 compare step + 1 node copied.
    k = MIB - 4
    doc: Any = {"a": [0] * k}
    budget = Budget(Limits())
    budget.total_weight = w = weight(doc)
    ops = _parse_all([{"op": "replace", "path": "/a", "value": 0}])
    with untraced():
        start = time.process_time()
        result = apply_ops(doc, w, ops, budget)
        elapsed = time.process_time() - start
    assert isinstance(result, Applied)
    assert budget.frame_work == MIB
    assert elapsed <= _MAX_UNIT_COST_VS_WALK * walk_seconds(MIB), elapsed
