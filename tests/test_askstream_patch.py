"""RFC 6902 applier: correctness against the RFC and a reference library."""

from __future__ import annotations

import copy
import json
import math
import time
import tracemalloc
from typing import Any, cast

import jsonpatch  # pyright: ignore[reportMissingTypeStubs]
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from pplx_agent_tools.askstream import patch as patch_mod
from pplx_agent_tools.askstream.jsonval import JsonError, loads, measure, weight
from pplx_agent_tools.askstream.patch import (
    Add,
    Applied,
    Budget,
    CapExceeded,
    Copy,
    Limits,
    Move,
    PatchOp,
    Rejected,
    Remove,
    Replace,
    apply_ops,
    parse_patch_op,
    parse_pointer,
)
from pplx_agent_tools.jsonval import JsonValue
from tests._patch_strategies import DOCS, get, op_for, untraced


def run(doc: Any, raw_ops: list[Any], limits: Limits | None = None) -> Any:
    """Parse and apply; returns the result, or "malformed" when an op does
    not parse. The raw ops are deep-copied: inserted values are owned by the
    document afterwards."""
    ops: list[PatchOp] = []
    for raw in copy.deepcopy(raw_ops):
        op = parse_patch_op(raw)
        if op is None:
            return "malformed"
        ops.append(op)
    budget = Budget(limits or Limits())
    budget.total_weight = weight(doc)
    return apply_ops(doc, weight(doc), ops, budget)


# --- pointers and op parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ()),
        ("/", ("",)),
        ("/a/b", ("a", "b")),
        ("/a~1b", ("a/b",)),
        ("/m~0n", ("m~n",)),
        ("/~01", ("~1",)),  # ~1 is unescaped before ~0
        ("/a/-", ("a", "-")),
        ("a", None),
        ("/~2", None),
        ("/~", None),
        (3, None),
        ("/a" * 128, ("a",) * 128),
        ("/a" * 129, None),
    ],
)
def test_parse_pointer(raw: object, expected: tuple[str, ...] | None) -> None:
    assert parse_pointer(raw) == expected


def test_parse_patch_op_shapes() -> None:
    assert parse_patch_op({"op": "add", "path": "/a", "value": None}) == Add(("a",), None, 4, 1, 0)
    assert parse_patch_op({"op": "remove", "path": "/a"}) == Remove(("a",))
    assert parse_patch_op({"op": "replace", "path": "", "value": [1]}) == Replace((), [1], 4, 2, 1)
    assert parse_patch_op({"op": "move", "from": "/a", "path": "/b"}) == Move(("a",), ("b",))
    assert parse_patch_op({"op": "copy", "from": "/a", "path": "/b"}) == Copy(("a",), ("b",))
    assert parse_patch_op({"op": "test", "path": "/a", "value": "x"}) == patch_mod.Test(
        ("a",), "x", 3
    )
    for bad in [
        {"op": "add", "path": "/a"},
        {"op": "move", "path": "/a"},
        {"op": "nope", "path": "/a"},
        {"op": "add", "value": 1},
        {"op": ["add"], "path": "/a", "value": 1},
        {"op": "add", "path": "/a", "value": (1,)},
        {"op": "add", "path": "/a", "value": {1: 2}},
        [],
        "add",
    ]:
        assert parse_patch_op(bad) is None


def test_value_nesting_is_capped() -> None:
    deep: object = 0
    for _ in range(128):
        deep = [deep]
    assert parse_patch_op({"op": "add", "path": "/a", "value": deep}) is not None
    assert parse_patch_op({"op": "add", "path": "/a", "value": [deep]}) is None


# --- RFC 6902 appendix A ----------------------------------------------------------------

_ERR = "error"

APPENDIX_A = [
    ("A.1", {"foo": "bar"}, [{"op": "add", "path": "/baz", "value": "qux"}],
     {"baz": "qux", "foo": "bar"}),
    ("A.2", {"foo": ["bar", "baz"]}, [{"op": "add", "path": "/foo/1", "value": "qux"}],
     {"foo": ["bar", "qux", "baz"]}),
    ("A.3", {"baz": "qux", "foo": "bar"}, [{"op": "remove", "path": "/baz"}], {"foo": "bar"}),
    ("A.4", {"foo": ["bar", "qux", "baz"]}, [{"op": "remove", "path": "/foo/1"}],
     {"foo": ["bar", "baz"]}),
    ("A.5", {"baz": "qux", "foo": "bar"}, [{"op": "replace", "path": "/baz", "value": "boo"}],
     {"baz": "boo", "foo": "bar"}),
    ("A.6", {"foo": {"bar": "baz", "waldo": "fred"}, "qux": {"corge": "grault"}},
     [{"op": "move", "from": "/foo/waldo", "path": "/qux/thud"}],
     {"foo": {"bar": "baz"}, "qux": {"corge": "grault", "thud": "fred"}}),
    ("A.7", {"foo": ["all", "grass", "cows", "eat"]},
     [{"op": "move", "from": "/foo/1", "path": "/foo/3"}],
     {"foo": ["all", "cows", "eat", "grass"]}),
    ("A.8", {"baz": "qux", "foo": ["a", 2, "c"]},
     [{"op": "test", "path": "/baz", "value": "qux"}, {"op": "test", "path": "/foo/1", "value": 2}],
     {"baz": "qux", "foo": ["a", 2, "c"]}),
    ("A.9", {"baz": "qux"}, [{"op": "test", "path": "/baz", "value": "bar"}], _ERR),
    ("A.10", {"foo": "bar"}, [{"op": "add", "path": "/child", "value": {"grandchild": {}}}],
     {"foo": "bar", "child": {"grandchild": {}}}),
    ("A.11", {"foo": "bar"}, [{"op": "add", "path": "/baz", "value": "qux", "xyz": 123}],
     {"foo": "bar", "baz": "qux"}),
    ("A.12", {"foo": "bar"}, [{"op": "add", "path": "/baz/bat", "value": "qux"}], _ERR),
    # A.13 (duplicate "op" members) is a JSON-text defect: the parser keeps one
    # member before any op exists, so it has no row at this layer.
    ("A.14", {"/": 9, "~1": 10}, [{"op": "test", "path": "/~01", "value": 10}],
     {"/": 9, "~1": 10}),
    ("A.15", {"/": 9, "~1": 10}, [{"op": "test", "path": "/~01", "value": "10"}], _ERR),
    ("A.16", {"foo": ["bar"]}, [{"op": "add", "path": "/foo/-", "value": ["abc", "def"]}],
     {"foo": ["bar", ["abc", "def"]]}),
]  # fmt: skip


@pytest.mark.parametrize(
    ("name", "doc", "ops", "expected"), APPENDIX_A, ids=[r[0] for r in APPENDIX_A]
)
def test_rfc6902_appendix_a(name: str, doc: Any, ops: list[Any], expected: Any) -> None:
    result = run(copy.deepcopy(doc), ops)
    if expected == _ERR:
        assert isinstance(result, Rejected), name
    else:
        assert isinstance(result, Applied), name
        assert result.doc == expected
        assert result.weight == weight(result.doc)


# --- differential against jsonpatch ---------------------------------------------------------


def _jsonpatch_skips(doc: Any, op: dict[str, Any]) -> bool:
    """Cases where jsonpatch departs from RFC 6902: it lets a value move into
    its own child when the value's parent is an array, it cannot put a value
    at the root of a document that is not an object, and it refuses to
    replace an object member named `-`."""
    if op["op"] in ("add", "move", "copy") and op["path"] == "" and not isinstance(doc, dict):
        return True
    if op["op"] == "replace" and op["path"].endswith("/-"):
        return True
    if op["op"] != "move":
        return False
    src, dst = parse_pointer(op["from"]), parse_pointer(op["path"])
    assert src is not None and dst is not None
    if len(dst) <= len(src) or dst[: len(src)] != src:
        return False
    try:
        return isinstance(get(doc, src[:-1]), list)
    except (KeyError, IndexError, ValueError, TypeError):
        return False


@st.composite
def _doc_and_ops(draw: st.DrawFn) -> tuple[Any, list[dict[str, Any]]]:
    doc = draw(DOCS)
    ref = copy.deepcopy(doc)
    ops: list[dict[str, Any]] = []
    for _ in range(draw(st.integers(1, 6))):
        op = draw(op_for(ref))
        if _jsonpatch_skips(ref, op):
            continue
        ops.append(op)
        try:
            ref = jsonpatch.apply_patch(ref, [copy.deepcopy(op)])
        except Exception:
            break  # the sequence now fails; later ops never run
    return doc, ops


@settings(max_examples=1000)
@given(_doc_and_ops())
def test_differential_against_jsonpatch(case: tuple[Any, list[dict[str, Any]]]) -> None:
    doc, ops = case
    try:
        expected: Any = jsonpatch.apply_patch(copy.deepcopy(doc), copy.deepcopy(ops))
    except Exception:
        expected = _ERR
    result = run(copy.deepcopy(doc), ops)
    event("rejected" if expected == _ERR else "applied")
    for op in ops:
        event(op["op"])
    if expected == _ERR:
        assert isinstance(result, Rejected), (doc, ops, result)
    else:
        assert isinstance(result, Applied), (doc, ops, result)
        assert result.doc == expected
        assert result.weight == weight(result.doc)


# --- arbitrary JSON as ops ----------------------------------------------------------------

_ANY_JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(), inner, max_size=4),
    max_leaves=16,
)
_OPISH = st.fixed_dictionaries(
    {},
    optional={
        "op": st.sampled_from(["add", "remove", "replace", "move", "copy", "test"]) | _ANY_JSON,
        "path": st.sampled_from(["", "/", "/a", "/a/0", "/a/-", "/0", "/a/01", "/~"]) | _ANY_JSON,
        "from": st.sampled_from(["", "/a", "/a/0", "/b"]) | _ANY_JSON,
        # Non-finite floats have no JSON form; parsing refuses them.
        "value": _ANY_JSON | st.sampled_from([math.nan, math.inf, -math.inf]),
    },
)


@given(_ANY_JSON, st.lists(_OPISH | _ANY_JSON, max_size=8))
def test_arbitrary_json_as_ops_never_raises(doc: Any, raws: list[Any]) -> None:
    ops = [op for op in (parse_patch_op(r) for r in raws) if op is not None]
    budget = Budget(Limits(work_per_frame=4096, ops_per_frame=6))
    result = apply_ops(doc, weight(doc), ops, budget)
    assert isinstance(result, (Applied, Rejected, CapExceeded))
    if isinstance(result, Applied):
        assert result.weight == weight(result.doc)


# --- throughput -----------------------------------------------------------------------------


def test_10k_chunk_appends_under_50ms() -> None:
    ops = [
        parse_patch_op({"op": "add", "path": "/chunks/-", "value": f"c{i} "}) for i in range(10_000)
    ]
    doc: JsonValue = {"chunks": []}
    budget = Budget(Limits(ops_per_frame=10_000))
    typed = [op for op in ops if op is not None]
    w = weight(doc)
    with untraced():
        start = time.perf_counter()
        result = apply_ops(doc, w, typed, budget)
        elapsed = time.perf_counter() - start
    assert isinstance(result, Applied)
    assert len(cast("list[str]", cast("dict[str, Any]", result.doc)["chunks"])) == 10_000
    assert elapsed < 0.050, elapsed


# --- rejections before allocation ----------------------------------------------------------------


def _peak_during(fn: Any) -> tuple[Any, int]:
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = fn()
        return out, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("/a/" + "9" * 200_000, "index_out_of_range"),
        ("/a/3", "index_out_of_range"),
        ("/a/01", "bad_index"),
        ("/a/-1", "bad_index"),
        ("/a/\u0663", "bad_index"),
        ("/a/-", "end_index"),
    ],
)
def test_bad_index_rejected_before_allocation(path: str, reason: str) -> None:
    op = parse_patch_op({"op": "replace", "path": path, "value": 0})
    assert op is not None
    doc: JsonValue = {"a": [1, 2, 3]}
    budget = Budget(Limits())
    result, peak = _peak_during(lambda: apply_ops(doc, weight(doc), [op], budget))
    assert result == Rejected(0, reason, weight({"a": [1, 2, 3]}))  # pyright: ignore[reportArgumentType]
    assert peak < 4096, peak
    assert doc == {"a": [1, 2, 3]}


def test_add_index_past_end_rejected() -> None:
    assert isinstance(run({"a": [1]}, [{"op": "add", "path": "/a/2", "value": 0}]), Rejected)
    assert isinstance(run({"a": [1]}, [{"op": "add", "path": "/a/1", "value": 0}]), Applied)


def test_129_segment_pointer_is_malformed_before_allocation() -> None:
    raw = "/a" * 129
    parsed, peak = _peak_during(lambda: parse_pointer(raw))
    assert parsed is None
    assert peak < 1024, peak
    assert run({}, [{"op": "add", "path": raw, "value": 1}]) == "malformed"
    assert run({}, [{"op": "move", "from": raw, "path": "/a"}]) == "malformed"


def test_remove_root_rejected() -> None:
    assert run({"a": 1}, [{"op": "remove", "path": ""}]) == Rejected(
        0, "remove_root", weight({"a": 1})
    )


def test_move_into_own_child_rejected() -> None:
    result = run({"a": [{"x": 1}]}, [{"op": "move", "from": "/a/0", "path": "/a/0/y"}])
    assert result == Rejected(0, "move_into_child", weight({"a": [{"x": 1}]}))


@pytest.mark.parametrize(
    ("ops", "expected"),
    [
        ([{"op": "move", "from": "/a/b/c", "path": "/a"}], {"a": [1, 2], "z": 3}),
        ([{"op": "move", "from": "/a/b", "path": ""}], {"c": [1, 2]}),
        ([{"op": "move", "from": "/a/b", "path": "/zz"}], {"a": {}, "z": 3, "zz": {"c": [1, 2]}}),
        ([{"op": "move", "from": "/a/b/c/0", "path": "/a/b/c/-"}], {"a": {"b": {"c": [2, 1]}}, "z": 3}),
        ([{"op": "copy", "from": "/a", "path": "/a/b/d"}],
         {"a": {"b": {"c": [1, 2], "d": {"b": {"c": [1, 2]}}}}, "z": 3}),
    ],
)  # fmt: skip
def test_move_and_copy_keep_weight_exact(ops: list[Any], expected: Any) -> None:
    result = run({"a": {"b": {"c": [1, 2]}}, "z": 3}, ops)
    assert isinstance(result, Applied)
    assert result.doc == expected
    assert result.weight == weight(expected)


def test_root_from_rows() -> None:
    assert isinstance(run({"a": 1}, [{"op": "move", "from": "", "path": ""}]), Applied)
    assert isinstance(run({"a": 1}, [{"op": "move", "from": "", "path": "/b"}]), Rejected)
    result = run({"a": 1}, [{"op": "copy", "from": "", "path": "/b"}])
    assert isinstance(result, Applied)
    assert result.doc == {"a": 1, "b": {"a": 1}}


def test_changed_flag() -> None:
    same = run({"a": [1, {"b": 2}]}, [{"op": "replace", "path": "/a", "value": [1, {"b": 2}]}])
    assert isinstance(same, Applied)
    assert same.changed == "noop"
    diff = run({"a": [1, {"b": 2}]}, [{"op": "replace", "path": "/a", "value": [1, {"b": 3}]}])
    assert isinstance(diff, Applied)
    assert diff.changed == "changed"
    assert run({"a": 1}, [{"op": "test", "path": "/a", "value": 1}]).changed == "noop"
    # JSON equality: true is not 1.
    assert isinstance(run({"a": True}, [{"op": "test", "path": "/a", "value": 1}]), Rejected)


@pytest.mark.parametrize(
    ("held", "tested", "equal"),
    [
        (1, 1.0, True),
        (1.0, 1, True),
        (-0.0, 0, True),
        (2**53, float(2**53), True),
        (1, 1.5, False),
        (True, 1, False),
        (1, True, False),
        (False, 0, False),
        (0, False, False),
        (None, 0, False),
        ("1", 1, False),
        (1, "1", False),
    ],
)
def test_json_number_equality(held: Any, tested: Any, equal: bool) -> None:
    """JSON has one number type: 1 and 1.0 are equal; true is not 1."""
    t = run({"a": [held]}, [{"op": "test", "path": "/a", "value": [tested]}])
    assert isinstance(t, Applied if equal else Rejected)
    r = run({"a": held}, [{"op": "replace", "path": "/a", "value": tested}])
    assert isinstance(r, Applied)
    assert r.changed == ("noop" if equal else "changed")


def test_rejected_weight_counts_the_earlier_ops() -> None:
    result = run(
        {"a": 1, "c": [1, 2]},
        [
            {"op": "add", "path": "/b", "value": "xyz"},
            {"op": "remove", "path": "/c/0"},
            {"op": "test", "path": "/a", "value": 2},
        ],
    )
    assert result == Rejected(2, "test_failed", weight({"a": 1, "c": [2], "b": "xyz"}))


def test_loads_rejects_non_finite_numbers() -> None:
    for raw in ["NaN", "Infinity", "-Infinity", "1e400", "-1e400", '{"a": [1, NaN]}', "[1e999]"]:
        assert loads(raw) == JsonError("syntax"), raw
    assert loads("1e308") == 1e308
    assert loads("-0.0") == 0.0
    for bad in [math.nan, math.inf, [1, -math.inf]]:
        assert measure(bad) is None
        assert parse_patch_op({"op": "add", "path": "/a", "value": bad}) is None


# --- jsonval ---------------------------------------------------------------------------------------


@given(_ANY_JSON)
def test_weight_within_one_byte_per_container_of_compact_json(v: Any) -> None:
    text = json.dumps(v, separators=(",", ":"))
    containers = text.count("[") + text.count("{")  # an upper bound (strings may hold them)
    assert len(text) <= weight(v) <= len(text) + containers


def test_loads() -> None:
    assert loads('{"a": [1]}') == {"a": [1]}
    assert loads("{") == JsonError("syntax")
    assert loads("[" * 128 + "]" * 128) is not None
    assert loads("[" * 129 + "]" * 129) == JsonError("depth")
    assert loads("[" * 200_000 + "]" * 200_000) == JsonError("depth")
    m = measure({"a": [1, "x"]})
    assert m is not None
    assert (m.weight, m.nodes) == (len('{"a":[1,"x"]}') + 2, 4)


# --- depth, pointer cost, ownership and the budget rule ---------------------------------------------


def _nested(depth: int) -> Any:
    v: Any = 0
    for _ in range(depth):
        v = [v]
    return v


# (segments in the target pointer, doc and op placing a value `v` there)
_DEPTH_ROWS: dict[str, tuple[int, Any]] = {
    "copy": (2, lambda v: ({"a": v, "b": []}, {"op": "copy", "from": "/a", "path": "/b/-"})),
    "add": (2, lambda v: ({"b": []}, {"op": "add", "path": "/b/-", "value": v})),
    "replace": (2, lambda v: ({"b": [0]}, {"op": "replace", "path": "/b/0", "value": v})),
    "move": (
        3,
        lambda v: ({"a": v, "b": {"c": {}}}, {"op": "move", "from": "/a", "path": "/b/c/d"}),
    ),
}


@pytest.mark.parametrize("name", list(_DEPTH_ROWS))
def test_op_past_max_depth_is_rejected_before_the_write(name: str) -> None:
    """No applied document nests deeper than `loads` accepts (128)."""
    segments, make = _DEPTH_ROWS[name]
    doc, op = make(_nested(129 - segments))
    before = copy.deepcopy(doc)
    assert run(doc, [op]) == Rejected(0, "too_deep", weight(before))
    assert doc == before
    doc, op = make(_nested(128 - segments))
    result = run(doc, [op])
    assert isinstance(result, Applied)
    assert loads(json.dumps(result.doc)) == result.doc


def test_long_pointer_text_is_charged_by_length() -> None:
    """A pointer segment is read like a key: 1 + len // 64 units, charged
    before the pointer is used."""
    key = "k" * 1_000_000
    op = parse_patch_op({"op": "add", "path": "/" + key, "value": 1})
    assert op is not None
    doc: JsonValue = {}
    budget = Budget(Limits(work_per_frame=1000))
    result = apply_ops(doc, 2, [op], budget)
    assert isinstance(result, CapExceeded), result
    assert (result.cap, result.observed, result.index) == ("work_per_frame", 1 + len(key) // 64, 0)
    assert doc == {}
    assert budget.frame_work == 0
    assert patch_mod.pointer_units(op.path) == 1 + len(key) // 64


def _reachable_ids(v: Any) -> set[int]:
    out: set[int] = set()
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, (dict, list)) and id(x) not in out:
            out.add(id(x))
            stack.extend(cast("dict[str, Any]", x).values() if isinstance(x, dict) else x)
    return out


@pytest.mark.parametrize("src", ["/a", "/b/0"])
def test_root_move_detaches_the_new_root_from_the_old_tree(src: str) -> None:
    doc: Any = {"a": {"x": [1]}, "b": [{"y": [2]}, 3]}
    result = run(doc, [{"op": "move", "from": src, "path": ""}])
    assert isinstance(result, Applied)
    assert result.weight == weight(result.doc)
    assert not _reachable_ids(result.doc) & _reachable_ids(doc)


_OTHER = 5000


def _scenarios() -> list[Any]:
    big = "y" * 1000
    return [
        pytest.param({"a": 1}, [{"op": "add", "path": "/b", "value": 2},
                                {"op": "replace", "path": "", "value": {"only": True}}],
                     Limits(), Applied, id="applied-add-then-root-replace"),
        pytest.param({"a": {"x": 1}, "b": 2}, [{"op": "move", "from": "/a", "path": ""}],
                     Limits(), Applied, id="applied-root-move"),
        pytest.param({"a": big, "c": 1}, [{"op": "remove", "path": "/a"},
                                          {"op": "test", "path": "/c", "value": 2}],
                     Limits(), Rejected, id="rejected-after-remove"),
        pytest.param({"a": big}, [{"op": "replace", "path": "", "value": [1]},
                                  {"op": "remove", "path": "/zz"}],
                     Limits(), Rejected, id="rejected-after-root-replace"),
        pytest.param({"a": big}, [{"op": "remove", "path": "/a"},
                                  {"op": "add", "path": "/b", "value": "z" * 2000}],
                     Limits(field_weight=1500), CapExceeded, id="cap-after-remove"),
        pytest.param({"a": 1}, [{"op": "replace", "path": "", "value": {"s": big}},
                                {"op": "remove", "path": "/s"}],
                     Limits(work_per_frame=15), CapExceeded, id="cap-after-root-replace"),
        pytest.param({"a": {"s": big}, "b": 1}, [{"op": "move", "from": "/a", "path": ""},
                                                 {"op": "test", "path": "", "value": {"s": big}}],
                     Limits(work_per_frame=30), CapExceeded, id="cap-after-root-move"),
        pytest.param({"a": 1}, [{"op": "add", "path": "/b", "value": 1}] * 3,
                     Limits(ops_per_frame=2), CapExceeded, id="cap-ops-per-frame"),
    ]  # fmt: skip


@pytest.mark.parametrize(("doc", "raw_ops", "limits", "kind"), _scenarios())
def test_every_result_lets_the_caller_resync_the_budget(
    doc: Any, raw_ops: list[Any], limits: Limits, kind: type
) -> None:
    """The `apply_ops` rule, using only the result: keep `Applied.doc` at its
    weight, or drop the field and subtract `result.weight`."""
    ops = [op for op in (parse_patch_op(r) for r in copy.deepcopy(raw_ops)) if op is not None]
    budget = Budget(limits)
    w = weight(doc)
    budget.total_weight = _OTHER + w
    result = apply_ops(doc, w, ops, budget)
    assert isinstance(result, kind), result
    if isinstance(result, Applied):
        kept = weight(result.doc)
        assert result.weight == kept
    else:
        budget.total_weight -= result.weight
        kept = 0
    assert budget.total_weight == _OTHER + kept


def test_inserted_values_are_owned_by_the_document() -> None:
    """The applier copies every value it inserts, so no node is shared
    between two places, two documents, or a document and an op."""
    raw: dict[str, Any] = {"n": 1}
    add = parse_patch_op({"op": "add", "path": "/a", "value": raw})
    rep = parse_patch_op({"op": "replace", "path": "/b", "value": raw})
    move = parse_patch_op({"op": "move", "from": "/a", "path": "/b/child"})
    assert isinstance(add, Add) and isinstance(rep, Replace) and move is not None
    docs: list[JsonValue] = []
    for _ in range(2):
        doc: JsonValue = {"b": 0}
        budget = Budget(Limits())
        result = apply_ops(doc, weight(doc), [add, rep, move], budget)
        assert isinstance(result, Applied)
        assert result.doc == {"b": {"n": 1, "child": {"n": 1}}}
        assert measure(result.doc) is not None  # a tree: no shared or cyclic node
        docs.append(result.doc)
    raw["n"] = 99
    assert docs[0] == docs[1] == {"b": {"n": 1, "child": {"n": 1}}}
    assert not _reachable_ids(docs[0]) & _reachable_ids(docs[1])
    assert not _reachable_ids(docs[0]) & _reachable_ids(raw)


def test_weight_walk_ends_on_shared_and_cyclic_values() -> None:
    shared: Any = [0]
    for _ in range(100):
        shared = [shared, shared]  # 2**100 paths through 101 containers
    with untraced():
        start = time.process_time()
        assert measure(shared, max_depth=2**62) is None
        assert weight(shared) == 0
        cyclic: Any = {"a": [1]}
        cyclic["a"].append(cyclic)
        assert measure(cyclic, max_depth=2**62) is None
        assert weight(cyclic) == 0
        assert time.process_time() - start < 1.0
