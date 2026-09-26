"""`decode_frame`: totality, depth, stage, ids, blocks and drift (plan §2.3;
U3 oracles 1, 7 and 8)."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import (
    IGNORED_ENVELOPE,
    AskFrame,
    BlockDiff,
    BlockMalformed,
    BlockSnapshot,
    EndOfStream,
    Frame,
    Heartbeat,
    Unparseable,
    UnparseableReason,
    decode_frame,
)
from pplx_agent_tools.askstream.ids import ReadWriteToken
from pplx_agent_tools.askstream.patch import Add, Copy, Move, Remove, Replace

UUID = "0d2b1c3a-1111-2222-3333-444455556666"
CTX = "5e8f0a7b-aaaa-bbbb-cccc-ddddeeeeffff"
TOKEN = "Zk3-q_9TOKENVALUEXYZ"
FRAME_TYPES = (Heartbeat, EndOfStream, Unparseable, AskFrame)
TAKEN_ENVELOPE = frozenset(
    {
        "status",
        "text_completed",
        "backend_uuid",
        "read_write_token",
        "cursor",
        "context_uuid",
        "reconnectable",
        "display_model",
        "text",
        "blocks",
    }
)


def _base() -> dict[str, Any]:
    return {
        "backend_uuid": UUID,
        "context_uuid": CTX,
        "read_write_token": TOKEN,
        "cursor": "c-1",
        "status": "PENDING",
        "text_completed": False,
        "reconnectable": True,
        "display_model": "turbo",
        "thread_url_slug": "slug",
        "blocks": [
            {"intended_usage": "ask_text", "markdown_block": {"chunks": ["a"]}},
            {
                "intended_usage": "web_results",
                "diff_block": {
                    "field": "web_result_block",
                    "patches": [{"op": "add", "path": "/web_results/0", "value": {"url": "u"}}],
                },
            },
        ],
    }


def _decode(obj: object) -> tuple[Frame, tuple[Drift, ...]]:
    return decode_frame(json.dumps(obj))


def _ask(obj: object) -> tuple[AskFrame, tuple[Drift, ...]]:
    frame, drift = _decode(obj)
    assert isinstance(frame, AskFrame)
    return frame, drift


def _check_total(raw: str) -> None:
    frame, drift = decode_frame(raw)
    assert isinstance(frame, FRAME_TYPES)
    assert isinstance(drift, tuple)
    assert all(isinstance(d, Drift) for d in drift)


JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(),
    lambda kids: (
        st.lists(kids, max_size=4) | st.dictionaries(st.text(max_size=8), kids, max_size=4)
    ),
    max_leaves=20,
)
OP_MEMBERS: dict[str, st.SearchStrategy[object]] = {
    "path": st.sampled_from(["", "/a", "/a/0", "x"]),
    "from": st.just("/b"),
    "value": JSON,
}
ENVELOPE_KEYS = st.sampled_from(sorted(TAKEN_ENVELOPE | {"uuid", "x"}))
BLOCK = st.fixed_dictionaries(
    {},
    optional={
        "intended_usage": st.sampled_from(["ask_text", "web_results", "zz"]) | JSON,
        "markdown_block": JSON,
        "diff_block": st.fixed_dictionaries(
            {},
            optional={
                "field": st.sampled_from(["markdown_block", "a.b.c"]) | JSON,
                "patches": st.lists(
                    st.fixed_dictionaries(
                        {
                            "op": st.sampled_from(
                                ["add", "remove", "replace", "move", "copy", "test", "zz"]
                            )
                        },
                        optional=OP_MEMBERS,
                    ),
                    max_size=3,
                )
                | JSON,
            },
        )
        | JSON,
    },
)
ENVELOPE = st.dictionaries(ENVELOPE_KEYS, JSON, max_size=6).flatmap(
    lambda d: st.lists(BLOCK, max_size=3).map(lambda bs: {**d, "blocks": bs})
) | st.dictionaries(ENVELOPE_KEYS, JSON, max_size=8)


@given(st.text())
def test_decode_frame_is_total_on_any_text(raw: str) -> None:
    _check_total(raw)


@given(st.binary())
def test_decode_frame_is_total_on_undecodable_bytes(raw: bytes) -> None:
    _check_total(raw.decode("utf-8", "surrogateescape"))


@settings(max_examples=300)
@given(JSON)
def test_decode_frame_is_total_on_any_json(value: object) -> None:
    _check_total(json.dumps(value))


@settings(max_examples=300)
@given(ENVELOPE)
def test_decode_frame_is_total_on_envelope_shaped_json(value: dict[str, object]) -> None:
    _check_total(json.dumps(value))


@pytest.mark.parametrize(
    "raw",
    [
        "1" * 5000,
        '{"text_completed": ' + "9" * 5000 + "}",
        "1e999",
        '{"cursor": -1e400}',
        "NaN",
        '{"status": Infinity}',
        '"\\ud800"',
        '{"\\udfff": 1}',
        "",
        "   ",
        "{",
        "[" * 50 + "]" * 50,
    ],
)
def test_decode_frame_is_total_on_edge_inputs(raw: str) -> None:
    _check_total(raw)


@pytest.mark.parametrize("n", [100_000, 129])
@pytest.mark.parametrize("open_close", [("[", "]"), ('{"a":', "}")])
def test_deep_nesting_is_unparseable_depth(n: int, open_close: tuple[str, str]) -> None:
    o, c = open_close
    frame, drift = decode_frame(o * n + "1" + c * n)
    assert frame == Unparseable("depth", len(o * n + "1" + c * n))
    assert drift == (Drift("unparseable_frame", name_of("depth")),)


def test_depth_128_is_accepted() -> None:
    frame, drift = decode_frame('{"a":' * 128 + "1" + "}" * 128)
    assert isinstance(frame, AskFrame)
    assert drift == (Drift("unknown_envelope_key", name_of("a")),)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("[]", "not_object"),
        ("null", "not_object"),
        ('"x"', "not_object"),
        ("{", "syntax"),
        ("", "syntax"),
    ],
)
def test_non_objects_are_unparseable(raw: str, reason: UnparseableReason) -> None:
    frame, drift = decode_frame(raw)
    assert frame == Unparseable(reason, len(raw))
    assert drift == (Drift("unparseable_frame", name_of(reason)),)


def test_empty_object_ends_the_stream() -> None:
    assert decode_frame("{}") == (EndOfStream(), ())


@pytest.mark.parametrize(
    ("status", "text_completed", "stage"),
    [
        ("PENDING", False, "pending"),
        ("PENDING", True, "text_complete"),
        ("PENDING", None, "pending"),
        ("COMPLETED", True, "completed"),
        ("COMPLETED", False, "completed"),
        ("FAILED", False, "failed"),
        (None, True, "text_complete"),
        (None, None, "pending"),
    ],
)
def test_stage_is_derived_once(status: str | None, text_completed: bool | None, stage: str) -> None:
    frame, drift = _ask({"status": status, "text_completed": text_completed, "text": "t"})
    assert frame.stage == stage
    assert frame.raw_status == status
    assert drift == ()


def test_unknown_status_is_other_with_drift() -> None:
    frame, drift = _ask({"status": "PAUSED"})
    assert frame.stage == "other"
    assert drift == (Drift("unknown_status", name_of("PAUSED")),)


def test_envelope_fields_are_read() -> None:
    frame, drift = _ask({**_base(), "reconnectable": False, "text": "T", "status": "COMPLETED"})
    assert drift == ()
    assert frame.backend_uuid == UUID
    assert frame.context_uuid == CTX
    assert frame.token == ReadWriteToken.parse(TOKEN)
    assert frame.cursor == "c-1"
    assert frame.reconnectable == "no"
    assert frame.display_model == "turbo"
    assert frame.text == "T"
    assert frame.size == len(
        json.dumps({**_base(), "reconnectable": False, "text": "T", "status": "COMPLETED"})
    )
    assert _ask({"status": "PENDING"})[0].reconnectable == "absent"


def test_null_counts_as_absent() -> None:
    keys = [*TAKEN_ENVELOPE, "uuid"]
    frame, drift = _ask({**dict.fromkeys(keys), "status": "PENDING"})
    assert drift == ()
    assert frame.backend_uuid is None
    assert frame.token is None
    assert frame.reconnectable == "absent"
    assert frame.blocks == ()


def test_uppercase_uuid_is_canonicalized() -> None:
    frame, _ = _ask({"backend_uuid": UUID.upper(), "status": "PENDING"})
    assert frame.backend_uuid == UUID


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("backend_uuid", "BU"),
        ("backend_uuid", 7),
        ("context_uuid", "not-a-uuid"),
        ("read_write_token", "has space"),
        ("read_write_token", "ü" * 3),
        ("cursor", ""),
        ("cursor", "x" * 2000),
        ("status", 1),
        ("text_completed", "yes"),
        ("reconnectable", 1),
        ("display_model", []),
        ("text", {}),
        ("blocks", True),
    ],
)
def test_wrong_typed_known_key_is_unexpected_type(key: str, value: object) -> None:
    frame, drift = _ask({"status": "PENDING", key: value})
    assert drift == (Drift("unexpected_type", name_of(key)),)
    assert frame.stage == "pending"


def test_token_value_never_shows_in_frame_repr() -> None:
    frame, _ = _ask(_base())
    assert TOKEN not in repr(frame)
    assert TOKEN not in str(frame)
    assert TOKEN not in f"{frame}"


def test_snapshot_and_diff_blocks() -> None:
    frame, drift = _ask(_base())
    assert drift == ()
    snap, diff = frame.blocks
    assert snap == BlockSnapshot(("ask_text", ("markdown_block",)), {"chunks": ["a"]})
    assert isinstance(diff, BlockDiff)
    assert diff.key == ("web_results", ("web_result_block",))
    (op,) = diff.ops
    assert isinstance(op, Add)
    assert op.path == ("web_results", "0")


def test_dotted_field_prefixes_every_op_path() -> None:
    patches = [
        {"op": "add", "path": "/1", "value": "x"},
        {"op": "replace", "path": "", "value": []},
        {"op": "remove", "path": "/0"},
        {"op": "move", "from": "/0", "path": "/2"},
        {"op": "copy", "from": "/1", "path": "/3"},
    ]
    block = {
        "intended_usage": "ask_text",
        "diff_block": {"field": "markdown_block.chunks", "patches": patches},
    }
    frame, drift = _ask({"status": "PENDING", "blocks": [block]})
    assert drift == ()
    (diff,) = frame.blocks
    assert isinstance(diff, BlockDiff)
    assert diff.key == ("ask_text", ("markdown_block",))
    add, rep, rem, mov, cop = diff.ops
    assert isinstance(add, Add) and add.path == ("chunks", "1")
    assert isinstance(rep, Replace) and rep.path == ("chunks",)
    assert isinstance(rem, Remove) and rem.path == ("chunks", "0")
    assert isinstance(mov, Move) and (mov.from_, mov.path) == (("chunks", "0"), ("chunks", "2"))
    assert isinstance(cop, Copy) and (cop.from_, cop.path) == (("chunks", "1"), ("chunks", "3"))


@pytest.mark.parametrize(
    ("block", "malformed"),
    [
        (
            {"intended_usage": "plan", "plan_block": [1]},
            BlockMalformed("plan", ("plan_block",), "snapshot_not_object"),
        ),
        (
            {"intended_usage": "plan", "diff_block": "x"},
            BlockMalformed("plan", (), "diff_not_object"),
        ),
        (
            {"intended_usage": "plan", "diff_block": {"field": 3, "patches": []}},
            BlockMalformed("plan", (), "field_not_string"),
        ),
        (
            {
                "intended_usage": "plan",
                "diff_block": {"field": "plan_block" + "." * 200, "patches": []},
            },
            BlockMalformed("plan", ("plan_block",), "field_too_deep"),
        ),
        (
            # 127 dotted segments below the field plus a 128-segment op path.
            {
                "intended_usage": "plan",
                "diff_block": {
                    "field": "plan_block" + ".a" * 127,
                    "patches": [{"op": "add", "path": "/b" * 128, "value": 1}],
                },
            },
            BlockMalformed("plan", ("plan_block", *("a",) * 127), "pointer_too_deep"),
        ),
        (
            {
                "intended_usage": "plan",
                "diff_block": {
                    "field": "plan_block.a",
                    "patches": [{"op": "copy", "from": "/b" * 128, "path": "/c"}],
                },
            },
            BlockMalformed("plan", ("plan_block", "a"), "pointer_too_deep"),
        ),
        (
            {"intended_usage": "plan", "diff_block": {"field": "plan_block", "patches": {}}},
            BlockMalformed("plan", ("plan_block",), "patches_not_list"),
        ),
        (
            {"intended_usage": "plan", "diff_block": {"field": "plan_block"}},
            BlockMalformed("plan", ("plan_block",), "patches_not_list"),
        ),
        (
            {
                "intended_usage": "plan",
                "diff_block": {
                    "field": "plan_block",
                    "patches": [
                        {"op": "add", "path": "/a", "value": 1},
                        {"op": "zz", "path": "/a"},
                    ],
                },
            },
            BlockMalformed("plan", ("plan_block",), "bad_op"),
        ),
        (
            {
                "intended_usage": "plan",
                "diff_block": {
                    "field": "plan_block",
                    "patches": [{"op": "add", "path": "a", "value": 1}],
                },
            },
            BlockMalformed("plan", ("plan_block",), "bad_op"),
        ),
    ],
)
def test_malformed_blocks(block: dict[str, object], malformed: BlockMalformed) -> None:
    frame, drift = _ask({"status": "PENDING", "blocks": [block]})
    assert frame.blocks == (malformed,)
    assert [d.kind for d in drift] == ["malformed_block"]


def test_block_without_usage_is_dropped_with_drift() -> None:
    frame, drift = _ask(
        {"status": "PENDING", "blocks": [{"markdown_block": {}}, {"intended_usage": 3}, 4]}
    )
    assert frame.blocks == ()
    assert Counter(drift) == Counter(
        [
            Drift("malformed_block", name_of("no_usage")),
            Drift("unexpected_type", name_of("intended_usage")),
            Drift("unexpected_type", name_of("blocks")),
        ]
    )


@pytest.mark.parametrize(
    "usage", ["ask_text_0_markdown", "ask_text_12_markdown", "workflow_root", "pro_search_steps"]
)
def test_known_usages_carry_no_drift(usage: str) -> None:
    _, drift = _ask(
        {"status": "PENDING", "blocks": [{"intended_usage": usage, "markdown_block": {}}]}
    )
    assert drift == ()


def test_unknown_usage_is_drift_and_still_decoded() -> None:
    frame, drift = _ask(
        {"status": "PENDING", "blocks": [{"intended_usage": "new_thing", "x_block": {}}]}
    )
    assert frame.blocks == (BlockSnapshot(("new_thing", ("x_block",)), {}),)
    assert drift == (Drift("unknown_usage", name_of("new_thing")),)


def test_unknown_key_names_are_scrubbed() -> None:
    _, drift = _ask({"status": "PENDING", UUID: 1, f"k_{TOKEN}": 2})
    assert all(UUID not in d.name and TOKEN not in d.name for d in drift)
    assert {d.kind for d in drift} == {"unknown_envelope_key"}


# Keys no level takes or ignores: no `_block` suffix, not a known key.
UNKNOWN_KEY = st.text(min_size=1, max_size=20).filter(
    lambda k: (
        not k.endswith("_block")
        and k
        not in TAKEN_ENVELOPE
        | IGNORED_ENVELOPE
        | {"intended_usage", "diff_block", "field", "patches"}
    )
)
WRONG_TYPED = {
    "status": 1,
    "text_completed": "no",
    "backend_uuid": "BU",
    "read_write_token": "bad token",
    "cursor": "",
    "context_uuid": 5,
    "reconnectable": "yes",
    "display_model": [],
    "text": {},
}


@settings(max_examples=200)
@given(
    st.lists(UNKNOWN_KEY, max_size=5, unique=True),
    st.lists(UNKNOWN_KEY, max_size=5, unique=True),
    st.lists(UNKNOWN_KEY, max_size=5, unique=True),
    st.sets(st.sampled_from(sorted(WRONG_TYPED))),
)
def test_ledger_counts_exactly_what_was_injected(
    env_keys: list[str], block_keys: list[str], diff_keys: list[str], wrong: set[str]
) -> None:
    obj = _base()
    obj.update(dict.fromkeys(env_keys, 0))
    obj.update({k: WRONG_TYPED[k] for k in wrong})
    obj["blocks"][0].update(dict.fromkeys(block_keys, 0))
    obj["blocks"][1]["diff_block"].update(dict.fromkeys(diff_keys, 0))
    _, drift = _decode(obj)
    expected = Counter(
        [Drift("unknown_envelope_key", name_of(k)) for k in env_keys]
        + [Drift("unknown_block_key", name_of(k)) for k in block_keys]
        + [Drift("unknown_diff_key", name_of(k)) for k in diff_keys]
        + [Drift("unexpected_type", name_of(k)) for k in wrong]
    )
    assert Counter(drift) == expected
