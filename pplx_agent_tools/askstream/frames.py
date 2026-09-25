"""One ask-family SSE `data` payload decoded into a typed frame.

`decode_frame` is total: any string gives a `Frame` and the drift items met
on the way. Each JSON object on the envelope, block and `diff_block` levels
is read through `_Obj`, where every key is taken, listed as ignored, or
counted as drift, so no key passes unseen. The content inside a `<name>_block`
or a patch value is not walked here; the projections read it.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar, final

from pplx_agent_tools.askstream.drift import Drift, DriftKind, name_of
from pplx_agent_tools.askstream.ids import (
    BackendUuid,
    ContextUuid,
    Cursor,
    ReadWriteToken,
    parse_backend_uuid,
    parse_context_uuid,
    parse_cursor,
)
from pplx_agent_tools.askstream.jsonval import JsonError, JsonValue, loads
from pplx_agent_tools.askstream.patch import (
    MAX_POINTER_SEGMENTS,
    Copy,
    Move,
    PatchOp,
    Pointer,
    parse_patch_op,
)

_T = TypeVar("_T")

__all__ = [
    "IGNORED_ENVELOPE",
    "KNOWN_USAGES",
    "AskFrame",
    "BlockDiff",
    "BlockMalformed",
    "BlockSnapshot",
    "BlockUpdate",
    "EndOfStream",
    "FieldKey",
    "Frame",
    "Heartbeat",
    "Reconnectable",
    "Stage",
    "Unparseable",
    "decode_frame",
    "known_usage",
]

# Derived once per frame from (status, text_completed).
Stage = Literal["pending", "text_complete", "completed", "failed", "other"]
Reconnectable = Literal["yes", "no", "absent"]
UnparseableReason = Literal["syntax", "depth", "not_object"]
MalformedReason = Literal[
    "snapshot_not_object",
    "diff_not_object",
    "field_not_string",
    "field_too_deep",
    "patches_not_list",
    "bad_op",
    "pointer_too_deep",
]
# (intended_usage, field). A dotted diff `field` keeps only its first part
# here; the rest is a pointer prefix on its ops, so every patch to one block
# lands in the same document.
FieldKey: TypeAlias = tuple[str, tuple[str, ...]]

# Envelope keys known and unused. Every entry occurs in a committed fixture,
# so dropping one fails the zero-drift fixture test.
IGNORED_ENVELOPE = frozenset(
    {
        "_extras",
        "access_level",
        "answer_modes",
        "async_rq_enabled",
        "attachments",
        "author_id",
        "author_username",
        "bookmark_state",
        "classifier_results",
        "entry_created_datetime",
        "entry_updated_datetime",
        "expect_search_results",
        "expiry_time",
        "final",
        "final_sse_message",
        "frontend_context_uuid",
        "frontend_uuid",
        "gpt4",
        "image_completions",
        "knowledge_cards",
        "message_mode",
        "mode",
        "personalized",
        "plan",
        "privacy_state",
        "prompt_source",
        "query_language",
        "query_source",
        "query_str",
        "related_queries",
        "related_query_items",
        "s3_social_preview_url",
        "search_focus",
        "search_implementation_mode",
        "search_mode",
        "source",
        "sources",
        "step_type",
        "structured_answer_block_usages",
        "telemetry_data",
        "thread_access",
        "thread_title",
        "thread_url_slug",
        "updated_datetime",
        "user_selected_model",
        "uuid",
        "widget_data",
    }
)

KNOWN_USAGES = frozenset(
    {
        "answer_assets_preview",
        "answer_tabs",
        "ask_text",
        "assets_answer_mode",
        "canvas_mode",
        "in_context_suggestions",
        "pending_followups",
        "plan",
        "pro_search_steps",
        "sources_answer_mode",
        "unified_assets",
        "web_results",
        "workflow_root",
    }
)
_NUMBERED_USAGE = re.compile(r"ask_text_[0-9]{1,4}_markdown")


def known_usage(usage: str) -> bool:
    return usage in KNOWN_USAGES or _NUMBERED_USAGE.fullmatch(usage) is not None


@final
@dataclass(frozen=True, slots=True)
class Heartbeat:
    pass


@final
@dataclass(frozen=True, slots=True)
class EndOfStream:
    pass


@final
@dataclass(frozen=True, slots=True)
class Unparseable:
    reason: UnparseableReason
    size: int


@final
@dataclass(frozen=True, slots=True)
class BlockSnapshot:
    key: FieldKey
    value: dict[str, JsonValue]


@final
@dataclass(frozen=True, slots=True)
class BlockDiff:
    key: FieldKey
    ops: tuple[PatchOp, ...]


@final
@dataclass(frozen=True, slots=True)
class BlockMalformed:
    usage: str
    # Empty when the field name itself could not be read.
    field: tuple[str, ...]
    reason: MalformedReason


BlockUpdate: TypeAlias = BlockSnapshot | BlockDiff | BlockMalformed


@final
@dataclass(frozen=True, slots=True)
class AskFrame:
    stage: Stage
    raw_status: str | None
    backend_uuid: BackendUuid | None
    token: ReadWriteToken | None
    cursor: Cursor | None
    context_uuid: ContextUuid | None
    reconnectable: Reconnectable
    display_model: str | None
    text: str | None
    blocks: tuple[BlockUpdate, ...]
    size: int


Frame: TypeAlias = Heartbeat | EndOfStream | Unparseable | AskFrame


@final
class _Obj:
    """Typed reads from one JSON object. A key present with value null
    counts as absent; a key present with another wrong type is recorded as
    `unexpected_type`. `rest` yields the keys never taken."""

    __slots__ = ("_d", "_drift", "_taken")

    def __init__(self, d: dict[str, JsonValue], drift: list[Drift]) -> None:
        self._d = d
        self._drift = drift
        self._taken: set[str] = set()

    def raw(self, key: str) -> JsonValue:
        self._taken.add(key)
        return self._d.get(key)

    def mistyped(self, key: str) -> None:
        self._drift.append(Drift("unexpected_type", name_of(key)))

    def str_(self, key: str) -> str | None:
        v = self.raw(key)
        if v is None or isinstance(v, str):
            return v
        self.mistyped(key)
        return None

    def bool_(self, key: str) -> bool | None:
        v = self.raw(key)
        if v is None or isinstance(v, bool):
            return v
        self.mistyped(key)
        return None

    def list_(self, key: str) -> list[JsonValue] | None:
        v = self.raw(key)
        if v is None or isinstance(v, list):
            return v
        self.mistyped(key)
        return None

    def rest(self) -> list[str]:
        return [k for k in self._d if k not in self._taken]

    def finish(self, ignored: frozenset[str], kind: DriftKind) -> None:
        self._drift.extend(Drift(kind, name_of(k)) for k in self.rest() if k not in ignored)


def _stage(status: str | None, text_completed: bool | None, drift: list[Drift]) -> Stage:
    # A frame with no status says nothing about the run ending.
    if status is None or status == "PENDING":
        return "text_complete" if text_completed else "pending"
    if status == "COMPLETED":
        return "completed"
    if status == "FAILED":
        return "failed"
    drift.append(Drift("unknown_status", name_of(status)))
    return "other"


def _prefixed(op: PatchOp, prefix: Pointer) -> PatchOp:
    if not prefix:
        return op
    if isinstance(op, (Move, Copy)):
        return dataclasses.replace(op, from_=prefix + op.from_, path=prefix + op.path)
    return dataclasses.replace(op, path=prefix + op.path)


def _field_op(raw: JsonValue, prefix: Pointer) -> PatchOp | MalformedReason:
    op = parse_patch_op(raw)
    if op is None:
        return "bad_op"
    op = _prefixed(op, prefix)
    # The dotted field and the op path are each within the limit; the
    # pointer they join into must be too.
    from_ = op.from_ if isinstance(op, (Move, Copy)) else ()
    if max(len(op.path), len(from_)) > MAX_POINTER_SEGMENTS:
        return "pointer_too_deep"
    return op


def _malformed(
    usage: str, field: tuple[str, ...], reason: MalformedReason, drift: list[Drift]
) -> BlockMalformed:
    drift.append(Drift("malformed_block", name_of(f"{usage}/{'.'.join(field)}:{reason}")))
    return BlockMalformed(usage, field, reason)


def _diff(usage: str, raw: JsonValue, drift: list[Drift]) -> BlockUpdate:
    if not isinstance(raw, dict):
        return _malformed(usage, (), "diff_not_object", drift)
    obj = _Obj(raw, drift)
    field_raw = obj.raw("field")
    patches = obj.raw("patches")
    obj.finish(frozenset(), "unknown_diff_key")
    if not isinstance(field_raw, str):
        return _malformed(usage, (), "field_not_string", drift)
    field = tuple(field_raw.split("."))
    if len(field) > MAX_POINTER_SEGMENTS:
        return _malformed(usage, (), "field_too_deep", drift)
    if not isinstance(patches, list):
        return _malformed(usage, field, "patches_not_list", drift)
    ops: list[PatchOp] = []
    for p in patches:
        op = _field_op(p, field[1:])
        if isinstance(op, str):
            return _malformed(usage, field, op, drift)
        ops.append(op)
    return BlockDiff((usage, field[:1]), tuple(ops))


def _block(raw: JsonValue, drift: list[Drift]) -> list[BlockUpdate]:
    if not isinstance(raw, dict):
        drift.append(Drift("unexpected_type", name_of("blocks")))
        return []
    obj = _Obj(raw, drift)
    usage = obj.str_("intended_usage")
    if usage is None:
        # Content with no usage has no field to land in.
        if raw.get("intended_usage") is None:
            drift.append(Drift("malformed_block", name_of("no_usage")))
        return []
    if not known_usage(usage):
        drift.append(Drift("unknown_usage", name_of(usage)))
    out: list[BlockUpdate] = []
    for key in obj.rest():
        if key == "diff_block":
            out.append(_diff(usage, obj.raw(key), drift))
        elif key.endswith("_block"):
            value = obj.raw(key)
            if isinstance(value, dict):
                out.append(BlockSnapshot((usage, (key,)), value))
            elif value is not None:
                out.append(_malformed(usage, (key,), "snapshot_not_object", drift))
    obj.finish(frozenset(), "unknown_block_key")
    return out


def _id(env: _Obj, key: str, parse: Callable[[object], _T | None]) -> _T | None:
    """An id through its smart constructor; a rejected value is drift."""
    raw = env.raw(key)
    value = parse(raw)
    if value is None and raw is not None:
        env.mistyped(key)
    return value


def _ask_frame(d: dict[str, JsonValue], size: int, drift: list[Drift]) -> AskFrame:
    env = _Obj(d, drift)
    status = env.str_("status")
    text_completed = env.bool_("text_completed")
    backend = _id(env, "backend_uuid", parse_backend_uuid)
    token = _id(env, "read_write_token", ReadWriteToken.parse)
    cursor = _id(env, "cursor", parse_cursor)
    context = _id(env, "context_uuid", parse_context_uuid)
    reconnectable = env.bool_("reconnectable")
    display_model = env.str_("display_model")
    text = env.str_("text")
    blocks_raw = env.list_("blocks") or []
    env.finish(IGNORED_ENVELOPE, "unknown_envelope_key")
    blocks = tuple(u for b in blocks_raw for u in _block(b, drift))
    return AskFrame(
        stage=_stage(status, text_completed, drift),
        raw_status=status,
        backend_uuid=backend,
        token=token,
        cursor=cursor,
        context_uuid=context,
        reconnectable="absent" if reconnectable is None else "yes" if reconnectable else "no",
        display_model=display_model,
        text=text,
        blocks=blocks,
        size=size,
    )


def decode_frame(raw: str) -> tuple[Frame, tuple[Drift, ...]]:
    """Decode one `data` payload. Never raises. `{}` ends the stream;
    anything that is not a JSON object nested at most 128 deep is
    `Unparseable`, with one `unparseable_frame` drift item."""
    size = len(raw)
    value = loads(raw)
    if isinstance(value, JsonError) or not isinstance(value, dict):
        reason: UnparseableReason = value.reason if isinstance(value, JsonError) else "not_object"
        return Unparseable(reason, size), (Drift("unparseable_frame", name_of(reason)),)
    if not value:
        return EndOfStream(), ()
    drift: list[Drift] = []
    frame = _ask_frame(value, size, drift)
    return frame, tuple(drift)
