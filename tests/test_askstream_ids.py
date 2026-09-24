"""Ids and the read-write token (plan §2.1, I13)."""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import pickle
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from pplx_agent_tools.askstream.ids import (
    BackendUuid,
    ReadWriteToken,
    ThreadRef,
    parse_backend_uuid,
    parse_context_uuid,
    parse_cursor,
)

REPO = Path(__file__).resolve().parent.parent
UUID = "0d2b1c3a-1111-2222-3333-444455556666"
TOKENS = st.from_regex(r"[A-Za-z0-9_-]{1,256}", fullmatch=True)
OTHER_RAW = "zzzzzzzz-other-token"


def _tok(raw: str) -> ReadWriteToken:
    tok = ReadWriteToken.parse(raw)
    assert tok is not None
    return tok


@dataclasses.dataclass(frozen=True)
class _Inner:
    token: ReadWriteToken | None
    blocks: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _FrameLike:
    """Stands in for a decoded frame that holds a token (frames land later)."""

    stage: str
    inner: _Inner
    ref: ThreadRef


def _holders(tok: ReadWriteToken) -> list[object]:
    ref = ThreadRef(BackendUuid(UUID), tok)
    return [tok, ref, _FrameLike("pending", _Inner(tok, ("a",)), ref)]


def _assert_hidden(render: Callable[[object], str], raw: str) -> None:
    """`render` never shows `raw`: its output does not depend on the token's
    value at all, and does not contain it."""
    for tok_holder, other_holder in zip(
        _holders(_tok(raw)), _holders(_tok(OTHER_RAW)), strict=True
    ):
        out = render(tok_holder)
        assert out == render(other_holder)
        assume(raw not in render(other_holder))
        assert raw not in out


# --- I13, one test per route ------------------------------------------------------


@given(TOKENS)
def test_repr_hides_token(raw: str) -> None:
    _assert_hidden(repr, raw)


@given(TOKENS)
def test_str_hides_token(raw: str) -> None:
    _assert_hidden(str, raw)


@given(TOKENS, st.text(max_size=20))
def test_format_any_spec_hides_token(raw: str, spec: str) -> None:
    tok = _tok(raw)
    assert format(tok, spec) == "ReadWriteToken(<redacted>)"
    assert f"{tok:{spec}}" == "ReadWriteToken(<redacted>)"
    _assert_hidden(lambda o: f"{o}", raw)


@given(TOKENS)
def test_percent_formatting_hides_token(raw: str) -> None:
    _assert_hidden(lambda o: "%s" % (o,), raw)  # noqa: UP031
    _assert_hidden(lambda o: "%r" % (o,), raw)  # noqa: UP031


def _asdict_or_self(o: object) -> object:
    return dataclasses.asdict(o) if dataclasses.is_dataclass(o) and not isinstance(o, type) else o


def _astuple_or_self(o: object) -> object:
    return dataclasses.astuple(o) if dataclasses.is_dataclass(o) and not isinstance(o, type) else o


@given(TOKENS)
def test_asdict_then_repr_hides_token(raw: str) -> None:
    _assert_hidden(lambda o: repr(_asdict_or_self(o)), raw)


@given(TOKENS)
def test_astuple_then_repr_hides_token(raw: str) -> None:
    _assert_hidden(lambda o: repr(_astuple_or_self(o)), raw)


@given(TOKENS)
def test_asdict_then_json_default_str_hides_token(raw: str) -> None:
    _assert_hidden(lambda o: json.dumps(_asdict_or_self(o), default=str), raw)


@given(TOKENS)
def test_astuple_then_json_default_str_hides_token(raw: str) -> None:
    _assert_hidden(lambda o: json.dumps(_astuple_or_self(o), default=str), raw)


@given(TOKENS)
def test_json_dumps_raises_type_error(raw: str) -> None:
    for holder in _holders(_tok(raw)):
        with pytest.raises(TypeError):
            json.dumps(_asdict_or_self(holder))


@given(TOKENS)
def test_pickle_raises_type_error(raw: str) -> None:
    for holder in _holders(_tok(raw)):
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with pytest.raises(TypeError, match="not picklable"):
                pickle.dumps(holder, protocol=protocol)


@given(TOKENS)
def test_copy_and_deepcopy_return_same_object(raw: str) -> None:
    tok = _tok(raw)
    assert copy.copy(tok) is tok
    assert copy.deepcopy(tok) is tok
    ref = ThreadRef(BackendUuid(UUID), tok)
    assert copy.deepcopy(ref).token is tok
    assert dataclasses.asdict(ref)["token"] is tok


@given(TOKENS)
def test_vars_raises(raw: str) -> None:
    with pytest.raises(TypeError):
        vars(_tok(raw))


@given(TOKENS, TOKENS)
def test_setattr_and_delattr_raise(raw: str, other: str) -> None:
    tok = _tok(raw)
    with pytest.raises(AttributeError):
        tok._v = other  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        del tok._v  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        tok.extra = 1  # pyright: ignore[reportAttributeAccessIssue]
    assert tok.reveal() == raw


def test_direct_construction_is_refused() -> None:
    with pytest.raises(TypeError):
        ReadWriteToken()


# --- parse and equality -------------------------------------------------------------


@given(TOKENS)
def test_parse_accepts_charset_and_reveals(raw: str) -> None:
    assert _tok(raw).reveal() == raw


_JSONISH = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text() | st.binary(),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(), inner, max_size=3),
    max_leaves=8,
)


@given(st.one_of(st.text(), st.binary(), st.integers(), st.none(), _JSONISH))
def test_parse_never_raises_and_rejects_outside_charset(raw: object) -> None:
    tok = ReadWriteToken.parse(raw)
    valid = (
        isinstance(raw, str)
        and 1 <= len(raw) <= 256
        and all(c.isascii() and (c.isalnum() or c in "_-") for c in raw)
    )
    assert (tok is not None) == valid


@pytest.mark.parametrize(
    "raw",
    [
        "t\u00f6k",
        "\u200b",
        UUID[:-1] + "\u0430",  # 36 characters, the last a Cyrillic a
        "",
        "a" * 257,
        "abc\n",
        "a b",
    ],
)
def test_parse_rejects_rows(raw: str) -> None:
    assert ReadWriteToken.parse(raw) is None


def test_parse_accepts_boundary_rows() -> None:
    assert ReadWriteToken.parse("a" * 256) is not None
    assert ReadWriteToken.parse(UUID) is not None
    assert ReadWriteToken.parse("TEST_RW_TOKEN") is not None


@given(TOKENS, TOKENS, st.one_of(st.text(), st.binary(), st.integers(), st.none()))
def test_equality_never_raises(a: str, b: str, other: object) -> None:
    ta, tb = _tok(a), _tok(b)
    assert (ta == tb) == (a == b)
    assert (ta == _tok(a)) is True
    assert (ta == other) is False
    assert (other == ta) is False
    assert (ta != other) is True
    if a == b:
        assert hash(ta) == hash(tb)


# --- other ids ------------------------------------------------------------------------


@pytest.mark.parametrize("parse", [parse_backend_uuid, parse_context_uuid])
def test_uuid_parsers_canonicalize(parse: Callable[[object], str | None]) -> None:
    assert parse(UUID.upper()) == UUID
    assert parse("{" + UUID + "}") == UUID
    assert parse(UUID.replace("-", "")) == UUID
    for bad in [None, 1, "", "not-a-uuid", UUID + "0", "\u0663" * 32, UUID * 3, b"x"]:
        assert parse(bad) is None


@given(st.one_of(st.text(), st.binary(), st.integers(), st.none(), _JSONISH))
def test_id_parsers_never_raise(raw: object) -> None:
    parse_backend_uuid(raw)
    parse_context_uuid(raw)
    parse_cursor(raw)


def test_parse_cursor() -> None:
    assert parse_cursor("abc") == "abc"
    assert parse_cursor("") is None
    assert parse_cursor("x" * 1025) is None
    assert parse_cursor(3) is None


# --- AST scans: `.reveal()` and `._v` -------------------------------------------------------

# The one place the raw token may be read: the delete request body.
_REVEAL_ALLOWED = {("pplx_agent_tools/wire.py", "AsyncTransport", "delete")}
_V_ALLOWED = {"pplx_agent_tools/askstream/ids.py"}


def _scanned_files() -> Iterator[tuple[str, str]]:
    for root in ("pplx_agent_tools", "scripts"):
        for path in sorted((REPO / root).rglob("*.py")):
            yield path.relative_to(REPO).as_posix(), path.read_text(encoding="utf-8")


def _violations(rel: str, source: str) -> list[str]:
    found: list[str] = []

    def walk(node: ast.AST, cls: str | None, fn: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            c, f = cls, fn
            if isinstance(child, ast.ClassDef):
                c, f = child.name, None
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f = child.name
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "reveal"
                and (rel, cls, fn) not in _REVEAL_ALLOWED
            ):
                found.append(f"{rel}:{child.lineno}: .reveal()")
            if isinstance(child, ast.Attribute) and child.attr == "_v" and rel not in _V_ALLOWED:
                found.append(f"{rel}:{child.lineno}: ._v")
            walk(child, c, f)

    walk(ast.parse(source), None, None)
    return found


def test_reveal_and_private_value_scans() -> None:
    files = list(_scanned_files())
    assert any(rel == "pplx_agent_tools/askstream/ids.py" for rel, _ in files)
    found = [v for rel, src in files for v in _violations(rel, src)]
    assert found == []


def test_scans_detect_violations() -> None:
    src = (
        "def f(t):\n    return t.reveal()\n"
        "class AsyncTransport:\n    def delete(self, r):\n        return r.token.reveal()\n"
        "x = tok._v\n"
    )
    assert _violations("pplx_agent_tools/other.py", src) == [
        "pplx_agent_tools/other.py:2: .reveal()",
        "pplx_agent_tools/other.py:5: .reveal()",
        "pplx_agent_tools/other.py:6: ._v",
    ]
    assert _violations("pplx_agent_tools/wire.py", src) == [
        "pplx_agent_tools/wire.py:2: .reveal()",
        "pplx_agent_tools/wire.py:6: ._v",
    ]
