"""Branded ids read from ask-family frames.

Each id has one smart constructor that returns None for anything it does not
accept, so a value of these types was always checked. `ReadWriteToken` is a
credential: no route (repr, str, format, dataclass walks, json, pickle, copy,
state and attribute introspection) yields its value, and `reveal()` is the
one reader.
"""

from __future__ import annotations

import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from typing import NewType, NoReturn, final

BackendUuid = NewType("BackendUuid", str)
ContextUuid = NewType("ContextUuid", str)
Cursor = NewType("Cursor", str)
ConnId = NewType("ConnId", int)

# Longest accepted spelling of a UUID ("urn:uuid:" plus 36); checked before
# `uuid.UUID` so an oversized value costs nothing.
_MAX_UUID_INPUT = 45
_MAX_CURSOR = 1024
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{1,256}")
_REDACTED = "ReadWriteToken(<redacted>)"


def _canonical_uuid(raw: object) -> str | None:
    # `isascii` first: `uuid.UUID` parses via `int(_, 16)`, which accepts
    # non-ASCII decimal digits.
    if not isinstance(raw, str) or len(raw) > _MAX_UUID_INPUT or not raw.isascii():
        return None
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return None


def parse_backend_uuid(raw: object) -> BackendUuid | None:
    """The canonical lowercase form of a UUID, else None."""
    v = _canonical_uuid(raw)
    return None if v is None else BackendUuid(v)


def parse_context_uuid(raw: object) -> ContextUuid | None:
    """The canonical lowercase form of a UUID, else None."""
    v = _canonical_uuid(raw)
    return None if v is None else ContextUuid(v)


def parse_cursor(raw: object) -> Cursor | None:
    """A non-empty string of at most 1024 characters, else None."""
    if not isinstance(raw, str) or not raw or len(raw) > _MAX_CURSOR:
        return None
    return Cursor(raw)


@final
class ReadWriteToken:
    """A thread's read-write token. Build with `parse`; read with `reveal`.

    A slots class rather than a dataclass: `dataclasses.asdict` recurses into
    dataclass fields, and would copy the raw value out.

    The instance never holds the raw text: `_v` is its bytes XORed with the
    random `_pad`. So every route that reads or prints stored state
    (`__getstate__`, `inspect.getmembers`, `"{0._v}".format`, a debugger)
    sees only masked bytes; unmasking takes a call, which only this class
    makes.
    """

    __slots__ = ("_pad", "_v")
    _pad: bytes
    _v: bytes

    def __init__(self) -> None:
        raise TypeError("use ReadWriteToken.parse")

    @classmethod
    def parse(cls, raw: object) -> ReadWriteToken | None:
        """A token from a string that fully matches `[A-Za-z0-9_-]{1,256}`
        (ASCII only), else None."""
        if not isinstance(raw, str) or _TOKEN_RE.fullmatch(raw) is None:
            return None
        data = raw.encode("ascii")
        pad = secrets.token_bytes(len(data))
        tok = object.__new__(cls)
        object.__setattr__(tok, "_pad", pad)
        object.__setattr__(tok, "_v", bytes(a ^ b for a, b in zip(data, pad, strict=True)))
        return tok

    def _raw(self) -> bytes:
        return bytes(a ^ b for a, b in zip(self._v, self._pad, strict=True))

    def reveal(self) -> str:
        return self._raw().decode("ascii")

    def __repr__(self) -> str:
        return _REDACTED

    def __str__(self) -> str:
        return _REDACTED

    def __format__(self, format_spec: str) -> str:
        return _REDACTED

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReadWriteToken):
            return NotImplemented
        return hmac.compare_digest(self._raw(), other._raw())

    def __hash__(self) -> int:
        return hash(self._raw())

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("ReadWriteToken is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("ReadWriteToken is immutable")

    def __copy__(self) -> ReadWriteToken:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> ReadWriteToken:
        return self

    def __getstate__(self) -> NoReturn:
        raise TypeError("ReadWriteToken is not picklable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("ReadWriteToken is not picklable")

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        raise TypeError("ReadWriteToken is not picklable")


@final
@dataclass(frozen=True, slots=True)
class ThreadRef:
    """What delete needs: the thread's uuid and its token."""

    uuid: BackendUuid
    token: ReadWriteToken
