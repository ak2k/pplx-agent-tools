"""Typed view of decoded JSON: the trust root for every wire response.

`JsonValue` is what a JSON parser can produce. Decoders accept `object` and
narrow only through the accessors below, so a field typed `str | None` can
only ever hold a `str` or `None`, whatever the server sends.
"""

from __future__ import annotations

from typing import TypeAlias, cast

JsonValue: TypeAlias = "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]"


def from_parser(value: object) -> JsonValue:
    """The one place an untyped parser result becomes `JsonValue`. Sound only
    for the output of a JSON parser, which is the only caller."""
    return cast("JsonValue", value)


def as_object(value: object) -> dict[str, object] | None:
    """`value` as a JSON object, or None. Non-str keys (impossible from a JSON
    parser, possible from a hand-built dict) are dropped rather than trusted."""
    if not isinstance(value, dict):
        return None
    items = cast("dict[object, object]", value).items()
    return {k: v for k, v in items if isinstance(k, str)}


def as_array(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return cast("list[object]", value)


def str_or_none(value: object) -> str | None:
    """A non-empty string, else None: an empty string carries no data here."""
    return value if isinstance(value, str) and value else None


def int_or_none(value: object) -> int | None:
    """An int, else None. bool is excluded: JSON `true` is not a count."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None
