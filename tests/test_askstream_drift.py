"""Drift names and the drift ledger (plan §2.8; U3 oracle 9; I13 on names)."""

from __future__ import annotations

import ast
import re
import uuid
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from pplx_agent_tools.askstream.drift import MAX_KEYS, MAX_NAME, Drift, DriftLedger, name_of

PKG = Path(__file__).resolve().parent.parent / "pplx_agent_tools"
# A name is safe characters and id markers; the cut at 64 may end mid-marker.
SAFE = re.compile(r"(?:[A-Za-z0-9_.:/-]|<id-like:[0-9]+>)*(?:<[a-z:0-9-]*)?")
ID_RUN = re.compile(r"[A-Za-z0-9_-]{16,}")


def _no_id_left(name: str) -> bool:
    return not any(any(c.isdigit() for c in m.group(0)) for m in ID_RUN.finditer(name))


@given(st.text(), st.uuids(), st.text(), st.booleans())
def test_name_of_hides_uuids(before: str, u: uuid.UUID, after: str, upper: bool) -> None:
    raw = str(u).upper() if upper else str(u)
    name = name_of(before + raw + after)
    assert raw not in name
    assert raw.replace("-", "") not in name


@given(
    st.text(max_size=40),
    st.from_regex(r"[A-Za-z0-9_-]*[0-9][A-Za-z0-9_-]*", fullmatch=True).filter(
        lambda s: len(s) >= 16
    ),
    st.text(max_size=40),
)
def test_name_of_hides_hex_and_base64url_runs(before: str, run: str, after: str) -> None:
    name = name_of(f"{before}.{run}.{after}")
    assert run not in name
    assert "<id-like:" in name


@given(
    st.from_regex(r"[0-9a-f]{16,64}", fullmatch=True).filter(lambda s: any(c.isdigit() for c in s))
)
def test_name_of_hides_hex_ids(hexid: str) -> None:
    assert name_of(f"key_{hexid}") == f"<id-like:{len(hexid) + 4}>"


@given(st.one_of(st.text(), st.binary().map(lambda b: b.decode("utf-8", "surrogateescape"))))
def test_name_of_output_is_safe_and_bounded(raw: str) -> None:
    name = name_of(raw)
    assert len(name) <= MAX_NAME
    assert SAFE.fullmatch(name), name
    assert _no_id_left(name)


def test_name_of_scrubs_nonprintable_and_non_ascii() -> None:
    assert name_of("a\x00b\nc\u2028d\ud800e") == "a_b_c_d_e"
    assert name_of("ключ") == "____"
    assert name_of("x" * 200) == "x" * MAX_NAME


def test_name_of_keeps_plain_names() -> None:
    for plain in (
        "unknown_key",
        "markdown_block",
        "ask_text",
        "thread_url_slug",
        "no_usage",
        "depth",
    ):
        assert name_of(plain) == plain


def test_name_of_hides_digit_runs_even_when_word_like() -> None:
    # The plan's rule is strict: readability of such names is traded for
    # never showing a token made of words and numbers.
    assert name_of("ask_text_0_markdown") == "<id-like:19>"
    assert "4155550123" not in name_of("please_call_4155550123")


def test_name_of_non_string_names_its_type() -> None:
    assert name_of(12345678901234567890) == "int"
    assert name_of(None) == "NoneType"


def test_ledger_is_bounded() -> None:
    ledger = DriftLedger()
    ledger.add(Drift("unknown_envelope_key", name_of(f"k{i}")) for i in range(1000))
    ledger.add([Drift("unknown_envelope_key", name_of("k0"))])
    assert len(ledger.items()) == MAX_KEYS
    assert ledger.overflow == 1000 - MAX_KEYS
    assert ledger.total == 1001
    assert dict(ledger.items())[Drift("unknown_envelope_key", name_of("k0"))] == 2


def test_drift_name_is_built_only_by_name_of() -> None:
    """`DriftName(...)` outside drift.py would bypass the scrubbing."""
    offenders: list[str] = []
    for path in PKG.rglob("*.py"):
        if path.name == "drift.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "DriftName"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
