"""Unit tests for verbs/snippets.py pure helpers."""

from __future__ import annotations

import struct

from pplx_agent_tools.verbs.snippets import (
    _fts5_escape,
    _split_paragraphs,
    _vec_to_blob,
)


def test_split_paragraphs_basic() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    assert _split_paragraphs(text) == ["First paragraph.", "Second paragraph.", "Third."]


def test_split_paragraphs_trims_whitespace() -> None:
    text = "\n\n  first\n\n  second  \n\n"
    assert _split_paragraphs(text) == ["first", "second"]


def test_split_paragraphs_collapses_multi_blank_lines() -> None:
    text = "one\n\n\n\n\ntwo"
    assert _split_paragraphs(text) == ["one", "two"]


def test_split_paragraphs_keeps_multiline_paragraphs_together() -> None:
    # A paragraph with single newlines (no blank line) stays one block
    text = "line a\nline b\nline c\n\nnext paragraph"
    blocks = _split_paragraphs(text)
    assert len(blocks) == 2
    assert "line a" in blocks[0] and "line c" in blocks[0]


def test_split_paragraphs_empty_input() -> None:
    assert _split_paragraphs("") == []
    assert _split_paragraphs("   \n\n  \n") == []


def test_fts5_escape_basic_terms() -> None:
    out = _fts5_escape("hello world")
    assert out == '"hello" OR "world"'


def test_fts5_escape_strips_special_chars() -> None:
    # Colons, quotes, parens would break FTS5 syntax — escape strips them
    out = _fts5_escape('hello: "world" (test)*')
    assert ":" not in out and "(" not in out
    assert "hello" in out and "world" in out and "test" in out


def test_fts5_escape_empty_input_safe() -> None:
    assert _fts5_escape("") == '""'
    assert _fts5_escape("!!!@@@###") == '""'


def test_fts5_escape_preserves_apostrophes_and_hyphens() -> None:
    out = _fts5_escape("don't anti-bot")
    assert "don't" in out
    assert "anti-bot" in out


def test_vec_to_blob_roundtrip() -> None:
    # 384-d MiniLM-style vector
    vec = [0.1, -0.2, 0.3, 0.0, 1.5]
    blob = _vec_to_blob(vec)
    assert isinstance(blob, bytes)
    # Each float32 is 4 bytes
    assert len(blob) == len(vec) * 4
    # Roundtrip: unpack should match within float32 precision
    unpacked = struct.unpack(f"<{len(vec)}f", blob)
    for orig, got in zip(vec, unpacked, strict=True):
        assert abs(orig - got) < 1e-6


def test_vec_to_blob_little_endian() -> None:
    # sqlite-vec requires little-endian; the format string starts with "<"
    vec = [1.0]
    blob = _vec_to_blob(vec)
    # 1.0 little-endian float32 = 00 00 80 3F
    assert blob == b"\x00\x00\x80\x3f"
