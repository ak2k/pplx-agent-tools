"""Unit tests for auth.py — cookie loading, shape normalization, perms enforcement."""

from __future__ import annotations

import json
import stat
import sys
import types
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pplx_agent_tools.auth import (
    DEFAULT_PROFILE,
    SUPPORTED_BROWSERS,
    _normalize,
    default_cookies_path,
    import_from_browser,
    load_cookies,
    resolve_profile,
    save_cookies,
)
from pplx_agent_tools.errors import AuthError

# ---------- profile resolution ----------


def test_resolve_profile_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PPLX_PROFILE", raising=False)
    assert resolve_profile() == DEFAULT_PROFILE == "default"


def test_resolve_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPLX_PROFILE", "work")
    assert resolve_profile() == "work"


def test_resolve_profile_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPLX_PROFILE", "work")
    assert resolve_profile("personal") == "personal"


def test_default_cookies_path_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("PPLX_PROFILE", raising=False)
    p = default_cookies_path()
    assert p == tmp_path / "perplexity" / "default" / "cookies.json"


def test_default_cookies_path_per_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = default_cookies_path("kanerai")
    assert p.parent.name == "kanerai"


# ---------- shape normalization ----------


def test_normalize_flat_dict() -> None:
    out = _normalize({"a": "1", "b": "2"}, source="t")
    assert out == {"a": "1", "b": "2"}


def test_normalize_allows_empty_value() -> None:
    assert _normalize({"x": ""}, source="t") == {"x": ""}


def test_normalize_cookie_editor_array() -> None:
    raw = [
        {"name": "a", "value": "1", "domain": ".x.com"},
        {"name": "b", "value": "2", "domain": ".x.com"},
    ]
    assert _normalize(raw, source="t") == {"a": "1", "b": "2"}


def test_normalize_array_missing_name_raises() -> None:
    with pytest.raises(AuthError) as ei:
        _normalize([{"value": "v"}], source="t")
    assert "missing 'name' or 'value'" in str(ei.value)


def test_normalize_array_missing_value_raises() -> None:
    with pytest.raises(AuthError):
        _normalize([{"name": "x"}], source="t")


def test_normalize_array_non_object_entry_raises() -> None:
    with pytest.raises(AuthError) as ei:
        _normalize(["just a string"], source="t")
    assert "not an object" in str(ei.value)


def test_normalize_rejects_non_dict_non_list() -> None:
    with pytest.raises(AuthError) as ei:
        _normalize("nope", source="t")
    assert "must be an object or array" in str(ei.value)


def test_normalize_empty_raises() -> None:
    with pytest.raises(AuthError) as ei:
        _normalize({}, source="t")
    assert "no cookies parsed" in str(ei.value)


def test_normalize_source_label_in_error() -> None:
    # Cookie values must never appear in errors — only the source label
    with pytest.raises(AuthError) as ei:
        _normalize("oops", source="$PPLX_COOKIES")
    assert "$PPLX_COOKIES" in str(ei.value)


# ---------- load_cookies: env precedence ----------


def test_load_cookies_inline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PPLX_COOKIES_PATH", raising=False)
    monkeypatch.setenv("PPLX_COOKIES", '{"foo": "bar"}')
    assert load_cookies() == {"foo": "bar"}


def test_load_cookies_inline_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PPLX_COOKIES_PATH", raising=False)
    monkeypatch.setenv("PPLX_COOKIES", "not json")
    with pytest.raises(AuthError) as ei:
        load_cookies()
    assert "$PPLX_COOKIES is not valid JSON" in str(ei.value)


def test_load_cookies_path_takes_precedence_over_inline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"from_file": "1"}))
    p.chmod(0o600)
    monkeypatch.setenv("PPLX_COOKIES_PATH", str(p))
    monkeypatch.setenv("PPLX_COOKIES", '{"from_inline": "1"}')
    assert load_cookies() == {"from_file": "1"}


def test_load_cookies_xdg_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("PPLX_COOKIES_PATH", raising=False)
    monkeypatch.delenv("PPLX_COOKIES", raising=False)
    cookies_dir = tmp_path / "perplexity" / "default"
    cookies_dir.mkdir(parents=True)
    p = cookies_dir / "cookies.json"
    p.write_text(json.dumps({"session": "x"}))
    p.chmod(0o600)
    assert load_cookies() == {"session": "x"}


def test_load_cookies_xdg_default_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("PPLX_COOKIES_PATH", raising=False)
    monkeypatch.delenv("PPLX_COOKIES", raising=False)
    with pytest.raises(AuthError) as ei:
        load_cookies()
    assert "pplx-auth import" in str(ei.value) or "pplx auth import" in str(ei.value)


# ---------- perms enforcement ----------


def _write_cookie_file(path: Path, mode: int) -> None:
    path.write_text(json.dumps({"a": "1"}))
    path.chmod(mode)


def test_load_cookies_world_readable_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = tmp_path / "cookies.json"
    _write_cookie_file(p, 0o644)
    monkeypatch.setenv("PPLX_COOKIES_PATH", str(p))
    with pytest.raises(AuthError) as ei:
        load_cookies()
    assert "world-accessible" in str(ei.value)


def test_load_cookies_group_readable_auto_chmod(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    p = tmp_path / "cookies.json"
    _write_cookie_file(p, 0o640)
    monkeypatch.setenv("PPLX_COOKIES_PATH", str(p))
    out = load_cookies()  # should succeed, with a stderr warning
    assert out == {"a": "1"}
    perms_after = stat.S_IMODE(p.stat().st_mode)
    assert perms_after == 0o600
    err = capsys.readouterr().err
    assert "group-accessible" in err
    assert "chmodding to 0600" in err


def test_load_cookies_perfect_perms_no_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    p = tmp_path / "cookies.json"
    _write_cookie_file(p, 0o600)
    monkeypatch.setenv("PPLX_COOKIES_PATH", str(p))
    load_cookies()
    assert capsys.readouterr().err == ""


# ---------- supported_browsers exposed ----------


def test_supported_browsers_includes_common() -> None:
    # These are the browsers explicitly documented in the SKILL.md / plan
    for name in ("brave", "chrome", "firefox", "safari", "edge"):
        assert name in SUPPORTED_BROWSERS


# ---------- defensive: missing file ----------


def test_load_cookies_path_nonexistent_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PPLX_COOKIES_PATH", "/no/such/path.json")
    monkeypatch.delenv("PPLX_COOKIES", raising=False)
    with pytest.raises(AuthError) as ei:
        load_cookies()
    assert "does not exist" in str(ei.value)


# ---------- save_cookies (rotation persistence) ----------


def test_save_cookies_writes_with_0600_perms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    dest = save_cookies({"a": "1", "b": "2"})
    assert dest.exists()
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600
    assert json.loads(dest.read_text()) == {"a": "1", "b": "2"}


def test_save_cookies_per_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    dest = save_cookies({"x": "1"}, profile="work")
    assert "work" in str(dest)
    assert json.loads(dest.read_text()) == {"x": "1"}


def test_save_cookies_creates_parent_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # No pre-existing perplexity/<profile>/ directory
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "fresh"))
    dest = save_cookies({"x": "1"})
    assert dest.parent.is_dir()


def test_save_cookies_atomic_replace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # First write
    dest = save_cookies({"v": "old"})
    inode_a = dest.stat().st_ino
    # Overwrite with new content
    dest2 = save_cookies({"v": "new"})
    assert dest == dest2
    assert json.loads(dest.read_text()) == {"v": "new"}
    # The .tmp file should not exist after rename
    assert not dest.with_name(dest.name + ".tmp").exists()
    # Inode should change (atomic rename creates new inode)
    inode_b = dest.stat().st_ino
    assert inode_a != inode_b


def test_save_cookies_then_load_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("PPLX_COOKIES_PATH", raising=False)
    monkeypatch.delenv("PPLX_COOKIES", raising=False)
    save_cookies({"session-token": "abc123", "csrf": "xyz"})
    assert load_cookies() == {"session-token": "abc123", "csrf": "xyz"}


# ---------- cookie-pair boundary (property) ----------

# Text biased toward the characters that split or inject a Cookie header.
# One sample from every class the gate refuses, plus allowed near-misses
# (space, non-ASCII, C1 control) so acceptance is exercised too.
_hostile_text = st.text(
    alphabet=st.sampled_from(
        [
            "a",
            "=",
            ";",
            "\r",
            "\n",
            "\x00",
            "\x01",
            "\t",
            "\x1f",
            "\x7f",
            "\ud800",
            "\udfff",
            " ",
            "é",
            "\x85",
            '"',
            ",",
        ]
    )
)
_safe_text = st.text(alphabet=st.sampled_from(["a", "Z", "0", " ", "é", "\x85", '"', ",", "-"]))
_bad_chars = ["=", ";", "\r", "\n", "\x00", "\x01", "\t", "\x1f", "\x7f", "\ud800", "\udfff"]
# Clean text with exactly one bad character, so each class is tested alone
# instead of hiding behind another one the gate already rejects.
_one_bad = st.builds(lambda a, c, b: a + c + b, _safe_text, st.sampled_from(_bad_chars), _safe_text)
_json_scalar = st.one_of(
    _one_bad,
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(max_size=10),
    _hostile_text,
)
_json_any = st.recursive(
    _json_scalar,
    lambda kids: st.one_of(
        st.lists(kids, max_size=3), st.dictionaries(st.text(max_size=5), kids, max_size=3)
    ),
    max_leaves=8,
)
_cookie_key = st.one_of(st.text(max_size=8), _hostile_text, _one_bad)
_dict_shape = st.dictionaries(_cookie_key, _json_any, max_size=5)
_entry_shape = st.one_of(
    _json_any,
    st.fixed_dictionaries({"name": st.one_of(_cookie_key, _json_any), "value": _json_any}),
)
_list_shape = st.lists(_entry_shape, max_size=5)


def _sendable(s: str) -> bool:
    # Stated independently of auth._cookie_text_ok so the test pins the rule.
    return not any(ord(c) < 0x20 or c in {"\x7f", ";"} or 0xD800 <= ord(c) <= 0xDFFF for c in s)


def _assert_clean(out: dict[str, str]) -> None:
    assert out
    for name, value in out.items():
        assert type(name) is str and type(value) is str
        assert name and "=" not in name
        assert _sendable(name) and _sendable(value)


@given(st.one_of(_dict_shape, _list_shape, _json_any))
def test_normalize_yields_clean_pairs_or_auth_error(payload: object) -> None:
    try:
        out = _normalize(payload, source="prop")
    except AuthError:
        return
    _assert_clean(out)


@given(st.one_of(_safe_text, _one_bad), st.one_of(_safe_text, _one_bad))
def test_single_pair_accepted_iff_sendable(name: str, value: str) -> None:
    expect_ok = bool(name) and "=" not in name and _sendable(name) and _sendable(value)
    try:
        out = _normalize({name: value}, source="prop")
    except AuthError:
        assert not expect_ok
        return
    assert expect_ok
    assert out == {name: value}


@pytest.mark.parametrize(
    "payload",
    [
        {"a": None},
        {"a": {"x": 1}},
        {"a": 1},
        {"a": "v\r\nX-Injected: 1"},
        {"a": "v\x00"},
        {"a": "v\x01"},
        {"a": "v\tw"},
        {"a": "v\x7f"},
        {"a": "\ud800"},
        {"\udc00": "v"},
        {"a=b": "v"},
        {"a\tb": "v"},
        {"a": "v;b=c"},
        {"c\r\nX": "v"},
        {"": "v"},
        [{"name": "a", "value": None}],
        [{"name": "a", "value": 1}],
        [{"name": "a;b", "value": "v"}],
    ],
)
def test_normalize_rejects_non_string_or_header_breaking(payload: object) -> None:
    with pytest.raises(AuthError):
        _normalize(payload, source="t")


@pytest.mark.parametrize("value", ["", "a b", "é", "\x85", '"x, y"', "x\\y"])
def test_normalize_accepts_sendable_values(value: str) -> None:
    assert _normalize({"a": value}, source="t") == {"a": value}


def test_normalize_error_omits_cookie_value() -> None:
    with pytest.raises(AuthError) as ei:
        _normalize({"a": "secret;x"}, source="t")
    assert "secret" not in str(ei.value)


# ---------- import_from_browser: rows pass the same gate ----------


def _fake_rookiepy(monkeypatch: pytest.MonkeyPatch, rows: object) -> None:
    mod = types.ModuleType("rookiepy")
    mod.brave = lambda _domains: rows  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "rookiepy", mod)


def test_import_from_browser_saves_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _fake_rookiepy(monkeypatch, [{"name": "a", "value": "1", "domain": ".perplexity.ai"}])
    dest = import_from_browser("brave")
    assert json.loads(dest.read_text()) == {"a": "1"}


def test_import_from_browser_skips_bad_rows_with_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    rows = [
        {"name": "session", "value": "good"},
        {"name": "nullish", "value": None},
        {"name": "crlf", "value": "SECRET1\r\nX: y"},
        {"name": "tabbed", "value": "SECRET2\tz"},
        {"name": "surr", "value": "SECRET3\ud800"},
        {"name": "a;b", "value": "SECRET4"},
        "not a row",
    ]
    _fake_rookiepy(monkeypatch, rows)
    dest = import_from_browser("brave")
    assert json.loads(dest.read_text()) == {"session": "good"}
    err = capsys.readouterr().err
    assert err.count("warning: skipping") == 6
    for name in ("nullish", "crlf", "tabbed", "surr"):
        assert repr(name) in err
    assert "has no value" in err
    assert "SECRET" not in err


def test_import_from_browser_all_rows_bad_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _fake_rookiepy(monkeypatch, [{"name": "a", "value": "x\r\ny"}])
    with pytest.raises(AuthError, match="sign in"):
        import_from_browser("brave")
    assert not default_cookies_path().exists()


def test_import_from_browser_empty_says_sign_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _fake_rookiepy(monkeypatch, [])
    with pytest.raises(AuthError, match="sign in"):
        import_from_browser("brave")
