"""Unit tests for auth.py — cookie loading, shape normalization, perms enforcement."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from pplx_agent_tools.auth import (
    DEFAULT_PROFILE,
    SUPPORTED_BROWSERS,
    _normalize,
    default_cookies_path,
    load_cookies,
    resolve_profile,
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


def test_normalize_flat_dict_coerces_values_to_str() -> None:
    out = _normalize({"x": 42}, source="t")
    assert out == {"x": "42"}


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
