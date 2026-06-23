"""Render tests for the quota + models verbs (text + JSON envelope)."""

from __future__ import annotations

from pplx_agent_tools.render import (
    render_models_json,
    render_models_text,
    render_quota_json,
    render_quota_text,
)
from pplx_agent_tools.verbs.models import ModeInfo, ModelCard, ModelInfo, ModelsResult
from pplx_agent_tools.verbs.quota import QuotaItem, QuotaResult

QUOTA = QuotaResult(
    free_queries=QuotaItem("free_queries", True, None),
    modes=[QuotaItem("research", True, None), QuotaItem("pro_search", True, None)],
    sources=[
        QuotaItem("bmj", True, None),
        QuotaItem("box", False, 0),
    ],
)

MODELS = ModelsResult(
    models=[ModelInfo("turbo", "Best", "Adapts", "search", "PERPLEXITY")],
    modes=[ModeInfo("research", "Research", "deep")],
    default_models={"search": "pplx_pro", "research": "pplx_alpha"},
    cards=[ModelCard("Claude Opus 4.8", "claude48opus", "claude48opusthinking", "max")],
)


def test_quota_text_shows_modes_and_notable_sources() -> None:
    out = render_quota_text(QUOTA)
    assert "research" in out
    assert "available" in out
    assert "1/2 available" in out  # sources summary
    assert "box" in out and "EXHAUSTED" in out and "0 remaining" in out
    assert "bmj" not in out  # available + unmetered → not spelled out


def test_quota_json_envelope() -> None:
    j = render_quota_json(QUOTA)
    assert j["_verb"] == "quota"
    assert "_pplx_tools_version" in j
    assert j["free_queries"] == {"name": "free_queries", "available": True}
    box = next(s for s in j["sources"] if s["name"] == "box")
    assert box["remaining"] == 0


def test_models_text() -> None:
    out = render_models_text(MODELS)
    assert "modes:" in out
    assert "turbo" in out and "[search]" in out
    assert "research" in out and "pplx_alpha" in out


def test_models_text_picker() -> None:
    out = render_models_text(MODELS)
    assert "model picker" in out
    assert "claude48opusthinking" in out and "[max]" in out


def test_models_json_envelope() -> None:
    j = render_models_json(MODELS)
    assert j["_verb"] == "models"
    assert j["models"][0]["key"] == "turbo"
    assert j["models"][0]["mode"] == "search"
    assert j["default_models"]["research"] == "pplx_alpha"
    assert j["picker"][0] == {
        "label": "Claude Opus 4.8",
        "base": "claude48opus",
        "thinking": "claude48opusthinking",
        "tier": "max",
    }


def test_empty_renders_are_safe() -> None:
    assert render_quota_text(QuotaResult(None, [], [])) == "(no quota data)"
    assert render_models_text(ModelsResult([], [], {})) == "(no model data)"
