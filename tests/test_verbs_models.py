"""Unit tests for verbs/models.py — decode of /rest/models/{config,modes}."""

from __future__ import annotations

import pytest

from pplx_agent_tools.errors import SchemaError
from pplx_agent_tools.verbs.models import decode_models_config, decode_modes

CONFIG = {
    "models": {
        "turbo": {
            "label": "Best",
            "description": "Adapts to each query",
            "mode": "search",
            "provider": "PERPLEXITY",
        },
        "gpt5": {"label": "GPT-5", "provider": "OPENAI"},
        "bad": "not a dict",
    },
    "default_models": {"search": "pplx_pro", "research": "pplx_alpha", "junk": 5},
}

MODES = {
    "modes": [
        {"id": "search", "label": "Search", "description": "fast answers"},
        {"id": "research", "label": "Research", "description": None},
        {"label": "no id — skipped"},
        "not a dict",
    ],
    "debug": None,
}


def test_config_models() -> None:
    infos, _ = decode_models_config(CONFIG)
    keys = [m.key for m in infos]
    assert keys == ["gpt5", "turbo"]  # sorted, "bad" skipped
    turbo = next(m for m in infos if m.key == "turbo")
    assert turbo.label == "Best"
    assert turbo.mode == "search"
    assert turbo.provider == "PERPLEXITY"
    gpt5 = next(m for m in infos if m.key == "gpt5")
    assert gpt5.description is None  # absent → None


def test_config_defaults_filters_non_str() -> None:
    _, defaults = decode_models_config(CONFIG)
    assert defaults == {"search": "pplx_pro", "research": "pplx_alpha"}  # "junk": 5 dropped


def test_config_non_dict_raises() -> None:
    with pytest.raises(SchemaError):
        decode_models_config([])


def test_modes_decode() -> None:
    modes = decode_modes(MODES)
    assert [m.id for m in modes] == ["search", "research"]  # id-less + non-dict skipped
    assert modes[0].label == "Search"
    assert modes[1].description is None


def test_modes_non_dict_raises() -> None:
    with pytest.raises(SchemaError):
        decode_modes("nope")


def test_modes_missing_list_tolerated() -> None:
    assert decode_modes({"debug": None}) == []
