"""pplx models verb: available models + modes via /rest/models/{config,modes}.

Stateless GETs (create no thread). Surfaces the model catalog, the mode catalog
(search / research / agentic_research / study / ...), and the default model per
mode — feeds `--model` / `--mode` validation for the research verb and lets an
agent discover what's selectable.

Shapes (observed 2026-06-22):
  /rest/models/config: {models: {<key>: {label, description, mode, provider}},
                        default_models: {<mode>: <model_key>}, config: [...], ...}
  /rest/models/modes:  {modes: [{id, label, description, subtitle, badge, ...}]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import SchemaError
from ..wire import Client

CONFIG_ENDPOINT = "/rest/models/config"
MODES_ENDPOINT = "/rest/models/modes"


@dataclass
class ModelInfo:
    key: str
    label: str | None
    description: str | None
    mode: str | None
    provider: str | None


@dataclass
class ModeInfo:
    id: str
    label: str | None
    description: str | None


@dataclass
class ModelCard:
    """A row of the UI model *picker* (config array): a base model + its optional
    thinking variant + the tier needed. e.g. label="Claude Opus 4.8",
    base="claude48opus", thinking="claude48opusthinking", tier="max"."""

    label: str
    base: str | None  # non_reasoning_model id
    thinking: str | None  # reasoning_model id (pass this id to use "thinking")
    tier: str | None  # subscription_tier (pro / max)


@dataclass
class ModelsResult:
    models: list[ModelInfo]
    modes: list[ModeInfo]
    default_models: dict[str, str]  # mode -> default model key
    cards: list[ModelCard] = field(default_factory=list)  # the picker (base/thinking/tier)
    warnings: list[str] = field(default_factory=list)


def models(client: Client) -> ModelsResult:
    """Fetch the model + mode catalog. Two stateless GETs, merged."""
    config_raw = client.get_json(CONFIG_ENDPOINT)
    model_infos, defaults = decode_models_config(config_raw)
    cards = decode_model_cards(config_raw)
    mode_infos = decode_modes(client.get_json(MODES_ENDPOINT))
    return ModelsResult(models=model_infos, modes=mode_infos, default_models=defaults, cards=cards)


def decode_model_cards(raw: Any) -> list[ModelCard]:
    """Pure decode of /rest/models/config `config` array → the picker rows."""
    if not isinstance(raw, dict):
        return []
    cards: list[ModelCard] = []
    for e in raw.get("config") or []:
        if not isinstance(e, dict):
            continue
        label = e.get("label")
        if not isinstance(label, str):
            continue
        cards.append(
            ModelCard(
                label=label,
                base=_str_or_none(e.get("non_reasoning_model")),
                thinking=_str_or_none(e.get("reasoning_model")),
                tier=_str_or_none(e.get("subscription_tier")),
            )
        )
    return cards


def decode_models_config(raw: Any) -> tuple[list[ModelInfo], dict[str, str]]:
    """Pure decode of /rest/models/config → (models, default_models)."""
    if not isinstance(raw, dict):
        raise SchemaError(f"unexpected response type from {CONFIG_ENDPOINT}: {type(raw).__name__}")
    models_raw = raw.get("models")
    infos: list[ModelInfo] = []
    if isinstance(models_raw, dict):
        for key, v in sorted(models_raw.items()):
            if not isinstance(v, dict):
                continue
            infos.append(
                ModelInfo(
                    key=key,
                    label=_str_or_none(v.get("label")),
                    description=_str_or_none(v.get("description")),
                    mode=_str_or_none(v.get("mode")),
                    provider=_str_or_none(v.get("provider")),
                )
            )
    defaults_raw = raw.get("default_models")
    defaults: dict[str, str] = {}
    if isinstance(defaults_raw, dict):
        defaults = {str(k): str(v) for k, v in defaults_raw.items() if isinstance(v, str)}
    return infos, defaults


def decode_modes(raw: Any) -> list[ModeInfo]:
    """Pure decode of /rest/models/modes → list[ModeInfo]."""
    if not isinstance(raw, dict):
        raise SchemaError(f"unexpected response type from {MODES_ENDPOINT}: {type(raw).__name__}")
    modes_raw = raw.get("modes")
    if not isinstance(modes_raw, list):
        return []
    out: list[ModeInfo] = []
    for m in modes_raw:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if not isinstance(mid, str):
            continue
        out.append(
            ModeInfo(
                id=mid,
                label=_str_or_none(m.get("label")),
                description=_str_or_none(m.get("description")),
            )
        )
    return out


def _str_or_none(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None
