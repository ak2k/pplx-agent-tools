"""Small hand-built ask frames for the block store and projection tests."""

from __future__ import annotations

import json
from typing import Any

from pplx_agent_tools.askstream.frames import AskFrame, decode_frame


def frame(
    *blocks: dict[str, Any],
    status: str = "PENDING",
    text_completed: bool = False,
    text: str | None = None,
) -> AskFrame:
    payload: dict[str, Any] = {
        "status": status,
        "text_completed": text_completed,
        "blocks": list(blocks),
    }
    if text is not None:
        payload["text"] = text
    f, drift = decode_frame(json.dumps(payload))
    assert isinstance(f, AskFrame), f
    assert drift == (), drift
    return f


def terminal(*blocks: dict[str, Any], text: str | None = None) -> AskFrame:
    return frame(*blocks, status="COMPLETED", text_completed=True, text=text)


def snap(usage: str, field: str, value: dict[str, Any]) -> dict[str, Any]:
    return {"intended_usage": usage, field: value}


def diff(usage: str, field: str, *ops: dict[str, Any]) -> dict[str, Any]:
    return {"intended_usage": usage, "diff_block": {"field": field, "patches": list(ops)}}


def add(path: str, value: Any) -> dict[str, Any]:
    return {"op": "add", "path": path, "value": value}


def replace(path: str, value: Any) -> dict[str, Any]:
    return {"op": "replace", "path": path, "value": value}


def remove(path: str) -> dict[str, Any]:
    return {"op": "remove", "path": path}


def md(chunks: list[str], offset: int | None = None, **extra: Any) -> dict[str, Any]:
    """An `ask_text` markdown snapshot block."""
    value: dict[str, Any] = {"chunks": chunks, **extra}
    if offset is not None:
        value["chunk_starting_offset"] = offset
    return snap("ask_text", "markdown_block", value)


def web(*urls: str, **extra: Any) -> dict[str, Any]:
    """A `web_results` snapshot block."""
    rows = [{"name": f"t{i}", "snippet": f"s{i}", "url": u} for i, u in enumerate(urls)]
    return snap("web_results", "web_result_block", {"web_results": rows, **extra})


def workflow_text(*items: tuple[list[str], str | None]) -> dict[str, Any]:
    """A workflow document with one step per text item (chunks, text)."""
    steps: list[dict[str, Any]] = []
    for chunks, text in items:
        tp: dict[str, Any] = {"chunks": chunks}
        if text is not None:
            tp["text"] = text
        steps.append({"items": [{"type": "WORKFLOW_ITEM_TEXT", "payload": {"text_payload": tp}}]})
    return {"steps": steps}


def report(body: str) -> dict[str, Any]:
    """A `unified_assets` document holding one report body."""
    return {"assets": [{"research_report": {"source_content": body}}]}
