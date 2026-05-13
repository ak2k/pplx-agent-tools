"""Render layer: pure functions from typed Result objects to text or JSON.

No I/O, no exceptions, fully deterministic — safe to snapshot-test.
"""

from __future__ import annotations

from typing import Any

from . import __version__
from .verbs.search import Hit, SearchResult


def render_search_text(result: SearchResult) -> str:
    """Numbered hit list, three lines per hit (title / URL / one-line snippet).

    Snippets are collapsed to a single line (some sources — Reddit, forum
    posts — have multi-line snippets that would break the format).
    """
    if not result.hits:
        return "(no results)"
    lines: list[str] = []
    for i, hit in enumerate(result.hits, start=1):
        lines.append(f"{i}. {hit.title}")
        lines.append(f"   {hit.url}")
        if hit.snippet:
            snippet = " ".join(hit.snippet.split())
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines[:-1]) if lines else ""


def render_search_json(result: SearchResult) -> dict[str, Any]:
    """Pass-through-ish shape: { hits, total, warnings?, _pplx_tools_version }."""
    out: dict[str, Any] = {
        "_pplx_tools_version": __version__,
        "query": result.query,
        "type": result.type,
        "hits": [_hit_to_json(h) for h in result.hits],
        "total": result.total,
    }
    if result.warnings:
        out["warnings"] = list(result.warnings)
    return out


def _hit_to_json(hit: Hit) -> dict[str, Any]:
    out: dict[str, Any] = {
        "url": hit.url,
        "title": hit.title,
    }
    if hit.domain is not None:
        out["domain"] = hit.domain
    if hit.snippet is not None:
        out["snippet"] = hit.snippet
    if hit.summary is not None:
        out["summary"] = hit.summary
    if hit.published_date is not None:
        out["published_date"] = hit.published_date
    if hit.images:
        out["images"] = list(hit.images)
    return out
