"""Render layer: pure functions from typed Result objects to text or JSON.

No I/O, no exceptions, fully deterministic — safe to snapshot-test.

This module is the single rendering registry for the CLI: every verb's
Result type has a `render_<verb>_text` / `render_<verb>_json` pair here,
and `cli_<verb>.py` imports them by name. Concentrating them in one file
trades fan-in coupling (this module imports from all verb modules) for
two upsides:

  1. Browsing the formatting decisions across verbs is a single-file read.
  2. Cross-verb consistency (timestamp formatting, truncation markers,
     version envelopes in JSON) is easy to enforce.

Adding a new verb means adding a new pair here — see the new-verb
checklist in `verbs/__init__.py` for the full file-edit list.
"""

from __future__ import annotations

from typing import Any

from . import __version__
from .verbs.fetch import FetchResult
from .verbs.search import Hit, SearchResult
from .verbs.snippets import SnippetsResult

# Reserved keys the envelope owns; payloads that try to set these are
# refused loudly so contract violations don't slip through silently.
_ENVELOPE_RESERVED_KEYS: tuple[str, ...] = ("_pplx_tools_version", "_verb", "warnings")


def envelope(
    verb: str,
    payload: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Wrap a verb's JSON payload with the agent-contract envelope.

    Every CLI verb's `--json` output goes through this. Guarantees:

    - `_pplx_tools_version` stamp is present (so agents can detect schema
      changes against their pinned tool version)
    - `_verb` discriminator is present (so a generic JSON consumer can
      branch without inspecting payload keys)
    - `warnings` lives under one consistent key when non-empty

    Reserved keys in `payload` are rejected to keep the envelope's
    invariants intact — a verb that wants to surface a `warnings` field
    must pass it through the `warnings` parameter, not the payload.
    """
    collisions = [k for k in _ENVELOPE_RESERVED_KEYS if k in payload]
    if collisions:
        raise ValueError(f"envelope payload cannot contain reserved keys: {collisions}")
    out: dict[str, Any] = {
        "_pplx_tools_version": __version__,
        "_verb": verb,
        **payload,
    }
    if warnings:
        out["warnings"] = list(warnings)
    return out


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
    """Pass-through-ish shape: envelope + { query, hits, total }."""
    return envelope(
        "search",
        {
            "query": result.query,
            "hits": [_hit_to_json(h) for h in result.hits],
            "total": result.total,
        },
        warnings=result.warnings,
    )


def render_snippets_text(result: SnippetsResult) -> str:
    """Per-URL block, header followed by relevance-ranked excerpts."""
    if not result.results:
        return "(no results)"
    parts: list[str] = []
    for ur in result.results:
        parts.append(f"# {ur.url}")
        if ur.error:
            parts.append(f"  error: {ur.error}")
        elif not ur.snippets:
            parts.append("  (no relevant snippets)")
        else:
            for s in ur.snippets:
                parts.append("")
                parts.append(s.text)
        parts.append("")
    return "\n".join(parts).rstrip()


def render_snippets_json(result: SnippetsResult) -> dict[str, Any]:
    return envelope(
        "snippets",
        {
            "query": result.query,
            "results": [
                {
                    "url": ur.url,
                    **({"error": ur.error} if ur.error else {}),
                    "snippets": [
                        {"text": s.text, "score": round(s.score, 5), "tokens": s.tokens}
                        for s in ur.snippets
                    ],
                }
                for ur in result.results
            ],
        },
        warnings=result.warnings,
    )


def render_fetch_text(result: FetchResult) -> str:
    """Header (title / URL / domain / extracted flag) followed by content."""
    header_lines: list[str] = []
    if result.title:
        header_lines.append(f"# {result.title}")
    header_lines.append(result.url)
    extra: list[str] = [f"domain: {result.domain}"]
    if result.published_date:
        extra.append(f"date: {result.published_date}")
    if result.is_extracted:
        extra.append("extracted: yes (LLM)")
    if not result.stream_complete:
        # Surfaced on the header line so a human eyeballing stdout doesn't
        # mistake a deadline-clipped partial answer for a complete one.
        # `cli_fetch` also emits a stderr warning for machine-parseable runs.
        extra.append("stream: incomplete (deadline or cut)")
    header_lines.append(" · ".join(extra))
    return "\n".join(header_lines) + "\n\n" + result.content


def render_fetch_json(result: FetchResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": result.url,
        "domain": result.domain,
        "is_extracted": result.is_extracted,
        "truncated": result.truncated,
        "stream_complete": result.stream_complete,
        "content": result.content,
    }
    if result.title is not None:
        payload["title"] = result.title
    if result.published_date is not None:
        payload["published_date"] = result.published_date
    return envelope("fetch", payload)


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
