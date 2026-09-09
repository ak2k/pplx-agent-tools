"""pplx research verb: Perplexity deep research via /rest/sse/perplexity_ask.

The flagship differentiated capability — multi-step, cited research, far beyond
`search`'s ranked hits. Same endpoint as `fetch --prompt`, but the deep behaviour
is selected by `model_preference` (NOT `params.mode` — see `_MODE_MODEL` and
docs/wire/perplexity-ask-research.md), with two important differences:

  1. Session-creating. Research is an ask-family verb, so it creates a thread.
     We send `is_incognito: true` (the thread never enters the user's history;
     verified) and still issue a best-effort `delete_thread` as a secondary
     guard. See CLAUDE.md → "Endpoint selection principle".
  2. Schematized response. Unlike copilot mode's incremental `markdown_block`
     chunks, research streams full-snapshot frames whose `text` field is a JSON
     list of `{step_type, content, uuid}` blocks. The FINAL block's
     `content.answer` holds only a short *cover note* ("I've compiled…, the full
     report includes…"); the report itself is a RESEARCH_ANSWER block asset
     (`assets[].research_report.source_content`), so the answer we return is the
     cover note followed by that body. Sources accumulate across SEARCH_RESULTS
     blocks' `content.web_results`. We keep the latest snapshot and decode it once.

Deep research takes ~90-120s (multi-round, ~40+ sources); the verb supports the
same `--timeout` → partial-result (exit 6) contract as `fetch --prompt`, with
bounded 429 retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..errors import SchemaError, StreamDeadlineError
from ..wire import Client
from ._ask_common import Source, base_ask_params, run_ask_stream, to_source

ENDPOINT = "/rest/sse/perplexity_ask"
DEFAULT_MODE = "research"

# Verified 2026-06-22: Perplexity selects deep-research behaviour by
# `model_preference`, NOT by `params.mode`. Sending params.mode="research" with
# model_preference="turbo" yields a plain copilot answer (1 search round); it's
# model_preference="pplx_alpha" that triggers real Deep Research (LOAD_SKILL +
# multiple SEARCH_WEB rounds + THOUGHT steps + far more sources). So we map the
# user-facing --mode to the model id that actually drives it, and keep
# params.mode coarse ("copilot"). Model ids here are the stable internal
# constants (ALPHA / AGENTIC_RESEARCH); the per-mode default_models can drift,
# but these identifiers have held across builds.
_RESEARCH_MODEL = "pplx_alpha"  # Deep Research: multi-round + reasoning
_COUNCIL_MODEL = "pplx_agentic_research"  # Model Council: multi-model cross-check
_MODE_MODEL = {
    "research": _RESEARCH_MODEL,
    "agentic_research": _COUNCIL_MODEL,
    "council": _COUNCIL_MODEL,  # friendly alias
}

# Default Model Council trio (mirrors /rest/models/config
# `agentic_research_compare_models`; can drift across builds — override with
# --council-models). Verified 2026-06-23: council STALLS forever unless
# `compare_model_preferences` is set (the web always sends it); with the trio it
# completes in ~80s and returns a FINAL block in the usual shape.
_DEFAULT_COUNCIL_MODELS = ["gpt55_thinking", "claude48opusthinking", "gemini31pro_high"]


def _model_for_mode(mode: str) -> str:
    """Map a user-facing --mode to its driving model_preference. An unknown value
    falls through as a literal model_preference so power users can pass a model id."""
    return _MODE_MODEL.get(mode, mode)


# ResearchSource is the shared `Source` (url/title/snippet) — alias kept for the
# verb's public API + existing tests/render references.
ResearchSource = Source


@dataclass
class ResearchResult:
    query: str
    answer: str
    sources: list[ResearchSource]
    mode: str
    # False iff the stream was cut before COMPLETED (deadline tripped / server cut).
    stream_complete: bool = True
    # True when the kept snapshot decodes shorter than the single LONGEST raw
    # snapshot seen (raw length is the cheap proxy — decoding all ~800 frames is
    # not). So it is a positive signal, never a completeness guarantee: a shrink
    # hidden under a frame whose raw text was larger goes unflagged.
    content_shortfall: bool = False
    warnings: list[str] = field(default_factory=list)


def _status_completed(event: dict[str, Any]) -> bool:
    """Completion predicate for research: `status == "COMPLETED"` only.

    The shared default also accepts `text_completed`, which Perplexity sets a
    few frames BEFORE the terminal repaint. That is right for delta callers, but
    research keeps whole snapshots: stopping early keeps a snapshot whose FINAL
    block is still being written."""
    data = event.get("data")
    return isinstance(data, dict) and data.get("status") == "COMPLETED"


def research(
    client: Client,
    query: str,
    *,
    mode: str = DEFAULT_MODE,
    model: str | None = None,
    council_models: list[str] | None = None,
    keep_thread: bool = False,
    timeout: float | None = None,
    progress: bool = False,
) -> ResearchResult:
    """Run a deep-research query through the ask endpoint in `mode`.

    `model` overrides the model_preference the `mode` would map to (power users
    only — a model incompatible with research fails fast). `council_models`
    (Model Council only) picks the cross-checked trio.

    `timeout` bounds wall-clock; on deadline-with-partial we return the partial
    answer with `stream_complete=False` (the agent contract is "always something
    plus a flag", exit 6). `keep_thread` preserves the incognito thread instead
    of deleting it (default deletes).

    Research streams full-snapshot frames, so we keep the *latest* `text` rather
    than concatenating deltas; the retry/deadline/heartbeat plumbing is shared
    (`_ask_common.run_ask_stream`).
    """
    model_preference = model or _model_for_mode(mode)
    # Model Council never completes unless compare_model_preferences is set, so
    # default to the trio when the user didn't pick one (see _DEFAULT_COUNCIL_MODELS).
    if model_preference == _COUNCIL_MODEL:
        if not council_models:
            council_models = list(_DEFAULT_COUNCIL_MODELS)
    else:
        # compare_model_preferences only applies to Model Council; drop it for any
        # other model so a stray --council-models doesn't ride along on a research req.
        council_models = None
    body = _build_research_body(query, model_preference, council_models=council_models)
    # `latest` is what we decode (the newest snapshot wins); `longest` is kept
    # only to detect a shortfall — a late frame can repaint a *smaller* snapshot.
    latest: dict[str, str | None] = {"text": None}
    longest: dict[str, str | None] = {"text": None}

    def _on_event(event: dict[str, Any]) -> None:
        data = event.get("data")
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            text = data["text"]
            latest["text"] = text
            best = longest["text"]
            if best is None or len(text) > len(best):
                longest["text"] = text

    state, deadline_tripped = run_ask_stream(
        client,
        ENDPOINT,
        body,
        on_event=_on_event,
        timeout=timeout,
        progress=progress,
        label="research",
        is_complete=_status_completed,
    )

    # Best-effort cleanup runs on EVERY exit path (success, FAILED, no-content,
    # or a decode error below) — delete_thread never raises, so doing it before
    # the error checks stops a FAILED/partial request from leaking the incognito
    # thread it created.
    if not keep_thread and state.backend_uuid and state.read_write_token:
        client.delete_thread(state.backend_uuid, state.read_write_token)

    if state.failed:
        raise SchemaError(
            f"research request on {ENDPOINT} returned status=FAILED; mode {mode!r} may "
            f"reject model_preference — check model↔mode compatibility via `pplx models`"
        )

    if latest["text"] is None:
        if deadline_tripped:
            raise StreamDeadlineError(
                f"research stream on {ENDPOINT} exceeded {timeout:.1f}s before any content"
            )
        raise SchemaError(f"no schematized text received from {ENDPOINT}")

    answer, sources = decode_research_text(latest["text"])
    warnings: list[str] = []
    content_shortfall = False
    best = longest["text"]
    if best is not None and best != latest["text"]:
        richer, _ = decode_research_text(best)
        if len(richer) > len(answer):
            content_shortfall = True
            warnings.append(
                f"kept snapshot decodes to {len(answer)} chars but an earlier frame "
                f"carried {len(richer)}; the report may be truncated"
            )

    return ResearchResult(
        query=query,
        answer=answer,
        sources=sources,
        mode=mode,
        stream_complete=state.saw_completed,
        content_shortfall=content_shortfall,
        warnings=warnings,
    )


def decode_research_text(text: str) -> tuple[str, list[ResearchSource]]:
    """Pure decode of a research snapshot `text` (JSON block list) → (answer, sources).

    Total function over parse success: returns (answer, sources); raises
    SchemaError if `text` isn't a JSON list of blocks. Tolerant of unknown
    step_types and missing fields — extra block kinds are ignored, partial
    snapshots yield whatever FINAL/SEARCH_RESULTS content is present so far.

    The FINAL block's `content.answer` is itself a JSON string wrapping
    `{answer: <markdown>, web_results: [<cited sources>], ...}` — we unwrap it.
    That markdown is only the cover note; the report body is the RESEARCH_ANSWER
    block's report asset, so the returned answer is cover note + body (in that
    reading order, not block order — RESEARCH_ANSWER precedes FINAL on the wire).
    Sources prefer the FINAL block's cited `web_results` (citation-aligned with
    the answer's [n] markers); we fall back to the intermediate SEARCH_RESULTS
    rounds when FINAL carries none.
    """
    try:
        blocks = json.loads(text)
    except (ValueError, TypeError) as e:
        raise SchemaError(f"research text is not JSON: {e}") from e
    if not isinstance(blocks, list):
        raise SchemaError(f"research text decoded to {type(blocks).__name__}, expected list")

    cover_parts: list[str] = []
    report_parts: list[str] = []
    final_web: list[Any] | None = None
    search_web: list[Any] = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        content = blk.get("content")
        if not isinstance(content, dict):
            continue
        step = blk.get("step_type")
        if step == "FINAL":
            markdown, cited = _unwrap_final_answer(content.get("answer"))
            if markdown:
                cover_parts.append(markdown)
            if cited:
                final_web = cited
        elif step == "RESEARCH_ANSWER":
            report_parts.extend(_report_bodies(blk))
        elif step == "SEARCH_RESULTS":
            wr = content.get("web_results")
            if isinstance(wr, list):
                search_web.extend(wr)

    cover = "\n\n".join(cover_parts).strip()
    # A body already quoted in the cover note would just be printed twice.
    answer_parts = cover_parts + [p for p in report_parts if p not in cover]
    chosen = final_web if final_web else search_web
    sources: list[Source] = []
    seen: set[str] = set()
    for wr in chosen:
        src = to_source(wr)
        if src is not None and src.url not in seen:
            seen.add(src.url)
            sources.append(src)
    return "\n\n".join(answer_parts).strip(), sources


def _report_bodies(blk: dict[str, Any]) -> list[str]:
    """A RESEARCH_ANSWER block → its report body/bodies (usually exactly one).

    The body is an asset: `assets[].research_report.source_content`. The block's
    own `content.answer` is empty in the observed builds but is honoured as a
    fallback, since that is where a non-asset build would put the same text.
    Returns [] when the block carries neither (a partial snapshot)."""
    bodies: list[str] = []
    for asset in blk.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        report = asset.get("research_report")
        if not isinstance(report, dict):
            continue
        body = report.get("source_content")
        if isinstance(body, str) and body.strip():
            bodies.append(body.strip())
    if bodies:
        return bodies
    content = blk.get("content")
    if isinstance(content, dict):
        inline = content.get("answer")
        if isinstance(inline, str) and inline.strip():
            return [inline.strip()]
    return []


def _unwrap_final_answer(raw: Any) -> tuple[str, list[Any]]:
    """FINAL `content.answer` → (markdown, cited web_results).

    Normally a JSON string `{"answer": <md>, "web_results": [...]}`; tolerate a
    plain-markdown string (returned as-is with no sources) so a server-side
    shape change degrades instead of crashing.
    """
    if not isinstance(raw, str) or not raw:
        return "", []
    try:
        inner = json.loads(raw)
    except (ValueError, TypeError):
        return raw, []  # already plain markdown
    if not isinstance(inner, dict):
        return raw, []
    md = inner.get("answer")
    web = inner.get("web_results")
    return (md if isinstance(md, str) else raw), (web if isinstance(web, list) else [])


def _build_research_body(
    query: str, model_preference: str, *, council_models: list[str] | None = None
) -> dict[str, Any]:
    """Ask-endpoint body for research. `model_preference` is what selects Deep
    Research (`pplx_alpha`) vs Model Council (`pplx_agentic_research`) — see
    `_MODE_MODEL`. `params.mode` stays "copilot" (coarse; the server derives the
    real mode from the model). `is_incognito` is True so the thread stays out of
    history. `council_models` (Model Council only) picks the cross-checked trio
    via `compare_model_preferences`; omitted → Perplexity's default trio."""
    params = base_ask_params(query, model_preference=model_preference)
    if council_models:
        params["compare_model_preferences"] = list(council_models)
    return {"query_str": query, "params": params}
