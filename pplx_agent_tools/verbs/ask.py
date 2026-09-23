"""pplx ask: Pro Search Q&A — a synthesized, cited answer to a question.

The front-door Perplexity experience: ask a question, Perplexity's LLM searches
the web and writes one cited answer (vs `search`, which returns raw ranked hits,
and `research`, which is the heavy multi-round path). Copilot mode on
/rest/sse/perplexity_ask — the same `markdown_block` stream `fetch --prompt`
consumes. Unlike fetch, ask reads on to the terminal COMPLETED frame, the only
one whose web_results list is in the order the answer's [n] citations index.

Model-selectable (`--model`): the answer-producing verb is where picking a
specific model (e.g. `claude48opusthinking`, a Max thinking variant) makes sense.
Default `turbo` ("Best — adapts to each query"). See `pplx models` for valid ids.

Session-creating but incognito (no history pollution) + best-effort cleanup, like
`research`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import NetworkError, SchemaError
from ..grounding import Grounding, check_grounding
from ..wire import Client
from ._ask_common import (
    COPILOT_SETTLE_SECONDS,
    AskStreamState,
    Source,
    apply_chunk_patch,
    base_ask_params,
    blocks_changed,
    cutoff_cause,
    cutoff_warnings,
    event_marks_completed,
    extract_chunk_patches,
    extract_web_results,
    no_content_error,
    release_thread,
    run_ask_stream,
    status_completed,
    to_source,
)

ENDPOINT = "/rest/sse/perplexity_ask"
DEFAULT_MODEL = "turbo"  # "Best — adapts to each query"
SOURCES_FRAME_MISSING = (
    "stream ended after the answer but before its final sources frame; "
    "[n] citations may not match the sources list"
)


@dataclass
class AskResult:
    query: str
    answer: str
    model: str
    # False iff the stream was cut before COMPLETED (deadline / stall / server cut).
    stream_complete: bool = True
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # "stall" | "deadline" when that bound cut the stream; None otherwise.
    cut_by: str | None = None
    # None when the caller disabled the check.
    grounding: Grounding | None = None


def ask(
    client: Client,
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    keep_thread: bool = False,
    timeout: float | None = None,
    stall_seconds: float | None = None,
    progress: bool = False,
    grounded_check: bool = True,
) -> AskResult:
    """Ask a question, get a synthesized cited answer (copilot mode).

    `model` is the `model_preference` (default `turbo`). `timeout` bounds
    wall-clock and `stall_seconds` the time without new content; when either
    trips with a partial answer we return it with `stream_complete=False`
    (exit 6) and a warning naming which one. `keep_thread` keeps the incognito
    thread. `grounded_check` attaches a `Grounding` verdict on whether the
    answer's figures and names appear in its sources.
    """
    body = _build_ask_body(query, model)
    chunks: dict[int, str] = {}
    sources: list[Source] = []
    text_done = False

    def on_event(event: dict[str, Any]) -> None:
        nonlocal text_done
        terminal = status_completed(event)
        for offset, run in extract_chunk_patches(event):
            apply_chunk_patch(chunks, offset, run, terminal=terminal)
        # Each search step emits its own web_results block; the COMPLETED
        # frame's block is the one the answer's [n] citations index, so the
        # latest non-empty block wins (deduped by URL).
        raw_results = extract_web_results(event)
        if raw_results:
            seen: set[str] = set()
            collected: list[Source] = []
            for raw in raw_results:
                src = to_source(raw)
                if src is not None and src.url not in seen:
                    seen.add(src.url)
                    collected.append(src)
            sources[:] = collected
        text_done = text_done or event_marks_completed(event)

    state = AskStreamState()
    try:
        run_ask_stream(
            client,
            ENDPOINT,
            body,
            state,
            on_event=on_event,
            timeout=timeout,
            stall_seconds=stall_seconds,
            progress=progress,
            label="ask",
            is_complete=status_completed,
            is_progress=blocks_changed(),
            settle_seconds=COPILOT_SETTLE_SECONDS,
        )
    except NetworkError:
        # After `text_completed` the answer is whole; losing the connection then
        # only costs the sources frame, handled below like a settle expiry.
        if not text_done:
            raise
    finally:
        release_thread(client, state, keep_thread=keep_thread)

    if state.failed:
        raise SchemaError(
            f"ask request on {ENDPOINT} returned status=FAILED; model {model!r} may be "
            f"invalid or not available on your plan — check `pplx models`"
        )

    content = "".join(chunks[i] for i in sorted(chunks)).strip()
    if not content and not state.saw_completed:
        raise no_content_error(label="ask", endpoint=ENDPOINT, timeout=timeout, cutoff=state.cutoff)

    # The answer is whole at `text_completed`; only the sources frame after it
    # was lost, so this is not a partial answer.
    sources_lost = text_done and not state.saw_completed
    return AskResult(
        query=query,
        answer=content,
        model=model,
        stream_complete=state.saw_completed or text_done,
        cut_by=None if sources_lost else cutoff_cause(state),
        sources=sources,
        warnings=[SOURCES_FRAME_MISSING] if sources_lost else cutoff_warnings(state),
        grounding=check_grounding(content, query, sources) if grounded_check else None,
    )


def _build_ask_body(query: str, model: str) -> dict[str, Any]:
    """Copilot ask body (query-only, no URL), model-selectable + incognito."""
    return {"query_str": query, "params": base_ask_params(query, model_preference=model)}
