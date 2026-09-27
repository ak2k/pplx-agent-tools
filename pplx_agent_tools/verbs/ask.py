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
from typing import Any, Literal, TypeAlias

from ..errors import SchemaError
from ..grounding import Grounding, Unchecked, check_grounding
from ..wire import Client
from ._ask_common import (
    COPILOT_SETTLE_SECONDS,
    AskStreamState,
    Source,
    apply_chunk_patch,
    base_ask_params,
    blocks_changed,
    cutoff_cause,
    cutoff_silence,
    cutoff_warnings,
    downgrade_verdict,
    extract_chunk_patches,
    extract_web_results,
    no_content_error,
    release_on_exit,
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


@dataclass(frozen=True)
class Finished:
    """The stream reached COMPLETED: the answer is whole and the sources are
    in the order its [n] citations index."""

    tag: Literal["finished"] = field(default="finished", init=False)


@dataclass(frozen=True)
class FinishedWithoutSources:
    """The answer is whole (`text_completed`) but the COMPLETED frame never
    arrived, so the sources are the last mid-stream list and [n] may not
    index them."""

    tag: Literal["finished_without_sources"] = field(default="finished_without_sources", init=False)


@dataclass(frozen=True)
class Cut:
    """The stream ended before the answer was whole. `silent_for` is set when
    the stall was the connection carrying no bytes at all for that long."""

    by: Literal["stall", "deadline", "drop", "server"]
    silent_for: float | None = None
    tag: Literal["cut"] = field(default="cut", init=False)


AskCompletion: TypeAlias = "Finished | FinishedWithoutSources | Cut"


@dataclass
class AskResult:
    query: str
    answer: str
    model: str
    completion: AskCompletion = field(default_factory=Finished)
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounding: Grounding = field(default_factory=lambda: Unchecked("disabled"))
    # True when the server ran a model other than `model`; None when no frame
    # named the model it ran.
    downgraded: bool | None = None
    # The last model the frames named; None when none did.
    served_model: str | None = None


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
    trips, or the connection drops, with a partial answer we return it with a
    `Cut` completion (exit 6) and a warning naming which one. `keep_thread`
    keeps the incognito thread. `grounded_check` attaches a `Grounding` verdict on whether the
    answer's figures and names appear in its sources.
    """
    body = _build_ask_body(query, model)
    chunks: dict[int, str] = {}
    sources: list[Source] = []

    def on_event(event: dict[str, Any]) -> None:
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

    state = AskStreamState()
    with release_on_exit(client, state, keep_thread=keep_thread, settles_after_text=True):
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

    if state.failed:
        raise SchemaError(
            f"ask request on {ENDPOINT} returned status=FAILED; model {model!r} may be "
            f"invalid or not available on your plan — check `pplx models`"
        )

    content = "".join(chunks[i] for i in sorted(chunks)).strip()
    if not content and not state.saw_completed:
        raise no_content_error(
            label="ask",
            endpoint=ENDPOINT,
            timeout=timeout,
            cutoff=state.cutoff,
            warnings=state.cleanup_warnings,
        )

    completion: AskCompletion
    if state.saw_completed:
        completion = Finished()
    elif state.text_completed:
        # The answer is whole at `text_completed`; only the sources frame after
        # it was lost (a settle expiry, a cut or a drop), so this is not a
        # partial answer.
        completion = FinishedWithoutSources()
    else:
        completion = Cut(cutoff_cause(state) or "server", cutoff_silence(state))
    downgraded, downgrade_warnings = downgrade_verdict(state, model)
    return AskResult(
        query=query,
        answer=content,
        model=model,
        completion=completion,
        sources=sources,
        warnings=(
            [SOURCES_FRAME_MISSING]
            if isinstance(completion, FinishedWithoutSources)
            else cutoff_warnings(state)
        )
        + downgrade_warnings
        + state.cleanup_warnings,
        grounding=(
            check_grounding(content, query, sources) if grounded_check else Unchecked("disabled")
        ),
        downgraded=downgraded,
        served_model=state.display_model,
    )


def _build_ask_body(query: str, model: str) -> dict[str, Any]:
    """Copilot ask body (query-only, no URL), model-selectable + incognito."""
    return {"query_str": query, "params": base_ask_params(query, model_preference=model)}
