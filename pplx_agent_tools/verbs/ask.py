"""pplx ask: Pro Search Q&A — a synthesized, cited answer to a question.

The front-door Perplexity experience: ask a question, Perplexity's LLM searches
the web and writes one cited answer (vs `search`, which returns raw ranked hits,
and `research`, which is the heavy multi-round path). Copilot mode on
/rest/sse/perplexity_ask — the same `markdown_block` stream `fetch --prompt`
consumes, so we reuse its chunk extractor.

Model-selectable (`--model`): the answer-producing verb is where picking a
specific model (e.g. `claude48opusthinking`, a Max thinking variant) makes sense.
Default `turbo` ("Best — adapts to each query"). See `pplx models` for valid ids.

Session-creating but incognito (no history pollution) + best-effort cleanup, like
`research`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import SchemaError, StreamDeadlineError
from ..wire import Client
from ._ask_common import (
    Source,
    base_ask_params,
    extract_chunks_from_event,
    extract_web_results,
    run_ask_stream,
    to_source,
)

ENDPOINT = "/rest/sse/perplexity_ask"
DEFAULT_MODEL = "turbo"  # "Best — adapts to each query"


@dataclass
class AskResult:
    query: str
    answer: str
    model: str
    # False iff the stream was cut before COMPLETED (deadline tripped / server cut).
    stream_complete: bool = True
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def ask(
    client: Client,
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    keep_thread: bool = False,
    timeout: float | None = None,
    progress: bool = False,
) -> AskResult:
    """Ask a question, get a synthesized cited answer (copilot mode).

    `model` is the `model_preference` (default `turbo`). `timeout` bounds
    wall-clock; on deadline-with-partial we return the partial answer with
    `stream_complete=False` (exit 6). `keep_thread` keeps the incognito thread.
    """
    body = _build_ask_body(query, model)
    chunks: list[str] = []
    sources: list[Source] = []

    def on_event(event: dict[str, Any]) -> None:
        chunks.extend(extract_chunks_from_event(event))
        # The copilot stream emits a `web_results` block carrying the cited
        # sources; the latest non-empty one wins (deduped by URL).
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

    state, deadline_tripped = run_ask_stream(
        client, ENDPOINT, body, on_event=on_event, timeout=timeout, progress=progress, label="ask"
    )

    # Best-effort cleanup runs on EVERY exit path (success, FAILED, no-content).
    # delete_thread never raises, so doing it before the error checks below stops
    # a FAILED/partial request from leaking the incognito thread it created.
    if not keep_thread and state.backend_uuid and state.read_write_token:
        client.delete_thread(state.backend_uuid, state.read_write_token)

    if state.failed:
        raise SchemaError(
            f"ask request on {ENDPOINT} returned status=FAILED; model {model!r} may be "
            f"invalid or not available on your plan — check `pplx models`"
        )

    content = "".join(chunks).strip()
    if not content and not state.saw_completed:
        if deadline_tripped:
            raise StreamDeadlineError(
                f"ask stream on {ENDPOINT} exceeded {timeout:.1f}s before any content"
            )
        raise SchemaError(f"no content received from {ENDPOINT}")

    return AskResult(
        query=query,
        answer=content,
        model=model,
        stream_complete=state.saw_completed,
        sources=sources,
    )


def _build_ask_body(query: str, model: str) -> dict[str, Any]:
    """Copilot ask body (query-only, no URL), model-selectable + incognito."""
    return {"query_str": query, "params": base_ask_params(query, model_preference=model)}
