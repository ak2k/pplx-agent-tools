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
from uuid import uuid4

from ..errors import SchemaError, StreamDeadlineError
from ..wire import Client
from ._ask_common import run_ask_stream
from .fetch import extract_chunks_from_event

ENDPOINT = "/rest/sse/perplexity_ask"
DEFAULT_MODEL = "turbo"  # "Best — adapts to each query"


@dataclass
class AskResult:
    query: str
    answer: str
    model: str
    # False iff the stream was cut before COMPLETED (deadline tripped / server cut).
    stream_complete: bool = True
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

    def on_event(event: dict[str, Any]) -> None:
        chunks.extend(extract_chunks_from_event(event))

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

    return AskResult(query=query, answer=content, model=model, stream_complete=state.saw_completed)


def _build_ask_body(query: str, model: str) -> dict[str, Any]:
    """Copilot ask body (query-only, no URL). Same shape as fetch's prompt body
    but model-selectable and incognito. `mode` stays "copilot"; the model is the
    real lever (see verbs/research.py for the model-as-mode finding)."""
    frontend_uuid = str(uuid4())
    return {
        "query_str": query,
        "params": {
            "query_source": "home",
            "prompt_source": "user",
            "source": "default",
            "version": "2.18",
            "language": "en-US",
            "timezone": "UTC",
            "search_focus": "internet",
            "sources": ["web"],
            "mode": "copilot",
            "model_preference": model,
            "frontend_uuid": frontend_uuid,
            "frontend_context_uuid": str(uuid4()),
            "client_search_results_cache_key": frontend_uuid,
            "use_schematized_api": True,
            "send_back_text_in_streaming_api": True,
            "skip_search_enabled": True,
            "is_incognito": True,
            "attachments": [],
            "mentions": [],
            "client_coordinates": None,
            "dsl_query": query,
        },
    }
