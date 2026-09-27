"""pplx research verb: Perplexity deep research via /rest/sse/perplexity_ask.

The flagship differentiated capability — multi-step, cited research, far beyond
`search`'s ranked hits. Same endpoint as `fetch --prompt`, but the deep behavior
is selected by `model_preference` (NOT `params.mode` — see `_MODE_MODEL` and
docs/wire/perplexity-ask-research.md), with two important differences:

  1. Session-creating. Research is an ask-family verb, so it creates a thread.
     We send `is_incognito: true` (the thread never enters the user's history;
     verified) and still issue a best-effort `delete_thread` as a secondary
     guard. See CLAUDE.md → "Endpoint selection principle". The exception is a
     stream cut by a dropped connection or total silence: the run goes on
     server-side without a listener, so its thread is kept for `pplx resume`
     (verbs/resume.py), and a local record (handles.py) lets a client killed
     outright find it again.
  2. Schematized response. Unlike copilot mode's incremental `markdown_block`
     chunks, research streams full-snapshot frames whose `text` field is a JSON
     list of `{step_type, content, uuid}` blocks. The FINAL block's
     `content.answer` holds only a short *cover note* ("I've compiled…, the full
     report includes…"); the report itself is a RESEARCH_ANSWER block asset
     (`assets[].research_report.source_content`), so the answer we return is the
     cover note followed by that body. Sources accumulate across SEARCH_RESULTS
     blocks' `content.web_results`. The latest PARSEABLE snapshot's decode is what
     we return.

Deep research takes ~90-120s for a focused question and far longer for a broad
one (multi-round, hundreds of sources); the verb supports the same `--timeout` /
stall → partial-result (exit 6) contract as `fetch --prompt`, with bounded 429
retry.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import PplxError, SchemaError, ThreadGoneError
from ..handles import ThreadHandle, ThreadStore, resume_command
from ..wire import Client
from ._ask_common import (
    RESEARCH_SILENCE_SECONDS,
    AskStreamState,
    Source,
    base_ask_params,
    cutoff_cause,
    cutoff_silence,
    cutoff_warnings,
    downgrade_verdict,
    error_notes,
    no_content_error,
    release_on_exit,
    run_ask_stream,
    status_completed,
    to_source,
)

ENDPOINT = "/rest/sse/perplexity_ask"
DEFAULT_MODE = "research"

# Verified 2026-06-22: Perplexity selects deep-research behavior by
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
    # False iff the stream was cut before COMPLETED (deadline / stall / server cut).
    stream_complete: bool = True
    # True when the kept snapshot's decoded report body is shorter than the
    # longest seen in any parseable snapshot, or when the stream's last frame
    # failed to decode at all. Frames that fail to decode never become the kept
    # snapshot, so this stays a positive signal rather than a completeness
    # guarantee.
    content_shortfall: bool = False
    warnings: list[str] = field(default_factory=list)
    # "stall" | "deadline" when that bound cut the stream, "drop" when the
    # connection died mid-stream; None otherwise.
    cut_by: str | None = None
    # Set with a "stall" cut_by when the connection carried no bytes at all
    # for that many seconds.
    silent_for: float | None = None
    # Questions the run asked the user; pplx cannot answer them, so the server
    # went on with its default answers.
    clarifying_questions: list[str] = field(default_factory=list)
    # The run asked clarifying questions but none of them could be read.
    clarifying_unreadable: bool = False
    # True when the server ran a model other than the one requested; None when
    # no frame named one, and always None for Model Council, whose reported
    # model has never been observed.
    downgraded: bool | None = None
    # The last model the frames named; None when none did.
    served_model: str | None = None
    # The command that fetches the finished report, set when a dropped
    # connection or total silence cut the stream and the thread was kept.
    resume: str | None = None


def _text_changed() -> Callable[[dict[str, Any]], bool]:
    """The stall-guard progress predicate for research's snapshot stream."""
    seen: set[int] = set()

    # Snapshot frames repeat (or replay older ones) while working or hung; only a
    # snapshot never seen before counts.
    def is_progress(event: dict[str, Any]) -> bool:
        data = event.get("data")
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str) or hash(text) in seen:
            return False
        seen.add(hash(text))
        return True

    return is_progress


@dataclass
class SnapshotReport:
    """The research consumer: folds full-snapshot frames into the newest
    report that decoded. `on_event` is the whole interface a stream driver
    needs, so another consumer can stand in for it.

    The kept snapshot is the newest one that DECODED — an unparseable frame
    never replaces it, so a garbage terminal repaint costs a warning instead
    of the whole ~2-minute run. `best_body` is the high-water mark used to
    detect a late frame repainting a *smaller* report: the report BODY,
    because cover note and body share one decoded answer and a repaint that
    grows the cover while losing body would otherwise read as growth. Raw
    frame length is no use either — the terminal repaint carries more sources
    and envelope fields than earlier frames, so it is raw-larger even when its
    body shrank. `best_total` is the fallback for streams that never carry a
    body block. Decoding every snapshot costs ~0.3s over an 800-frame stream.
    """

    text: str | None = None
    answer: str = ""
    sources: list[ResearchSource] = field(default_factory=list)
    body_len: int = 0
    questions: list[str] | None = None
    best_body: int = 0
    best_total: int = 0
    saw_body: bool = False
    last_frame_decoded: bool = False
    last_raw: str | None = None

    def on_event(self, event: dict[str, Any]) -> None:
        data = event.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            return
        text: str = data["text"]
        self.last_raw = text
        try:
            cover_parts, report_parts, sources, questions = _decode_parts(text)
        except SchemaError:
            self.last_frame_decoded = False
            return
        # The body measure is the report asset itself, NOT what survives the
        # join's cover dedupe: a cover note that quotes the report in full would
        # otherwise measure zero and read as a total loss of the report.
        body = "\n\n".join(report_parts).strip()
        self.text = text
        self.answer = _join_answer(cover_parts, report_parts)
        self.sources = sources
        self.body_len = len(body)
        self.questions = questions
        self.last_frame_decoded = True
        if body:
            self.saw_body = True
        self.best_body = max(self.best_body, len(body))
        self.best_total = max(self.best_total, len(self.answer))


def research(
    client: Client,
    query: str,
    *,
    mode: str = DEFAULT_MODE,
    model: str | None = None,
    council_models: list[str] | None = None,
    keep_thread: bool = False,
    timeout: float | None = None,
    stall_seconds: float | None = None,
    progress: bool = False,
    profile: str | None = None,
) -> ResearchResult:
    """Run a deep-research query through the ask endpoint in `mode`.

    `model` overrides the model_preference the `mode` would map to (power users
    only — a model incompatible with research fails fast). `council_models`
    (Model Council only) picks the cross-checked trio.

    `timeout` bounds wall-clock and `stall_seconds` the time without new content;
    when either trips with a partial we return it with `stream_complete=False`
    and a warning naming which one (the agent contract is "always something plus
    a flag", exit 6). `keep_thread` preserves the incognito thread instead
    of deleting it (default deletes). A dropped connection or total silence
    keeps the thread whatever `keep_thread` says, and the result or error
    names the `pplx resume` command for it. `profile` scopes the local thread
    record and is named in that command.

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
    report = SnapshotReport()
    state = AskStreamState()
    handle = ThreadHandle(ThreadStore(profile), prompt=query, mode=mode, model=model_preference)
    read_report(
        client,
        state,
        handle,
        report.on_event,
        endpoint=ENDPOINT,
        body=body,
        label="research",
        keep_thread=keep_thread,
        timeout=timeout,
        stall_seconds=stall_seconds,
        progress=progress,
    )
    kept = state.backend_uuid if state.kept and not state.deleted else None
    return finish_report(
        report,
        state,
        handle,
        label="research",
        endpoint=ENDPOINT,
        query=query,
        mode=mode,
        requested_model=None if model_preference == _COUNCIL_MODEL else model_preference,
        timeout=timeout,
        resume=resume_command(kept, profile) if kept else None,
    )


def read_report(
    client: Client,
    state: AskStreamState,
    handle: ThreadHandle,
    on_event: Callable[[dict[str, Any]], None],
    *,
    endpoint: str,
    body: dict[str, Any],
    label: str,
    keep_thread: bool,
    timeout: float | None,
    stall_seconds: float | None,
    progress: bool,
    opener: Callable[..., Iterator[dict[str, Any]]] | None = None,
    keep_on_raise: bool = False,
) -> None:
    """Drive one research stream into `on_event` under research's cleanup
    policy, keeping `handle`'s record current.

    The record is written before the frame that first names the thread is
    consumed, and settled from what cleanup did, so a process killed in
    between leaves it `running`. A run that may go on server-side after the
    read ends (a dropped connection, total silence, or a terminate that could
    not be sent or failed) keeps its thread (`state.kept`, see
    `release_on_exit`); every other end is cleaned up as for any ask-family
    run, and so is an exception unless `keep_on_raise`. `opener` replaces the
    default POST of `body` to `endpoint` (see `run_ask_stream`); `endpoint`
    then only names the stream in messages.
    """

    def recording(event: dict[str, Any]) -> None:
        handle.observe(state.backend_uuid, state.read_write_token)
        on_event(event)

    raised = True
    gone = False
    try:
        with release_on_exit(
            client, state, keep_thread=keep_thread, keep_live=True, keep_on_raise=keep_on_raise
        ):
            run_ask_stream(
                client,
                endpoint,
                body,
                state,
                on_event=recording,
                timeout=timeout,
                stall_seconds=stall_seconds,
                progress=progress,
                label=label,
                is_complete=status_completed,
                is_progress=_text_changed(),
                silence_seconds=RESEARCH_SILENCE_SECONDS,
                opener=opener,
            )
        raised = False
    except ThreadGoneError:
        gone = True
        raise
    finally:
        _settle(handle, state, gone=gone)
        if raised:
            # An exception in flight has no result or error text of ours to carry these.
            for warning in handle.warnings:
                print(f"warning: {warning}", file=sys.stderr)


def _settle(handle: ThreadHandle, state: AskStreamState, *, gone: bool) -> None:
    """Bring the record in line with what cleanup did: kept while the thread
    can be resumed, removed once it is deleted, gone or finished with, and
    left as it was when no frame named the thread."""
    if gone or state.deleted:
        handle.forget()
    elif state.kept:
        handle.keep()
    elif state.backend_uuid is not None:
        handle.forget()


def kept_warning(command: str) -> str:
    return (
        "the research run may still be going on server-side, so its thread was kept: get "
        f"the finished report with `{command}` within about 24 h; do not re-run, which "
        "spends another research unit"
    )


def finish_report(
    report: SnapshotReport,
    state: AskStreamState,
    handle: ThreadHandle,
    *,
    label: str,
    endpoint: str,
    query: str,
    mode: str,
    requested_model: str | None,
    timeout: float | None,
    resume: str | None,
    notes: Sequence[str] = (),
) -> ResearchResult:
    """The result of a stream `read_report` drove into `report`, or the error
    for one that yielded nothing usable.

    `requested_model` is judged against the model the frames named (None skips
    the downgrade check). `resume`, set when the thread was kept, reaches the
    result's warnings and `resume` field, or the error's text and `resume`
    attribute, so an agent reading either document finds it. `notes` lead
    the warnings, or end the error's text, like those.
    """
    lead = [*notes, kept_warning(resume)] if resume else list(notes)
    error_tail = lead + state.cleanup_warnings + handle.warnings
    try:
        if state.failed:
            raise SchemaError(
                f"{label} request on {endpoint} returned status=FAILED; mode {mode!r} may "
                f"reject model_preference — check model↔mode compatibility via `pplx models`"
                f"{error_notes(handle.warnings)}"
            )

        if report.text is None:
            if report.last_raw is not None:
                # Text arrived but no frame ever parsed — the decode error is the
                # honest diagnosis, so re-raise it rather than reporting no content.
                _raise_undecodable(report.last_raw, error_tail)
            raise no_content_error(
                label=label,
                endpoint=endpoint,
                timeout=timeout,
                cutoff=state.cutoff,
                warnings=error_tail,
            )

        if not state.saw_completed and not report.answer and not report.sources:
            # A cut stream whose only snapshot is the empty INITIAL_QUERY step has
            # nothing to salvage; report it as a retryable cutoff, as `ask` does.
            # Questions the run asked before the cut go in the error, so the retry
            # can answer them.
            raise no_content_error(
                label=label,
                endpoint=endpoint,
                timeout=timeout,
                cutoff=state.cutoff,
                warnings=_clarifying_warnings(report.questions, no_answer=True) + error_tail,
            )
    except PplxError as e:
        e.resume = resume
        raise
    content_shortfall, warnings = _shortfall_verdict(report)
    warnings += _clarifying_warnings(report.questions)
    downgraded: bool | None = None
    if requested_model is not None:
        downgraded, downgrade_warnings = downgrade_verdict(state, requested_model)
        warnings += downgrade_warnings

    return ResearchResult(
        query=query,
        answer=report.answer,
        sources=report.sources,
        mode=mode,
        stream_complete=state.saw_completed,
        cut_by=cutoff_cause(state),
        silent_for=cutoff_silence(state),
        content_shortfall=content_shortfall,
        warnings=cutoff_warnings(state)
        + lead
        + warnings
        + state.cleanup_warnings
        + handle.warnings,
        clarifying_questions=report.questions or [],
        clarifying_unreadable=report.questions is not None and not report.questions,
        downgraded=downgraded,
        served_model=state.display_model,
        resume=resume,
    )


def _clarifying_warnings(questions: list[str] | None, *, no_answer: bool = False) -> list[str]:
    """The warning for clarifying questions the run asked (None: it asked
    none). pplx cannot answer them: a run that went on used the server's
    defaults; with `no_answer` the run ended before any answer arrived."""
    outcome = "no answer arrived" if no_answer else "proceeded on the server's default answers"
    if questions:
        return [f"research asked clarifying questions and {outcome}: " + "; ".join(questions)]
    if questions is not None:
        return [f"research asked clarifying questions pplx could not read and {outcome}"]
    return []


def _shortfall_verdict(report: SnapshotReport) -> tuple[bool, list[str]]:
    """Did the kept snapshot lose content? → (flag, warnings).

    Judged on the report BODY whenever any frame carried one, since the cover
    note moves independently of the report; a stream of FINAL-only snapshots has
    no body to compare and falls back to the whole decoded answer.
    """
    warnings: list[str] = []
    answer_len = len(report.answer)
    if report.saw_body:
        if report.body_len < report.best_body:
            warnings.append(
                f"kept snapshot's report body decodes to {report.body_len} chars but an earlier "
                f"frame carried {report.best_body}; the report may be truncated"
            )
    elif answer_len < report.best_total:
        warnings.append(
            f"kept snapshot decodes to {answer_len} chars but an earlier frame "
            f"carried {report.best_total}; the report may be truncated"
        )
    if not report.last_frame_decoded:
        warnings.append(
            "the last frame of the stream failed to decode; returning the newest "
            f"parseable snapshot ({answer_len} chars), which may be truncated"
        )
    return bool(warnings), warnings


def _raise_undecodable(text: str, warnings: Sequence[str]) -> None:
    """Raise the decode error for `text`, ending with `warnings`: cleanup has
    already run by then, so a run it could not stop is named here."""
    try:
        decode_research_text(text)
    except SchemaError as e:
        raise SchemaError(f"{e}{error_notes(warnings)}") from e


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
    cover_parts, report_parts, sources, _ = _decode_parts(text)
    return _join_answer(cover_parts, report_parts), sources


def _join_answer(cover_parts: list[str], report_parts: list[str]) -> str:
    """Cover note parts + report body parts → the answer we return.

    Dropping a body the cover note already quotes verbatim is a RENDERING
    concern and lives only here — the report is present either way, so no
    caller should read the drop as missing content.
    """
    cover = "\n\n".join(cover_parts).strip()
    kept_reports = [p for p in report_parts if p not in cover]
    return "\n\n".join(cover_parts + kept_reports).strip()


def _decode_parts(
    text: str,
) -> tuple[list[str], list[str], list[ResearchSource], list[str] | None]:
    """`decode_research_text` before the join: (cover parts, report body parts,
    sources, clarifying question texts). The question texts are None when the
    snapshot asked none, and empty when it asked some that could not be read.

    Split out so a caller can measure the report BODY on its own — the joined
    answer mixes cover note and body, and a snapshot that grows the cover while
    losing body keeps the total steady. The parts are raw: `_join_answer` owns
    every rendering decision made on top of them.
    """
    try:
        blocks = json.loads(text)
    except (ValueError, TypeError) as e:
        raise SchemaError(f"research text is not JSON: {e}") from e
    if not isinstance(blocks, list):
        raise SchemaError(f"research text decoded to {type(blocks).__name__}, expected list")

    cover_parts: list[str] = []
    report_parts: list[str] = []
    questions: list[str] | None = None
    final_web: list[Any] | None = None
    search_web: list[Any] = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        step = blk.get("step_type")
        # The report body is an ASSET, not `content` — so RESEARCH_ANSWER must be
        # dispatched ahead of the content guard below, or a block with a null
        # `content` and a perfectly good report asset drops the whole report.
        if step == "RESEARCH_ANSWER":
            report_parts.extend(_report_bodies(blk))
            continue
        # Dispatched ahead of the guard for the same reason: a clarifying step
        # with an unreadable `content` still means the run went on without answers.
        if step == "RESEARCH_CLARIFYING_QUESTIONS":
            questions = (questions or []) + _question_texts(blk.get("content"))
            continue
        content = blk.get("content")
        if not isinstance(content, dict):
            continue
        if step == "FINAL":
            markdown, cited = _unwrap_final_answer(content.get("answer"))
            if markdown:
                cover_parts.append(markdown)
            if cited:
                final_web = cited
        elif step == "SEARCH_RESULTS":
            wr = content.get("web_results")
            if isinstance(wr, list):
                search_web.extend(wr)

    chosen = final_web if final_web else search_web
    sources: list[Source] = []
    seen: set[str] = set()
    for wr in chosen:
        src = to_source(wr)
        if src is not None and src.url not in seen:
            seen.add(src.url)
            sources.append(src)
    return cover_parts, report_parts, sources, questions


def _question_texts(content: Any) -> list[str]:
    """A RESEARCH_CLARIFYING_QUESTIONS block's `content` → its question texts.
    Total over JSON shape, like `_report_bodies`."""
    raw = content.get("questions") if isinstance(content, dict) else None
    out: list[str] = []
    for q in raw if isinstance(raw, list) else []:
        text = q.get("question_text") if isinstance(q, dict) else None
        if isinstance(text, str) and text.strip():
            out.append(text.strip())
    return out


def _report_bodies(blk: dict[str, Any]) -> list[str]:
    """A RESEARCH_ANSWER block → its report body/bodies (usually exactly one).

    The body is an asset: `assets[].research_report.source_content`. The block's
    own `content.answer` is empty in the observed builds but is honored as a
    fallback, since that is where a non-asset build would put the same text.
    Returns [] when the block carries neither (a partial snapshot).

    Total over JSON shape: every decode path runs inside the stream callback,
    which guards only SchemaError, so a TypeError here would escape the stream
    loop and strand the thread it created."""
    bodies: list[str] = []
    assets = blk.get("assets")
    for asset in assets if isinstance(assets, list) else []:
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
