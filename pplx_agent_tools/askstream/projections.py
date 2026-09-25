"""What a verb returns, read from the tracked block documents.

Every projection is a pure function of a `TrackedView`, which can only name
the fields in `READS`. The answer is read from exactly one source per run,
chosen by the verb's `AnswerPaths`; `answer_text` is the one statement of
which text a verb returns and whether its citation numbers are final.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, final

from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import FieldKey
from pplx_agent_tools.askstream.jsonval import JsonValue

ReadKey = Literal["ask_text", "workflow", "web_results", "unified_assets"]

READS: Mapping[ReadKey, FieldKey] = MappingProxyType(
    {
        "ask_text": ("ask_text", ("markdown_block",)),
        "workflow": ("workflow_root", ("workflow_block",)),
        "web_results": ("web_results", ("web_result_block",)),
        "unified_assets": ("unified_assets", ("unified_assets_block",)),
    }
)
READ_FIELDS: frozenset[FieldKey] = frozenset(READS.values())

AnswerPaths = Literal["ask_text_or_workflow", "ask_text_only"]

CITATIONS_NOT_FINAL = "the answer's citation numbers may not match `sources`"

WORKFLOW_TEXT = "WORKFLOW_ITEM_TEXT"


class TrackedView(Protocol):
    def get(self, key: ReadKey) -> JsonValue | None:
        """The field's document, or None when it is absent or desynced."""
        ...


@final
@dataclass(frozen=True, slots=True)
class AskTextPath:
    pass


@final
@dataclass(frozen=True, slots=True)
class WorkflowPath:
    pass


@final
@dataclass(frozen=True, slots=True)
class NoAnswer:
    pass


AnswerSource: TypeAlias = AskTextPath | WorkflowPath | NoAnswer


@final
@dataclass(frozen=True, slots=True)
class Answer:
    source: AnswerSource
    # The chunk join, raw.
    streamed: str
    # `ask_text`: the terminal block's `answer`; workflow: `text_payload.text`.
    # None when absent or empty.
    final: str | None


@final
@dataclass(frozen=True, slots=True)
class AnswerText:
    text: str
    citations_final: bool


@final
@dataclass(frozen=True, slots=True)
class WebSource:
    url: str
    title: str | None
    snippet: str | None


NO_ANSWER = Answer(NoAnswer(), "", None)


def _obj(v: JsonValue | None) -> dict[str, JsonValue]:
    return v if isinstance(v, dict) else {}


def _list(v: JsonValue | None) -> list[JsonValue]:
    return v if isinstance(v, list) else []


def _text(v: JsonValue | None) -> str | None:
    return v if isinstance(v, str) and v else None


def _join(chunks: JsonValue | None) -> str:
    # Gaps are null: a chunk placed past the end keeps its index.
    return "".join(c for c in _list(chunks) if isinstance(c, str))


def _has_text(a: Answer) -> bool:
    return bool(a.streamed) or a.final is not None


def _ask_text(view: TrackedView) -> Answer:
    doc = _obj(view.get("ask_text"))
    return Answer(AskTextPath(), _join(doc.get("chunks")), _text(doc.get("answer")))


def _workflow_items(view: TrackedView) -> list[Answer]:
    out: list[Answer] = []
    for step in _list(_obj(view.get("workflow")).get("steps")):
        for item in _list(_obj(step).get("items")):
            it = _obj(item)
            if it.get("type") == WORKFLOW_TEXT:
                tp = _obj(_obj(it.get("payload")).get("text_payload"))
                out.append(Answer(WorkflowPath(), _join(tp.get("chunks")), _text(tp.get("text"))))
    return out


def answer(view: TrackedView, paths: AnswerPaths) -> tuple[Answer, tuple[Drift, ...]]:
    """The run's answer from the one source `paths` allows.

    `ask_text` wins whenever it has text. Research (`ask_text_only`) never
    reads workflow text items: there they are per-step finding summaries."""
    at = _ask_text(view)
    if paths == "ask_text_only":
        return (at if _has_text(at) else NO_ANSWER), ()
    items = _workflow_items(view)
    drift: list[Drift] = []
    if _has_text(at):
        if any(_has_text(i) for i in items):
            drift.append(Drift("projection_ambiguous", name_of("answer:both_paths")))
        return at, tuple(drift)
    if not items or not _has_text(items[-1]):
        return NO_ANSWER, ()
    if len(items) > 1:
        drift.append(Drift("projection_ambiguous", name_of("answer:workflow_items")))
    return items[-1], tuple(drift)


def answer_text(ans: Answer, completed: bool) -> AnswerText:
    """The text a verb returns, stripped, and whether its citation numbers
    are final. `ask_text`'s `answer` is final only on the terminal frame;
    the workflow `text` equals the chunk join once present."""
    match ans.source:
        case NoAnswer():
            return AnswerText("", True)
        case AskTextPath():
            final = ans.final if completed else None
        case WorkflowPath():
            final = ans.final
    if final is not None:
        return AnswerText(final.strip(), True)
    return AnswerText(ans.streamed.strip(), False)


def citation_warnings(t: AnswerText) -> tuple[str, ...]:
    """`CITATIONS_NOT_FINAL` exactly when non-empty text has streamed
    citation numbers, for any outcome."""
    return (CITATIONS_NOT_FINAL,) if t.text and not t.citations_final else ()


def sources(view: TrackedView) -> tuple[WebSource, ...]:
    """The current `web_results` list in order, deduplicated by URL; an entry
    without a non-empty string `url` is skipped. Accepts `name` or `title`.
    A run's sources are `latest_sources`, which this list can empty."""
    out: list[WebSource] = []
    seen: set[str] = set()
    for raw in _list(_obj(view.get("web_results")).get("web_results")):
        r = _obj(raw)
        url = _text(r.get("url"))
        if url is None or url in seen:
            continue
        seen.add(url)
        title = r.get("name") or r.get("title")
        snippet = r.get("snippet")
        out.append(
            WebSource(
                url,
                title if isinstance(title, str) else None,
                snippet if isinstance(snippet, str) else None,
            )
        )
    return tuple(out)


def latest_sources(held: tuple[WebSource, ...], view: TrackedView) -> tuple[WebSource, ...]:
    """A run's sources after a frame: the latest non-empty `web_results`
    list wins, so an empty, absent or desynced one keeps `held`."""
    if _list(_obj(view.get("web_results")).get("web_results")):
        return sources(view)
    return held


def _asset_bodies(assets: JsonValue | None) -> Iterator[str]:
    for asset in _list(assets):
        body = _obj(_obj(asset).get("research_report")).get("source_content")
        if isinstance(body, str) and body.strip():
            yield body.strip()


def report_parts(view: TrackedView) -> list[str]:
    """The research report bodies from the first path that holds one:
    `unified_assets`, else the workflow step items' `sources_payload`."""
    parts = list(_asset_bodies(_obj(view.get("unified_assets")).get("assets")))
    if parts:
        return parts
    for step in _list(_obj(view.get("workflow")).get("steps")):
        for item in _list(_obj(step).get("items")):
            sp = _obj(_obj(_obj(item).get("payload")).get("sources_payload"))
            parts.extend(_asset_bodies(sp.get("assets")))
    return parts


def report_body(view: TrackedView) -> str:
    return "\n\n".join(report_parts(view))


def research_answer(view: TrackedView, completed: bool) -> AnswerText:
    """A research result built from the projections, for a run whose
    terminal frame carried no `text` or never arrived.

    The cover note comes from `answer_text`; the report body is appended
    unless the cover already quotes it, as the verb joins a decoded `text`.
    A body read before the terminal frame is not final either."""
    ans, _ = answer(view, "ask_text_only")
    cover = answer_text(ans, completed)
    parts = report_parts(view)
    kept = [p for p in parts if p not in cover.text]
    text = "\n\n".join(([cover.text] if cover.text else []) + kept).strip()
    return AnswerText(text, cover.citations_final and (completed or not parts))


_MARKER = re.compile(r"\[\s*\d+(?:\s*[,\-\u2013]\s*\d+)*\s*\]")
_DIGITS = re.compile(r"\d+")


def _mask_citations(s: str) -> str:
    return _MARKER.sub(lambda m: _DIGITS.sub("#", m.group(0)), s)


def citations_renumbered(a: str, b: str) -> bool:
    """True when `a` and `b` differ, and only in the digits of `[n]`
    citation markers (any digit count, so `[9]` to `[10]` counts)."""
    return a != b and _mask_citations(a) == _mask_citations(b)


def projection_missing(view: TrackedView) -> tuple[Drift, ...]:
    """A read path absent from a document that is present and in sync."""
    checks: tuple[tuple[ReadKey, str], ...] = (
        ("ask_text", "chunks"),
        ("workflow", "steps"),
        ("web_results", "web_results"),
        ("unified_assets", "assets"),
    )
    out: list[Drift] = []
    for key, member in checks:
        doc = view.get(key)
        if doc is not None and not isinstance(_obj(doc).get(member), list):
            out.append(Drift("projection_missing", name_of(f"{key}/{member}")))
    return tuple(out)
