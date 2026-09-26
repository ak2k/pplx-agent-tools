"""Projections (plan §2.4.2): the answer source per verb, the returned text
and its citation warning, sources and the report body (U3 oracle 11, I24)."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.jsonval import JsonValue
from pplx_agent_tools.askstream.projections import (
    CITATIONS_NOT_FINAL,
    Answer,
    AnswerText,
    AskTextPath,
    NoAnswer,
    ReadKey,
    WorkflowPath,
    answer,
    answer_text,
    citation_warnings,
    citations_renumbered,
    projection_missing,
    report_body,
    research_answer,
    sources,
)
from pplx_agent_tools.verbs._ask_common import to_source
from tests._askframes import report, workflow_text

BOTH_PATHS = Drift("projection_ambiguous", name_of("answer:both_paths"))
WORKFLOW_ITEMS = Drift("projection_ambiguous", name_of("answer:workflow_items"))


class View:
    def __init__(self, **docs: Any) -> None:
        self.docs: dict[str, Any] = docs

    def get(self, key: ReadKey) -> JsonValue | None:
        return self.docs.get(key)


def ask_text(chunks: list[str], answer: str | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {"chunks": chunks}
    if answer is not None:
        doc["answer"] = answer
    return doc


# --- answer source (oracle 11, I24) -------------------------------------------------


def test_workflow_path_when_ask_text_has_no_text() -> None:
    view = View(workflow=workflow_text((["An", "swer"], None)))
    assert answer(view, "ask_text_or_workflow") == (Answer(WorkflowPath(), "Answer", None), ())


def test_ask_text_path_when_only_ask_text_has_text() -> None:
    view = View(ask_text=ask_text(["A", "B"], "AB"))
    assert answer(view, "ask_text_or_workflow") == (Answer(AskTextPath(), "AB", "AB"), ())


def test_both_paths_with_text_read_ask_text_and_flag_one_ambiguity() -> None:
    view = View(ask_text=ask_text(["A"]), workflow=workflow_text((["W"], "W")))
    ans, drift = answer(view, "ask_text_or_workflow")
    assert ans.source == AskTextPath()
    assert drift == (BOTH_PATHS,)


def test_two_workflow_text_items_read_the_last_and_flag_one_ambiguity() -> None:
    view = View(workflow=workflow_text((["first"], "first"), (["second"], None)))
    ans, drift = answer(view, "ask_text_or_workflow")
    assert ans == Answer(WorkflowPath(), "second", None)
    assert drift == (WORKFLOW_ITEMS,)


def test_research_reads_ask_text_only_with_no_ambiguity() -> None:
    summaries = workflow_text((["step one"], None), (["step two"], None))
    view = View(ask_text=ask_text(["Cover"]), workflow=summaries)
    assert answer(view, "ask_text_only") == (Answer(AskTextPath(), "Cover", None), ())


def test_research_before_its_first_ask_text_chunk_has_no_answer() -> None:
    """Research's step summaries arrive long before `ask_text`; they are
    never its answer."""
    view = View(ask_text=ask_text([]), workflow=workflow_text((["summary"], None)))
    assert answer(view, "ask_text_only") == (Answer(NoAnswer(), "", None), ())


def test_empty_view_has_no_answer() -> None:
    for paths in ("ask_text_or_workflow", "ask_text_only"):
        assert answer(View(), paths)[0].source == NoAnswer()


# --- returned text and CITATIONS_NOT_FINAL (§2.4.2 answer_text) -----------------------


@pytest.mark.parametrize(
    ("ans", "completed", "expected"),
    [
        (Answer(AskTextPath(), "a [1] ", "a [2]"), True, AnswerText("a [2]", True)),
        (Answer(AskTextPath(), "a [1] ", "a [2]"), False, AnswerText("a [1]", False)),
        (Answer(AskTextPath(), "a [1]", None), True, AnswerText("a [1]", False)),
        (Answer(WorkflowPath(), "w", "w!"), False, AnswerText("w!", True)),
        (Answer(WorkflowPath(), "w", "w!"), True, AnswerText("w!", True)),
        (Answer(WorkflowPath(), " w ", None), True, AnswerText("w", False)),
        (Answer(NoAnswer(), "", None), False, AnswerText("", True)),
    ],
)
def test_answer_text_rule(ans: Answer, completed: bool, expected: AnswerText) -> None:
    assert answer_text(ans, completed) == expected


@pytest.mark.parametrize(
    ("text", "warned"),
    [
        (AnswerText("x", False), True),
        (AnswerText("x", True), False),
        (AnswerText("", False), False),
        (AnswerText("", True), False),
    ],
)
def test_citation_warning_exactly_on_nonempty_text_with_streamed_numbers(
    text: AnswerText, warned: bool
) -> None:
    assert citation_warnings(text) == ((CITATIONS_NOT_FINAL,) if warned else ())


@pytest.mark.parametrize(
    ("a", "b", "renumbered"),
    [
        ("x [1] y [2].", "x [2] y [1].", True),
        ("x [9].", "x [10].", True),
        ("x [1, 2].", "x [3, 4].", True),
        ("x [1].", "x [1].", False),
        ("x [1].", "y [1].", False),
        ("x [1].", "x [1]!", False),
        ("x [a].", "x [b].", False),
        ("x 1.", "x 2.", False),
    ],
)
def test_citations_renumbered_rows(a: str, b: str, renumbered: bool) -> None:
    assert citations_renumbered(a, b) is renumbered


# --- a cut research run's partial (§3.6 via answer_text) ------------------------------


def test_cut_research_cover_only_carries_the_warning() -> None:
    view = View(ask_text=ask_text(["Cover [1]"]))
    t = research_answer(view, completed=False)
    assert t == AnswerText("Cover [1]", False)
    assert citation_warnings(t) == (CITATIONS_NOT_FINAL,)


def test_cut_research_body_only_carries_the_warning() -> None:
    t = research_answer(View(unified_assets=report("Body [3]")), completed=False)
    assert t == AnswerText("Body [3]", False)
    assert citation_warnings(t) == (CITATIONS_NOT_FINAL,)


def test_completed_research_without_text_joins_final_cover_and_body() -> None:
    view = View(ask_text=ask_text(["Cov"], "Cover"), unified_assets=report("Body"))
    t = research_answer(view, completed=True)
    assert t == AnswerText("Cover\n\nBody", True)
    assert citation_warnings(t) == ()


def test_research_body_quoted_by_the_cover_is_not_repeated() -> None:
    view = View(ask_text=ask_text([], "See: Body"), unified_assets=report("Body"))
    assert research_answer(view, completed=True).text == "See: Body"


def test_research_with_nothing_is_empty_and_unwarned() -> None:
    t = research_answer(View(), completed=False)
    assert (t, citation_warnings(t)) == (AnswerText("", True), ())


# --- sources and report body -----------------------------------------------------------

ROW = st.fixed_dictionaries(
    {},
    optional={
        "url": st.sampled_from(["", "https://a.example/", "https://b.example/", 3]),
        "name": st.sampled_from(["", "N", None]),
        "title": st.sampled_from(["T", 7]),
        "snippet": st.sampled_from(["S", None]),
    },
)


@given(st.lists(ROW | st.just("not a row"), max_size=6))
def test_sources_equal_to_source_deduplicated_by_url(rows: list[Any]) -> None:
    expected: list[Any] = []
    seen: set[str] = set()
    for raw in rows:
        src = to_source(raw)
        if src is not None and src.url not in seen:
            seen.add(src.url)
            expected.append(src)
    got = sources(View(web_results={"web_results": rows}))
    assert [(s.url, s.title, s.snippet) for s in got] == [
        (s.url, s.title, s.snippet) for s in expected
    ]


def test_report_body_prefers_unified_assets() -> None:
    wf = {"steps": [{"items": [{"payload": {"sources_payload": report("wf body")}}]}]}
    assert report_body(View(unified_assets=report(" ua body "), workflow=wf)) == "ua body"
    assert report_body(View(unified_assets=report(" "), workflow=wf)) == "wf body"
    assert report_body(View()) == ""


def test_projection_missing_names_each_absent_read_path() -> None:
    view = View(ask_text={}, web_results={"web_results": "x"}, workflow={"steps": []})
    names = {d.name for d in projection_missing(view)}
    assert names == {name_of("ask_text/chunks"), name_of("web_results/web_results")}
    assert all(d.kind == "projection_missing" for d in projection_missing(view))
