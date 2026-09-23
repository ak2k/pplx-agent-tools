"""Unit tests for verbs/ask.py — copilot stream accumulation + render."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pplx_agent_tools.errors import (
    EXIT_GENERIC,
    EXIT_NETWORK,
    NetworkError,
    RateLimitError,
    SchemaError,
    StreamDeadlineError,
    exit_code,
)
from pplx_agent_tools.grounding import (
    Grounded,
    Grounding,
    Unchecked,
    Ungrounded,
    check_grounding,
)
from pplx_agent_tools.render import grounding_summary, render_ask_json, render_ask_text
from pplx_agent_tools.verbs._ask_common import (
    Source,
    apply_chunk_patch,
    extract_chunks_from_event,
    extract_web_results,
)
from pplx_agent_tools.verbs.ask import (
    SOURCES_FRAME_MISSING,
    AskCompletion,
    AskResult,
    Cut,
    Finished,
    FinishedWithoutSources,
    _build_ask_body,
    ask,
)

from ._doubles import _TestClientBase


def _chunk_event(text: str) -> dict[str, Any]:
    """A copilot SSE event carrying one markdown_block chunk."""
    return {
        "data": {
            "backend_uuid": "BU",
            "read_write_token": "RW",
            "blocks": [{"intended_usage": "ask_text", "markdown_block": {"chunks": [text]}}],
        }
    }


def _patch_event(offset: int | None, chunks: list[str], **data: Any) -> dict[str, Any]:
    """A copilot SSE event carrying `chunks` placed at chunk index `offset`."""
    block: dict[str, Any] = {"chunks": chunks}
    if offset is not None:
        block["chunk_starting_offset"] = offset
    return {
        "data": {
            "backend_uuid": "BU",
            "read_write_token": "RW",
            "blocks": [{"intended_usage": "ask_text", "markdown_block": block}],
            **data,
        }
    }


def _web_results_event(results: list[dict[str, Any]]) -> dict[str, Any]:
    """A copilot SSE event carrying the cited-sources block."""
    return {
        "data": {
            "blocks": [
                {"intended_usage": "web_results", "web_result_block": {"web_results": results}}
            ]
        }
    }


class _FakeClient(_TestClientBase):
    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        raise_deadline: bool = False,
        raise_network: bool = False,
    ) -> None:
        super().__init__()
        self._events = events
        self._raise_deadline = raise_deadline
        self._raise_network = raise_network
        self.deleted: list[tuple[str, str]] = []

    def sse_post(  # type: ignore[override]
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        is_progress: Callable[[dict[str, Any]], bool] | None = None,
        stall_window: Callable[[], float | None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        yield from self._events
        if self._raise_deadline:
            raise StreamDeadlineError("simulated deadline")
        if self._raise_network:
            raise NetworkError("simulated mid-stream connection reset")

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


def _complete() -> list[dict[str, Any]]:
    return [_chunk_event("Hello "), _chunk_event("world."), {"data": {"status": "COMPLETED"}}]


def test_ask_accumulates_and_cleans_up() -> None:
    client = _FakeClient(_complete())
    result = ask(client, "hi")
    assert result.answer == "Hello world."
    assert result.completion == Finished()
    assert result.model == "turbo"
    assert client.deleted == [("BU", "RW")]  # incognito thread cleaned up by default


def test_ask_keep_thread_skips_cleanup() -> None:
    client = _FakeClient(_complete())
    ask(client, "hi", keep_thread=True)
    assert client.deleted == []


def test_ask_extracts_sources_from_web_results_block() -> None:
    client = _FakeClient(
        [
            _web_results_event(
                [
                    {"url": "https://a", "name": "A", "snippet": "sa"},
                    {"url": "https://b", "name": "B"},
                    {"url": "https://a", "name": "dup"},  # duplicate URL
                    {"name": "no url — skipped"},
                ]
            ),
            _chunk_event("the answer"),
            {"data": {"status": "COMPLETED"}},
        ]
    )
    result = ask(client, "q")
    assert result.answer == "the answer"
    assert [s.url for s in result.sources] == ["https://a", "https://b"]  # deduped, ordered
    assert result.sources[0].title == "A"
    assert result.sources[0].snippet == "sa"


_MULTI_STEP = Path(__file__).parent / "fixtures" / "ask" / "multi-step-sources.events.jsonl"


def _multi_step_events() -> list[dict[str, Any]]:
    return [json.loads(line) for line in _MULTI_STEP.read_text().splitlines()]


def test_ask_sources_come_from_the_completed_frame_in_citation_order() -> None:
    """Each search step emits its own web_results block; the COMPLETED frame
    re-sends them reordered so that [n] is sources[n-1]. The early
    `text_completed` frame must not end the read before it."""
    result = ask(_FakeClient(_multi_step_events()), "q")
    assert [s.url for s in result.sources] == [
        f"https://{n}.example/p" for n in ("echo", "bravo", "alpha", "charlie", "delta")
    ]
    # The COMPLETED frame repaints every chunk from offset 0: placed, not appended.
    assert result.answer == "Price is $45 at shop A [1]; score 93 per B [2]."
    assert result.completion == Finished()
    assert result.warnings == []


def test_ask_stream_ending_at_text_completed_is_whole_but_warns() -> None:
    events = [e for e in _multi_step_events() if e["data"]["status"] != "COMPLETED"]
    result = ask(_FakeClient(events), "q")
    assert result.answer == "Price is $45 at shop A [1]; score 93 per B [2]."
    assert result.completion == FinishedWithoutSources()
    assert result.warnings == [SOURCES_FRAME_MISSING]
    j = render_ask_json(result)
    assert (j["stream_complete"], j["cut_by"], j["sources_complete"]) == (True, None, False)
    # The last search step's block is all that arrived.
    assert [s.url for s in result.sources] == ["https://delta.example/p", "https://echo.example/p"]


def test_terminal_repaint_drops_a_stale_tail() -> None:
    events = [
        _patch_event(0, ["A "]),
        _patch_event(1, ["B "]),
        _patch_event(2, ["C"], text_completed=True),
        _patch_event(0, ["A ", "B"], status="COMPLETED"),
    ]
    assert ask(_FakeClient(events), "q").answer == "A B"


def test_chunk_past_the_end_keeps_its_place() -> None:
    events = [
        _patch_event(3, ["D"]),
        _patch_event(0, ["A "]),
        _patch_event(1, ["B "]),
        _patch_event(2, ["C "]),
        {"data": {"status": "COMPLETED"}},
    ]
    assert ask(_FakeClient(events), "q").answer == "A B C D"


def test_midstream_patch_does_not_truncate() -> None:
    events = [
        _patch_event(0, ["A"]),
        _patch_event(1, ["B"]),
        _patch_event(2, ["C"]),
        _patch_event(0, ["a"]),
        {"data": {"status": "COMPLETED"}},
    ]
    assert ask(_FakeClient(events), "q").answer == "aBC"


def test_terminal_repaint_without_offset_is_not_appended() -> None:
    events = [
        _patch_event(None, ["A "]),
        _patch_event(None, ["B"]),
        _patch_event(None, ["A ", "B"], status="COMPLETED"),
    ]
    assert ask(_FakeClient(events), "q").answer == "A B"


def test_empty_terminal_repaint_keeps_the_answer() -> None:
    events = [_patch_event(0, ["A"]), _patch_event(0, [], status="COMPLETED")]
    assert ask(_FakeClient(events), "q").answer == "A"


@given(st.lists(st.text(min_size=1, max_size=3), min_size=1, max_size=12), st.randoms())
def test_offset_runs_in_any_order_rebuild_the_answer(parts: list[str], rnd: Any) -> None:
    """Mid-stream runs may arrive in any order; the terminal repaint then
    matches whatever they built, never duplicating or dropping a chunk."""
    runs = [(i, parts[i : i + 2]) for i in range(0, len(parts), 2)]
    rnd.shuffle(runs)
    chunks: dict[int, str] = {}
    for offset, run in runs:
        apply_chunk_patch(chunks, offset, run, terminal=False)
    assert [chunks[i] for i in sorted(chunks)] == parts
    apply_chunk_patch(chunks, 0, parts[: len(parts) // 2 or 1], terminal=True)
    assert [chunks[i] for i in sorted(chunks)] == parts[: len(parts) // 2 or 1]


def test_network_error_after_text_completed_keeps_the_whole_answer() -> None:
    """Only the sources frame was outstanding, so the answer is returned as
    complete with the citation-order warning, as when the settle window runs out."""
    events = [e for e in _multi_step_events() if e["data"]["status"] != "COMPLETED"]
    client = _FakeClient(events, raise_network=True)
    result = ask(client, "q")
    assert result.answer == "Price is $45 at shop A [1]; score 93 per B [2]."
    assert result.completion == FinishedWithoutSources()
    assert result.warnings == [SOURCES_FRAME_MISSING]
    assert client.deleted == [("BU", "RW")]


def test_ask_attaches_grounding_verdict() -> None:
    events = [
        _web_results_event(
            [{"url": "https://www.vivino.com/", "name": "Vivino", "snippet": "Wine"}]
        ),
        _chunk_event("It is rated 4.2 by 3,100 users [1]."),
        {"data": {"status": "COMPLETED"}},
    ]
    result = ask(_FakeClient(events), "q")
    assert isinstance(result.grounding, Ungrounded)
    assert result.grounding.ungrounded_terms == ("4.2", "3,100")
    disabled = ask(_FakeClient(events), "q", grounded_check=False).grounding
    assert disabled == Unchecked("disabled")


def test_ask_partial_on_deadline() -> None:
    client = _FakeClient([_chunk_event("partial")], raise_deadline=True)
    result = ask(client, "hi", timeout=30)
    assert result.completion == Cut("deadline")
    assert result.answer == "partial"


def test_ask_no_content_no_completion_raises() -> None:
    client = _FakeClient([{"data": {"foo": "bar"}}])
    with pytest.raises(SchemaError):
        ask(client, "hi")


def test_ask_failed_status_reaps_thread_then_raises() -> None:
    # FAILED frame carries the thread IDs — cleanup must still run (no leak) even
    # though ask() then raises. (Regression: cleanup used to be skipped by the raise.)
    client = _FakeClient(
        [{"data": {"backend_uuid": "BU", "read_write_token": "RW", "status": "FAILED"}}]
    )
    with pytest.raises(SchemaError, match="FAILED"):
        ask(client, "hi", model="bogus_model")
    assert client.deleted == [("BU", "RW")]


def test_ask_deadline_before_any_content_raises() -> None:
    client = _FakeClient([], raise_deadline=True)
    with pytest.raises(StreamDeadlineError):
        ask(client, "hi", timeout=30)


def test_ask_midstream_network_error_reaps_thread_then_raises() -> None:
    """The thread ids arrived before the transport died, so the incognito thread
    exists and only this process knows how to delete it."""
    client = _FakeClient([_chunk_event("partial")], raise_network=True)

    with pytest.raises(NetworkError) as excinfo:
        ask(client, "hi")

    assert not isinstance(excinfo.value, StreamDeadlineError)
    assert exit_code(excinfo.value) == EXIT_NETWORK
    assert client.deleted == [("BU", "RW")]


def test_ask_deadline_and_closed_empty_texts_are_distinguishable() -> None:
    """Both end with no answer, but only the deadline is worth retrying with a
    larger --timeout, so the text and the exit code have to say which it was."""
    starved = _FakeClient([], raise_deadline=True)
    with pytest.raises(StreamDeadlineError) as deadline:
        ask(starved, "hi", timeout=30)

    empty = _FakeClient([{"data": {"foo": "bar"}}])
    with pytest.raises(SchemaError) as closed:
        ask(empty, "hi")

    assert str(deadline.value) == (
        "ask stream on /rest/sse/perplexity_ask exceeded 30.0s deadline "
        "before the first content arrived"
    )
    assert str(closed.value) == "ask stream on /rest/sse/perplexity_ask closed with no content"
    assert exit_code(deadline.value) == EXIT_NETWORK
    assert exit_code(closed.value) == EXIT_GENERIC


def test_ask_model_passthrough_to_body() -> None:
    body = _build_ask_body("q", "claude48opusthinking")
    assert body["params"]["model_preference"] == "claude48opusthinking"
    assert body["params"]["mode"] == "copilot"
    assert body["params"]["is_incognito"] is True


def test_ask_family_bodies_share_one_base() -> None:
    """ask/research/fetch --prompt bodies all delegate to base_ask_params, so a
    field added to one without the others is a bug. Lock the shared key set."""
    from pplx_agent_tools.verbs.fetch import _build_chat_body
    from pplx_agent_tools.verbs.research import _build_research_body

    ask_p = _build_ask_body("q", "turbo")["params"]
    research_p = _build_research_body("q", "pplx_alpha")["params"]
    fetch_p = _build_chat_body("q")["params"]
    base_keys = set(ask_p) - {"compare_model_preferences"}
    assert set(research_p) == base_keys
    assert set(fetch_p) == base_keys
    for k in ("mode", "search_focus", "sources", "is_incognito", "use_schematized_api"):
        assert ask_p[k] == research_p[k] == fetch_p[k]


def test_render_ask_text_and_json() -> None:
    result = AskResult(query="q", answer="The answer.", model="turbo")
    assert render_ask_text(result) == "The answer."
    j = render_ask_json(result)
    assert j["_verb"] == "ask"
    assert j["model"] == "turbo"
    assert j["answer"] == "The answer."
    assert j["stream_complete"] is True


def test_render_ask_text_incomplete_marker() -> None:
    result = AskResult("q", "partial", "turbo", Cut("server"))
    assert "stream: incomplete" in render_ask_text(result)


_COMPLETIONS: list[tuple[AskCompletion, bool, str | None, bool, str | None]] = [
    (Finished(), True, None, True, None),
    (FinishedWithoutSources(), True, None, False, None),
    (Cut("stall"), False, "stall", False, "stream: incomplete (stall: no new content)"),
    (Cut("deadline"), False, "deadline", False, "stream: incomplete (deadline)"),
    (Cut("server"), False, None, False, "stream: incomplete (server cut)"),
]


@pytest.mark.parametrize(
    ("completion", "stream_complete", "cut_by", "sources_complete", "marker"), _COMPLETIONS
)
def test_render_ask_completion_states(
    completion: AskCompletion,
    stream_complete: bool,
    cut_by: str | None,
    sources_complete: bool,
    marker: str | None,
) -> None:
    """`stream_complete` and `cut_by` keep their shape; `sources_complete`
    alone tells a clean finish from one whose sources frame never arrived."""
    result = AskResult("q", "A", "turbo", completion)
    j = render_ask_json(result)
    assert (j["stream_complete"], j["cut_by"], j["sources_complete"]) == (
        stream_complete,
        cut_by,
        sources_complete,
    )
    lines = render_ask_text(result).splitlines()
    assert [ln for ln in lines if ln.startswith("stream:")] == ([marker] if marker else [])


def test_completion_table_covers_every_variant() -> None:
    assert {type(c) for c, *_ in _COMPLETIONS} == {Finished, FinishedWithoutSources, Cut}


def test_render_ask_with_sources() -> None:
    from pplx_agent_tools.verbs._ask_common import Source

    result = AskResult("q", "Answer.", "turbo", Finished(), [Source("https://a", "A", "snip")])
    out = render_ask_text(result)
    assert "— sources (1) —" in out and "[1] A" in out and "https://a" in out
    j = render_ask_json(result)
    assert j["sources"][0] == {"url": "https://a", "title": "A", "snippet": "snip"}


def _src(url: str, snippet: str) -> list[Source]:
    return [Source(url, "t", snippet)]


_LOW_SUPPORT = "figures/names appear in a cited source's title or snippet"
# Every variant (and every ungrounded reason) with the exact JSON it renders to.
_VARIANTS: list[tuple[str, Grounding, dict[str, Any]]] = [
    (
        "grounded",
        check_grounding("It costs $45, $60 and $75.", "q", _src("https://a.test/p", "$45, $60")),
        {
            "grounded": True,
            "grounding_reasons": [],
            "ungrounded_terms": ["$75"],
            "checked_terms": 3,
        },
    ),
    (
        "no_sources",
        check_grounding("It sold 4,500 units in 2023.", "q", []),
        {
            "grounded": False,
            "grounding_reasons": ["no sources"],
            "ungrounded_terms": ["4,500", "2023"],
            "checked_terms": 2,
        },
    ),
    (
        "site_roots",
        check_grounding("It is 4.0.", "q", _src("https://a.test/", "rated 4.0")),
        {
            "grounded": False,
            "grounding_reasons": ["every cited URL is a site root"],
            "ungrounded_terms": [],
            "checked_terms": 1,
        },
    ),
    (
        "low_support",
        check_grounding(
            "Figures: 101, 202, 303, 404 and 505.", "q", _src("https://a.test/p", "101")
        ),
        {
            "grounded": False,
            "grounding_reasons": [f"1 of 5 {_LOW_SUPPORT}"],
            "ungrounded_terms": ["202", "303", "404", "505"],
            "checked_terms": 5,
        },
    ),
    (
        "site_roots+low_support",
        check_grounding("It is 4.0 and costs $55.", "q", _src("https://a.test/", "s")),
        {
            "grounded": False,
            "grounding_reasons": ["every cited URL is a site root", f"0 of 2 {_LOW_SUPPORT}"],
            "ungrounded_terms": ["4.0", "$55"],
            "checked_terms": 2,
        },
    ),
    (
        "no_checkable_terms",
        check_grounding("Yes.", "q", []),
        {
            "grounded": None,
            "grounding_reasons": ["no checkable figures or names"],
            "ungrounded_terms": [],
            "checked_terms": 0,
        },
    ),
    (
        "disabled",
        Unchecked("disabled"),
        {
            "grounded": None,
            "grounding_reasons": ["check disabled"],
            "ungrounded_terms": [],
            "checked_terms": 0,
        },
    ),
]


@pytest.mark.parametrize(
    ("name", "grounding", "expected"), _VARIANTS, ids=[v[0] for v in _VARIANTS]
)
def test_render_ask_grounding_variants(
    name: str, grounding: Grounding, expected: dict[str, Any]
) -> None:
    result = AskResult("q", "Answer.", "turbo", grounding=grounding)
    j = render_ask_json(result)
    assert {k: j[k] for k in expected} == expected
    marker = [ln for ln in render_ask_text(result).splitlines() if ln.startswith("grounded:")]
    if isinstance(grounding, Ungrounded):
        reasons = "; ".join(expected["grounding_reasons"])
        terms = ", ".join(expected["ungrounded_terms"])
        assert marker == [
            f"grounded: no ({reasons}; unsupported: {terms})"
            if terms
            else f"grounded: no ({reasons})"
        ]
    else:
        assert marker == []


def test_variant_table_covers_every_variant() -> None:
    assert {type(g) for _, g, _ in _VARIANTS} == {Grounded, Ungrounded, Unchecked}


def test_render_ask_json_check_disabled_is_the_default() -> None:
    j = render_ask_json(AskResult("q", "Answer.", "turbo"))
    assert j["grounded"] is None
    assert j["grounding_reasons"] == ["check disabled"]


def test_grounding_summary_caps_listed_terms() -> None:
    g = check_grounding(" ".join(f"x {n}," for n in range(100, 112)), "q", [])
    assert isinstance(g, Ungrounded)
    assert grounding_summary(g).endswith("106, 107 (+4 more)")


# ---------- run_ask_stream 429 retry/exhaustion (exercised via ask) ----------


class _RateLimitClient(_TestClientBase):
    """Raises RateLimitError on the first `fail_times` sse_post calls (retry_after=0
    so backoff is instant), then yields normal complete events."""

    def __init__(self, fail_times: int, events: list[dict[str, Any]]) -> None:
        super().__init__()
        self._fail_times = fail_times
        self._events = events
        self._calls = 0
        self.deleted: list[tuple[str, str]] = []

    def sse_post(  # type: ignore[override]
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        is_progress: Callable[[dict[str, Any]], bool] | None = None,
        stall_window: Callable[[], float | None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RateLimitError("429", retry_after=0.0)
        yield from self._events

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


def test_ask_retries_on_rate_limit_then_succeeds() -> None:
    client = _RateLimitClient(2, _complete())  # fail twice, succeed on the 3rd attempt
    result = ask(client, "hi")
    assert result.answer == "Hello world."
    assert result.completion == Finished()


def test_ask_rate_limit_exhausted_reraises() -> None:
    client = _RateLimitClient(3, _complete())  # fail all 3 attempts
    with pytest.raises(RateLimitError):
        ask(client, "hi")


# ---------- block extractors stay total ----------


@pytest.mark.parametrize("blocks", [True, 5, 3.5])
def test_block_extractors_survive_truthy_non_list_blocks(blocks: object) -> None:
    """`blocks` is server-supplied, and `... or []` only absorbs the falsy shapes:
    a truthy scalar reached the `for` and raised TypeError, which both extractors
    document as impossible and every caller relies on mid-stream."""
    event: dict[str, Any] = {"event": None, "data": {"blocks": blocks}}
    assert extract_chunks_from_event(event) == []
    assert extract_web_results(event) == []
