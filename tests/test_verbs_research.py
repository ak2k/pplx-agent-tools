"""Unit tests for verbs/research.py — schematized block decode + render + orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pplx_agent_tools.errors import (
    EXIT_GENERIC,
    EXIT_NETWORK,
    NetworkError,
    SchemaError,
    StreamDeadlineError,
    exit_code,
)
from pplx_agent_tools.render import render_research_json, render_research_text
from pplx_agent_tools.verbs.research import (
    ResearchResult,
    ResearchSource,
    _build_research_body,
    _model_for_mode,
    decode_research_text,
    research,
)

from ._doubles import _TestClientBase

# JSON-like recursive values, mirroring tests/test_fuzz_robustness.py — bounded
# leaves keep individual inputs small (breadth of shape, not size).
_json_leaf = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.text(max_size=30),
)
_json_value = st.recursive(
    _json_leaf,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=15), children, max_size=5),
    ),
    max_leaves=15,
)


def _snapshot(answer: str = "QUIC is a protocol. [1]") -> str:
    """A research `text` snapshot: block list with a JSON-wrapped FINAL answer."""
    blocks = [
        {"step_type": "INITIAL_QUERY", "content": {"query": "q"}},
        {
            "step_type": "SEARCH_RESULTS",
            "content": {"web_results": [{"url": "https://mid", "name": "mid"}]},
        },
        {
            "step_type": "FINAL",
            "content": {
                "answer": json.dumps(
                    {"answer": answer, "web_results": [{"url": "https://cited", "name": "Cited"}]}
                )
            },
        },
    ]
    return json.dumps(blocks)


class _FakeClient(_TestClientBase):
    """Yields canned research SSE events; records delete_thread calls."""

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
        self, path: str, body: dict[str, Any], *, max_total_seconds: float | None = None
    ) -> Iterator[dict[str, Any]]:
        yield from self._events
        if self._raise_deadline:
            raise StreamDeadlineError("simulated deadline")
        if self._raise_network:
            raise NetworkError("simulated mid-stream connection reset")

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


BLOCKS = [
    {"step_type": "INITIAL_QUERY", "content": {"goal_id": None, "query": "q"}, "uuid": "1"},
    {
        "step_type": "SEARCH_WEB",
        "content": {"goal_id": "0", "queries": [{"query": "q"}]},
        "uuid": "2",
    },
    {
        "step_type": "SEARCH_RESULTS",
        "content": {
            "goal_id": "0",
            "web_results": [
                {"url": "https://a", "name": "A", "snippet": "sa"},
                {"url": "https://b", "name": "B"},
            ],
        },
        "uuid": "3",
    },
    {
        "step_type": "SEARCH_RESULTS",
        "content": {
            "goal_id": "1",
            "web_results": [
                {"url": "https://a", "name": "A dup"},  # duplicate URL
                {"url": "https://c", "title": "C"},  # title (not name)
                {"name": "no url — skipped"},
            ],
        },
        "uuid": "4",
    },
    {
        "step_type": "FINAL",
        "content": {"goal_id": None, "answer": "# Answer\nbody [1][2]"},
        "uuid": "5",
    },
]


def test_decode_answer_and_sources() -> None:
    answer, sources = decode_research_text(json.dumps(BLOCKS))
    assert answer == "# Answer\nbody [1][2]"
    assert [s.url for s in sources] == ["https://a", "https://b", "https://c"]  # deduped, ordered
    assert sources[0].title == "A"
    assert sources[0].snippet == "sa"
    assert sources[2].title == "C"  # picked up from "title" key


def test_decode_unwraps_json_wrapped_final_answer() -> None:
    """The real FINAL shape: content.answer is a JSON string wrapping
    {answer: markdown, web_results: [...cited]}. Cited web_results win over
    the intermediate SEARCH_RESULTS rounds."""
    blocks = [
        {
            "step_type": "SEARCH_RESULTS",
            "content": {"web_results": [{"url": "https://intermediate", "name": "mid"}]},
        },
        {
            "step_type": "FINAL",
            "content": {
                "answer": json.dumps(
                    {
                        "answer": "QUIC is a transport protocol. [1]",
                        "web_results": [{"url": "https://cited", "name": "Cited", "snippet": "s"}],
                    }
                )
            },
        },
    ]
    answer, sources = decode_research_text(json.dumps(blocks))
    assert answer == "QUIC is a transport protocol. [1]"  # unwrapped markdown, not the JSON blob
    assert [s.url for s in sources] == ["https://cited"]  # FINAL cited set, not intermediate


def test_decode_multiple_final_concatenated() -> None:
    blocks = [
        {"step_type": "FINAL", "content": {"answer": "part one"}},
        {"step_type": "FINAL", "content": {"answer": "part two"}},
    ]
    answer, _ = decode_research_text(json.dumps(blocks))
    assert answer == "part one\n\npart two"


def test_decode_ignores_unknown_steps_and_bad_blocks() -> None:
    blocks = ["not a dict", {"step_type": "MYSTERY", "content": {"x": 1}}, {"no": "content"}]
    answer, sources = decode_research_text(json.dumps(blocks))
    assert answer == ""
    assert sources == []


def test_decode_non_json_raises() -> None:
    with pytest.raises(SchemaError):
        decode_research_text("{not json")


def test_decode_non_list_raises() -> None:
    with pytest.raises(SchemaError):
        decode_research_text(json.dumps({"step_type": "FINAL"}))


def test_render_text_has_answer_and_sources() -> None:
    result = ResearchResult(
        query="q",
        answer="The answer.",
        sources=[ResearchSource("https://a", "A", None), ResearchSource("https://b", None, None)],
        mode="research",
    )
    out = render_research_text(result)
    assert "The answer." in out
    assert "— sources (2) —" in out
    assert "[1] A" in out and "https://a" in out
    assert "[2] https://b" in out  # no title → url as label


def test_render_text_incomplete_marker() -> None:
    result = ResearchResult("q", "partial", [], "research", stream_complete=False)
    assert "stream: incomplete" in render_research_text(result)


def test_render_json_envelope() -> None:
    result = ResearchResult(
        query="q",
        answer="A",
        sources=[ResearchSource("https://a", "A", "snip")],
        mode="research",
    )
    j = render_research_json(result)
    assert j["_verb"] == "research"
    assert j["mode"] == "research"
    assert j["answer"] == "A"
    assert j["sources"][0] == {"url": "https://a", "title": "A", "snippet": "snip"}
    assert j["stream_complete"] is True


# ---------- orchestration (research()) with a fake SSE client ----------


def _complete_events() -> list[dict[str, Any]]:
    return [
        {"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": _snapshot()}},
        {"data": {"text": _snapshot(), "status": "COMPLETED"}},
    ]


def test_research_completes_parses_and_cleans_up() -> None:
    client = _FakeClient(_complete_events())
    result = research(client, "what is quic")
    assert result.stream_complete is True
    assert result.answer == "QUIC is a protocol. [1]"  # unwrapped markdown
    assert [s.url for s in result.sources] == ["https://cited"]  # FINAL cited set
    assert result.mode == "research"
    assert client.deleted == [("BU", "RW")]  # incognito thread cleaned up by default


def test_research_keep_thread_skips_cleanup() -> None:
    client = _FakeClient(_complete_events())
    research(client, "q", keep_thread=True)
    assert client.deleted == []


def test_research_partial_on_deadline_returns_incomplete() -> None:
    # One snapshot arrives, then the stream trips the overall deadline.
    client = _FakeClient(
        [{"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": _snapshot()}}],
        raise_deadline=True,
    )
    result = research(client, "q", timeout=30)
    assert result.stream_complete is False  # never saw COMPLETED
    assert result.answer == "QUIC is a protocol. [1]"  # partial answer still returned
    assert client.deleted == [("BU", "RW")]


def test_research_deadline_before_any_content_raises() -> None:
    client = _FakeClient([], raise_deadline=True)
    with pytest.raises(StreamDeadlineError):
        research(client, "q", timeout=30)


def test_research_midstream_network_error_reaps_thread_then_raises() -> None:
    """A research run is minutes long, so a transport failure part-way through
    is a likely exit — and the thread ids are already in hand when it happens."""
    client = _FakeClient(
        [{"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": _snapshot()}}],
        raise_network=True,
    )

    with pytest.raises(NetworkError) as excinfo:
        research(client, "q")

    assert not isinstance(excinfo.value, StreamDeadlineError)
    assert exit_code(excinfo.value) == EXIT_NETWORK
    assert client.deleted == [("BU", "RW")]


def test_research_deadline_and_closed_empty_texts_are_distinguishable() -> None:
    """Same wording as ask and fetch --prompt: the deadline says retry longer,
    the closed-empty stream says do not."""
    starved = _FakeClient([], raise_deadline=True)
    with pytest.raises(StreamDeadlineError) as deadline:
        research(starved, "q", timeout=30)

    empty = _FakeClient([{"data": {"status": "COMPLETED"}}])
    with pytest.raises(SchemaError) as closed:
        research(empty, "q")

    assert str(deadline.value) == (
        "research stream on /rest/sse/perplexity_ask exceeded 30.0s deadline "
        "before the first content arrived"
    )
    assert str(closed.value) == (
        "research stream on /rest/sse/perplexity_ask closed with no content"
    )
    assert exit_code(deadline.value) == EXIT_NETWORK
    assert exit_code(closed.value) == EXIT_GENERIC


def test_research_completed_without_text_raises_schema() -> None:
    client = _FakeClient([{"data": {"status": "COMPLETED"}}])
    with pytest.raises(SchemaError):
        research(client, "q")


def test_research_failed_status_raises_clear_error() -> None:
    # Real shape when model_preference is incompatible with the mode: a single
    # frame with text present but status=FAILED + mode dropped to CONCISE.
    client = _FakeClient(
        [
            {
                "data": {
                    "backend_uuid": "BU",
                    "read_write_token": "RW",
                    "text": "{}",
                    "status": "FAILED",
                    "mode": "CONCISE",
                    "text_completed": False,
                }
            }
        ]
    )
    with pytest.raises(SchemaError, match="FAILED"):
        research(client, "q", mode="research")
    assert client.deleted == [("BU", "RW")]  # thread reaped even on FAILED (no leak)


def test_model_for_mode_maps_to_driving_model() -> None:
    # model_preference (not params.mode) is the real selector.
    assert _model_for_mode("research") == "pplx_alpha"
    assert _model_for_mode("agentic_research") == "pplx_agentic_research"
    assert _model_for_mode("council") == "pplx_agentic_research"  # friendly alias
    assert _model_for_mode("pplx_asi") == "pplx_asi"  # unknown → literal passthrough


def test_build_body_drives_via_model_preference() -> None:
    body = _build_research_body("q", _model_for_mode("research"))
    assert body["params"]["model_preference"] == "pplx_alpha"  # the deep-research model
    assert body["params"]["mode"] == "copilot"  # coarse; server derives mode from the model
    assert body["params"]["is_incognito"] is True
    assert "compare_model_preferences" not in body["params"]  # only set for council


def test_build_body_council_models() -> None:
    body = _build_research_body(
        "q", "pplx_agentic_research", council_models=["gpt55_thinking", "claude48opusthinking"]
    )
    assert body["params"]["compare_model_preferences"] == ["gpt55_thinking", "claude48opusthinking"]


def test_research_model_override_bypasses_mode_mapping() -> None:
    captured: dict[str, Any] = {}

    class _Cap(_FakeClient):
        def sse_post(self, path: str, body: dict[str, Any], *, max_total_seconds=None):  # type: ignore[override]
            captured["mp"] = body["params"]["model_preference"]
            return iter(self._events)

    research(_Cap(_complete_events()), "q", mode="research", model="o4mini")
    assert captured["mp"] == "o4mini"  # --model wins over the mode default (pplx_alpha)


def test_research_council_auto_sends_default_trio() -> None:
    captured: dict[str, Any] = {}

    class _Cap(_FakeClient):
        def sse_post(self, path: str, body: dict[str, Any], *, max_total_seconds=None):  # type: ignore[override]
            captured["mp"] = body["params"]["model_preference"]
            captured["compare"] = body["params"].get("compare_model_preferences")
            return iter(self._events)

    research(_Cap(_complete_events()), "q", mode="council")
    assert captured["mp"] == "pplx_agentic_research"
    # council STALLS without compare_model_preferences, so the verb defaults the trio.
    assert captured["compare"] == ["gpt55_thinking", "claude48opusthinking", "gemini31pro_high"]


def test_research_council_explicit_models_override_default() -> None:
    captured: dict[str, Any] = {}

    class _Cap(_FakeClient):
        def sse_post(self, path: str, body: dict[str, Any], *, max_total_seconds=None):  # type: ignore[override]
            captured["compare"] = body["params"].get("compare_model_preferences")
            return iter(self._events)

    research(_Cap(_complete_events()), "q", mode="council", council_models=["m1", "m2", "m3"])
    assert captured["compare"] == ["m1", "m2", "m3"]


def test_research_passes_model_preference_into_body() -> None:
    captured: dict[str, Any] = {}

    class _BodyCapture(_FakeClient):
        def sse_post(self, path: str, body: dict[str, Any], *, max_total_seconds=None):  # type: ignore[override]
            captured["model_preference"] = body["params"]["model_preference"]
            captured["is_incognito"] = body["params"]["is_incognito"]
            return iter(self._events)

    client = _BodyCapture(_complete_events())
    research(client, "q", mode="agentic_research")
    assert captured["model_preference"] == "pplx_agentic_research"
    assert captured["is_incognito"] is True


# ---------- report body (RESEARCH_ANSWER asset) ----------


def _report_blocks(cover: str, body: str, *, inline: str = "") -> str:
    """The real deep-research shape: a RESEARCH_ANSWER block whose report body is
    an asset, plus a FINAL block holding only the cover note."""
    blocks = [
        {
            "step_type": "RESEARCH_ANSWER",
            "content": {"goal_id": "8", "answer": inline, "title": "T", "url": "https://asset"},
            "assets": [
                {
                    "asset_type": "RESEARCH_REPORT",
                    "research_report": {"name": "T", "source_content": body},
                }
            ],
            "uuid": "9",
        },
        {"step_type": "FINAL", "content": {"answer": cover}, "uuid": "10"},
    ]
    return json.dumps(blocks)


def test_decode_includes_report_body_after_cover_note() -> None:
    """The FINAL block only carries a cover note; dropping the RESEARCH_ANSWER
    asset is what made `research` return a summary that described a missing
    report."""
    answer, _ = decode_research_text(
        _report_blocks("I compiled a report. The full report includes tables.", "# Report\nbody")
    )
    assert answer == "I compiled a report. The full report includes tables.\n\n# Report\nbody"


def test_decode_falls_back_to_inline_research_answer() -> None:
    answer, _ = decode_research_text(_report_blocks("cover", "", inline="# Inline body"))
    assert answer == "cover\n\n# Inline body"


def test_decode_does_not_duplicate_body_quoted_in_cover() -> None:
    answer, _ = decode_research_text(_report_blocks("cover: # Report\nbody", "# Report\nbody"))
    assert answer == "cover: # Report\nbody"


def test_decode_partial_research_answer_without_body() -> None:
    blocks = [{"step_type": "RESEARCH_ANSWER", "content": {"answer": ""}, "assets": []}]
    answer, _ = decode_research_text(json.dumps(blocks))
    assert answer == ""


# ---------- completion predicate + shortfall flag ----------


def test_research_reads_past_text_completed_to_the_repaint() -> None:
    """`text_completed` fires before the terminal COMPLETED repaint. Research
    keeps whole snapshots, so it must consume the repaint, not stop early."""
    events = [
        {"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": _snapshot("early")}},
        {"data": {"text": _snapshot("early"), "text_completed": True}},
        {"data": {"text": _snapshot("final repaint"), "status": "COMPLETED"}},
    ]
    result = research(_FakeClient(events), "q")
    assert result.answer == "final repaint"
    assert result.stream_complete is True
    assert result.content_shortfall is False


def test_research_flags_content_shortfall_when_final_snapshot_shrinks() -> None:
    long_answer = "x" * 500
    events = [
        {
            "data": {
                "backend_uuid": "BU",
                "read_write_token": "RW",
                "text": _snapshot(long_answer),
            }
        },
        {"data": {"text": _snapshot("tiny"), "status": "COMPLETED"}},
    ]
    result = research(_FakeClient(events), "q")
    # The stream DID complete — the honest signal is a separate flag, not a lie
    # about stream_complete.
    assert result.stream_complete is True
    assert result.content_shortfall is True
    assert result.answer == "tiny"  # the newest snapshot is still what we return
    assert result.warnings and "truncated" in result.warnings[0]


def test_research_no_shortfall_when_snapshots_grow() -> None:
    result = research(_FakeClient(_complete_events()), "q")
    assert result.content_shortfall is False
    assert result.warnings == []


def test_render_text_content_shortfall_marker() -> None:
    result = ResearchResult("q", "short", [], "research", content_shortfall=True)
    assert "content: may be incomplete" in render_research_text(result)


def test_render_json_reports_content_shortfall() -> None:
    result = ResearchResult("q", "short", [], "research", content_shortfall=True)
    assert render_research_json(result)["content_shortfall"] is True
    assert render_research_json(ResearchResult("q", "a", [], "research"))["content_shortfall"] is (
        False
    )


def test_research_incomplete_when_stream_ends_at_text_completed() -> None:
    """A stream that stops at `text_completed` never sent the repaint research
    waits for, so its answer is partial and must say so. The shared default
    predicate treated this same stream as a clean finish (exit 0)."""
    client = _FakeClient(
        [
            {
                "data": {
                    "backend_uuid": "BU",
                    "read_write_token": "RW",
                    "text": _snapshot("partial"),
                }
            },
            {"data": {"text": _snapshot("partial"), "text_completed": True}},
        ]
    )
    result = research(client, "q")

    assert result.stream_complete is False
    assert result.answer == "partial"
    assert result.content_shortfall is False
    assert client.deleted == [("BU", "RW")], "the incognito thread is still cleaned up"


def test_decode_research_answer_survives_null_content() -> None:
    """The report body is an ASSET, so a RESEARCH_ANSWER block with a null
    `content` still carries a full report. Running the `content` isinstance guard
    ahead of the step dispatch dropped it — the shipped bug, reproduced."""
    blocks = [
        {
            "step_type": "RESEARCH_ANSWER",
            "content": None,
            "assets": [{"research_report": {"source_content": "# Report\nbody"}}],
        },
        {"step_type": "FINAL", "content": {"answer": "cover"}},
    ]
    answer, _ = decode_research_text(json.dumps(blocks))
    assert answer == "cover\n\n# Report\nbody"


def test_research_shortfall_survives_an_unparseable_longest_frame() -> None:
    """A snapshot that fails to decode is skipped by the shortfall tracker: it
    says nothing about the kept snapshot and must not sink the run, even though
    its raw text is by far the largest in the stream."""
    events = [
        {"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": "not json " * 600}},
        {"data": {"text": _snapshot("the real answer"), "status": "COMPLETED"}},
    ]
    result = research(_FakeClient(events), "q")

    assert result.answer == "the real answer"
    assert result.content_shortfall is False
    assert result.warnings == []


def _padded_snapshot(answer: str, *, pad_results: int) -> str:
    """A snapshot whose raw size is inflated by SEARCH_RESULTS metadata.

    Models the production repaint: the COMPLETED frame carries more sources and
    envelope fields than earlier frames, so its RAW length grows even when its
    report body shrinks."""
    blocks = [
        {
            "step_type": "SEARCH_RESULTS",
            "content": {
                "web_results": [
                    {"url": f"https://pad/{i}", "name": f"pad {i}", "snippet": "s" * 40}
                    for i in range(pad_results)
                ]
            },
        },
        {
            "step_type": "FINAL",
            "content": {
                "answer": json.dumps(
                    {"answer": answer, "web_results": [{"url": "https://cited", "name": "Cited"}]}
                )
            },
        },
    ]
    return json.dumps(blocks)


def test_research_flags_shortfall_when_the_repaint_grows_in_metadata() -> None:
    """The terminal repaint can carry MORE raw bytes than earlier frames (extra
    SEARCH_RESULTS, envelope fields) while its report body shrinks or vanishes.
    Judging by raw frame size never compared those two, so the drop shipped as
    exit 0."""
    early = _padded_snapshot("x" * 500, pad_results=0)
    repaint = _padded_snapshot("tiny", pad_results=12)
    assert len(repaint) > len(early), "the premise: the repaint is raw-larger, body-smaller"

    events = [
        {"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": early}},
        {"data": {"text": repaint, "status": "COMPLETED"}},
    ]
    result = research(_FakeClient(events), "q")

    assert result.stream_complete is True
    assert result.answer == "tiny", "the latest snapshot is still what we return"
    assert result.content_shortfall is True
    assert result.warnings and "truncated" in result.warnings[0]


def test_research_flags_body_loss_masked_by_a_growing_cover_note() -> None:
    """The cover note and the report body live in one decoded answer, so a
    repaint that GROWS the cover while LOSING report body can keep the total at
    or above the maximum seen. The report is the body; judge on it."""
    early = _report_blocks("", "x" * 10_000)
    repaint = _report_blocks("c" * 1_200, "y" * 9_500)
    assert len(repaint) > len(early), "the premise: the repaint is total-larger, body-smaller"

    events = [
        {"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": early}},
        {"data": {"text": repaint, "status": "COMPLETED"}},
    ]
    result = research(_FakeClient(events), "q")

    assert result.stream_complete is True
    assert result.content_shortfall is True
    assert result.warnings and "truncated" in result.warnings[0]
    assert "9500" in result.warnings[0] and "10000" in result.warnings[0]


def test_research_no_shortfall_when_only_the_cover_note_shrinks() -> None:
    """A shorter cover note over an intact report is not a truncated report."""
    body = "b" * 5_000
    events = [
        {
            "data": {
                "backend_uuid": "BU",
                "read_write_token": "RW",
                "text": _report_blocks("c" * 1_200, body),
            }
        },
        {"data": {"text": _report_blocks("c" * 10, body), "status": "COMPLETED"}},
    ]
    result = research(_FakeClient(events), "q")

    assert result.content_shortfall is False
    assert result.warnings == []


def test_research_keeps_the_last_parseable_snapshot_when_the_repaint_is_garbage() -> None:
    """A malformed terminal repaint used to overwrite a perfectly good snapshot
    and blow up the whole ~2-minute run with SchemaError."""
    events = [
        {
            "data": {
                "backend_uuid": "BU",
                "read_write_token": "RW",
                "text": _snapshot("the real answer"),
            }
        },
        {"data": {"status": "COMPLETED", "text": "not json"}},
    ]
    result = research(_FakeClient(events), "q")

    assert result.answer == "the real answer"
    assert result.content_shortfall is True
    assert result.warnings and "decode" in result.warnings[0]


def test_research_raises_when_no_frame_ever_parsed() -> None:
    events = [
        {"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": "not json"}},
        {"data": {"text": "still not json", "status": "COMPLETED"}},
    ]
    with pytest.raises(SchemaError):
        research(_FakeClient(events), "q")


def test_research_no_shortfall_when_the_cover_note_quotes_the_whole_body() -> None:
    """The join drops a body already quoted in the cover note, but the report is
    still fully there — measuring the post-dedupe parts read that complete answer
    as a body of zero and flagged it."""
    body = "# Report\n" + "b" * 5_000
    events = [
        {
            "data": {
                "backend_uuid": "BU",
                "read_write_token": "RW",
                "text": _report_blocks("cover", body),
            }
        },
        {"data": {"text": _report_blocks("Here it is: " + body, body), "status": "COMPLETED"}},
    ]
    result = research(_FakeClient(events), "q")

    assert result.content_shortfall is False
    assert result.warnings == []
    assert result.answer == "Here it is: " + body, "the body is not repeated after the cover"


# ---------- decode totality ----------


def test_research_survives_a_malformed_assets_field_mid_stream() -> None:
    """Decoding now runs inside the stream callback, which catches SchemaError
    only — a decode that raised TypeError escaped the stream loop before the
    thread cleanup ran and lost the COMPLETED frame that followed."""
    bad = json.dumps([{"step_type": "RESEARCH_ANSWER", "assets": True, "content": None}])
    client = _FakeClient(
        [
            {"data": {"backend_uuid": "BU", "read_write_token": "RW", "text": bad}},
            {"data": {"text": _snapshot("the real answer"), "status": "COMPLETED"}},
        ]
    )
    result = research(client, "q")

    assert result.answer == "the real answer"
    assert client.deleted == [("BU", "RW")], "the incognito thread is still cleaned up"


@pytest.mark.parametrize("assets", [True, 3, "str", {"a": 1}, None, [1, "x", None]])
def test_decode_tolerates_any_assets_shape(assets: Any) -> None:
    blocks = [
        {"step_type": "RESEARCH_ANSWER", "assets": assets, "content": {"answer": "inline"}},
        {"step_type": "FINAL", "content": {"answer": "cover"}},
    ]
    answer, _ = decode_research_text(json.dumps(blocks))
    assert answer == "cover\n\ninline"


@given(_json_value)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_decode_research_text_returns_or_raises_schema(payload: Any) -> None:
    """Totality: over ANY JSON input, decode returns a (str, list[Source]) pair
    or raises SchemaError. A TypeError/AttributeError/KeyError here reaches the
    stream callback, which only guards against SchemaError."""
    try:
        answer, sources = decode_research_text(json.dumps(payload))
    except SchemaError:
        return
    assert isinstance(answer, str)
    assert all(isinstance(s, ResearchSource) for s in sources)
