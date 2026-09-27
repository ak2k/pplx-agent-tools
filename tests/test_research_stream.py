"""Research over the diff-mode stream: fixture replays (oracle 1), the
outcome's mapping onto v0.8's cutoff errors, the answer read from the
projections when no `text` decoded, and the cleanup legs."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from pplx_agent_tools.askstream.fsm import Done, Known, NoIds, ReconnectReason, UuidOnly
from pplx_agent_tools.askstream.outcome import Completed, Cut, EndedEarly
from pplx_agent_tools.askstream.projections import CITATIONS_NOT_FINAL, research_answer
from pplx_agent_tools.errors import (
    AuthError,
    NetworkError,
    PplxError,
    SchemaError,
    StreamDeadlineError,
    StreamFirstContentError,
    StreamSilenceError,
    StreamStallError,
)
from pplx_agent_tools.verbs._ask_common import Source, cutoff_cause, cutoff_silence
from pplx_agent_tools.verbs._research_stream import ResearchRun, release
from pplx_agent_tools.verbs.research import ENDPOINT, _shortfall_verdict, decode_research_text
from tests._driver import (
    HEARTBEAT,
    FakeClient,
    FakeClock,
    Item,
    Script,
    fixture_items,
    message,
    paced,
    run_research,
)
from tests._fsm import CTX, KNOWN, TOKEN_RAW, UUID

FIXTURE_UUID = "00000000-0000-4000-8000-000000000001"
FIXTURE_CTX = "00000000-0000-4000-8000-000000000002"
FIXTURE_TOKEN = "TEST_RW_TOKEN"
MODEL = "pplx_alpha"
STOPPED = [(FIXTURE_UUID, FIXTURE_CTX, MODEL)]
DELETED = [(FIXTURE_UUID, FIXTURE_TOKEN)]
MAY_BE_LIVE = "the run may still be running on the server and using quota"
P3_INITIAL = fixture_items("p3-research-initial")


def _terminal_text(items: list[Item]) -> str:
    return next(
        i["data"]["text"] for i in reversed(items) if i["data"].get("status") == "COMPLETED"
    )


def _sources(run: ResearchRun) -> list[Source]:
    return [Source(s.url, s.title, s.snippet) for s in run.store.run_sources]


def _refused(n: int, e: PplxError) -> list[Script]:
    return [[(0.0, e)] for _ in range(n)]


# --- oracle 1: replays -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stem", "legacy"),
    [
        ("weather-nowcasting-apis", True),
        ("ocio-fees-final-only", True),
        ("p1-A", False),
        ("p1-B", False),
    ],
)
def test_replay_answers_with_the_terminal_text(stem: str, legacy: bool) -> None:
    items = fixture_items(stem, legacy=legacy)
    client = FakeClient(FakeClock(), initials=[paced(items)])
    run = run_research(client)
    answer, sources = decode_research_text(_terminal_text(items))
    assert answer
    assert run.answer == answer
    assert run.sources == sources
    assert run.terminal_text == _terminal_text(items)
    assert run.state.saw_completed
    assert run.state.cutoff is None
    assert run.warnings == []
    assert run.thread == "deleted"
    assert (client.terminated, client.deleted) == ([], DELETED)


def test_replay_p3_drop_then_reconnect_answers_with_the_reconnect_terminal_text() -> None:
    reconnect = fixture_items("p3-research-reconnect1")
    initial = [*paced(P3_INITIAL), (26.0, NetworkError("reset"))]
    client = FakeClient(FakeClock(), initials=[initial], reconnects=[paced(reconnect, start=30.0)])
    run = run_research(client)
    answer, sources = decode_research_text(_terminal_text(reconnect))
    assert answer
    assert (run.answer, run.sources) == (answer, sources)
    assert run.driver.reconnect_opens == 1
    assert run.state.saw_completed
    assert run.state.cutoff is None
    assert (client.terminated, client.deleted) == ([], DELETED)


# --- the answer from the projections -----------------------------------------------------------


def test_a_deadline_cut_before_the_terminal_frame_answers_from_the_projections() -> None:
    items = fixture_items("p1-A")[:-1]
    client = FakeClient(
        FakeClock(), initials=[[*paced(items), (150.0, StreamDeadlineError("deadline"))]]
    )
    run = run_research(client, timeout=150.0)
    projected = research_answer(run.store, False)
    assert projected.text
    assert not projected.citations_final
    assert run.answer == projected.text
    assert run.sources == _sources(run)
    assert run.sources
    assert run.warnings == [CITATIONS_NOT_FINAL]
    assert run.questions is None
    cutoff = run.state.cutoff
    assert isinstance(cutoff, StreamDeadlineError)
    assert str(cutoff) == f"research stream on {ENDPOINT} exceeded 150.0s deadline"
    assert cutoff.since_progress == run.driver.since_progress
    assert cutoff_cause(run.state) == "deadline"
    assert not run.state.saw_completed
    assert run.thread == "deleted"
    assert (client.terminated, client.deleted) == (STOPPED, DELETED)


def test_an_undecodable_terminal_text_keeps_the_projection_and_flags_the_last_frame() -> None:
    items = fixture_items("p1-A")
    bad = message({**items[-1]["data"], "text": "{not json"})
    client = FakeClient(FakeClock(), initials=[paced([*items[:-1], bad])])
    run = run_research(client)
    assert run.state.saw_completed
    assert run.answer == research_answer(run.store, True).text
    assert run.answer
    assert run.saw["last_frame_decoded"] is False
    _, warnings = _shortfall_verdict(
        answer_len=len(run.answer), body_len=run.body_len, best=run.best, saw=run.saw
    )
    assert any("the last frame of the stream failed to decode" in w for w in warnings)


def test_an_undecodable_text_with_an_empty_projection_raises_the_decode_error() -> None:
    client = FakeClient(
        FakeClock(), initials=[[(1.0, message({"status": "COMPLETED", "text": "{not json"}))]]
    )
    with pytest.raises(SchemaError, match="research text is not JSON"):
        run_research(client)


# --- the shortfall verdict ---------------------------------------------------------------------


def _verdict(run: ResearchRun) -> tuple[bool, list[str]]:
    return _shortfall_verdict(
        answer_len=len(run.answer), body_len=run.body_len, best=run.best, saw=run.saw
    )


def _p3_client() -> FakeClient:
    initial = [*paced(P3_INITIAL), (26.0, NetworkError("reset"))]
    reconnect = paced(fixture_items("p3-research-reconnect1"), start=30.0)
    return FakeClient(FakeClock(), initials=[initial], reconnects=[reconnect])


@pytest.mark.parametrize(
    ("stem", "legacy"),
    [
        ("weather-nowcasting-apis", True),
        ("ocio-fees-final-only", True),
        ("p1-A", False),
        ("p1-B", False),
        ("p3", False),
    ],
)
def test_a_completed_capture_is_not_flagged_short(stem: str, legacy: bool) -> None:
    if stem == "p3":
        client = _p3_client()
    else:
        client = FakeClient(FakeClock(), initials=[paced(fixture_items(stem, legacy=legacy))])
    run = run_research(client)
    assert run.state.saw_completed and run.consumer.text is not None
    assert _verdict(run) == (False, [])


def test_a_terminal_text_whose_report_body_is_shorter_than_the_streamed_one_is_flagged() -> None:
    items = fixture_items("p1-A")
    terminal = items[-1]["data"]
    blocks = json.loads(terminal["text"])
    cut_bodies = 0
    for block in blocks:
        if block.get("step_type") != "RESEARCH_ANSWER":
            continue
        for asset in block.get("assets") or []:
            report = asset.get("research_report") or {}
            body = report.get("source_content")
            if isinstance(body, str):
                report["source_content"] = body[: len(body) // 4]
                cut_bodies += 1
    assert cut_bodies == 1
    cut = message({**terminal, "text": json.dumps(blocks)})
    run = run_research(FakeClient(FakeClock(), initials=[paced([*items[:-1], cut])]))
    assert run.state.saw_completed
    assert run.body_len < run.consumer.report_high
    flagged, warnings = _verdict(run)
    assert flagged
    assert warnings == [
        f"kept snapshot's report body decodes to {run.body_len} chars but an earlier frame "
        f"carried {run.consumer.report_high}; the report may be truncated"
    ]


# --- outcome to cutoff -------------------------------------------------------------------------


def test_a_stall_the_reconnects_cannot_recover_is_a_stall_cut() -> None:
    beats: Script = [(float(t), HEARTBEAT) for t in range(27, 60)]
    initial = [*paced(P3_INITIAL), *beats]
    client = FakeClient(
        FakeClock(), initials=[initial], reconnects=_refused(3, NetworkError("refused"))
    )
    run = run_research(client, stall_seconds=20.0)
    cutoff = run.state.cutoff
    assert isinstance(cutoff, StreamStallError)
    assert not isinstance(cutoff, StreamSilenceError)
    assert cutoff.seconds == 20.0
    assert str(cutoff) == f"SSE stream on {ENDPOINT} stalled: no new content for 20.0s"
    assert (cutoff_cause(run.state), cutoff_silence(run.state)) == ("stall", None)
    assert run.driver.trigger == "stall"
    assert run.thread == "deleted"
    assert (client.terminated, client.deleted) == (STOPPED, DELETED)


def test_no_first_content_raises_the_first_content_error() -> None:
    beats: Script = [(float(t), HEARTBEAT) for t in range(20, 120, 20)]
    client = FakeClient(FakeClock(), initials=[beats])
    with pytest.raises(StreamFirstContentError) as caught:
        run_research(client)
    assert str(caught.value) == (
        f"research stream on {ENDPOINT} sent no first content within 90.0s"
    )
    assert caught.value.seconds == 90.0
    assert (client.terminated, client.deleted) == ([], [])


def test_lost_after_a_silent_reconnect_still_reads_as_a_drop() -> None:
    initial = [*paced(P3_INITIAL), (26.0, NetworkError("reset"))]
    silent = StreamSilenceError("went silent", 90.0)
    client = FakeClient(FakeClock(), initials=[initial], reconnects=_refused(3, silent))
    run = run_research(client)
    assert type(run.state.cutoff) is NetworkError
    assert str(run.state.cutoff) == "stream dropped"
    assert cutoff_cause(run.state) == "drop"


def test_ended_early_by_the_server_returns_the_partial_without_a_cutoff() -> None:
    client = FakeClient(FakeClock(), initials=[paced(P3_INITIAL)])
    run = run_research(client)
    assert isinstance(run.driver.state, Done)
    assert run.driver.state.outcome == EndedEarly(3, "server")
    assert run.sources
    assert run.state.cutoff is None
    assert not run.state.saw_completed
    assert run.thread == "deleted"
    assert (client.terminated, client.deleted) == (STOPPED, DELETED)


def test_ended_early_by_the_server_with_nothing_raises_closed_with_no_content() -> None:
    client = FakeClient(FakeClock(), initials=[paced(P3_INITIAL[:3])])
    with pytest.raises(SchemaError) as caught:
        run_research(client)
    assert type(caught.value) is SchemaError
    assert str(caught.value) == f"research stream on {ENDPOINT} closed with no content"
    assert (client.terminated, client.deleted) == (STOPPED, DELETED)


def test_a_reconnect_refused_for_auth_with_nothing_raises_the_auth_error() -> None:
    refused = AuthError("expired")
    initial = [*paced(P3_INITIAL[:3]), (4.0, NetworkError("reset"))]
    client = FakeClient(FakeClock(), initials=[initial], reconnects=[[(5.0, refused)]])
    seen: list[ResearchRun] = []
    with pytest.raises(AuthError) as caught:
        run_research(client, observe=seen.append)
    assert caught.value is refused
    assert isinstance(seen[0].driver.state, Done)
    assert seen[0].driver.state.outcome == EndedEarly(1, "auth")
    assert seen[0].thread == "kept"
    assert (client.terminated, client.deleted) == ([], [])


@pytest.mark.parametrize(
    ("e", "expected"),
    [
        (NetworkError("POST failed"), "POST failed"),
        (
            StreamSilenceError("went silent", 90.0),
            f"research stream on {ENDPOINT} went silent: no bytes for 90.0s "
            "before the first content arrived",
        ),
    ],
)
def test_an_initial_network_failure_raises_as_no_content(e: NetworkError, expected: str) -> None:
    client = FakeClient(FakeClock(), initials=[[(0.5, e)]])
    with pytest.raises(NetworkError) as caught:
        run_research(client)
    assert type(caught.value) is type(e)
    assert str(caught.value) == expected


def test_an_initial_rejection_raises_the_original_error() -> None:
    e = SchemaError("unexpected status 404")
    client = FakeClient(FakeClock(), initials=[[(0.5, e)]])
    with pytest.raises(SchemaError) as caught:
        run_research(client)
    assert caught.value is e


def test_a_rejection_mid_stream_cleans_up_and_warns_on_stderr() -> None:
    e = SchemaError("16 MiB")
    client = FakeClient(FakeClock(), initials=[[*paced(P3_INITIAL[:5]), (6.0, e)]])
    client.terminate_ok = False
    err = io.StringIO()
    with pytest.raises(SchemaError, match=r"^16 MiB$") as caught:
        run_research(client, err=err)
    assert type(caught.value) is SchemaError
    assert (client.terminated, client.deleted) == (STOPPED, DELETED)
    assert err.getvalue() == f"warning: {MAY_BE_LIVE}: the request to stop it failed\n"


def test_a_failed_run_returns_for_research_to_raise() -> None:
    failed = message({"status": "FAILED", "text": "[]"})
    client = FakeClient(FakeClock(), initials=[[*paced(P3_INITIAL[:5]), (6.0, failed)]])
    run = run_research(client)
    assert run.state.failed
    assert not run.state.saw_completed
    assert (client.terminated, client.deleted) == ([], DELETED)


def test_clarifying_questions_of_a_run_with_no_answer_end_the_error() -> None:
    blocks = [
        {
            "step_type": "RESEARCH_CLARIFYING_QUESTIONS",
            "content": {"questions": [{"question_text": "Which region?"}]},
        }
    ]
    frame = message({"status": "PENDING", "text": json.dumps(blocks)})
    client = FakeClient(FakeClock(), initials=[[(1.0, frame), (2.0, NetworkError("reset"))]])
    with pytest.raises(NetworkError) as caught:
        run_research(client)
    assert str(caught.value) == (
        "reset; research asked clarifying questions and no answer arrived: Which region?"
    )


# --- hooks -------------------------------------------------------------------------------------


def test_observe_sees_the_run_before_an_error_is_raised() -> None:
    seen: list[ResearchRun] = []
    client = FakeClient(FakeClock(), initials=[paced(P3_INITIAL[:3])])
    with pytest.raises(SchemaError):
        run_research(client, observe=seen.append)
    assert len(seen) == 1
    assert seen[0].thread == "deleted"


def test_on_data_sees_every_frame_payload() -> None:
    seen: list[dict[str, object]] = []
    items = fixture_items("p1-B")
    run_research(FakeClient(FakeClock(), initials=[paced(items)]), on_data=seen.append)
    assert seen == [i["data"] for i in items]


# --- release -----------------------------------------------------------------------------------


def _release(
    last: Any,
    trigger: ReconnectReason | None = None,
    *,
    gone: bool = False,
    model: str | None = MODEL,
    keep: bool = False,
    ok: bool = True,
) -> tuple[FakeClient, tuple[str, list[str]]]:
    client = FakeClient(FakeClock(), terminate_ok=ok)
    got = release(client, last, trigger, gone=gone, display_model=model, keep_thread=keep)
    return client, got


def test_release_before_the_run_started_sends_nothing() -> None:
    client, got = _release(None)
    assert got == ("none", [])
    assert (client.terminated, client.deleted) == ([], [])


@pytest.mark.parametrize("trigger", ["drop", "silence"])
def test_release_keeps_a_thread_lost_to_a_drop_or_silence(trigger: ReconnectReason) -> None:
    client, got = _release(Done(Cut("stall", 240.0, 3), KNOWN), trigger)
    assert got == ("kept", [])
    assert (client.terminated, client.deleted) == ([], [])


def test_release_of_a_cut_terminates_then_deletes() -> None:
    client, got = _release(Done(Cut("deadline", 3600.0, 0), KNOWN))
    assert got == ("deleted", [])
    assert client.terminated == [(UUID, CTX, MODEL)]
    assert client.deleted == [(UUID, TOKEN_RAW)]


def test_release_with_keep_thread_terminates_but_does_not_delete() -> None:
    client, got = _release(Done(Cut("deadline", 3600.0, 0), KNOWN), keep=True)
    assert got == ("cleaned", [])
    assert (client.terminated, client.deleted) == ([(UUID, CTX, MODEL)], [])


def test_release_of_a_completed_run_only_deletes() -> None:
    client, got = _release(Done(Completed(0), KNOWN))
    assert got == ("deleted", [])
    assert (client.terminated, client.deleted) == ([], [(UUID, TOKEN_RAW)])


@pytest.mark.parametrize(
    ("ids", "model", "missing"),
    [
        (UuidOnly(UUID, None), MODEL, "context uuid"),
        (Known(KNOWN.ref, None), None, "context uuid"),
        (UuidOnly(UUID, CTX), None, "display model"),
    ],
)
def test_release_names_what_a_terminate_lacked(
    ids: UuidOnly | Known, model: str | None, missing: str
) -> None:
    client, got = _release(Done(Cut("deadline", 3600.0, 0), ids), model=model)
    expected = f"{MAY_BE_LIVE}: no {missing} arrived, so pplx could not ask it to stop"
    assert got == ("kept", [expected])
    assert client.terminated == []
    assert client.deleted == []


def test_release_warns_when_the_terminate_request_fails() -> None:
    client, got = _release(Done(Cut("deadline", 3600.0, 0), KNOWN), ok=False)
    assert got == ("kept", [f"{MAY_BE_LIVE}: the request to stop it failed"])
    assert client.deleted == []


def test_release_with_no_thread_sends_nothing() -> None:
    client, got = _release(Done(Cut("first_content", 90.0, 0), NoIds()), "first_content")
    assert got == ("cleaned", [])
    assert (client.terminated, client.deleted) == ([], [])
