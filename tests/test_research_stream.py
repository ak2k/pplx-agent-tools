"""Research over the diff-mode stream: fixture replays (oracle 1), the
outcome's mapping onto v0.8's cutoff errors, the answer read from the
projections when no `text` decoded, and the cleanup legs."""

from __future__ import annotations

import io
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pplx_agent_tools.askstream.blocks import BlockStore, FrameApplied
from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.driver import ConnBounds
from pplx_agent_tools.askstream.frames import AskFrame
from pplx_agent_tools.askstream.fsm import Done, Known, NoIds, ReconnectReason, UuidOnly
from pplx_agent_tools.askstream.outcome import Completed, Cut, EndedEarly
from pplx_agent_tools.askstream.patch import Limits
from pplx_agent_tools.askstream.projections import CITATIONS_NOT_FINAL, research_answer
from pplx_agent_tools.errors import (
    AuthError,
    NetworkError,
    PplxError,
    ResourceLimitError,
    SchemaError,
    StreamDeadlineError,
    StreamFirstContentError,
    StreamSilenceError,
    StreamStallError,
)
from pplx_agent_tools.handles import ThreadHandle, ThreadStore
from pplx_agent_tools.verbs import _research_stream
from pplx_agent_tools.verbs._ask_common import AskStreamState, Source, cutoff_cause, cutoff_silence
from pplx_agent_tools.verbs._research_stream import ResearchRun, _unread, release, research_stream
from pplx_agent_tools.verbs.research import (
    DECODER,
    ENDPOINT,
    _shortfall_verdict,
    decode_research_text,
    finish_report,
    kept_warning,
    research,
)
from tests._askframes import diff, frame, replace, report, snap
from tests._doubles import FakeTime
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


def _cut_p1a(*, reject_report_patch: bool) -> ResearchRun:
    """p1-A's first 60 frames, then a drop no reconnect recovers."""
    items = fixture_items("p1-A")[:60]
    if reject_report_patch:
        items = json.loads(json.dumps(items))
        for block in items[11]["data"]["blocks"]:
            if block.get("intended_usage") == "unified_assets":
                block["diff_block"]["patches"][0]["path"] = "/no/such/path"
    initial = [*paced(items), (100.0, NetworkError("reset"))]
    client = FakeClient(FakeClock(), initials=[initial], reconnects=_refused(3, NetworkError("x")))
    return run_research(client)


def test_a_partial_whose_report_patch_was_rejected_is_flagged_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    whole, cut = _cut_p1a(reject_report_patch=False), _cut_p1a(reject_report_patch=True)
    assert len(cut.answer) < len(whole.answer)
    results = [
        finish_report(
            run,
            ThreadHandle(ThreadStore()),
            label="research",
            endpoint=ENDPOINT,
            query="q",
            mode="research",
            requested_model=None,
            timeout=3600.0,
            resume=None,
        )
        for run in (whole, cut)
    ]
    assert not results[0].content_shortfall
    assert not any("could not be applied" in w for w in results[0].warnings)
    assert results[1].content_shortfall
    assert [w for w in results[1].warnings if "could not be applied" in w] == [
        "a patch to the report could not be applied (missing_target), so it stopped "
        "updating and the answer may be missing content"
    ]


def _report_store(*frames: AskFrame) -> tuple[BlockStore, Counter[Drift]]:
    store = BlockStore("ask_text_only", Limits())
    drift: Counter[Drift] = Counter()
    first = frame(diff("unified_assets", "unified_assets_block", replace("", report("Body"))))
    for f in (first, *frames):
        applied = store.apply_frame(f)
        assert isinstance(applied, FrameApplied)
        drift.update(applied.drift)
    return store, drift


def test_a_field_a_snapshot_brought_back_in_sync_is_not_unread() -> None:
    rejected = frame(diff("unified_assets", "unified_assets_block", replace("/no/such", "x")))
    resynced = frame(snap("unified_assets", "unified_assets_block", report("Body more")))
    store, drift = _report_store(rejected)
    assert [d.kind for d in drift] == ["patch_rejected"]
    assert len(_unread(store, drift)) == 1
    store, drift = _report_store(rejected, resynced)
    assert [d.kind for d in drift] == ["patch_rejected"]
    assert _unread(store, drift) == []


def test_a_read_path_missing_from_its_document_is_unread() -> None:
    store, _ = _report_store()
    missing = Counter([Drift("projection_missing", name_of("unified_assets/assets"))])
    assert _unread(store, missing) == [
        "the report arrived in a shape pplx could not read (unified_assets/assets), so the "
        "answer may be missing content"
    ]
    other = Counter([Drift("projection_missing", name_of("text"))])
    assert _unread(store, other) == []


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


@pytest.mark.parametrize("stopped", [True, False])
def test_a_rejection_mid_stream_cleans_up_and_warns_on_stderr(stopped: bool) -> None:
    """A run pplx could not stop keeps its thread: no exception is in flight,
    only a rejection."""
    e = SchemaError("16 MiB")
    client = FakeClient(FakeClock(), initials=[[*paced(P3_INITIAL[:5]), (6.0, e)]])
    client.terminate_ok = stopped
    err = io.StringIO()
    with pytest.raises(SchemaError, match=r"^16 MiB$") as caught:
        run_research(client, err=err)
    assert type(caught.value) is SchemaError
    if stopped:
        assert (client.terminated, client.deleted) == (STOPPED, DELETED)
        assert err.getvalue() == ""
    else:
        assert (client.terminated, client.deleted) == (STOPPED, [])
        assert err.getvalue() == f"warning: {MAY_BE_LIVE}: the request to stop it failed\n"


@pytest.mark.parametrize("stopped", [True, False])
def test_a_cap_on_the_first_frame_that_names_the_thread_goes_through_the_cleanup_rules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stopped: bool
) -> None:
    """The capped frame is the only one to name the thread: its ids still
    reach cleanup, and a run pplx could not stop keeps its record."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    ops = [replace("/chunks/0", "x") for _ in range(Limits().ops_per_frame + 1)]
    data = {
        **P3_INITIAL[0]["data"],
        "read_write_token": FIXTURE_TOKEN,
        "blocks": [diff("ask_text", "markdown_block", *ops)],
    }
    client = FakeClient(FakeClock(), initials=[[(1.0, message(data))]])
    client.terminate_ok = stopped
    store = ThreadStore()
    with pytest.raises(ResourceLimitError):
        research_stream(
            client,
            client.open_initial,
            DECODER,
            endpoint=ENDPOINT,
            handle=ThreadHandle(store, prompt="q"),
            clock=client.clock,
            sleep=client.clock.sleep,
            rand=lambda: 0.5,
            err=io.StringIO(),
        )
    assert (client.terminated, client.deleted) == (STOPPED, DELETED if stopped else [])
    record = store.load(FIXTURE_UUID)
    assert (record.status if record is not None else None) == (None if stopped else "kept")


def _auth_refused_before_first_content(trigger: str) -> FakeClient:
    """The first frame names the thread and carries no content; then a drop,
    or no content until the first-content bound, and the reconnect is
    refused for expired cookies."""
    first = paced(P3_INITIAL[:1])
    if trigger == "drop":
        initial = [*first, (5.0, NetworkError("reset"))]
    else:
        initial = [*first, *((15.0 * i, HEARTBEAT) for i in range(1, 11))]
    return FakeClient(FakeClock(), initials=[initial], reconnects=[[(0.0, AuthError("expired"))]])


@pytest.mark.parametrize(
    ("trigger", "stopped", "legs", "kept"),
    [
        ("drop", False, ([], []), True),
        ("first_content", False, (STOPPED, []), True),
        # The first frame carries no read_write_token, so there is no delete.
        ("first_content", True, (STOPPED, []), False),
    ],
)
def test_a_reconnect_refused_for_auth_before_first_content_goes_through_the_keep_rules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    trigger: str,
    stopped: bool,
    legs: tuple[list[Any], list[Any]],
    kept: bool,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    client = _auth_refused_before_first_content(trigger)
    client.terminate_ok = stopped
    store = ThreadStore()
    state = AskStreamState()
    with pytest.raises(AuthError):
        research_stream(
            client,
            client.open_initial,
            DECODER,
            endpoint=ENDPOINT,
            timeout=3600.0,
            stall_seconds=240.0,
            state=state,
            handle=ThreadHandle(store, prompt="q"),
            clock=client.clock,
            sleep=client.clock.sleep,
            rand=lambda: 0.5,
            err=io.StringIO(),
        )
    assert (client.terminated, client.deleted) == legs
    assert state.kept is kept
    record = store.load(FIXTURE_UUID)
    assert (record.status if record is not None else None) == ("kept" if kept else None)


class _Posting(FakeClient):
    def sse_post(
        self,
        endpoint: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        silence_seconds: float,
    ) -> Iterator[Item]:
        return self.open_initial(ConnBounds(max_total_seconds, stall_seconds, silence_seconds))


def test_research_names_the_resume_command_on_the_error_of_a_kept_rejected_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("PPLX_PROFILE", raising=False)
    monkeypatch.setattr(_research_stream, "time", FakeTime())
    drop = _auth_refused_before_first_content("drop")
    client = _Posting(drop.clock, initials=drop.initials, reconnects=drop.reconnects)
    with pytest.raises(AuthError) as caught:
        research(client, "q")  # pyright: ignore[reportArgumentType]
    command = f"pplx resume --profile default {FIXTURE_UUID}"
    assert caught.value.resume == command
    assert f"`{command}`" in capsys.readouterr().err
    record = ThreadStore().load(FIXTURE_UUID)
    assert record is not None and record.status == "kept"


def test_a_partial_ended_by_an_auth_refused_reconnect_names_the_expiry_beside_the_resume_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The resume command fails the same way until the cookies are refreshed,
    and the partial has no cut_by to say why it ended."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("PPLX_PROFILE", raising=False)
    monkeypatch.setattr(_research_stream, "time", FakeTime())
    initial = [*paced(P3_INITIAL), (26.0, NetworkError("reset"))]
    client = _Posting(FakeClock(), initials=[initial], reconnects=[[(0.0, AuthError("x"))]])
    result = research(client, "q")  # pyright: ignore[reportArgumentType]
    command = f"pplx resume --profile default {FIXTURE_UUID}"
    assert (result.resume, result.cut_by, result.stream_complete) == (command, None, False)
    kept = result.warnings.index(kept_warning(command))
    assert result.warnings[kept + 1] == (
        "the stream ended early because a reconnect was refused: session cookies expired; "
        f"refresh with pplx auth, then get the report with `{command}`"
    )


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
