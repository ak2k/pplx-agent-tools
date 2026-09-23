"""Fixture-replay tests for verbs/research.py.

Feeds a real (sanitized, trimmed) deep-research SSE stream through `research()`
to catch upstream schema regressions and snapshot-decode bugs that the synthetic
blocks in `test_verbs_research.py` can't reproduce.

Observed shape (weather-nowcasting-apis.events.jsonl — 7 of 802 captured frames):
  - frames 0-2: scaffolding then the growing snapshot (search rounds, thoughts,
    the report body streaming into a RESEARCH_ANSWER asset)
  - frame 3: first `text_completed: True`, status still PENDING
  - frame 5: status COMPLETED carrying the authoritative repaint
  - frame 6: `{}` — the stream's final empty frame

Nested `web_results` lists are capped at 10 per list by the sanitizer, so source
counts here are a floor, not the real stream's total — assert on shape, not size.
This fixture predates the citation-aligned cap: its report cites up to [34]
against 10 kept sources, and only a fresh capture can restore the rest.

ocio-fees-final-only.events.jsonl (7 of 958 frames, a 14-minute broad run) is
the other build's shape: no RESEARCH_ANSWER block at all — the report arrives
whole as FINAL's `content.answer` — plus an auto-skipped
RESEARCH_CLARIFYING_QUESTIONS step. Its text_completed frame carries an EMPTY
FINAL `web_results`; only the COMPLETED repaint carries the cited list.

Two defects are pinned here, both of which shipped a plausible-looking wrong
answer (exit 0, ~1.2k chars) instead of the 9.7k-char report:
  1. the report body lives in the RESEARCH_ANSWER block's report asset, not in
     FINAL's `content.answer` (which holds only the cover note);
  2. the shared completion predicate accepts `text_completed`, which fires
     before the terminal repaint — research overrides it (`status_completed`).

Regenerate with scripts/re-capture-research.py + scripts/re-sanitize-research-fixture.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest

from pplx_agent_tools.verbs._ask_common import event_marks_completed
from pplx_agent_tools.verbs.research import decode_research_text, research
from tests._doubles import _TestClientBase

FIXTURES = Path(__file__).parent / "fixtures" / "research"
SANITIZER_SCRIPT = Path(__file__).parent.parent / "scripts" / "re-sanitize-research-fixture.py"
CAPTURE_SCRIPT = Path(__file__).parent.parent / "scripts" / "re-capture-research.py"

# Matches the sentinels in scripts/re-sanitize-research-fixture.py.
# Drift between this file and the script is caught by
# test_sentinels_match_sanitizer_script below — keep them in lockstep.
SENTINEL_BACKEND_UUID = "00000000-0000-4000-8000-000000000001"
SENTINEL_RW_TOKEN = "TEST_RW_TOKEN"


class FixtureClient(_TestClientBase):
    """Yields canned SSE events from a sanitized .events.jsonl fixture.

    Each line in the fixture is the `data` payload of one SSE event; we wrap it
    in the `{event, data}` envelope shape that the real `sse_post` yields, and
    count how many the verb pulled (the completion predicate is exactly a
    statement about where the verb stops).
    """

    def __init__(self, fixture_path: Path) -> None:
        super().__init__()
        self._events: list[Any] = [
            json.loads(line) for line in fixture_path.read_text().splitlines() if line.strip()
        ]
        self.consumed = 0
        self.deleted: list[tuple[str, str]] = []

    def sse_post(  # type: ignore[override]
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        is_progress: Callable[[dict[str, Any]], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        for payload in self._events:
            self.consumed += 1
            yield {"event": "message", "data": payload}

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


@pytest.fixture
def weather_fixture() -> Path:
    path = FIXTURES / "weather-nowcasting-apis.events.jsonl"
    assert path.exists(), f"missing fixture: {path}"
    return path


@pytest.fixture
def ocio_fixture() -> Path:
    path = FIXTURES / "ocio-fees-final-only.events.jsonl"
    assert path.exists(), f"missing fixture: {path}"
    return path


def _payloads(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _final_inner(payload: dict[str, Any]) -> dict[str, Any]:
    blocks = json.loads(payload["text"])
    final = next(b for b in blocks if b.get("step_type") == "FINAL")
    return json.loads(final["content"]["answer"])


def _headings(answer: str) -> list[str]:
    return [line for line in answer.splitlines() if line.startswith("#")]


def test_research_replays_real_stream(weather_fixture: Path) -> None:
    """The answer must be exactly the COMPLETED repaint's decode.

    Size/shape thresholds alone do NOT pin the completion predicate: the earlier
    `text_completed` frame decodes to 9684 chars and the COMPLETED one to 9688,
    so both clear any `> 5000` bar. Equality is what discriminates them, and the
    expectation is computed from the fixture so it survives regeneration."""
    payloads = [
        json.loads(line) for line in weather_fixture.read_text().splitlines() if line.strip()
    ]
    completed = next(p for p in payloads if p.get("status") == "COMPLETED")
    expected_answer, expected_sources = decode_research_text(completed["text"])

    client = FixtureClient(weather_fixture)
    result = research(client, "compare weather nowcasting APIs")

    assert result.stream_complete is True
    assert result.content_shortfall is False
    assert result.answer == expected_answer
    assert [s.url for s in result.sources] == [s.url for s in expected_sources]
    assert len(result.answer) > 5000, "the report body, not just the cover note"
    assert len(_headings(result.answer)) >= 3
    assert len(result.sources) > 0


def test_research_replay_deletes_thread(weather_fixture: Path) -> None:
    client = FixtureClient(weather_fixture)
    research(client, "compare weather nowcasting APIs")
    assert client.deleted == [(SENTINEL_BACKEND_UUID, SENTINEL_RW_TOKEN)]


def test_research_consumes_the_completed_repaint(weather_fixture: Path) -> None:
    """The verb must run past the first `text_completed` frame. The shared
    default predicate stops there; research's own stops only at COMPLETED."""
    payloads = [
        json.loads(line) for line in weather_fixture.read_text().splitlines() if line.strip()
    ]
    first_text_completed = next(
        i for i, p in enumerate(payloads) if event_marks_completed({"data": p})
    )
    completed = next(i for i, p in enumerate(payloads) if p.get("status") == "COMPLETED")
    assert first_text_completed < completed, "fixture must span the early-completion window"

    client = FixtureClient(weather_fixture)
    research(client, "compare weather nowcasting APIs")
    assert client.consumed == completed + 1


def test_final_block_alone_is_only_a_cover_note(weather_fixture: Path) -> None:
    """The regression this fixture exists for: FINAL's `content.answer` is a
    ~1.2k-char note that *describes* a report it does not contain. Decoding it
    alone is the shape the verb used to return with `stream_complete: true`."""
    payloads = [
        json.loads(line) for line in weather_fixture.read_text().splitlines() if line.strip()
    ]
    completed = next(p for p in payloads if p.get("status") == "COMPLETED")
    blocks = json.loads(completed["text"])
    cover = ""
    for blk in blocks:
        if blk.get("step_type") == "FINAL":
            cover = json.loads(blk["content"]["answer"])["answer"]
    assert 0 < len(cover) < 2000
    assert not _headings(cover)

    answer, _ = decode_research_text(completed["text"])
    assert answer.startswith(cover)
    assert len(answer) > 4 * len(cover)
    assert len(_headings(answer)) >= 3


def test_research_replays_final_only_stream(ocio_fixture: Path) -> None:
    """A build with no RESEARCH_ANSWER asset: the whole report is FINAL's answer,
    and every [n] in it must resolve against the sources the verb returns."""
    payloads = _payloads(ocio_fixture)
    completed = next(p for p in payloads if p.get("status") == "COMPLETED")
    steps = {b.get("step_type") for b in json.loads(completed["text"])}
    assert "RESEARCH_ANSWER" not in steps
    assert "RESEARCH_CLARIFYING_QUESTIONS" in steps
    inner = _final_inner(completed)

    client = FixtureClient(ocio_fixture)
    result = research(client, "compare OCIO fees")

    assert result.stream_complete is True
    assert result.content_shortfall is False
    assert result.answer == inner["answer"].strip()
    assert len(result.answer) > 12000, "the full report, not a partial repaint"
    assert client.consumed == payloads.index(completed) + 1
    assert client.deleted == [(SENTINEL_BACKEND_UUID, SENTINEL_RW_TOKEN)]

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result.answer)}
    web = inner["web_results"]
    assert cited and max(cited) <= len(web), "the fixture kept every cited source"
    urls = {s.url for s in result.sources}
    assert {web[n - 1]["url"] for n in cited} <= urls


def _load_script(name: str, path: Path) -> ModuleType:
    """Import a script by path: `scripts/` is not a package and the module names
    are hyphenated, so neither a plain import nor a relative one reaches them."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sanitizer() -> ModuleType:
    return _load_script("re_sanitize_fixture", SANITIZER_SCRIPT)


def test_capture_label_must_stay_inside_the_scratch_dir() -> None:
    """A raw capture holds a live thread token and account ids. An unvalidated
    --label escaped re-fixtures/ and landed one in the tracked fixture tree."""
    capture = _load_script("re_capture_research", CAPTURE_SCRIPT)

    for bad in (
        "",
        "..",
        "../../tests/fixtures/research/raw",
        "/tmp/raw",
        "sub/dir",
        "back\\slash",
        ".hidden",
        "a..b",
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            capture.label_arg(bad)

    for ok in ("weather-nowcasting-apis-2026-06-22T10-00-00Z", "run2.take3"):
        assert capture.label_arg(ok) == ok


def test_sanitizer_redacts_identity_keys_of_every_type() -> None:
    """An `author_*`/`user_*` key is identity whatever its value's JSON type.
    Gating the rule on `str` let an int, a list or a dict ride out unscrubbed."""
    san = _sanitizer()
    out = san._scrub(
        {
            "user_id": 481516,
            "user_aliases": ["adam", "ak2k"],
            "author_profile": {"backend_uuid": "real-uuid", "bio": "reach me at a@b.example"},
            "user_deleted_at": None,
            "user_selected_model": "pplx_alpha",
        }
    )

    assert out["user_id"] == "REDACTED", "an int id is still an id"
    assert out["user_aliases"] == ["REDACTED", "REDACTED"]
    assert out["author_profile"] == "REDACTED", "a nested identity object goes wholesale"
    assert out["user_deleted_at"] is None, "a null carries no identity — don't invent a field"
    assert out["user_selected_model"] == "pplx_alpha", "PREFIX_EXEMPT: a model id, not an identity"


def test_sanitizer_scrubs_canned_policy_signed_urls() -> None:
    """CloudFront's canned-policy signature carries no `Policy=` — only
    `Signature=` + `Key-Pair-Id=` — so a `Policy=`-only pattern let a live
    credential survive into a committed fixture."""
    san = _sanitizer()
    canned = "https://cdn.example/report.md?Expires=1&Signature=AbC123&Key-Pair-Id=APKAEXAMPLE"

    assert san._scrub({"url": canned})["url"] == san.SENTINEL_REPORT_URL
    assert san._scrub({"url": f"{canned}&Policy=eyJ"})["url"] == san.SENTINEL_REPORT_URL
    # …without swallowing ordinary cited sources, which the tests read as data.
    plain = "https://example.com/article?utm_source=x"
    assert san._scrub({"url": plain})["url"] == plain


def test_sanitizer_keeps_compact_json_parseable() -> None:
    """A `\\S*` URL run walked straight through the closing quote of a URL
    embedded in compact JSON, leaving an unterminated string behind."""
    san = _sanitizer()
    doc = '{"url":"https://d1.cloudfront.net/r.md?Signature=AAA&Key-Pair-Id=K1","final":true}'

    scrubbed = san._scrub_string(doc)

    assert json.loads(scrubbed) == {"url": san.SENTINEL_REPORT_URL, "final": True}


def test_sanitizer_scrubs_vendor_prefixed_signature_params() -> None:
    """S3 SigV4 and GCS prefix the credential param, so a bare `[?&]Signature=`
    alternation let a live pre-signed URL through."""
    san = _sanitizer()
    amz = "https://bucket.s3.amazonaws.com/r.md?X-Amz-Credential=AKIA%2F1&X-Amz-Signature=dead"
    goog = "https://storage.googleapis.com/b/r.md?X-Goog-Signature=deadbeef"

    assert san._scrub({"url": amz})["url"] == san.SENTINEL_REPORT_URL
    assert san._scrub({"url": goog})["url"] == san.SENTINEL_REPORT_URL
    plain = "https://example.com/article?utm_source=x"
    assert san._scrub({"url": plain})["url"] == plain, "ordinary cited sources survive"


def test_sanitizer_scrubs_a_credential_spanning_a_chunk_boundary() -> None:
    """`chunks` re-ships the answer in ~23-char slices; per-string scrubbing
    redacted the whole `answer` and kept the same credential verbatim across a
    slice boundary."""
    san = _sanitizer()
    url = "https://cdn.example/r.md?Signature=SECRETSIG&Key-Pair-Id=K1"
    chunks = [url[i : i + 23] for i in range(0, len(url), 23)]
    assert len(chunks) > 1

    out = san._scrub({"chunks": chunks})

    assert "SECRETSIG" not in "".join(out["chunks"])
    assert san.SENTINEL_REPORT_URL in "".join(out["chunks"])
    # A clean capture keeps its slicing, so re-running the sanitizer is stable.
    clean = ["hello ", "world"]
    assert san._scrub({"chunks": clean})["chunks"] == clean


def test_sanitizer_select_never_indexes_past_a_short_capture() -> None:
    """The mid-stream floors (1, 2) exceed a two-frame capture's length."""
    san = _sanitizer()
    keep = san._select([{"text": "a"}, {"text": "b", "text_completed": True}])

    assert keep == [0, 1]


def _research_frame(
    *, cover: str = "", body: str = "", status: str = "PENDING", text_completed: bool = False
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = [{"step_type": "INITIAL_QUERY", "content": {}}]
    if body:
        blocks.append(
            {
                "step_type": "RESEARCH_ANSWER",
                "content": {},
                "assets": [{"research_report": {"source_content": body}}],
            }
        )
    if cover:
        blocks.append({"step_type": "FINAL", "content": {"answer": json.dumps({"answer": cover})}})
    return {
        "status": status,
        "text_completed": text_completed,
        "text": json.dumps(blocks),
    }


def _shrinking_stream(*, body: bool) -> list[dict[str, Any]]:
    """12 frames; the longest answer is frame 7, which neither frame 0, the
    mid-stream picks (3, 6) nor the tail (9..11) covers."""
    sizes = [0, 10, 20, 30, 40, 50, 60, 500, 70, 80, 90, 100]
    frames = []
    for i, n in enumerate(sizes):
        text = "x" * n
        frames.append(
            _research_frame(
                cover="note" if body else text,
                body=text if body else "",
                status="COMPLETED" if i == 10 else "PENDING",
                text_completed=i >= 9,
            )
        )
    frames[-1] = {}
    return frames


@pytest.mark.parametrize("body", [True, False], ids=["report-body", "final-only"])
def test_sanitizer_select_keeps_the_high_water_frame(tmp_path: Path, body: bool) -> None:
    """The verb flags a shortfall against the longest report it saw; a reducer
    that drops that frame turns a shortfall in the raw stream into a clean replay."""
    san = _sanitizer()
    raw = _shrinking_stream(body=body)

    keep = san._select(raw)

    assert 7 in keep
    reduced = tmp_path / "reduced.events.jsonl"
    reduced.write_text("".join(json.dumps(raw[i]) + "\n" for i in keep))
    full = tmp_path / "full.events.jsonl"
    full.write_text("".join(json.dumps(p) + "\n" for p in raw))
    for path in (full, reduced):
        assert research(FixtureClient(path), "q").content_shortfall is True


def test_sanitizer_measure_matches_the_verb(weather_fixture: Path, ocio_fixture: Path) -> None:
    """The reducer picks high-water frames by its own decode; if it drifts from
    the verb's, it keeps the wrong frame and the shortfall stops reproducing."""
    from pplx_agent_tools.verbs.research import _decode_parts, _join_answer

    san = _sanitizer()
    frames = _payloads(weather_fixture) + _payloads(ocio_fixture)
    frames += _shrinking_stream(body=True) + _shrinking_stream(body=False)
    # A FINAL block that exists before its answer has streamed.
    empty_final = {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": ""})}}
    with_body = _research_frame(body="report")
    with_body["text"] = json.dumps([*json.loads(with_body["text"]), empty_final])
    frames.append(with_body)
    measured = 0
    for payload in frames:
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            assert san._measure(payload) == (0, 0)
            continue
        cover, reports, _ = _decode_parts(payload["text"])
        expected = (len("\n\n".join(reports).strip()), len(_join_answer(cover, reports)))
        assert san._measure(payload) == expected
        measured += expected != (0, 0)
    assert measured > 10


def test_sanitizer_keeps_every_cited_final_source() -> None:
    """FINAL's web_results is indexed by the answer's [n] markers, so a flat cap
    of 10 left [11..N] pointing at nothing."""
    san = _sanitizer()
    web = [{"url": f"https://example.com/{i}", "name": str(i)} for i in range(40)]

    def final_web(answer: str) -> list[Any]:
        inner = {"answer": answer, "web_results": web}
        blocks = [{"step_type": "FINAL", "content": {"answer": json.dumps(inner)}}]
        out = json.loads(san._scrub_research_text(json.dumps(blocks)))
        return json.loads(out[0]["content"]["answer"])["web_results"]

    assert final_web("see [3] and [23][2]") == web[:23]
    assert final_web("no citations yet") == web[: san.MAX_WEB_RESULTS]


def test_sanitizer_caps_repeated_steps_but_not_the_answer() -> None:
    san = _sanitizer()
    steps = [{"step_type": "THOUGHT", "content": {"n": i}} for i in range(9)]
    steps += [{"step_type": "FINAL", "content": {}} for _ in range(6)]

    out = san._scrub({"steps": steps})["steps"]

    kinds = [s["step_type"] for s in out]
    assert kinds.count("THOUGHT") == san.MAX_STEP_BLOCKS
    assert [s["content"]["n"] for s in out[: san.MAX_STEP_BLOCKS]] == [0, 1, 2, 3]
    assert kinds.count("FINAL") == 6


def test_sanitizer_redacts_account_metadata() -> None:
    san = _sanitizer()
    extras = {"country": "US", "subscription_tier": "max", "payment_tier": "paid", "next": None}
    telemetry = {"country": "US", "region": "us-east-1"}

    out = san._scrub_payload({"_extras": extras, "telemetry_data": telemetry})

    assert out["_extras"] == {
        "country": "REDACTED",
        "subscription_tier": "REDACTED",
        "payment_tier": "REDACTED",
        "next": None,
    }
    assert out["telemetry_data"] == {"country": "REDACTED", "region": "us-east-1"}


def test_committed_fixtures_carry_no_account_metadata(
    weather_fixture: Path, ocio_fixture: Path
) -> None:
    san = _sanitizer()
    for path in (weather_fixture, ocio_fixture):
        for payload in _payloads(path):
            for parent in ("_extras", "telemetry_data"):
                meta = payload.get(parent) if isinstance(payload, dict) else None
                if not isinstance(meta, dict):
                    continue
                for key in san.ACCOUNT_KEYS:
                    assert meta.get(key) in (None, san.SENTINEL_ACCOUNT_VALUE), (path, key)


def test_sentinels_match_sanitizer_script() -> None:
    """Defensive: detect drift between the sanitizer's SENTINELS dict and the
    constants this test asserts against, which would otherwise only show up as a
    mysterious failure after the next regeneration."""
    assert SANITIZER_SCRIPT.exists(), f"missing script: {SANITIZER_SCRIPT}"
    script = SANITIZER_SCRIPT.read_text()
    assert SENTINEL_BACKEND_UUID in script, (
        f"SENTINEL_BACKEND_UUID {SENTINEL_BACKEND_UUID!r} not in sanitizer script"
    )
    assert SENTINEL_RW_TOKEN in script, (
        f"SENTINEL_RW_TOKEN {SENTINEL_RW_TOKEN!r} not in sanitizer script"
    )


def test_sanitizer_redacts_a_nested_identity_object_outright() -> None:
    """A dict under an identity key used to recurse "keeping the shape" — so a
    profile object whose own keys match neither SENTINELS nor the prefix rule
    rode out whole, ids and all."""
    san = _sanitizer()
    out = san._scrub(
        {
            "user_profile": {"id": 481516, "username": "X", "display_name": "Y"},
            "user_roles": [{"id": 1, "name": "admin"}],
            "user_empty": {},
            "user_blank": "",
            "user_none": None,
        }
    )

    assert out["user_profile"] == "REDACTED", "the whole object goes, not its matching leaves"
    assert out["user_roles"] == ["REDACTED"]
    # Empty containers and blanks carry shape but no secret; substituting them
    # would churn the fixture and hide the wire shape.
    assert out["user_empty"] == {}
    assert out["user_blank"] == ""
    assert out["user_none"] is None


class _StubResearchClient:
    """Stands in for `wire.Client` inside the capture script: yields canned
    frames, records the thread it is asked to delete. No network, no cookies."""

    deleted: ClassVar[list[tuple[str, str]]] = []

    @classmethod
    def from_default_cookies(cls, profile: str | None = None) -> _StubResearchClient:
        return cls()

    def sse_post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
        stall_seconds: float | None = None,
        is_progress: Callable[[dict[str, Any]], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        yield {"event": "message", "data": {"backend_uuid": "BU", "read_write_token": "RW"}}

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:
        type(self).deleted.append((entry_uuid, read_write_token))
        return True


class _UnwritableFile:
    """A capture file whose first flush fails, like a full disk."""

    def __enter__(self) -> _UnwritableFile:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        raise OSError(28, "No space left on device")


def _unwritable_capture(path: Path) -> _UnwritableFile:
    """Create the capture for real, then fail on flush.

    The file has to exist or the cleanup skips its `out_path.exists()` chmod
    entirely, which is the branch these tests are here to cover."""
    path.touch()
    return _UnwritableFile()


def test_capture_deletes_the_thread_when_the_first_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frame carrying the thread ids is the same frame being written, so
    reading them after the write left the cleanup blind to a failed flush and
    the created thread outlived the run."""
    capture = _load_script("re_capture_research", CAPTURE_SCRIPT)
    _StubResearchClient.deleted = []
    monkeypatch.setattr(capture, "Client", _StubResearchClient)
    monkeypatch.setattr(capture, "OUT_DIR", tmp_path)
    monkeypatch.setattr(capture, "_create_capture_file", _unwritable_capture)

    with pytest.raises(OSError, match="No space left"):
        capture.main(["q", "--label", "cap"])

    assert _StubResearchClient.deleted == [("BU", "RW")]


def test_capture_deletes_the_thread_when_the_cleanup_chmod_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk failure that breaks the write can break the chmod on the same
    partial capture — and that chmod sits between the run and the delete, so a
    raise there used to strand the thread it had already identified."""
    capture = _load_script("re_capture_research", CAPTURE_SCRIPT)
    _StubResearchClient.deleted = []
    monkeypatch.setattr(capture, "Client", _StubResearchClient)
    monkeypatch.setattr(capture, "OUT_DIR", tmp_path)
    monkeypatch.setattr(capture, "_create_capture_file", _unwritable_capture)

    def _failing_chmod(self: Path, mode: int, **kwargs: object) -> None:
        raise OSError(5, "Input/output error")

    # Path.chmod, not the script's own creation path: `_create_capture_file`
    # pins the mode with os.fchmod and stays usable.
    monkeypatch.setattr(Path, "chmod", _failing_chmod)

    with pytest.raises(OSError) as excinfo:
        capture.main(["q", "--label", "cap"])

    assert _StubResearchClient.deleted == [("BU", "RW")], "the delete runs regardless"
    assert "Input/output error" in str(excinfo.value)
    assert "No space left" in str(excinfo.value.__context__), "the write failure is not lost"


def test_capture_file_is_created_unreadable_to_others(tmp_path: Path) -> None:
    """The capture holds a live read_write_token from its first frame, so the
    mode has to be right at creation — the finally-block chmod only lands after
    a 90-120 s stream has finished writing it."""
    capture = _load_script("re_capture_research", CAPTURE_SCRIPT)
    path = tmp_path / "capture.events.jsonl"

    with capture._create_capture_file(path) as f:
        f.write("{}\n")
        f.flush()
        assert path.stat().st_mode & 0o777 == 0o600, "mode is wrong WHILE the stream runs"

    with pytest.raises(FileExistsError):
        capture._create_capture_file(path)  # exclusive create is preserved
