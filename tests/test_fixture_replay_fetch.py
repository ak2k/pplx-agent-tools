"""Fixture-replay tests for verbs/fetch.py.

Feeds a real (sanitized) chat-fetch-prompt SSE stream through `_fetch_with_prompt`
to catch upstream schema regressions and accumulation bugs that synthetic
events in `test_verbs_fetch.py` can't reproduce.

Observed streaming pattern (example-com-prompt.events.jsonl, 23 events):
  - events 0-7: scaffolding (plan, pro_search_steps, web_results) — no ask_text
  - events 8-20: one delta chunk per event in `ask_text`, status=PENDING
  - event 20: also sets `text_completed: True` — terminates the verb loop early
  - event 21: status=COMPLETED carrying a full 13-chunk REPAINT of ask_text
    — never reached by the verb (and must not be, or the answer would double)

If Perplexity ever drops `text_completed` and only sends `status: COMPLETED`,
this test will fail loudly with a doubled answer string — exactly the upstream
drift this fixture is here to catch.

Three further captures cover the answer shapes that differ from a short, clean
summary: a paywalled article the model declines to quote, a prompt whose honest
answer is "there is no such table on this page", and a 5 kB sectioned answer
with a table. Each pins one invariant about how that shape maps to an exit code.

Capture with `scripts/re-capture-fetch-prompt.py`, then sanitize with
`scripts/re-sanitize-fetch-fixture.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pplx_agent_tools import cli_fetch, cli_runner
from pplx_agent_tools.errors import EXIT_GENERIC, EXIT_NETWORK, EXIT_OK, NetworkError
from pplx_agent_tools.verbs.fetch import _fetch_with_prompt
from tests._doubles import _TestClientBase

FIXTURES = Path(__file__).parent / "fixtures" / "fetch-url"
SANITIZER_SCRIPT = Path(__file__).parent.parent / "scripts" / "re-sanitize-fetch-fixture.py"
CAPTURE_SCRIPT = Path(__file__).parent.parent / "scripts" / "re-capture-fetch-prompt.py"

# Matches the sentinels in scripts/re-sanitize-fetch-fixture.py.
# Drift between this file and the script is caught by
# test_sentinels_appear_in_sanitizer_source below — keep them in lockstep.
SENTINEL_BACKEND_UUID = "00000000-0000-4000-8000-000000000001"
SENTINEL_RW_TOKEN = "TEST_RW_TOKEN"
SENTINEL_UUID = "00000000-0000-4000-8000-000000000003"
SENTINEL_REDACTED = "REDACTED"
SENTINEL_EMAIL = "redacted@example.invalid"

# The example.com fetch's expected answer text. Derived from the
# `markdown_block.answer` field on the captured COMPLETED event — this is
# Perplexity's authoritative join of the delta chunks.
EXPECTED_ANSWER = (
    "This domain is for use in illustrative examples in documents. You may use this\n\n"
    "domain in literature without prior coordination or asking for permission.\n\n"
    "More information..."
)


class FixtureClient(_TestClientBase):
    """Yields canned SSE events from a sanitized .events.jsonl fixture.

    Each line in the fixture is the `data` payload of one SSE event; we wrap
    it in the `{event, data}` envelope shape that the real `sse_post` yields.
    """

    def __init__(self, fixture_path: Path) -> None:
        super().__init__()
        self._events: list[dict[str, Any]] = [
            json.loads(line) for line in fixture_path.read_text().splitlines() if line.strip()
        ]
        self.deleted: list[tuple[str, str]] = []

    def sse_post(  # type: ignore[override]
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        for payload in self._events:
            yield {"event": "message", "data": payload}

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


@pytest.fixture
def example_com_fixture() -> Path:
    path = FIXTURES / "example-com-prompt.events.jsonl"
    assert path.exists(), f"missing fixture: {path}"
    return path


def test_fetch_with_prompt_replays_real_stream(example_com_fixture: Path) -> None:
    client = FixtureClient(example_com_fixture)
    result = _fetch_with_prompt(
        client, "https://example.com", "summarize", "example.com", max_chars=None
    )

    assert result.is_extracted is True
    assert result.url == "https://example.com"
    assert result.domain == "example.com"
    # The verb must reconstruct exactly the answer Perplexity served — no
    # double-counting from the COMPLETED event's repaint, no dropped chunks.
    assert result.content == EXPECTED_ANSWER


def test_fetch_with_prompt_captures_thread_identifiers(example_com_fixture: Path) -> None:
    client = FixtureClient(example_com_fixture)
    _fetch_with_prompt(client, "https://example.com", "summarize", "example.com", max_chars=None)
    # Default cleanup deletes the thread using the (backend_uuid, read_write_token)
    # pair carried by the SSE events.
    assert client.deleted == [(SENTINEL_BACKEND_UUID, SENTINEL_RW_TOKEN)]


def test_fetch_with_prompt_keep_thread_skips_cleanup(example_com_fixture: Path) -> None:
    client = FixtureClient(example_com_fixture)
    _fetch_with_prompt(
        client,
        "https://example.com",
        "summarize",
        "example.com",
        max_chars=None,
        keep_thread=True,
    )
    assert client.deleted == []


def test_fetch_with_prompt_truncates_real_stream(example_com_fixture: Path) -> None:
    client = FixtureClient(example_com_fixture)
    result = _fetch_with_prompt(
        client, "https://example.com", "summarize", "example.com", max_chars=20
    )
    assert len(result.content) == 20
    assert result.truncated is True
    assert result.content == EXPECTED_ANSWER[:20]


def test_no_completed_marker_returns_partial_with_flag() -> None:
    """Fixture: 4 events with chunks, none of which carries COMPLETED.
    Verb must return the accumulated chunks with stream_complete=False —
    matches real-world "server cut the connection" recovery.
    """
    from pplx_agent_tools.verbs.fetch import _fetch_with_prompt

    client = FixtureClient(FIXTURES / "no-completed-marker.events.jsonl")
    result = _fetch_with_prompt(
        client, "https://example.com", "summarize", "example.com", max_chars=None
    )
    assert result.content == "Partial answer before cut"
    assert result.stream_complete is False
    # Even without COMPLETED, the verb captured backend_uuid/read_write_token
    # from the first event and will attempt thread cleanup.
    assert client.deleted == [("fake-uuid-123", "fake-token-456")]


def test_empty_stream_raises_schema_error() -> None:
    """Fixture: zero SSE events. No content + no COMPLETED → SchemaError
    because we have nothing to return and no signal that the stream
    finished. Distinct from the deadline-trip case (StreamDeadlineError).
    """
    from pplx_agent_tools.errors import SchemaError
    from pplx_agent_tools.verbs.fetch import _fetch_with_prompt

    client = FixtureClient(FIXTURES / "empty-stream.events.jsonl")
    with pytest.raises(SchemaError, match="no markdown_block content"):
        _fetch_with_prompt(
            client, "https://example.com", "summarize", "example.com", max_chars=None
        )


def test_sentinels_appear_in_sanitizer_source() -> None:
    """Supplementary text check: the sentinel literals this file asserts on
    still exist in the script. It cannot show they are USED — the behavioral
    tests below do that — but it names the drift cheaply when a literal is
    renamed on one side only.
    """
    assert SANITIZER_SCRIPT.exists(), f"missing script: {SANITIZER_SCRIPT}"
    script = SANITIZER_SCRIPT.read_text()
    # Each sentinel that the replay test asserts on MUST appear verbatim
    # in the script. Substring match is fine — the script wraps these in
    # the SENTINELS dict.
    for name, value in (
        ("SENTINEL_BACKEND_UUID", SENTINEL_BACKEND_UUID),
        ("SENTINEL_RW_TOKEN", SENTINEL_RW_TOKEN),
        ("SENTINEL_UUID", SENTINEL_UUID),
        ("SENTINEL_REDACTED", SENTINEL_REDACTED),
        ("SENTINEL_EMAIL", SENTINEL_EMAIL),
    ):
        assert value in script, f"{name} {value!r} not in sanitizer script"


# ---------- the sanitizer's behavior, not its source text ----------
#
# The scripts are not importable modules (hyphenated names, PEP 723 headers),
# so they are loaded by path. Asserting on `_scrub_node` is what makes a
# redaction regression fail here: a sentinel can stay spelled in the source
# while nothing applies it any more.


def _load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sanitizer() -> ModuleType:
    return _load_script(SANITIZER_SCRIPT, "re_sanitize_fetch_fixture")


@pytest.fixture(scope="module")
def capture() -> ModuleType:
    return _load_script(CAPTURE_SCRIPT, "re_capture_fetch_prompt")


# A payload carrying an identity under every JSON type the wire can use.
DIRTY_PAYLOAD = {
    "author_profile": {"account_id": "acct-secret-123", "name": "Alice"},
    "user_id": 123456,
    "user_ids": [1, 2],
    "author_username": "alice",
    "backend_uuid": "0e2a1f7c-real-uuid",
    "read_write_token": "live-session-token",
    "user_selected_model": "turbo",
    "uuid": "",
    "status": "COMPLETED",
    "thread_title": "mail alice@corp.example for the numbers",
}


def test_scrub_redacts_identities_under_every_json_type(sanitizer: ModuleType) -> None:
    """The defect this pins: dispatching on the value's type before the key
    let a dict, a list or a number under an identity key ride out verbatim.
    """
    out = sanitizer._scrub_node(dict(DIRTY_PAYLOAD))

    assert out["author_profile"] == SENTINEL_REDACTED
    assert "acct-secret-123" not in json.dumps(out)
    assert out["user_id"] == SENTINEL_REDACTED
    assert out["user_ids"] == SENTINEL_REDACTED
    assert out["author_username"] == "test_user"
    assert out["backend_uuid"] == SENTINEL_BACKEND_UUID
    assert out["read_write_token"] == SENTINEL_RW_TOKEN


def test_scrub_preserves_settings_and_shape(sanitizer: ModuleType) -> None:
    out = sanitizer._scrub_node(dict(DIRTY_PAYLOAD))
    # A model id is a request setting, not a person; the empty uuid is an
    # un-started plan step, and inventing a value there would hide the shape.
    assert out["user_selected_model"] == "turbo"
    assert out["uuid"] == ""
    assert out["status"] == "COMPLETED"


def test_scrub_redacts_emails_in_prose(sanitizer: ModuleType) -> None:
    out = sanitizer._scrub_node(dict(DIRTY_PAYLOAD))
    assert "alice@corp.example" not in json.dumps(out)
    assert SENTINEL_EMAIL in out["thread_title"]


def test_scrub_redacts_nested_and_embedded_identities(sanitizer: ModuleType) -> None:
    """Identifiers nest inside blocks and inside `text`, which is itself a
    JSON document serialized into a string — the walk must reach both.
    """
    payload = {
        "blocks": [{"plan": {"uuid": "real-uuid", "author_id": {"id": "x"}}}],
        "text": json.dumps([{"backend_uuid": "real", "user_id": 7}]),
    }
    out = sanitizer._scrub_node(payload)

    assert out["blocks"][0]["plan"]["uuid"] == SENTINEL_UUID
    assert out["blocks"][0]["plan"]["author_id"] == "00000000-0000-4000-8000-00000000000a"
    inner = json.loads(out["text"])
    assert inner[0]["backend_uuid"] == SENTINEL_BACKEND_UUID
    assert inner[0]["user_id"] == SENTINEL_REDACTED


def test_scrub_is_idempotent(sanitizer: ModuleType) -> None:
    """Re-sanitizing a committed fixture is the standing leak check, so a
    second pass must be a no-op rather than a diff.
    """
    once = sanitizer._scrub_node(dict(DIRTY_PAYLOAD))
    assert sanitizer._scrub_node(once) == once


def test_scrub_redacts_an_email_spanning_a_chunk_boundary(sanitizer: ModuleType) -> None:
    """`chunks` re-ships the answer in ~23-char slices; per-string scrubbing
    redacted the whole answer string and kept the same address verbatim across
    a slice boundary.
    """
    out = sanitizer._scrub({"chunks": ["mail alice@", "corp.example for it"]})

    assert "alice@corp.example" not in "".join(out["chunks"])
    assert SENTINEL_EMAIL in "".join(out["chunks"])
    # A clean capture keeps its slicing, so re-running the sanitizer is stable.
    clean = ["hello ", "world"]
    assert sanitizer._scrub({"chunks": clean})["chunks"] == clean


@pytest.mark.parametrize(
    "label",
    ["../x", "a/b", ".", "..", "", "/abs/path", "..\\x", "../../tests/fixtures/fetch-url/x"],
)
def test_capture_label_rejects_path_like_values(capture: ModuleType, label: str) -> None:
    """A capture is UNSANITIZED — it carries the session thread token and
    account ids. A path-like label would write it outside the gitignored
    capture directory, e.g. into tests/fixtures/.
    """
    with pytest.raises(argparse.ArgumentTypeError):
        capture._fixture_label(label)


def test_capture_label_accepts_a_bare_basename(capture: ModuleType) -> None:
    assert capture._fixture_label("paywalled-article") == "paywalled-article"


# ---------- capture-variety fixtures, driven through the CLI ----------
#
# These go through `cli_fetch.main` rather than `_fetch_with_prompt` because the
# invariant each one pins is an exit code, and the exit code is decided by
# `cli_fetch._finalize`, not by the verb.

PAYWALLED_URL = "https://www.wsj.com/articles/even-chinas-property-stalwart-isnt-immune-from-the-crisis-19799863"
PAYWALLED_PROMPT = "Quote the opening two paragraphs of this article verbatim."
NO_RESULTS_URL = "https://example.com"
NO_RESULTS_PROMPT = (
    "List every pricing tier in the pricing table on this page, with the monthly price for each."
)
MULTI_BLOCK_URL = "https://en.wikipedia.org/wiki/SQLite"
MULTI_BLOCK_PROMPT = (
    "Summarize this page under three titled sections: ## Overview, ## Notable features, "
    "## Limitations. Then add a markdown table of four notable releases with columns "
    "Version, Year, Highlight."
)

PAYWALLED_FIXTURE = FIXTURES / "paywalled-article-prompt.events.jsonl"
NO_RESULTS_FIXTURE = FIXTURES / "no-results-prompt.events.jsonl"
MULTI_BLOCK_FIXTURE = FIXTURES / "multi-block-prompt.events.jsonl"


def _load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _completed_answer(path: Path) -> str:
    """Perplexity's own join of the delta chunks, off the COMPLETED event.

    The verb must reconstruct exactly this from the streamed chunks; deriving
    the expectation from the capture rather than hard-coding prose keeps the
    assertion honest when the fixture is re-captured.
    """
    for event in reversed(_load_events(path)):
        if event.get("status") != "COMPLETED":
            continue
        for block in event.get("blocks") or []:
            if not isinstance(block, dict) or block.get("intended_usage") != "ask_text":
                continue
            mb = block.get("markdown_block")
            if isinstance(mb, dict) and isinstance(mb.get("answer"), str):
                return mb["answer"].strip()
    raise AssertionError(f"no COMPLETED ask_text markdown_block.answer in {path}")


def _run_cli_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    client: Any,
    argv: list[str],
) -> tuple[int, dict[str, Any], str]:
    """Drive `pplx fetch --prompt ... --json` against a canned client.

    Returns the exit code, the parsed stdout envelope (success or error shape)
    and stderr, which is where the CLI puts warnings and error prose.
    """
    monkeypatch.setattr(
        cli_runner.Client,
        "from_default_cookies",
        classmethod(lambda cls, **_: client),
    )
    rc = cli_fetch.main([*argv, "--json"])
    cap = capsys.readouterr()
    return rc, json.loads(cap.out), cap.err


def test_paywalled_refusal_is_still_a_complete_stream(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model that answers "I can't read this, it's paywalled" has produced a
    normal, finished answer. Only the stream's own health decides the exit code,
    so a refusal must not be reported as a partial or failed fetch.
    """
    events = _load_events(PAYWALLED_FIXTURE)
    assert all(e.get("status") != "FAILED" for e in events)

    client = FixtureClient(PAYWALLED_FIXTURE)
    rc, payload, _ = _run_cli_json(
        monkeypatch, capsys, client, [PAYWALLED_URL, "--prompt", PAYWALLED_PROMPT]
    )
    assert rc == EXIT_OK
    assert payload["stream_complete"] is True
    assert payload["content"].strip() != ""


def test_no_results_prose_is_content_not_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "Nothing found" is an answer. The empty-results case must be exit 0 with
    the prose as content — only a stream with zero events is an error.
    """
    client = FixtureClient(NO_RESULTS_FIXTURE)
    rc, payload, _ = _run_cli_json(
        monkeypatch, capsys, client, [NO_RESULTS_URL, "--prompt", NO_RESULTS_PROMPT]
    )
    assert rc == EXIT_OK
    assert payload["stream_complete"] is True
    assert payload["content"].strip() != ""


def test_empty_stream_is_the_error_case_not_no_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Contrast with the test above: zero SSE events is a SchemaError (exit 1)."""
    client = FixtureClient(FIXTURES / "empty-stream.events.jsonl")
    rc, payload, _ = _run_cli_json(
        monkeypatch, capsys, client, ["https://example.com", "--prompt", "summarize"]
    )
    assert rc == EXIT_GENERIC
    assert payload["error"]["type"] == "SchemaError"


def test_multi_block_answer_is_not_double_counted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 5 kB sectioned answer with a table, streamed over ~100 events: the
    case where any accumulation bug — a re-read of the COMPLETED event's full
    repaint, or a second `intended_usage` slipping past the chunk filter —
    shows up as a visibly doubled answer rather than a subtle off-by-one.
    """
    events = _load_events(MULTI_BLOCK_FIXTURE)
    usages = {
        b.get("intended_usage")
        for e in events
        for b in (e.get("blocks") or [])
        if isinstance(b, dict)
    }
    assert "ask_text" in usages

    client = FixtureClient(MULTI_BLOCK_FIXTURE)
    rc, payload, _ = _run_cli_json(
        monkeypatch, capsys, client, [MULTI_BLOCK_URL, "--prompt", MULTI_BLOCK_PROMPT]
    )
    assert rc == EXIT_OK
    assert payload["stream_complete"] is True
    assert payload["content"] == _completed_answer(MULTI_BLOCK_FIXTURE)


@pytest.mark.parametrize(
    "fixture",
    [PAYWALLED_FIXTURE, NO_RESULTS_FIXTURE, MULTI_BLOCK_FIXTURE],
    ids=["paywalled", "no-results", "multi-block"],
)
def test_truncation_never_flips_stream_complete(
    fixture: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Truncation is a presentation choice made after the stream finished;
    conflating it with an incomplete stream would hand agents exit 6 for a
    perfectly healthy fetch.
    """
    client = FixtureClient(fixture)
    rc, payload, _ = _run_cli_json(
        monkeypatch,
        capsys,
        client,
        ["https://example.com", "--prompt", "tldr", "--max-chars", "20"],
    )
    assert rc == EXIT_OK
    assert payload["stream_complete"] is True
    assert len(payload["content"]) == 20


class MidStreamFailureClient(_TestClientBase):
    """Yields `fail_after` events from a fixture, then dies like a dropped
    connection — the shape `wire.sse_post` now reports as a NetworkError.
    """

    def __init__(self, fixture_path: Path, fail_after: int) -> None:
        super().__init__()
        self._events = _load_events(fixture_path)
        self._fail_after = fail_after
        self.deleted: list[tuple[str, str]] = []

    def sse_post(  # type: ignore[override]
        self,
        path: str,
        body: dict[str, Any],
        *,
        max_total_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        for payload in self._events[: self._fail_after]:
            yield {"event": "message", "data": payload}
        raise NetworkError(f"SSE stream on {path} failed mid-stream: connection reset")

    def delete_thread(self, entry_uuid: str, read_write_token: str) -> bool:  # type: ignore[override]
        self.deleted.append((entry_uuid, read_write_token))
        return True


def test_midstream_network_error_exits_four_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], example_com_fixture: Path
) -> None:
    """The agent contract is a typed error with a documented exit code; a
    transport failure part-way through a stream must not surface as a crash.
    """
    client = MidStreamFailureClient(example_com_fixture, fail_after=5)
    rc, payload, err = _run_cli_json(
        monkeypatch, capsys, client, ["https://example.com", "--prompt", "summarize"]
    )
    assert rc == EXIT_NETWORK
    assert payload["error"]["type"] == "NetworkError"
    assert payload["error"]["exit_code"] == EXIT_NETWORK
    assert "Traceback" not in err
    assert err.startswith("pplx fetch: ")
