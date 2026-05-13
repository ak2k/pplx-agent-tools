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

Regenerate the fixture with `scripts/re-sanitize-fetch-fixture.py`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pplx_agent_tools.verbs.fetch import _fetch_with_prompt
from pplx_agent_tools.wire import Client

FIXTURES = Path(__file__).parent / "fixtures" / "fetch-url"

# Matches the sentinels in scripts/re-sanitize-fetch-fixture.py
SENTINEL_BACKEND_UUID = "00000000-0000-4000-8000-000000000001"
SENTINEL_RW_TOKEN = "TEST_RW_TOKEN"

# The example.com fetch's expected answer text. Derived from the
# `markdown_block.answer` field on the captured COMPLETED event — this is
# Perplexity's authoritative join of the delta chunks.
EXPECTED_ANSWER = (
    "This domain is for use in illustrative examples in documents. You may use this\n\n"
    "domain in literature without prior coordination or asking for permission.\n\n"
    "More information..."
)


class FixtureClient(Client):
    """Yields canned SSE events from a sanitized .events.jsonl fixture.

    Each line in the fixture is the `data` payload of one SSE event; we wrap
    it in the `{event, data}` envelope shape that the real `sse_post` yields.
    """

    def __init__(self, fixture_path: Path) -> None:
        self._cookies = {"fake": "cookie"}
        self._base_url = "https://www.perplexity.ai"
        self._timeout = 1.0
        self._events: list[dict[str, Any]] = [
            json.loads(line) for line in fixture_path.read_text().splitlines() if line.strip()
        ]
        self.deleted: list[tuple[str, str]] = []

    def sse_post(self, path: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:  # type: ignore[override]
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
    _fetch_with_prompt(
        client, "https://example.com", "summarize", "example.com", max_chars=None
    )
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
