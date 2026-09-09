"""Fixture-replay tests for verbs/research.py.

Feeds a real (sanitized, trimmed) deep-research SSE stream through `research()`
to catch upstream schema regressions and snapshot-decode bugs that the synthetic
blocks in `test_verbs_research.py` can't reproduce.

Observed shape (weather-nowcasting-apis.events.jsonl — 8 of 802 captured frames):
  - frames 0-3: scaffolding then the growing snapshot (search rounds, thoughts,
    the report body streaming into a RESEARCH_ANSWER asset)
  - frame 4: first `text_completed: True`, status still PENDING
  - frame 6: status COMPLETED carrying the authoritative repaint
  - frame 7: `{}` — the stream's final empty frame

Two defects are pinned here, both of which shipped a plausible-looking wrong
answer (exit 0, ~1.2k chars) instead of the 9.7k-char report:
  1. the report body lives in the RESEARCH_ANSWER block's report asset, not in
     FINAL's `content.answer` (which holds only the cover note);
  2. the shared completion predicate accepts `text_completed`, which fires
     before the terminal repaint — research overrides it (`_status_completed`).

Regenerate with scripts/re-capture-research.py + scripts/re-sanitize-research-fixture.py.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pplx_agent_tools.verbs._ask_common import event_marks_completed
from pplx_agent_tools.verbs.research import decode_research_text, research
from tests._doubles import _TestClientBase

FIXTURES = Path(__file__).parent / "fixtures" / "research"
SANITIZER_SCRIPT = Path(__file__).parent.parent / "scripts" / "re-sanitize-research-fixture.py"

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


def _headings(answer: str) -> list[str]:
    return [line for line in answer.splitlines() if line.startswith("#")]


def test_research_replays_real_stream(weather_fixture: Path) -> None:
    client = FixtureClient(weather_fixture)
    result = research(client, "compare weather nowcasting APIs")

    assert result.stream_complete is True
    assert result.content_shortfall is False
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
