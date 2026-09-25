"""The committed SSE fixtures through the askstream decoder (U3 oracles 2
and 7). Legacy fixtures hold one JSON payload per line, bare or wrapped as
`{"event", "data"}`; each line is rebuilt as the SSE block it came from."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from pplx_agent_tools.askstream.drift import Drift, name_of
from pplx_agent_tools.askstream.frames import (
    IGNORED_ENVELOPE,
    AskFrame,
    BlockMalformed,
    Frame,
    Heartbeat,
    Unparseable,
    decode_frame,
)
from pplx_agent_tools.askstream.sse import decode_event
from pplx_agent_tools.wire import _SSEFramer

FIXTURES = Path(__file__).parent / "fixtures"
EVENT_FIXTURES = sorted(FIXTURES.rglob("*.events.jsonl"))
IDS = [str(p.relative_to(FIXTURES)) for p in EVENT_FIXTURES]

# Hand-written fixtures whose placeholder `backend_uuid` ("BU",
# "fake-uuid-123") is not a UUID; each frame carrying one yields exactly this.
PLACEHOLDER_ID = Drift("unexpected_type", name_of("backend_uuid"))
EXPECTED_DRIFT = {
    "ask/multi-step-sources.events.jsonl": Counter({PLACEHOLDER_ID: 7}),
    "fetch-url/no-completed-marker.events.jsonl": Counter({PLACEHOLDER_ID: 1}),
}


def _blocks(path: Path) -> Iterator[str]:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and set(record) == {"event", "data"}:
            yield f"event: {record['event']}\ndata: {json.dumps(record['data'])}"
        else:
            yield f"data: {line}"


def _decoded(path: Path) -> list[tuple[Frame, tuple[Drift, ...]]]:
    return [decode_event(b) for b in _blocks(path)]


def test_corpus_is_present() -> None:
    assert len(EVENT_FIXTURES) >= 8


@pytest.mark.parametrize("path", EVENT_FIXTURES, ids=IDS)
def test_fixture_decodes_with_no_unparseable_or_malformed(path: Path) -> None:
    for frame, _ in _decoded(path):
        assert not isinstance(frame, Unparseable)
        if isinstance(frame, AskFrame):
            assert not any(isinstance(b, BlockMalformed) for b in frame.blocks)


@pytest.mark.parametrize("path", EVENT_FIXTURES, ids=IDS)
def test_fixture_decodes_with_empty_drift_ledger(path: Path) -> None:
    drift = Counter(d for _, items in _decoded(path) for d in items)
    assert drift == EXPECTED_DRIFT.get(str(path.relative_to(FIXTURES)), Counter())


def test_every_ignored_envelope_key_occurs_in_a_fixture() -> None:
    """Dropping any key from `IGNORED_ENVELOPE` must make a fixture drift."""
    seen: set[str] = set()
    for path in EVENT_FIXTURES:
        for block in _blocks(path):
            payload = json.loads(block.split("data: ", 1)[1])
            if isinstance(payload, dict):
                seen.update(payload)
    assert IGNORED_ENVELOPE - seen == set()


@pytest.mark.parametrize("seed", range(3))
def test_framer_output_decodes_like_the_lines(seed: int) -> None:
    """The byte framer, fed CRLF lines, heartbeats and random chunk cuts,
    yields blocks that decode to the same frames as each payload alone."""
    rng = random.Random(seed)
    path = FIXTURES / "fetch-url" / "multi-block-prompt.events.jsonl"
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    stream = "".join(f": ping\r\n\r\ndata: {ln}\r\n\r\n" for ln in lines).encode()
    framer = _SSEFramer()
    blocks: list[str] = []
    pos = 0
    while pos < len(stream):
        step = rng.randint(1, 8192)
        blocks.extend(framer.feed(stream[pos : pos + step]))
        pos += step
    frames = [decode_event(b)[0] for b in blocks]
    assert frames[1::2] == [decode_frame(ln)[0] for ln in lines]
    assert all(isinstance(f, Heartbeat) for f in frames[0::2])
