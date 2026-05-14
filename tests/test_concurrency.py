"""Concurrency-shape tests for snippets per-host grouping and fetch jitter.

These lock the two concurrency fixes from Move 5/6:

  1. snippets._fetch_all groups URLs by host, parallelizes across hosts,
     serializes within a host (no 6-way burst at a single origin).
  2. fetch._rate_limit_backoff applies ±15% multiplicative jitter so N
     parallel callers honoring the same retry-after don't reform the
     thundering herd at wake-up time.

Both fixes are local — they don't solve inter-process coordination,
which would require shared state (file lock / token bucket). The agent
contract is that batch callers honor exit-3 themselves.
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from typing import Any
from unittest.mock import patch

from pplx_agent_tools.errors import RateLimitError
from pplx_agent_tools.verbs.fetch import (
    _BACKOFF_JITTER_HIGH,
    _BACKOFF_JITTER_LOW,
    _RATE_LIMIT_BACKOFF_CAP,
    _RATE_LIMIT_DEFAULT_BACKOFF,
    _rate_limit_backoff,
)

# ====================================================================
# Snippets per-host grouping
# ====================================================================


def _patch_fetch_page_with_timing() -> tuple[list[tuple[str, float, float]], Any]:
    """Returns (timings_log, patch_target) for monkey-patching fetch_page.

    Each call records (url, start_time, end_time). Each call sleeps 50ms
    so concurrent invocations on different hosts are distinguishable from
    serialized invocations on the same host.
    """
    log: list[tuple[str, float, float]] = []
    log_lock = threading.Lock()

    def stub(url: str, domain: str, *, max_chars: int | None, session: Any = None) -> Any:
        from pplx_agent_tools.verbs.fetch import FetchResult

        del session  # ignored — Session-reuse correctness has its own test
        start = time.monotonic()
        time.sleep(0.05)
        end = time.monotonic()
        with log_lock:
            log.append((url, start, end))
        return FetchResult(
            url=url, title=None, domain=domain, content=f"body-{url}", is_extracted=False
        )

    return log, stub


def _max_concurrent_for_host(log: list[tuple[str, float, float]], host: str) -> int:
    """How many calls on `host` overlapped in time at any moment."""
    host_calls = [(s, e) for (url, s, e) in log if host in url]
    max_overlap = 0
    for i, (s_i, e_i) in enumerate(host_calls):
        overlap = 1
        for j, (s_j, e_j) in enumerate(host_calls):
            if i == j:
                continue
            # Two windows overlap iff start_j < end_i AND start_i < end_j
            if s_j < e_i and s_i < e_j:
                overlap += 1
        max_overlap = max(max_overlap, overlap)
    return max_overlap


def test_fetch_all_serializes_same_host_calls() -> None:
    """6 URLs from one host must execute serially (max-concurrent == 1)."""
    from pplx_agent_tools.verbs.snippets import _fetch_all

    urls = [f"https://example.com/page-{i}" for i in range(6)]
    log, stub = _patch_fetch_page_with_timing()

    with patch("pplx_agent_tools.verbs.fetch.fetch_page", side_effect=stub):
        out = _fetch_all(urls)

    assert len(out) == 6
    # All 6 same-host calls must have ZERO time overlap with each other
    assert _max_concurrent_for_host(log, "example.com") == 1


def test_fetch_all_parallelizes_across_hosts() -> None:
    """3 URLs on 3 different hosts must execute concurrently."""
    from pplx_agent_tools.verbs.snippets import _fetch_all

    urls = [
        "https://a.example/x",
        "https://b.example/y",
        "https://c.example/z",
    ]
    log, stub = _patch_fetch_page_with_timing()

    with patch("pplx_agent_tools.verbs.fetch.fetch_page", side_effect=stub):
        _fetch_all(urls)

    # Total wall-clock must be roughly one stub-sleep (50ms), not 3x
    # — proves the three calls happened in parallel.
    earliest_start = min(s for _, s, _ in log)
    latest_end = max(e for _, _, e in log)
    span = latest_end - earliest_start
    # Generous bound for CI slowness; serial would be >=150ms.
    assert span < 0.12, f"hosts didn't parallelize (span={span:.3f}s)"


def test_fetch_all_preserves_input_order() -> None:
    """Output order matches input order even when host groups finish out of order."""
    from pplx_agent_tools.verbs.snippets import _fetch_all

    urls = [
        "https://a.example/1",
        "https://b.example/2",
        "https://a.example/3",
        "https://c.example/4",
        "https://b.example/5",
    ]
    _, stub = _patch_fetch_page_with_timing()

    with patch("pplx_agent_tools.verbs.fetch.fetch_page", side_effect=stub):
        out = _fetch_all(urls)

    out_urls = [row[0] for row in out]
    assert out_urls == urls


def test_fetch_all_reuses_one_session_per_host() -> None:
    """Same-host URLs share one curl_cffi Session (TCP reuse + HTTP/2);
    different-host URLs get different Sessions. The point of grouping
    by host is BOTH politeness AND speed — without session reuse, each
    serial call would still cost a full handshake.
    """
    from pplx_agent_tools.verbs.fetch import FetchResult
    from pplx_agent_tools.verbs.snippets import _fetch_all

    seen: list[tuple[str, int]] = []

    def stub(url: str, domain: str, *, max_chars: int | None, session: Any = None) -> FetchResult:
        # Track which Session instance (by id) each URL was fetched with.
        seen.append((url, id(session) if session is not None else 0))
        return FetchResult(url=url, title=None, domain=domain, content="x", is_extracted=False)

    urls = [
        "https://a.example/1",
        "https://a.example/2",
        "https://b.example/3",
        "https://a.example/4",
    ]
    with patch("pplx_agent_tools.verbs.fetch.fetch_page", side_effect=stub):
        _fetch_all(urls)

    by_url = dict(seen)
    # All 3 URLs on a.example share the same Session (same id)
    a_session_ids = {by_url[u] for u in urls if "a.example" in u}
    assert len(a_session_ids) == 1
    # The single a.example session is NOT zero (i.e. one was passed in)
    assert 0 not in a_session_ids
    # b.example used a different Session instance
    assert by_url["https://b.example/3"] not in a_session_ids


def test_fetch_all_errors_within_host_dont_skip_remaining() -> None:
    """One URL fails on host A; the next URL on host A still gets fetched.

    The verb captures errors per-URL rather than aborting the host group,
    so retry-friendly callers can see which specific URLs failed.
    """
    from pplx_agent_tools.verbs.fetch import FetchResult
    from pplx_agent_tools.verbs.snippets import _fetch_all

    def stub(url: str, domain: str, *, max_chars: int | None, session: Any = None) -> Any:
        del session  # accept new kwarg; this test isn't about session reuse
        if "fail" in url:
            from pplx_agent_tools.errors import NetworkError

            raise NetworkError(f"forced fail on {url}")
        return FetchResult(url=url, title=None, domain=domain, content="ok", is_extracted=False)

    urls = [
        "https://example.com/ok-1",
        "https://example.com/fail-2",
        "https://example.com/ok-3",
    ]
    with patch("pplx_agent_tools.verbs.fetch.fetch_page", side_effect=stub):
        out = _fetch_all(urls)

    by_url = {row[0]: row for row in out}
    assert by_url["https://example.com/ok-1"][2] is None  # no error
    assert "forced fail" in by_url["https://example.com/fail-2"][2]  # captured
    assert by_url["https://example.com/ok-3"][2] is None  # NOT skipped


# ====================================================================
# Fetch rate-limit jitter
# ====================================================================


def test_rate_limit_backoff_applies_jitter_to_retry_after() -> None:
    """100 samples of `retry_after=5` should land in [4.25, 5.75]."""
    err = RateLimitError("limited", retry_after=5.0)
    samples = [_rate_limit_backoff(err, remaining=None) for _ in range(100)]
    assert all(_BACKOFF_JITTER_LOW * 5.0 <= s <= _BACKOFF_JITTER_HIGH * 5.0 for s in samples)
    # Some spread is required — if all 100 samples landed at the exact
    # same value, jitter would be ineffective.
    assert len(set(samples)) > 50


def test_rate_limit_backoff_jitters_default_when_no_retry_after() -> None:
    """When the 429 didn't carry retry-after, we use the default + jitter."""
    err = RateLimitError("limited", retry_after=None)
    samples = [_rate_limit_backoff(err, remaining=None) for _ in range(50)]
    lo = _BACKOFF_JITTER_LOW * _RATE_LIMIT_DEFAULT_BACKOFF
    hi = _BACKOFF_JITTER_HIGH * _RATE_LIMIT_DEFAULT_BACKOFF
    assert all(lo <= s <= hi for s in samples)


def test_rate_limit_backoff_respects_cap_even_with_jitter() -> None:
    """A wildly large retry-after is capped BEFORE jitter — jitter never
    pushes us above the cap x _BACKOFF_JITTER_HIGH boundary.
    """
    err = RateLimitError("limited", retry_after=99999.0)
    samples = [_rate_limit_backoff(err, remaining=None) for _ in range(50)]
    upper = _BACKOFF_JITTER_HIGH * _RATE_LIMIT_BACKOFF_CAP
    assert all(s <= upper for s in samples)


def test_rate_limit_backoff_respects_remaining_deadline() -> None:
    """When the wall-clock budget is smaller than the jittered backoff,
    we clip to remaining rather than oversleeping past the deadline.
    """
    err = RateLimitError("limited", retry_after=10.0)
    # remaining=0.5s, jittered backoff would be 8.5-11.5s
    samples = [_rate_limit_backoff(err, remaining=0.5) for _ in range(20)]
    assert all(s <= 0.5 for s in samples)


def test_rate_limit_backoff_distribution_centered() -> None:
    """Sanity: mean of jittered samples is roughly the base. This is the
    sleep-aware check that "we still respect the server" — jitter spreads,
    doesn't bias.
    """
    random.seed(42)  # determinism for the assertion bounds
    err = RateLimitError("limited", retry_after=10.0)
    samples = [_rate_limit_backoff(err, remaining=None) for _ in range(500)]
    mean = sum(samples) / len(samples)
    # Expected mean is 10.0; allow generous bounds for 500-sample noise.
    assert 9.7 <= mean <= 10.3, f"jitter biased mean to {mean:.2f}"


def test_rate_limit_backoff_clamps_negative_to_zero() -> None:
    """If remaining is negative (impossible budget), return 0 not a negative."""
    err = RateLimitError("limited", retry_after=5.0)
    assert _rate_limit_backoff(err, remaining=-1.0) == 0.0


# Quiet the unused-import linters; defaultdict here documents the
# defaultdict-based grouping in _fetch_all (single-source-of-truth helper).
_ = defaultdict
