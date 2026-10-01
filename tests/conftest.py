"""pytest config + shared fixtures for pplx-agent-tools."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

LIVE_ENV = "PPLX_LIVE_TESTS"

# The fixture sanitizers (loaded by path) import their shared sibling module as
# top-level `_fixture_account`, as they do when run directly from scripts/.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# The property tests assert that nothing crashes, not that it runs fast: Hypothesis'
# 200 ms per-example deadline and its too_slow health check trip on a loaded CI
# runner or a shared dev box with no code change, so both are off suite-wide.
settings.register_profile(
    "no-deadline", deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("no-deadline")


@pytest.fixture(scope="session", autouse=True)
def _private_state_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Research records its threads under $XDG_STATE_HOME; no test may write
    to the real one, and a build sandbox may have no writable home at all."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("xdg-state")))
        yield


@pytest.fixture(scope="session", autouse=True)
def _private_cookies(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """No test may read or write the developer's cookie files, whatever PPLX_*
    they export: a test that needs a cookie source or profile sets its own, and
    profile files default to a tmp dir."""
    with pytest.MonkeyPatch.context() as mp:
        for name in ("PPLX_COOKIES_PATH", "PPLX_COOKIES", "PPLX_PROFILE"):
            mp.delenv(name, raising=False)
        mp.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg-config")))
        yield


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: a benchmark row; deselect with -m 'not slow'")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests in tests/test_live_*.py unless PPLX_LIVE_TESTS=1."""
    if os.environ.get(LIVE_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"set {LIVE_ENV}=1 to run live tests")
    for item in items:
        if "test_live_" in item.nodeid or item.get_closest_marker("live"):
            item.add_marker(skip)
