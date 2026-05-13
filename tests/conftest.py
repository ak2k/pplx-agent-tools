"""pytest config + shared fixtures for pplx-agent-tools."""

from __future__ import annotations

import os

import pytest

LIVE_ENV = "PPLX_LIVE_TESTS"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests in tests/test_live_*.py unless PPLX_LIVE_TESTS=1."""
    if os.environ.get(LIVE_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"set {LIVE_ENV}=1 to run live tests")
    for item in items:
        if "test_live_" in item.nodeid or item.get_closest_marker("live"):
            item.add_marker(skip)
