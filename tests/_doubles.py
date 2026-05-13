"""Shared test-double base for `Client` subclasses.

Several tests subclass `pplx_agent_tools.wire.Client` to substitute the
network-touching methods (`post_json`, `sse_post`, `delete_thread`) with
canned responses. None of them want a real curl_cffi Session, but they
also can't simply skip `Client.__init__` — that trips CodeQL's
"missing super().__init__" rule on every new subclass.

`_TestClientBase` resolves both: it calls `super().__init__` with
throwaway cookies (CodeQL-clean) and centralises that setup so adding a
new attribute to `Client.__init__` only requires one update here.
"""

from __future__ import annotations

from pplx_agent_tools.wire import Client


class _TestClientBase(Client):
    """Inherit from this instead of `Client` directly when writing a test
    double. Subclasses call `super().__init__()` (no args) to inherit the
    throwaway-cookie setup; CodeQL sees the chained super() and is happy.
    """

    def __init__(self) -> None:
        # Allocates a curl_cffi Session we never use — subclasses override
        # every method that would read `_session`.
        super().__init__({"x": "y"})
