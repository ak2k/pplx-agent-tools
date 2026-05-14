"""pplx-agent-tools: agent toolkit for Perplexity backed by web-session cookies.

Version is derived from git tags via hatch-vcs at build time; the resolved
string lives in the generated `_version.py` (gitignored, written by the
build backend). Importing it lazily here keeps `__version__` a stable
public attribute without forcing setuptools-scm into the runtime path.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:
    # No build artifact yet (e.g. `python -c 'import pplx_agent_tools'`
    # in a fresh source tree without `pip install -e .`). Fall back to
    # installed-package metadata, then to a dev sentinel as last resort.
    try:
        from importlib.metadata import version as _v

        __version__ = _v("pplx-agent-tools")
    except Exception:
        __version__ = "0.0.0+unknown"
