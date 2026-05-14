"""pplx-agent-tools: agent toolkit for Perplexity backed by web-session cookies.

Version is derived from git tags via hatch-vcs at build time and baked into
the package's wheel metadata. `importlib.metadata.version("pplx-agent-tools")`
reads from that — works after any standard install (`pip`, `uv`, `nix`), no
extra file-generation hook needed.

If the package isn't installed (e.g. `python -c 'import pplx_agent_tools'`
straight from a source tree without `pip install -e .`), falls back to a
dev sentinel.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v

try:
    __version__ = _v("pplx-agent-tools")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
