"""pplx skill-path: print the absolute path to the bundled SKILL.md.

Useful for setting up agent discovery on uv-tool-installed setups:

    ln -sf "$(pplx skill-path)" ~/.claude/skills/pplx-agent-tools/SKILL.md

Nix users get this wiring automatically via the home-manager module.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from .cli_types import PplxArgumentParser


def find_skill_path() -> Path | None:
    """Find SKILL.md across install modes.

    1. Wheel/sdist install: hatch's force-include copies SKILL.md into the
       package. `importlib.resources.files` resolves it.
    2. Editable / development install: package dir has no SKILL.md;
       fall back to the repo root (one level up).
    """
    try:
        bundled = files("pplx_agent_tools") / "SKILL.md"
        if bundled.is_file():
            return Path(str(bundled))
    except Exception:
        pass
    editable = Path(__file__).resolve().parent.parent / "SKILL.md"
    if editable.is_file():
        return editable
    return None


def build_parser() -> PplxArgumentParser:
    return PplxArgumentParser(
        prog="pplx skill-path",
        description="Print the absolute path to the bundled SKILL.md.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print SKILL.md path."""
    build_parser().parse_args(argv)
    path = find_skill_path()
    if path is None:
        print(
            "pplx skill-path: SKILL.md not found in package or repo root; "
            "reinstall or check your install method.",
            file=sys.stderr,
        )
        return 1
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
