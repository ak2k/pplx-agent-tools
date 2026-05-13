"""Top-level `pplx` dispatcher.

Routes `pplx <verb> [args]` to the appropriate verb's main(). Each verb
keeps its own argparse parser; this module is a thin router so verbs stay
independently testable.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from . import cli_auth, cli_fetch, cli_search, cli_snippets

VERBS: dict[str, tuple[Callable[[Sequence[str] | None], int], str]] = {
    "search": (cli_search.main, "Search Perplexity for ranked web hits"),
    "fetch": (cli_fetch.main, "Fetch a URL (use --prompt for LLM extraction)"),
    "snippets": (cli_snippets.main, "Extract query-relevant excerpts from N URLs"),
    "auth": (cli_auth.main, "Manage Perplexity web-session cookies"),
}


def _top_level_help() -> str:
    lines = [
        "usage: pplx <verb> [args]",
        "",
        "Perplexity agent toolkit, backed by your Pro subscription's web session.",
        "",
        "verbs:",
    ]
    for name, (_, desc) in VERBS.items():
        lines.append(f"  {name:10s} {desc}")
    lines.append("")
    lines.append("Run `pplx <verb> --help` for details on each verb.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(_top_level_help())
        return 0
    cmd = argv[0]
    rest = list(argv[1:])
    if cmd not in VERBS:
        print(_top_level_help(), file=sys.stderr)
        print(f"\npplx: unknown verb {cmd!r}", file=sys.stderr)
        return 1
    fn, _ = VERBS[cmd]
    return fn(rest)


if __name__ == "__main__":
    raise SystemExit(main())
