"""pplx search: agent-facing search CLI for Perplexity.

Queries Perplexity's realtime search via /rest/realtime/search-web (see
docs/wire/search-web.md). Multi-query is server-side native — positional
args become the `queries[]` field in one round-trip; the server merges and
dedupes results.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .cli_runner import run_verb
from .render import render_search_json, render_search_text
from .verbs.search import search_many


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pplx search",
        description="Search Perplexity using your Pro subscription's web session.",
    )
    parser.add_argument("query", nargs="+", help="one or more search queries")
    parser.add_argument("-n", "--limit", type=int, default=10, help="result count (default: 10)")
    parser.add_argument("-j", "--json", action="store_true", help="output JSON")
    parser.add_argument(
        "--profile",
        help="cookie profile (default: $PPLX_PROFILE or 'default')",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_verb(
        "search",
        args,
        requires_auth=True,
        run=lambda client: search_many(client, list(args.query), limit=args.limit),
        render_text=render_search_text,
        render_json=render_search_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
