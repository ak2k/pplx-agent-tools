"""pplx search: agent-facing search CLI for Perplexity.

Queries Perplexity's realtime search via /rest/realtime/search-web (see
docs/wire/search-web.md). Multi-query is server-side native — positional
args become the `queries[]` field in one round-trip; the server merges and
dedupes results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .errors import PplxError, exit_code
from .render import render_search_json, render_search_text
from .verbs.search import search_many
from .wire import Client


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

    try:
        client = Client.from_default_cookies(profile=args.profile)
    except PplxError as e:
        print(f"pplx search: {e}", file=sys.stderr)
        return exit_code(e)

    queries: list[str] = list(args.query)

    try:
        merged = search_many(client, queries, limit=args.limit)
    except PplxError as e:
        print(f"pplx search: {e}", file=sys.stderr)
        return exit_code(e)

    if args.json:
        print(json.dumps(render_search_json(merged), indent=2))
    else:
        print(render_search_text(merged))

    for w in merged.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
