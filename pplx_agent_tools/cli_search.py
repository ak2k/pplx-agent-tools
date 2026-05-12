"""pplx-search: agent-facing search CLI for Perplexity.

Planning stub. See the design plan in ak2k/nix-config:docs/plans/pplx-agent-tools.md.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pplx-search",
        description=(
            "Search Perplexity using your Pro subscription's web session. "
            "Planning stub — no implementation yet."
        ),
    )
    parser.add_argument("query", nargs="*", help="search query (one or more for web)")
    parser.add_argument(
        "-t",
        "--type",
        default="web",
        choices=["web", "academic", "images", "videos", "shopping"],
        help="search type (default: web)",
    )
    parser.add_argument("-n", "--limit", type=int, default=10, help="result count (default: 10)")
    parser.add_argument("--country", default="US", help="country code (default: US)")
    parser.add_argument("--domains", help="comma-separated include domains (web only)")
    parser.add_argument("--excluded-domains", help="comma-separated exclude domains (web only)")
    parser.add_argument("-j", "--json", action="store_true", help="output JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    print("pplx-search: not implemented yet. See README.md.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
