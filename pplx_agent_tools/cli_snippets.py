"""pplx snippets: batched query-relevant excerpts from N URLs.

Hybrid retrieval (BM25 + semantic) over locally-fetched content. See
verbs/snippets.py and docs/wire/snippets.md for the rationale.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .errors import PplxError, exit_code
from .render import render_snippets_json, render_snippets_text
from .verbs.snippets import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOKENS_PER_PAGE,
    snippets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pplx snippets",
        description=(
            "Fetch N URLs and extract query-relevant paragraphs from each "
            "using hybrid retrieval (BM25 keyword + semantic vector)."
        ),
    )
    parser.add_argument("query", help="query to retrieve against")
    parser.add_argument("url", nargs="+", help="one or more URLs")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"total token budget across all snippets (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--max-tokens-per-page",
        type=int,
        default=DEFAULT_MAX_TOKENS_PER_PAGE,
        help=f"per-URL token budget (default: {DEFAULT_MAX_TOKENS_PER_PAGE})",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"fastembed model name (default: {DEFAULT_EMBED_MODEL})",
    )
    parser.add_argument("-j", "--json", action="store_true", help="output JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result = snippets(
            args.query,
            list(args.url),
            max_tokens=args.max_tokens,
            max_tokens_per_page=args.max_tokens_per_page,
            embed_model=args.embed_model,
        )
    except PplxError as e:
        print(f"pplx snippets: {e}", file=sys.stderr)
        return exit_code(e)

    if args.json:
        print(json.dumps(render_snippets_json(result), indent=2))
    else:
        print(render_snippets_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
