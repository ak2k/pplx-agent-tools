"""pplx fetch: URL → cleaned content (optional LLM extraction via --prompt).

Plain mode fetches the URL locally (curl_cffi + trafilatura); --prompt mode
routes through Perplexity's chat endpoint so its LLM can fetch+answer in
one round-trip. See docs/wire/fetch-url.md for the rationale.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .errors import PplxError, exit_code
from .render import render_fetch_json, render_fetch_text
from .verbs.fetch import fetch
from .wire import Client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pplx fetch",
        description=(
            "Fetch a URL and return cleaned content. With --prompt, route "
            "through Perplexity's LLM for extraction in one round-trip."
        ),
    )
    parser.add_argument("url", help="URL to fetch")
    parser.add_argument(
        "--prompt",
        help="if set, ask Perplexity's LLM to fetch + answer about the URL",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="cap content length (chars); truncation reported in output",
    )
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
        print(f"pplx fetch: {e}", file=sys.stderr)
        return exit_code(e)

    try:
        result = fetch(
            client,
            args.url,
            prompt=args.prompt,
            max_chars=args.max_chars,
        )
    except PplxError as e:
        print(f"pplx fetch: {e}", file=sys.stderr)
        return exit_code(e)

    if args.json:
        print(json.dumps(render_fetch_json(result), indent=2))
    else:
        print(render_fetch_text(result))

    if result.truncated:
        print(
            f"warning: content truncated at {args.max_chars} chars",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
