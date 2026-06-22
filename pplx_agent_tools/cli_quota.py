"""pplx quota: show subscription rate-limit / availability.

Stateless GET against /rest/rate-limit/status — useful for an agent loop to
check whether an expensive mode (research) is still available before firing.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .cli_runner import run_verb
from .render import render_quota_json, render_quota_text
from .verbs.quota import quota


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pplx quota",
        description="Show Perplexity rate-limit / availability for your subscription.",
    )
    parser.add_argument("-j", "--json", action="store_true", help="output JSON")
    parser.add_argument(
        "--profile",
        help="cookie profile (default: $PPLX_PROFILE or 'default')",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_verb(
        "quota",
        args,
        requires_auth=True,
        run=quota,
        render_text=render_quota_text,
        render_json=render_quota_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
