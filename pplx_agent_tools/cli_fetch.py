"""pplx fetch: URL → cleaned content (optional LLM extraction via --prompt).

Plain mode fetches the URL locally (curl_cffi + trafilatura); --prompt mode
routes through Perplexity's chat endpoint so its LLM can fetch+answer in
one round-trip. See docs/wire/fetch-url.md for the rationale.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from .cli_runner import run_verb
from .errors import EXIT_OK, EXIT_PARTIAL
from .render import render_fetch_json, render_fetch_text
from .verbs.fetch import FetchResult, fetch

_DEFAULT_PROMPT_TIMEOUT_SECONDS = 180.0


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
    parser.add_argument(
        "--keep-thread",
        action="store_true",
        help=(
            "for --prompt mode: keep the chat thread in your Perplexity UI. "
            "Default behavior deletes it post-call so agent runs don't pollute "
            "thread history. Also honors $PPLX_KEEP_THREADS=1."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "for --prompt mode: overall wall-clock deadline (seconds). On "
            "deadline trip, any accumulated content is returned with a "
            "'stream: incomplete' marker. Default: 180s (override via "
            "$PPLX_FETCH_TIMEOUT or 0 to disable). Ignored in plain-fetch mode."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help=(
            "for --prompt mode: emit a heartbeat dot to stderr per ~10 SSE "
            "events so concurrent backgrounded calls show liveness. Also "
            "honors $PPLX_PROGRESS=1."
        ),
    )
    return parser


def _resolve_timeout(arg: float | None) -> float | None:
    """CLI flag → env var → default. 0 means 'disable the deadline'."""
    if arg is not None:
        return None if arg <= 0 else arg
    env = os.environ.get("PPLX_FETCH_TIMEOUT")
    if env is not None:
        try:
            v = float(env)
        except ValueError:
            print(
                f"pplx fetch: ignoring non-numeric $PPLX_FETCH_TIMEOUT={env!r}",
                file=sys.stderr,
            )
            return _DEFAULT_PROMPT_TIMEOUT_SECONDS
        return None if v <= 0 else v
    return _DEFAULT_PROMPT_TIMEOUT_SECONDS


def _finalize(result: FetchResult, max_chars: int | None) -> int:
    """Verb-specific stderr warnings + EXIT_PARTIAL for incomplete streams.

    Truncation and stream-incomplete go to stderr so machine consumers can
    grep them without parsing the rendered header. Exit 6 (EXIT_PARTIAL) on
    incomplete-stream lets scripts using `$?` detect a salvaged-but-partial
    response distinct from a clean success — partial content is still on
    stdout for callers that can use it.
    """
    if result.truncated:
        print(f"warning: content truncated at {max_chars} chars", file=sys.stderr)
    if not result.stream_complete:
        print(
            "warning: stream did not reach COMPLETED (deadline or server cut); "
            "partial content returned (exit 6)",
            file=sys.stderr,
        )
        return EXIT_PARTIAL
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keep_thread = args.keep_thread or os.environ.get("PPLX_KEEP_THREADS") == "1"
    progress = args.progress or os.environ.get("PPLX_PROGRESS") == "1"
    timeout = _resolve_timeout(args.timeout) if args.prompt else None

    return run_verb(
        "fetch",
        args,
        requires_auth=True,
        run=lambda client: fetch(
            client,
            args.url,
            prompt=args.prompt,
            max_chars=args.max_chars,
            keep_thread=keep_thread,
            timeout=timeout,
            progress=progress,
        ),
        render_text=render_fetch_text,
        render_json=render_fetch_json,
        finalize=lambda result: _finalize(result, args.max_chars),
    )


if __name__ == "__main__":
    raise SystemExit(main())
