"""pplx ask: ask a question, get a synthesized cited answer (Pro Search).

The front-door Perplexity Q&A — `search` returns sources, `ask` returns an
answer. Model-selectable via --model (see `pplx models`). Session-creating but
incognito; supports the --timeout -> partial (exit 6) contract.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from .cli_runner import resolve_model, resolve_timeout, run_verb
from .errors import EXIT_OK, EXIT_PARTIAL
from .render import render_ask_json, render_ask_text
from .verbs.ask import DEFAULT_MODEL, AskResult, ask

_DEFAULT_TIMEOUT_SECONDS = 120.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pplx ask",
        description="Ask Perplexity a question and get a synthesized, cited answer.",
    )
    parser.add_argument("query", help="the question to ask")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"model_preference (default: {DEFAULT_MODEL} = 'Best', or $PPLX_ASK_MODEL "
            "/ $PPLX_MODEL). Pass a model id from `pplx models` — incl. thinking "
            "variants like 'claude48opusthinking' (Max). An invalid model fails fast."
        ),
    )
    parser.add_argument("-j", "--json", action="store_true", help="output JSON")
    parser.add_argument(
        "--profile",
        help="cookie profile (default: $PPLX_PROFILE or 'default')",
    )
    parser.add_argument(
        "--keep-thread",
        action="store_true",
        help="keep the (incognito) thread instead of deleting it. Honors $PPLX_KEEP_THREADS=1.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "overall wall-clock deadline (seconds). On deadline trip, any partial "
            f"answer is returned + 'stream: incomplete' marker (exit 6). Default: "
            f"{_DEFAULT_TIMEOUT_SECONDS:.0f}s ($PPLX_ASK_TIMEOUT, or 0 to disable)."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="emit a heartbeat dot to stderr per ~10 SSE events. Honors $PPLX_PROGRESS=1.",
    )
    return parser


def _finalize(result: AskResult) -> int:
    if not result.stream_complete:
        print(
            "warning: ask stream did not reach COMPLETED (deadline or cut); "
            "partial answer returned (exit 6)",
            file=sys.stderr,
        )
        return EXIT_PARTIAL
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keep_thread = args.keep_thread or os.environ.get("PPLX_KEEP_THREADS") == "1"
    progress = args.progress or os.environ.get("PPLX_PROGRESS") == "1"
    timeout = resolve_timeout(args.timeout, "PPLX_ASK_TIMEOUT", _DEFAULT_TIMEOUT_SECONDS, "ask")
    model = resolve_model(args.model, ("PPLX_ASK_MODEL", "PPLX_MODEL"), DEFAULT_MODEL)

    return run_verb(
        "ask",
        args,
        requires_auth=True,
        run=lambda client: ask(
            client,
            args.query,
            model=model,
            keep_thread=keep_thread,
            timeout=timeout,
            progress=progress,
        ),
        render_text=render_ask_text,
        render_json=render_ask_json,
        finalize=_finalize,
    )


if __name__ == "__main__":
    raise SystemExit(main())
