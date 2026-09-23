"""pplx ask: ask a question, get a synthesized cited answer (Pro Search).

The front-door Perplexity Q&A — `search` returns sources, `ask` returns an
answer. Model-selectable via --model (see `pplx models`). Session-creating but
incognito; supports the --timeout -> partial (exit 6) contract.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from .cli_runner import resolve_model, resolve_timeout, run_verb
from .cli_types import PplxArgumentParser, duration
from .errors import EXIT_OK, EXIT_PARTIAL
from .render import grounding_summary, render_ask_json, render_ask_text
from .verbs._ask_common import COPILOT_STALL_SECONDS
from .verbs.ask import DEFAULT_MODEL, AskResult, ask

# A hard cap, not the expected duration. A thinking model on a long multi-part
# prompt can think for minutes and answer at ~5 min, so a tight cap turns a
# slow answer into no answer; a hung stream is cut by the stall guard instead.
# Kept under the 10 min an agent's foreground shell command is allowed.
_DEFAULT_TIMEOUT_SECONDS = 540.0


def build_parser() -> PplxArgumentParser:
    parser = PplxArgumentParser(
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
        type=duration,
        default=None,
        help=(
            "overall wall-clock deadline (seconds). On deadline trip, any partial "
            f"answer is returned + 'stream: incomplete' marker (exit 6). Default: "
            f"{_DEFAULT_TIMEOUT_SECONDS:.0f}s ($PPLX_ASK_TIMEOUT, or 0 to disable)."
        ),
    )
    parser.add_argument(
        "--stall-timeout",
        type=duration,
        default=None,
        help=(
            "cut the stream after this many seconds without new content (server "
            "heartbeats and repeated frames don't count); a partial answer is "
            "returned (exit 6), none "
            f"exits 4. Default: {COPILOT_STALL_SECONDS:.0f}s ($PPLX_STALL_TIMEOUT, "
            "or 0 to disable)."
        ),
    )
    parser.add_argument(
        "--no-grounded-check",
        action="store_true",
        help=(
            "skip the check that the answer's figures and names appear in its "
            "cited sources' titles/snippets (on by default; never changes the exit code)."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="emit a heartbeat dot to stderr per ~10 SSE events. Honors $PPLX_PROGRESS=1.",
    )
    return parser


def _finalize(result: AskResult) -> int:
    if result.grounding is not None and result.grounding.grounded is False:
        print(
            f"warning: ask answer not grounded in its sources: "
            f"{grounding_summary(result.grounding)}",
            file=sys.stderr,
        )
    if not result.stream_complete:
        print(
            "warning: ask stream did not reach COMPLETED (deadline, stall or cut); "
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
    stall_seconds = resolve_timeout(
        args.stall_timeout, "PPLX_STALL_TIMEOUT", COPILOT_STALL_SECONDS, "ask"
    )
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
            stall_seconds=stall_seconds,
            progress=progress,
            grounded_check=not args.no_grounded_check,
        ),
        render_text=render_ask_text,
        render_json=render_ask_json,
        finalize=_finalize,
    )


if __name__ == "__main__":
    raise SystemExit(main())
