"""pplx research: Perplexity deep research (multi-step, cited).

Routes through /rest/sse/perplexity_ask in research mode. Session-creating but
runs incognito (no history pollution) + best-effort thread cleanup. Supports the
same --timeout / --stall-timeout → partial-result (exit 6) contract as
`pplx fetch --prompt`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from .cli_runner import resolve_timeout, run_verb
from .cli_types import PplxArgumentParser, duration
from .errors import EXIT_OK, EXIT_PARTIAL
from .render import render_research_json, render_research_text
from .verbs._ask_common import DEFAULT_STALL_SECONDS
from .verbs.research import DEFAULT_MODE, ResearchResult, research

# A hard cap, not the expected duration: a focused question finishes in
# ~90-120s but a broad one can run past 30 minutes, and a hung backend is
# caught by the stall guard long before this.
_DEFAULT_TIMEOUT_SECONDS = 3600.0


def build_parser() -> PplxArgumentParser:
    parser = PplxArgumentParser(
        prog="pplx research",
        description="Run Perplexity deep research (multi-step, cited) on a query.",
    )
    parser.add_argument("query", help="research question")
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        help=(
            f"research depth (default: {DEFAULT_MODE} = Perplexity Deep Research). "
            "'council' (aka 'agentic_research') = Model Council: 3 frontier models "
            "cross-checked (~80s; pick them with --council-models, else a default "
            "trio is sent). An unknown value is passed through as a raw model_preference."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "override the model_preference the --mode maps to (power users; a model "
            "incompatible with research fails fast). research accepts pplx_alpha / "
            "o4mini — see `pplx models`."
        ),
    )
    parser.add_argument(
        "--council-models",
        default=None,
        metavar="A,B,C",
        help=(
            "Model Council only (--mode council): comma-separated model ids to "
            "cross-check (e.g. gpt55_thinking,claude48opusthinking,gemini31pro_high). "
            "Omitted → Perplexity's default trio."
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
        help=(
            "keep the (incognito) research thread instead of deleting it. "
            "Default deletes it post-call. Also honors $PPLX_KEEP_THREADS=1."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=duration,
        default=None,
        help=(
            "overall wall-clock deadline (seconds). On deadline trip, any "
            "accumulated answer is returned with a 'stream: incomplete' marker "
            f"(exit 6). Default: {_DEFAULT_TIMEOUT_SECONDS:.0f}s "
            "(override via $PPLX_RESEARCH_TIMEOUT or 0 to disable)."
        ),
    )
    parser.add_argument(
        "--stall-timeout",
        type=duration,
        default=None,
        help=(
            "cut the stream after this many seconds without new content (server "
            "heartbeats and repeated frames don't count); a partial report is "
            "returned (exit 6), none "
            f"exits 4. Default: {DEFAULT_STALL_SECONDS:.0f}s ($PPLX_STALL_TIMEOUT, "
            "or 0 to disable)."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="emit a heartbeat dot to stderr per ~10 SSE events. Honors $PPLX_PROGRESS=1.",
    )
    return parser


def _finalize(result: ResearchResult) -> int:
    if not result.stream_complete:
        print(
            "warning: research stream did not reach COMPLETED (deadline, stall or cut); "
            "partial answer returned (exit 6)",
            file=sys.stderr,
        )
        return EXIT_PARTIAL
    if result.content_shortfall:
        # Stream finished, answer didn't: a plausible-looking short report is
        # the failure mode worth an exit code. Which of the several causes fired
        # is in `result.warnings`, which run_verb prints — stay generic here
        # rather than asserting one of them.
        print(
            "warning: research answer may be incomplete; see the warnings above for why (exit 6)",
            file=sys.stderr,
        )
        return EXIT_PARTIAL
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keep_thread = args.keep_thread or os.environ.get("PPLX_KEEP_THREADS") == "1"
    progress = args.progress or os.environ.get("PPLX_PROGRESS") == "1"
    timeout = resolve_timeout(
        args.timeout, "PPLX_RESEARCH_TIMEOUT", _DEFAULT_TIMEOUT_SECONDS, "research"
    )
    stall_seconds = resolve_timeout(
        args.stall_timeout, "PPLX_STALL_TIMEOUT", DEFAULT_STALL_SECONDS, "research"
    )
    council_models = (
        [m.strip() for m in args.council_models.split(",") if m.strip()]
        if args.council_models
        else None
    )

    return run_verb(
        "research",
        args,
        requires_auth=True,
        run=lambda client: research(
            client,
            args.query,
            mode=args.mode,
            model=args.model,
            council_models=council_models,
            keep_thread=keep_thread,
            timeout=timeout,
            stall_seconds=stall_seconds,
            progress=progress,
        ),
        render_text=render_research_text,
        render_json=render_research_json,
        finalize=_finalize,
    )


if __name__ == "__main__":
    raise SystemExit(main())
