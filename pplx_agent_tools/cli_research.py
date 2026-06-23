"""pplx research: Perplexity deep research (multi-step, cited).

Routes through /rest/sse/perplexity_ask in research mode. Session-creating but
runs incognito (no history pollution) + best-effort thread cleanup. Supports the
same --timeout → partial-result (exit 6) contract as `pplx fetch --prompt`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from .cli_runner import run_verb
from .errors import EXIT_OK, EXIT_PARTIAL
from .render import render_research_json, render_research_text
from .verbs.research import DEFAULT_MODE, ResearchResult, research

# Research is slower than copilot ask (~10-60s); give it a longer default leash.
_DEFAULT_TIMEOUT_SECONDS = 300.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pplx research",
        description="Run Perplexity deep research (multi-step, cited) on a query.",
    )
    parser.add_argument("query", help="research question")
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        help=(
            f"research depth (default: {DEFAULT_MODE} = Perplexity Deep Research). "
            "'council' (aka 'agentic_research') = Model Council: multiple frontier "
            "models cross-checked. EXPERIMENTAL — observed to run >7 min without "
            "completing and its answer block isn't decoded yet, so it usually times "
            "out empty; prefer 'research'. An unknown value is passed through as a "
            "raw model_preference."
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
        type=float,
        default=None,
        help=(
            "overall wall-clock deadline (seconds). On deadline trip, any "
            "accumulated answer is returned with a 'stream: incomplete' marker "
            f"(exit 6). Default: {_DEFAULT_TIMEOUT_SECONDS:.0f}s "
            "(override via $PPLX_RESEARCH_TIMEOUT or 0 to disable)."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="emit a heartbeat dot to stderr per ~10 SSE events. Honors $PPLX_PROGRESS=1.",
    )
    return parser


def _resolve_timeout(arg: float | None) -> float | None:
    """CLI flag → env var → default. 0 means 'disable the deadline'."""
    if arg is not None:
        return None if arg <= 0 else arg
    env = os.environ.get("PPLX_RESEARCH_TIMEOUT")
    if env is not None:
        try:
            v = float(env)
        except ValueError:
            print(
                f"pplx research: ignoring non-numeric $PPLX_RESEARCH_TIMEOUT={env!r}",
                file=sys.stderr,
            )
            return _DEFAULT_TIMEOUT_SECONDS
        return None if v <= 0 else v
    return _DEFAULT_TIMEOUT_SECONDS


def _finalize(result: ResearchResult) -> int:
    if not result.stream_complete:
        print(
            "warning: research stream did not reach COMPLETED (deadline or cut); "
            "partial answer returned (exit 6)",
            file=sys.stderr,
        )
        return EXIT_PARTIAL
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keep_thread = args.keep_thread or os.environ.get("PPLX_KEEP_THREADS") == "1"
    progress = args.progress or os.environ.get("PPLX_PROGRESS") == "1"
    timeout = _resolve_timeout(args.timeout)
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
            progress=progress,
        ),
        render_text=render_research_text,
        render_json=render_research_json,
        finalize=_finalize,
    )


if __name__ == "__main__":
    raise SystemExit(main())
