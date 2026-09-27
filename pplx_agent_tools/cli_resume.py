"""pplx resume: get the report of a research run whose connection dropped or
whose client was killed, by reattaching to its thread.

Takes the thread uuid `pplx research` printed, or `--last` for the newest
resumable thread recorded for the profile. Creates no thread; the resumed
thread is deleted once its report is read, as research does.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from .cli_research import _DEFAULT_TIMEOUT_SECONDS, _finalize
from .cli_runner import resolve_timeout, run_verb
from .cli_types import PplxArgumentParser, duration, thread_id
from .handles import ThreadStore
from .render import render_resume_json, render_resume_text
from .verbs._ask_common import DEFAULT_STALL_SECONDS
from .verbs.research import ResearchResult
from .verbs.resume import last_resumable, resume
from .wire import Client


def build_parser() -> PplxArgumentParser:
    parser = PplxArgumentParser(
        prog="pplx resume",
        description=(
            "Get the report of a research run whose connection dropped or whose client "
            "was killed: reattach to its thread (kept about 24 h) and read it to the end."
        ),
    )
    parser.add_argument(
        "uuid",
        nargs="?",
        type=thread_id,
        help="the thread to resume, as `pplx research` printed it",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="resume the newest resumable research thread recorded for the profile",
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
            "keep the thread after its report is read instead of deleting it. "
            "Also honors $PPLX_KEEP_THREADS=1."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=duration,
        default=None,
        help=(
            "overall wall-clock deadline (seconds) for a thread still running. "
            f"Default: {_DEFAULT_TIMEOUT_SECONDS:.0f}s "
            "(override via $PPLX_RESEARCH_TIMEOUT or 0 to disable)."
        ),
    )
    parser.add_argument(
        "--stall-timeout",
        type=duration,
        default=None,
        help=(
            "cut the stream after this many seconds without new content. "
            f"Default: {DEFAULT_STALL_SECONDS:.0f}s ($PPLX_STALL_TIMEOUT, or 0 to disable)."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="emit a heartbeat dot to stderr per ~10 SSE events. Honors $PPLX_PROGRESS=1.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.uuid is None) == (not args.last):
        parser.error("give a thread uuid or --last, not both")
    keep_thread = args.keep_thread or os.environ.get("PPLX_KEEP_THREADS") == "1"
    progress = args.progress or os.environ.get("PPLX_PROGRESS") == "1"
    timeout = resolve_timeout(
        args.timeout, "PPLX_RESEARCH_TIMEOUT", _DEFAULT_TIMEOUT_SECONDS, "resume"
    )
    stall_seconds = resolve_timeout(
        args.stall_timeout, "PPLX_STALL_TIMEOUT", DEFAULT_STALL_SECONDS, "resume"
    )
    store = ThreadStore(args.profile)

    def run(client: Client) -> ResearchResult:
        uuid, notes = (args.uuid, []) if args.uuid else _last(store)
        return resume(
            client,
            uuid,
            store=store,
            keep_thread=keep_thread,
            timeout=timeout,
            stall_seconds=stall_seconds,
            progress=progress,
            notes=notes,
        )

    return run_verb(
        "resume",
        args,
        requires_auth=True,
        run=run,
        render_text=render_resume_text,
        render_json=render_resume_json,
        finalize=_finalize,
    )


def _last(store: ThreadStore) -> tuple[str, list[str]]:
    record, notes = last_resumable(store)
    return record.backend_uuid, notes


if __name__ == "__main__":
    raise SystemExit(main())
