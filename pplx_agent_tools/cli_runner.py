"""Generic verb runner: lifts the agent-contract ceremony out of each cli_X.py.

Every CLI verb has the same skeleton: parse args → set up client (if it
needs auth) → call the verb → render result → emit warnings → return
exit code. Repeating that in five files is bug-magnet territory — the
silent break in PR #5 (JSON envelope drift) happened in part because
each verb's render path was independent.

`run_verb()` owns the skeleton. Each `cli_X.py` shrinks to:

  1. its argparse builder
  2. a `run(client)` lambda that invokes the verb
  3. text/json renderers (already in render.py)
  4. an optional `finalize(result)` for verb-specific tail behavior
     (e.g. fetch's truncated/partial warnings + EXIT_PARTIAL)

Verbs that don't need auth (snippets, plain fetch) pass `requires_auth=False` and
their `run` callable receives `client=None` (typically ignored). The
overloads narrow `client` to `Client` (non-None) when `requires_auth=True`
so verb callees that demand a non-None client don't need cast/assert
boilerplate at every call site.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from argparse import ArgumentTypeError, Namespace
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeVar, overload

from .cli_types import DISABLED, Duration, duration
from .errors import EXIT_GENERIC, EXIT_OK, PplxError, exit_code
from .render import envelope
from .wire import Client

R = TypeVar("R")


def resolve_model(arg: str | None, env_vars: Sequence[str], default: str) -> str:
    """Resolve a model preference: explicit --model flag → env vars (in order) →
    default. Lets a user set e.g. $PPLX_ASK_MODEL=claude48opusthinking once
    instead of passing --model every call, while the flag still wins per-call."""
    if arg:
        return arg
    for ev in env_vars:
        v = os.environ.get(ev)
        if v:
            return v
    return default


def resolve_timeout(arg: Duration | None, env_var: str, default: float, verb: str) -> float | None:
    """Resolve a wall-clock deadline: --timeout flag → env var → default. Returns
    None when disabled (0 or negative). An env value `duration` rejects is warned
    about and ignored. Shared by the ask-family CLIs."""
    if arg is None:
        env = os.environ.get(env_var)
        if env is None:
            return default
        try:
            arg = duration(env)
        except ArgumentTypeError as e:
            print(f"pplx {verb}: ignoring ${env_var}={env!r}: {e}", file=sys.stderr)
            return default
    return None if arg is DISABLED else arg


def _emit_error(name: str, err: PplxError, args: Namespace) -> int:
    """Print the error to stderr and, under --json, a parseable error envelope
    to stdout, then return the mapped exit code.

    Without this, a `--json` run that fails writes nothing to stdout and the
    reason only to stderr, forcing pure-JSON consumers to scrape stderr text.
    The error envelope mirrors the success shape (`_pplx_tools_version`,
    `_verb`) so a consumer can branch on the presence of an `error` key.
    """
    print(f"pplx {name}: {err}", file=sys.stderr)
    return _emit_error_envelope(name, type(err).__name__, str(err), exit_code(err), args)


def _emit_error_envelope(
    name: str, type_name: str, message: str, code: int, args: Namespace
) -> int:
    if getattr(args, "json", False):
        error_obj = {"type": type_name, "message": message, "exit_code": code}
        print(json.dumps(envelope(name, {"error": error_obj}), indent=2))
    return code


@overload
def run_verb(
    name: str,
    args: Namespace,
    *,
    requires_auth: Literal[True],
    run: Callable[[Client], R],
    render_text: Callable[[R], str],
    render_json: Callable[[R], dict[str, Any]],
    finalize: Callable[[R], int] | None = None,
) -> int: ...


@overload
def run_verb(
    name: str,
    args: Namespace,
    *,
    requires_auth: Literal[False],
    run: Callable[[Client | None], R],
    render_text: Callable[[R], str],
    render_json: Callable[[R], dict[str, Any]],
    finalize: Callable[[R], int] | None = None,
) -> int: ...


def run_verb(
    name: str,
    args: Namespace,
    *,
    requires_auth: bool,
    run: Callable[..., R],
    render_text: Callable[[R], str],
    render_json: Callable[[R], dict[str, Any]],
    finalize: Callable[[R], int] | None = None,
) -> int:
    """Execute a verb end-to-end with the standard agent contract.

    Contract maintained here (not by individual verbs):
    - Errors of type `PplxError` map to documented exit codes; any other
      exception is a bug: traceback to stderr, `InternalError` envelope, exit 1
    - JSON output goes through `render_json` (which uses `envelope()`)
    - Text output is the default; `--json` swaps to JSON
    - Result `.warnings` (if present) emit as `warning: <msg>` to stderr
    - Exit code defaults to `EXIT_OK`; `finalize` may override it

    `finalize(result) -> int` runs after rendering and returns the final
    exit code. Use it for verb-specific stderr warnings + non-zero exit
    codes that depend on the result (e.g. `EXIT_PARTIAL` when a fetch
    stream didn't reach COMPLETED). If `finalize` is None, the runner
    returns `EXIT_OK` after a successful render.
    """
    # Set once stdout carries the success document, so a later failure does
    # not append a second JSON document to it.
    wrote_stdout = False
    try:
        client: Client | None = None
        if requires_auth:
            try:
                client = Client.from_default_cookies(profile=getattr(args, "profile", None))
            except PplxError as e:
                return _emit_error(name, e, args)

        try:
            result = run(client)
        except PplxError as e:
            return _emit_error(name, e, args)

        if getattr(args, "json", False):
            print(json.dumps(render_json(result), indent=2))
        else:
            print(render_text(result))
        wrote_stdout = True

        for w in getattr(result, "warnings", []):
            print(f"warning: {w}", file=sys.stderr)

        if finalize is not None:
            return finalize(result)
        return EXIT_OK
    except Exception as e:
        # A bug, not an expected failure: keep the traceback for the report,
        # but still honor the --json contract of one parseable document.
        traceback.print_exc()
        print(f"pplx {name}: internal error: {type(e).__name__}: {e}", file=sys.stderr)
        if wrote_stdout:
            return EXIT_GENERIC
        return _emit_error_envelope(
            name, "InternalError", f"{type(e).__name__}: {e}", EXIT_GENERIC, args
        )
