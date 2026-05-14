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

Verbs that don't need auth (snippets) pass `requires_auth=False` and
their `run` callable receives `client=None` (typically ignored). The
overloads narrow `client` to `Client` (non-None) when `requires_auth=True`
so verb callees that demand a non-None client don't need cast/assert
boilerplate at every call site.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from collections.abc import Callable
from typing import Any, Literal, TypeVar, overload

from .errors import EXIT_OK, PplxError, exit_code
from .wire import Client

R = TypeVar("R")


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
    - Errors of type `PplxError` map to documented exit codes
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
    client: Client | None = None
    if requires_auth:
        try:
            client = Client.from_default_cookies(profile=getattr(args, "profile", None))
        except PplxError as e:
            print(f"pplx {name}: {e}", file=sys.stderr)
            return exit_code(e)

    try:
        result = run(client)
    except PplxError as e:
        print(f"pplx {name}: {e}", file=sys.stderr)
        return exit_code(e)

    if getattr(args, "json", False):
        print(json.dumps(render_json(result), indent=2))
    else:
        print(render_text(result))

    for w in getattr(result, "warnings", []):
        print(f"warning: {w}", file=sys.stderr)

    if finalize is not None:
        return finalize(result)
    return EXIT_OK
