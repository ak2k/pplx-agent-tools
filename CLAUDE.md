# pplx-agent-tools — project notes

## Releasing changes

**Code changes here do NOT reach `ak2k-skills` consumers until a tag is cut.**
The `ak2k-skills` flake pins this repo to a specific tag (e.g.
`github:ak2k/pplx-agent-tools/v0.1.0`); Renovate auto-opens a bump PR
on every new GitHub release matching `vX.Y.Z`.

**Release recipe** (after merging a behavior change to `main`):

```bash
git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z
```

That's it — no source-file edits, no version strings to keep in sync.
The tag IS the version: `hatch-vcs` derives it at build time from
either git history (local dev) or the substituted `.git_archival.txt`
(GitHub-fetched tarballs, which is how Nix flake-consumes us via
`github:ak2k/pplx-agent-tools/vX.Y.Z`). Pushing the tag triggers
`.github/workflows/release.yml`, which creates the matching GitHub
**Release** — `ak2k-skills`' Renovate watches Releases (not bare tags), so
the Release is what opens the bump PR there (within ~an hour). A tag with no
Release never reaches consumers.

Between tags, `pplx --version` reports a dev marker like
`0.3.2.dev2+ga1b2c3d` — informative ("you're on a non-released build")
without anyone editing a string.

**When to tag:** any user-visible change to verbs / CLI / SKILL.md / wire
behavior. Skip for pure-internal refactors that don't change agent or
human consumer behavior.

**SemVer:** breaking changes to verb signatures, CLI flags, or JSON output
shapes bump the **minor** while we're pre-1.0 (everything is "unstable"
per the `Development Status :: 1 - Planning` classifier). Fixes that
preserve all shapes bump the **patch**.

## Commands

- Run tests: `uv run --extra dev pytest -q`
- Lint + format check: `uv run --extra dev ruff check . && uv run --extra dev ruff format --check .`
- Typecheck: `uv run --extra dev basedpyright pplx_agent_tools/ tests/`
- Coverage: `uv run --extra dev pytest --cov` (gated at `fail_under = 80`)
- Dead code: `uv run --extra dev vulture`

**CI is `nix flake check`** (`ci.yml`; CodeQL and gitleaks run as separate
workflows). It builds `checks.{package,nix-fmt,lint,typecheck,deadcode,tests}`:
`nix build .#default` (package), `nixfmt --check flake.nix` (nix-fmt),
`ruff check` + `ruff format --check` (lint), `basedpyright` (typecheck), `vulture`
(deadcode), and `pytest --cov` (tests), and **stops at the first failing check**
(so failures surface one at a time). The commands above skip nixfmt and the
package build — run `nix flake check` to reproduce the exact gate before pushing.

`nix flake check` builds from a snapshot of the **git** tree: untracked files are
invisible to it, so `git add` anything new before running it or the check runs
against a tree that is missing it.

`.python-version` pins uv to 3.12 because `rookiepy` publishes cp310–cp312 wheels
only — on 3.13 uv falls back to building it from source and fails. Inside
`nix develop` the shell's `UV_PYTHON` points at the flake's dev venv and takes
precedence over `.python-version`; the pin is what makes a bare `uv run` work outside that shell.

The `[tool.pyright]` table in `pyproject.toml` still configures the type
checker — basedpyright reads the same config keys as pyright (it's a
fork). Don't be confused by the `pyright` table name and `basedpyright`
binary; they're intentionally compatible. The modules listed in that table's
`strict` array (`askstream/`, `jsonval.py`, `cli_runner.py`, …) are checked in
strict mode; `auth.py`, `grounding.py` and `handles.py` opt in with a
`# pyright: strict` first line. Code there that decodes wire JSON takes `object` /
`JsonValue` and reads it through the `jsonval` accessors, never `Any`, because
strict mode does not flag `Any`.

## Layout

All paths are under `pplx_agent_tools/`.

- `cli.py` — the `pplx <verb>` dispatcher (`VERBS`).
- `cli_<verb>.py` — one argparse front end per verb.
- `cli_runner.py` — `run_verb`, which every `cli_<verb>.main` calls: it builds
  the `Client`, renders text or JSON, maps `PplxError` to its exit code and
  writes the JSON error envelope.
- `verbs/<verb>.py` — verb logic; each returns a typed `*Result` dataclass
  (`resume` returns it with the held thread).
  `verbs/_ask_common.py` is shared by the ask-family verbs (`ask`, `research`,
  `fetch --prompt`); `verbs/_research_stream.py` runs, salvages and cleans up a
  research stream for `research` and `resume`.
- `askstream/` — the diff-mode SSE decoder and run lifecycle: typed frames, an
  RFC 6902 patch applier with work caps, the block store and projections, and the
  lifecycle as a pure state machine with per-verb policy and cleanup. `research`
  and `resume` run on it through `_research_stream`; `ask` and `fetch --prompt`
  use `_ask_common.run_ask_stream`.
- `render.py` — single rendering registry: every verb with a Result type (all
  but `auth` and `skill-path`) has a `render_<verb>_{text,json}` pair here, and
  JSON goes through `envelope()`. Concentrated on purpose; see module docstring.
- `wire.py` — HTTP/SSE `Client` (curl_cffi chrome impersonation, status-code
  branching to typed exceptions, SSE reconnect and terminate).
- `auth.py` — cookie loading + perms enforcement.
- `errors.py` — typed errors, each mapped to an exit code 1–5, and the
  `EXIT_*` constants 0–6 (6, a partial result, comes from a verb's `finalize`).
  The codes are part of the CLI contract that SKILL.md documents.
- `netguard.py` — SSRF guard: checks every user-supplied URL that `fetch` or
  `snippets` fetches locally, redirect targets included.
- `handles.py` — local records of research threads, read by `pplx resume --last`.
- `grounding.py` — checks whether an `ask` answer's figures and names appear in
  its sources.

## Endpoint selection principle (stateless-first)

**Prefer the realtime / GET endpoints; treat ask-family endpoints as the
exception that requires the terminate-and-delete cleanup discipline.**

Perplexity's surface splits into two classes:

- **Stateless** — create nothing server-side, leave no trace in the user's
  Library. `/rest/realtime/*` (`search-web`, `search-youtube`, `query-video`)
  and read-only `GET`s (`rate-limit/status`, `models/config`, `models/modes`). `search-web`'s
  `session_id` is a throwaway UUID per call. Plain `fetch` + `snippets` are fully
  local. **This is the default class for new verbs** — verified to create zero
  threads (before/after `GET /rest/thread/list_recent`).
- **Ask-family / session-creating** — `/rest/sse/perplexity_ask` and anything
  scoped to an `entry_uuid`. These create a thread ("entry") in the user's
  history. `ask`, `research` and `fetch --prompt` use it, all building the body
  with `_ask_common.base_ask_params`. `resume` creates nothing: it reattaches to
  an existing research thread through `/rest/sse/perplexity_ask/reconnect/{uuid}`.

Discipline for the ask-family exception:

- Set `params.is_incognito: true` in the ask body. Verified: an incognito ask
  **never appears in `list_recent`**, whereas `is_incognito: false` does. This is
  strictly better than the legacy create-then-delete (no history pollution even
  if cleanup fails). Deleting the thread by UUID is a second safeguard; the
  incognito flag alone keeps it out of the history.
- On exit, terminate the run if it may still be live
  (`/rest/sse/perplexity_terminate`), then delete the thread, unless
  `--keep-thread` is passed, `$PPLX_KEEP_THREADS=1` is set, or the research case
  below applies. `ask` and `fetch --prompt` get this from
  `_ask_common.release_on_exit`, and `research` from `_research_stream.release`.
  `resume` gets the terminate there and the delete from
  `verbs/resume.HeldThread.release`, which runs only after the report is flushed
  to stdout. A new ask-family verb must call one of these; a bare
  `delete_thread` skips the terminate.
- `research` keeps the thread when the run may still be going on the server: the
  stream dropped or hit the silence timeout, or pplx stopped reading and could
  not terminate the run with no exception in flight (after Ctrl-C it still
  deletes). A record in `$XDG_STATE_HOME/perplexity/<profile>/threads/` is
  written when a frame first names the thread and removed when cleanup finishes,
  unless the run stays resumable. `pplx resume --last` picks a record left
  `running` (client killed) or marked `kept`; a completed run kept with
  `--keep-thread` leaves no record.

Gotcha — **the stateless GETs answer 200 to an anonymous caller.** With no
cookies (or an expired session), `rate-limit/status` returns a fully populated
table with every mode and source `{"available": false, "remaining_detail":
{"kind": "exact", "remaining": 0}}` — byte-for-byte the shape of an exhausted
account — and `models/config` returns the public catalog. The wire layer's
401/403 → `AuthError` mapping never fires on these. A verb whose payload is only
meaningful for *this* session must call `client.auth_session()` first (`quota`
does; captured fixture at `tests/fixtures/rate-limit-status/anonymous.json`).

Gotcha — **the image/video/news "variant search" endpoints are NOT stateless.**
`/rest/media/search-images-and-videos` requires `{query, entry_uuid,
read_write_token}` and `/rest/sources/search/news` requires `{entry_uuid, limit,
page}` (both 422 without it). They are SERP tabs that enrich an *existing* answer
thread — i.e. downstream of an ask entry, not standalone searches. There is no
stateless multi-result image/video/news search. `realtime/query-video` is
"analyze a specific `video_url`", not a search. See
`docs/wire/endpoint-gap-analysis.md`.

## Adding a new verb

Each verb has three files, plus an entry in `cli.VERBS`. Adding `pplx widget`
means:

1. **`verbs/widget.py`** — define `WidgetResult` (dataclass) and a top-level
   `widget(client: Client, ...) -> WidgetResult` function. Raise typed
   exceptions from `errors.py` (`SchemaError`, `NetworkError`, etc.) on
   failure; never return None or raise generic Exception.
2. **`render.py`** — add `render_widget_text(result) -> str` and
   `render_widget_json(result) -> dict[str, Any]`. The JSON function returns
   `envelope("widget", payload, warnings=result.warnings)`. It stamps
   `_pplx_tools_version` and `_verb`, adds `warnings` when that argument is
   non-empty, and raises `ValueError` if the payload sets any of the three.
3. **`cli_widget.py`** — define `build_parser() -> PplxArgumentParser` (from
   `cli_types`, which also holds argument validators such as `positive_int`) and
   `main(argv: Sequence[str] | None) -> int`. The parser needs `-j/--json` and
   `--profile` as `cli_search.py` has them: `run_verb` reads both with silent
   defaults, so a parser without them prints text and uses the default profile.
   `main()` calls `run_verb("widget", args, requires_auth=..., run=...,
   render_text=..., render_json=...)`. Pass `finalize` for an exit code that
   depends on the result, such as `EXIT_PARTIAL`.

Then import `cli_widget` in `cli.py` and add
`"widget": (cli_widget.main, "...one-line description...")` to `cli.VERBS` so
`pplx widget` is dispatchable. Run `pplx --help` after to confirm the verb is
listed.

**Why three files:** verb logic, render, and CLI parsing have different
change cadences and are independently testable; co-locating them in one
file couples those cadences. `render.py` as a single registry keeps
cross-verb formatting decisions (timestamps, JSON envelope, truncation
markers) visible in one place.

## Test doubles

`tests/_doubles.py` defines `_TestClientBase` — a `Client` subclass that
calls `super().__init__({"x": "y"})` to satisfy CodeQL's
missing-super-init rule. Inherit test doubles from `_TestClientBase`
(not `Client` directly) and call `super().__init__()` in their `__init__`.
The base also stubs `terminate` and `sse_reconnect`, but not `delete_thread`:
a double whose run reaches cleanup with a thread id and token must override
`delete_thread`, or cleanup sends a real DELETE.
