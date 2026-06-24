# pplx-agent-tools — project notes

## Releasing changes

**Code changes here do NOT reach `ak2k-skills` consumers until a tag is cut.**
The `ak2k-skills` flake pins this repo to a specific tag (e.g.
`github:ak2k/pplx-agent-tools/v0.1.0`); Renovate auto-opens a bump PR
on every new GitHub release matching `vX.Y.Z`.

**Release recipe** (after merging a behaviour change to `main`):

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
behaviour. Skip for pure-internal refactors that don't change agent or
human consumer behaviour.

**SemVer:** breaking changes to verb signatures, CLI flags, or JSON output
shapes bump the **minor** while we're pre-1.0 (everything is "unstable"
per the `Development Status :: 1 - Planning` classifier). Fixes that
preserve all shapes bump the **patch**.

## Commands

- Run tests: `uv run --extra dev pytest -q`
- Lint + format check: `uv run --extra dev ruff check . && uv run --extra dev ruff format --check .`
- Typecheck: `uv run --extra dev basedpyright pplx_agent_tools/`
- Coverage: `uv run --extra dev pytest --cov` (gated at `fail_under = 60`)

**CI runs all four** — `ruff check`, `ruff format --check`, `basedpyright`,
and `pytest`. Running only `ruff check` locally misses the formatter;
always run both before pushing.

The `[tool.pyright]` table in `pyproject.toml` still configures the type
checker — basedpyright reads the same config keys as pyright (it's a
fork). Don't be confused by the `pyright` table name and `basedpyright`
binary; they're intentionally compatible.

## Layout

- `pplx_agent_tools/verbs/{search,fetch,snippets}.py` — verb logic; each
  returns a typed `*Result` dataclass.
- `pplx_agent_tools/render.py` — single rendering registry: every verb has
  a `render_<verb>_{text,json}` pair here. Concentrated on purpose; see
  module docstring.
- `pplx_agent_tools/wire.py` — HTTP/SSE `Client` (curl_cffi chrome
  impersonation, status-code branching to typed exceptions).
- `pplx_agent_tools/auth.py` — cookie loading + perms enforcement.

## Endpoint selection principle (stateless-first)

**Prefer the realtime / GET endpoints; treat ask-family endpoints as the
exception that requires the delete-thread cleanup discipline.**

Perplexity's surface splits into two classes:

- **Stateless** — create nothing server-side, leave no trace in the user's
  Library. `/rest/realtime/*` (`search-web`, `search-youtube`, `query-video`)
  and read-only `GET`s (`rate-limit/status`, `models/config`). `search-web`'s
  `session_id` is a throwaway UUID per call. Plain `fetch` + `snippets` are fully
  local. **This is the default class for new verbs** — verified to create zero
  threads (before/after `GET /rest/thread/list_recent`).
- **Ask-family / session-creating** — `/rest/sse/perplexity_ask` and anything
  scoped to an `entry_uuid`. These create a thread ("entry") in the user's
  history. Only `fetch --prompt` uses this today.

Discipline for the ask-family exception:

- Set `params.is_incognito: true` in the ask body. Verified: an incognito ask
  **never appears in `list_recent`**, whereas `is_incognito: false` does. This is
  strictly better than the legacy create-then-delete (no history pollution even
  if cleanup fails). The `backend_uuid` is still deletable by UUID as
  belt-and-suspenders, but deletion stops being load-bearing.
- Keep `delete_thread` cleanup as the secondary guard (`keep_thread=False`).

Gotcha — **the image/video/news "variant search" endpoints are NOT stateless.**
`/rest/media/search-images-and-videos` requires `{query, entry_uuid,
read_write_token}` and `/rest/sources/search/news` requires `{entry_uuid, limit,
page}` (both 422 without it). They are SERP tabs that enrich an *existing* answer
thread — i.e. downstream of an ask entry, not standalone searches. There is no
stateless multi-result image/video/news search. `realtime/query-video` is
"analyze a specific `video_url`", not a search. See
`docs/wire/endpoint-gap-analysis.md`.

## Adding a new verb

A verb lives in three files. Adding `pplx widget` means:

1. **`verbs/widget.py`** — define `WidgetResult` (dataclass) and a top-level
   `widget(client: Client, ...) -> WidgetResult` function. Raise typed
   exceptions from `errors.py` (`SchemaError`, `NetworkError`, etc.) on
   failure; never return None or raise generic Exception.
2. **`render.py`** — add `render_widget_text(result) -> str` and
   `render_widget_json(result) -> dict[str, Any]`. The JSON branch must
   include `"_pplx_tools_version": __version__` for agent consumers.
3. **`cli_widget.py`** — define `build_parser() -> argparse.ArgumentParser`
   and `main(argv: Sequence[str] | None) -> int`. `main()` builds a Client,
   calls the verb, catches `PplxError` to print + return `exit_code(e)`,
   then dispatches to the right `render_widget_*` based on `--json`.

Then register `("widget", cli_widget.main, "...one-line description...")`
in `cli.VERBS` so `pplx widget` is dispatchable. Run `pplx --help` after
to confirm the verb is listed.

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
