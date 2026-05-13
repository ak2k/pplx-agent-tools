# pplx-agent-tools — project notes

## Releasing changes

**Code changes here do NOT reach `ak2k-skills` consumers until a tag is cut.**
The `ak2k-skills` flake pins this repo to a specific tag (e.g.
`github:ak2k/pplx-agent-tools/v0.1.0`); Renovate auto-opens a bump PR
on every new GitHub release matching `vX.Y.Z`.

**Release recipe** (after merging a behaviour change to `main`):

1. Bump the version in **both**:
   - `pyproject.toml` — `version = "..."`
   - `pplx_agent_tools/__init__.py` — `__version__ = "..."`
   - `uv.lock` will auto-update on the next `uv sync` (or commit it directly via `uv lock --upgrade-package pplx-agent-tools`).
2. Commit (`Release vX.Y.Z`) and push to `main`. Wait for CI green.
3. `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`. Renovate picks up the new tag within an hour and opens a PR on `ak2k-skills`.

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
- Typecheck: `uv run --extra dev pyright pplx_agent_tools/`
- Coverage: `uv run --extra dev pytest --cov` (gated at `fail_under = 60`)

**CI runs all four** — `ruff check`, `ruff format --check`, `pyright`, and
`pytest`. Running only `ruff check` locally misses the formatter; always
run both before pushing.

## Layout

- `pplx_agent_tools/verbs/{search,fetch,snippets}.py` — verb logic; each
  returns a typed `*Result` dataclass.
- `pplx_agent_tools/render.py` — single rendering registry: every verb has
  a `render_<verb>_{text,json}` pair here. Concentrated on purpose; see
  module docstring.
- `pplx_agent_tools/wire.py` — HTTP/SSE `Client` (curl_cffi chrome
  impersonation, status-code branching to typed exceptions).
- `pplx_agent_tools/auth.py` — cookie loading + perms enforcement.
- See `verbs/__init__.py` docstring for the new-verb plug-in checklist.

## Test doubles

`tests/_doubles.py` defines `_TestClientBase` — a `Client` subclass that
calls `super().__init__({"x": "y"})` to satisfy CodeQL's
missing-super-init rule. Inherit test doubles from `_TestClientBase`
(not `Client` directly) and call `super().__init__()` in their `__init__`.
