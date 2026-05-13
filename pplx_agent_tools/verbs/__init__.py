"""Verb layer: per-verb request shape + response parsing → typed Result.

Each verb owns the knowledge of which Perplexity endpoint to call, what to
send, and how to parse the response into a typed Result. The transport seam
(`wire.Client`) is passed in so tests can swap a fake.

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
   and `main(argv: Sequence[str] | None) -> int`. main() builds a Client,
   calls the verb, catches PplxError to print + return exit_code(e), then
   dispatches to the right render_widget_* based on `--json`.

Then register `("widget", cli_widget.main, "...one-line description...")`
in `cli.VERBS` so `pplx widget` is dispatchable. Run `pplx --help` after
to confirm the verb is listed.

## Why three files

- Verb logic, render, and CLI parsing are all independently testable
  surfaces with different change cadences. Co-locating them in one file
  couples those cadences.
- `render.py` as a registry (see its docstring) keeps cross-verb
  formatting decisions visible in one place.
"""
