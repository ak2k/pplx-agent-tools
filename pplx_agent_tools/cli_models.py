"""pplx models: list available models + modes and the default model per mode.

Stateless GETs against /rest/models/{config,modes}. Surfaces what `--model` /
`--mode` accept for the research verb.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .cli_runner import run_verb
from .render import render_models_json, render_models_text
from .verbs.models import models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pplx models",
        description="List Perplexity models, modes, and the default model per mode.",
    )
    parser.add_argument("-j", "--json", action="store_true", help="output JSON")
    parser.add_argument(
        "--profile",
        help="cookie profile (default: $PPLX_PROFILE or 'default')",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_verb(
        "models",
        args,
        requires_auth=True,
        run=models,
        render_text=render_models_text,
        render_json=render_models_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
