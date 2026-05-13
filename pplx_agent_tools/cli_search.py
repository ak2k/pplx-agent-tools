"""pplx-search: agent-facing search CLI for Perplexity.

Queries Perplexity's web-session search via /rest/sse/perplexity_ask (see
docs/wire/search-web.md). Multi-query: positional args fire concurrently
and results are merged / deduped client-side by URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

from .errors import PplxError, exit_code
from .render import render_search_json, render_search_text
from .verbs.search import SEARCH_TYPES, SearchResult, search
from .wire import Client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pplx-search",
        description="Search Perplexity using your Pro subscription's web session.",
    )
    parser.add_argument("query", nargs="+", help="one or more search queries")
    parser.add_argument(
        "-t",
        "--type",
        default="web",
        choices=list(SEARCH_TYPES),
        help="search type (default: web; non-web types are Step 9)",
    )
    parser.add_argument("-n", "--limit", type=int, default=10, help="result count (default: 10)")
    parser.add_argument("--country", default="US", help="country code (default: US)")
    parser.add_argument("--domains", help="comma-separated include domains (web only)")
    parser.add_argument(
        "--excluded-domains", help="comma-separated exclude domains (web only)"
    )
    parser.add_argument("-j", "--json", action="store_true", help="output JSON")
    parser.add_argument(
        "--profile",
        help="cookie profile (default: $PPLX_PROFILE or 'default')",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    domains = _csv(args.domains)
    excluded_domains = _csv(args.excluded_domains)

    try:
        client = Client.from_default_cookies(profile=args.profile)
    except PplxError as e:
        print(f"pplx-search: {e}", file=sys.stderr)
        return exit_code(e)

    queries: list[str] = list(args.query)

    try:
        merged = _run_queries(
            client,
            queries,
            search_type=args.type,
            limit=args.limit,
            country=args.country,
            domains=domains,
            excluded_domains=excluded_domains,
        )
    except PplxError as e:
        print(f"pplx-search: {e}", file=sys.stderr)
        return exit_code(e)

    if args.json:
        print(json.dumps(render_search_json(merged), indent=2))
    else:
        print(render_search_text(merged))

    for w in merged.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def _csv(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    return [s.strip() for s in arg.split(",") if s.strip()]


def _run_queries(
    client: Client,
    queries: list[str],
    *,
    search_type: str,
    limit: int,
    country: str,
    domains: list[str] | None,
    excluded_domains: list[str] | None,
) -> SearchResult:
    """Run each query (concurrent for N > 1), merge by URL with first-seen wins."""
    if len(queries) == 1:
        return search(
            client,
            queries[0],
            search_type=search_type,
            limit=limit,
            country=country,
            domains=domains,
            excluded_domains=excluded_domains,
        )

    results: list[SearchResult] = [None] * len(queries)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
        futures = {
            pool.submit(
                search,
                client,
                q,
                search_type=search_type,
                limit=limit,
                country=country,
                domains=domains,
                excluded_domains=excluded_domains,
            ): i
            for i, q in enumerate(queries)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    seen: set[str] = set()
    merged_hits = []
    warnings: list[str] = []
    for r in results:
        warnings.extend(r.warnings)
        for h in r.hits:
            if h.url in seen:
                continue
            seen.add(h.url)
            merged_hits.append(h)

    merged_hits = merged_hits[: limit * len(queries)]
    return SearchResult(
        query=" | ".join(queries),
        type=search_type,
        hits=merged_hits,
        total=len(merged_hits),
        warnings=sorted(set(warnings)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
