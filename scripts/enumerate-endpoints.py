#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["curl_cffi>=0.7"]
# ///
"""Enumerate perplexity.ai's /rest/ + /api/ endpoint surface via SPA bundle walk.

Perplexity migrated from a Next.js app (`_next/static/chunks/*.js`) to a
Vite/Rolldown SPA (`_spa/assets/*.js`) served from pplx-next-static-public.
The May-2026 enumeration walked `_next/static`; that path is now dead, which is
why a naive homepage grep finds zero endpoints today.

This walks the new layout:
  1. fetch the 12 KB app shell from www.perplexity.ai (curl_cffi passes CF where
     a Playwright browser is hard-blocked by CF's managed challenge),
  2. seed from the `_spa/assets/*.js` <script>/<link> URLs in the shell,
  3. BFS the chunk graph (Rolldown chunks reference each other by filename),
  4. grep every chunk for /rest/ + /api/ endpoint literals, inferring the HTTP
     method from the call-site context.

CF-free, no browser, no account-side actions — just static asset reads. Output:
a committed endpoint inventory + a diff against what pplx-agent-tools exposes.

Usage:
  scripts/enumerate-endpoints.py
  scripts/enumerate-endpoints.py --max-chunks 800
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from curl_cffi import requests

COOKIES_FILE = Path.home() / ".config/perplexity/default/cookies.json"
REPO = Path(__file__).resolve().parent.parent
TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
# Output is a sanitized inventory (paths/methods/chunk names only — no cookies,
# headers, or response bodies), so it lives under tracked docs/, not the
# gitignored re-fixtures/ scratch dir.
OUTDIR = REPO / "docs" / "wire" / "endpoint-inventory"

CDN_BASE = "https://pplx-next-static-public.perplexity.ai/_spa/assets/"
SHELL_URL = "https://www.perplexity.ai/"

# Already exposed by the CLI, or RE'd + documented in the May plan. Drives the diff.
KNOWN = {
    "/api/auth/session": "auth keepalive (EXPOSED via `pplx auth`)",
    "/rest/realtime/search-web": "web search (EXPOSED via `pplx search`)",
    "/rest/sse/perplexity_ask": "ask/chat SSE (EXPOSED via `pplx fetch --prompt`)",
    "/rest/realtime/search-youtube": "single-best-match video nav (RE'd, deferred)",
    "/rest/uploads/create_upload_url": "file uploads (RE'd, deferred Phase 2)",
}

CHUNK_REF = re.compile(r"""["'`]((?:\./)?[A-Za-z0-9][A-Za-z0-9._-]*-[A-Za-z0-9_-]{8}\.js)["'`]""")
EP_LITERAL = re.compile(r"""["'`](/(?:rest|api)/[A-Za-z0-9_./:${}-]+)["'`]""")
# method inference from a window of text around the endpoint literal
METHOD_NEAR = re.compile(
    r"""(?:\.(get|post|put|patch|delete)\b|method\s*[:=]\s*["'`](GET|POST|PUT|PATCH|DELETE)|EventSource|new\s+WebSocket)""",
    re.I,
)


def session() -> requests.Session:
    s = requests.Session(impersonate="chrome")
    if COOKIES_FILE.exists():
        for k, v in json.loads(COOKIES_FILE.read_text()).items():
            s.cookies.set(k, v, domain=".perplexity.ai")
    return s


def get(s: requests.Session, url: str) -> str:
    # one retry: the CDN occasionally drops a connection mid-walk (curl 35), and
    # a silently-dropped chunk silently undercounts the surface.
    for attempt in (1, 2):
        try:
            r = s.get(url, timeout=40)
            return r.text if r.status_code == 200 else ""
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                sys.stderr.write(f"  ! {url[:90]}: {e}\n")
                return ""
    return ""


def seed_chunks(s: requests.Session) -> set[str]:
    html = get(s, SHELL_URL)
    sys.stderr.write(f"shell: {len(html)} bytes\n")
    urls = set(re.findall(r"https://pplx-next-static-public\.perplexity\.ai/_spa/assets/[^\"']+\.js", html))
    # also relative refs in the shell
    for m in CHUNK_REF.findall(html):
        urls.add(CDN_BASE + m.lstrip("./"))
    return urls


def bfs(s: requests.Session, seeds: set[str], max_chunks: int) -> dict[str, str]:
    """Fetch the chunk graph breadth-first; return {url: body}."""
    seen: dict[str, str] = {}
    frontier = set(seeds)
    while frontier and len(seen) < max_chunks:
        batch = list(frontier)[: max_chunks - len(seen)]
        frontier = set()
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            bodies = dict(zip(batch, ex.map(lambda u: get(s, u), batch)))
        for url, body in bodies.items():
            seen[url] = body
            for ref in CHUNK_REF.findall(body):
                u = CDN_BASE + ref.lstrip("./")
                if u not in seen:
                    frontier.add(u)
        sys.stderr.write(f"  bfs: {len(seen)} fetched, {len(frontier)} queued\n")
    return seen


def extract(bodies: dict[str, str]) -> dict[str, dict]:
    eps: dict[str, dict] = {}
    for url, body in bodies.items():
        chunk = url.rsplit("/", 1)[-1]
        for m in EP_LITERAL.finditer(body):
            path = m.group(1)
            # normalize template paths to a stable key
            lo = max(0, m.start() - 90)
            hi = min(len(body), m.end() + 30)
            window = body[lo:hi]
            rec = eps.setdefault(path, {"methods": set(), "chunks": set(), "dynamic": "${" in path})
            rec["chunks"].add(chunk.split("-")[0])  # logical chunk name, drop hash
            for mm in METHOD_NEAR.finditer(window):
                meth = (mm.group(1) or mm.group(2) or "").upper()
                if meth:
                    rec["methods"].add(meth)
                elif "EventSource" in mm.group(0):
                    rec["methods"].add("SSE")
                elif "WebSocket" in mm.group(0):
                    rec["methods"].add("WS")
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=800)
    args = ap.parse_args()

    s = session()
    seeds = seed_chunks(s)
    sys.stderr.write(f"seeds: {len(seeds)} chunks\n")
    if not seeds:
        sys.stderr.write("no seed chunks found — shell layout may have changed again\n")
        return 1
    bodies = bfs(s, seeds, args.max_chunks)
    total_bytes = sum(len(b) for b in bodies.values())
    sys.stderr.write(f"fetched {len(bodies)} chunks, {total_bytes//1024} KiB\n")

    eps = extract(bodies)
    by_ns: dict[str, list[str]] = defaultdict(list)
    for p in eps:
        by_ns["/".join(p.strip("/").split("/")[:2])].append(p)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "captured_at": TS,
        "chunks_fetched": len(bodies),
        "bytes": total_bytes,
        "endpoint_count": len(eps),
        "endpoints": {
            p: {
                "methods": sorted(r["methods"]),
                "chunks": sorted(r["chunks"]),
                "dynamic": r["dynamic"],
                "known": KNOWN.get(p),
            }
            for p, r in sorted(eps.items())
        },
    }
    (OUTDIR / f"{TS}.json").write_text(json.dumps(summary, indent=2))

    new = sorted(p for p in eps if p not in KNOWN)
    lines = [
        f"# perplexity.ai endpoint inventory — {TS}",
        "",
        f"Source: Vite/Rolldown SPA bundle walk (`_spa/assets/`), {len(bodies)} chunks, "
        f"{total_bytes//1024} KiB. CF-free; no browser.",
        "",
        f"- distinct endpoint literals: **{len(eps)}**",
        f"- already exposed/known: {sum(1 for p in eps if p in KNOWN)}",
        f"- **NEW (not exposed, not in May plan): {len(new)}**",
        "",
        "## NEW endpoints",
        "",
        "| endpoint | methods | seen in chunks |",
        "|---|---|---|",
    ]
    for p in new:
        r = eps[p]
        meth = ", ".join(sorted(r["methods"])) or "?"
        ch = ", ".join(sorted(r["chunks"])[:4])
        lines.append(f"| `{p}` | {meth} | {ch} |")
    lines += ["", "## Known / already handled", ""]
    for p, why in KNOWN.items():
        seen = "✓ in bundle" if p in eps else "— not in bundle"
        meth = ", ".join(sorted(eps[p]["methods"])) if p in eps else ""
        lines.append(f"- `{p}` — {why}  [{seen} {meth}]")
    lines += ["", "## All endpoints by namespace", ""]
    for k in sorted(by_ns):
        lines.append(f"\n### {k} ({len(by_ns[k])})")
        for p in sorted(by_ns[k]):
            r = eps[p]
            meth = ",".join(sorted(r["methods"])) or "?"
            lines.append(f"- `{p}` [{meth}]")
    (OUTDIR / f"{TS}.md").write_text("\n".join(lines) + "\n")

    sys.stderr.write(f"\n{len(eps)} endpoints ({len(new)} new). Inventory: {OUTDIR/(TS+'.md')}\n")
    print(OUTDIR / f"{TS}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
