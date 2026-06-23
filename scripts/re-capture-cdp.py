#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["curl_cffi>=0.7"]
# ///
"""CDP-driven endpoint capture for perplexity.ai — KNOWN-BLOCKED, kept for record.

OUTCOME (2026-06-22): this does NOT work against perplexity.ai. A Playwright /
agent-browser Chromium hits Cloudflare's managed-challenge loop on every HTML
page route (`GET /` -> 403 -> cdn-cgi/challenge-platform -> still 403): the
automation browser's signals + TLS fingerprint don't match the cf_clearance
cookie (issued by the user's real browser), so CF never serves the document.

Use `scripts/enumerate-endpoints.py` instead — a CF-free static bundle walk that
gives fuller breadth. This script is retained because:
  - it documents *why* CDP is the wrong tool (the CF wall is the finding), and
  - it's a working CDP capture harness (HAR + resource-timing + body-shape) that
    would apply if driven through Comet (Perplexity's own browser, which CF
    trusts) for the request-BODY depth pass the static walk can't provide.

When it can reach the app it merges two signals:
  - FIRED endpoints   (from the HAR: method + path + request-body key-shape)
  - REFERENCED endpoints (from grepping the loaded JS bundles)

Auth: injects ~/.config/perplexity/default/cookies.json as Playwright state.
Only endpoint paths, methods, and request-body *key names* are persisted — never
cookie values, auth headers, or response bodies (those stay in the local HAR,
which is NOT committed).

Usage:
  scripts/re-capture-cdp.py                 # full sweep
  scripts/re-capture-cdp.py --no-search     # skip the interactive search trigger
  scripts/re-capture-cdp.py --keep-har      # leave the raw HAR in the out dir
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from curl_cffi import requests

COOKIES_FILE = Path.home() / ".config/perplexity/default/cookies.json"
REPO = Path(__file__).resolve().parent.parent
TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
OUTDIR = REPO / "re-fixtures" / "endpoint-sweep" / TS

# Product surfaces to visit. Some paths are guesses; a redirect/404 still reveals
# routing + any bootstrap XHRs. Order: cheap reads first, search trigger last.
SURFACES: list[tuple[str, str]] = [
    ("home", "https://www.perplexity.ai/"),
    ("discover", "https://www.perplexity.ai/discover"),
    ("library", "https://www.perplexity.ai/library"),
    ("spaces", "https://www.perplexity.ai/spaces"),
    ("tasks", "https://www.perplexity.ai/tasks"),
    ("automations", "https://www.perplexity.ai/automations"),
    ("finance", "https://www.perplexity.ai/finance"),
    ("labs", "https://www.perplexity.ai/labs"),
    ("pages", "https://www.perplexity.ai/pages"),
    ("shopping", "https://www.perplexity.ai/shopping"),
    ("account", "https://www.perplexity.ai/account/details"),
    ("settings", "https://www.perplexity.ai/settings/account"),
    ("settings-api", "https://www.perplexity.ai/settings/api"),
]

# Already exposed by the CLI, or RE'd + documented in the plan. Used for the diff.
KNOWN = {
    "/api/auth/session": "auth keepalive (exposed)",
    "/rest/realtime/search-web": "pplx search (exposed)",
    "/rest/sse/perplexity_ask": "pplx fetch --prompt (exposed)",
    "/rest/realtime/search-youtube": "RE'd: single-best-match video nav (deferred)",
    "/rest/uploads/create_upload_url": "RE'd: file uploads (deferred Phase 2)",
}

CAP_RESOURCE_JS = """
(() => {
  const out = {js: [], rest: []};
  for (const e of performance.getEntriesByType('resource')) {
    try {
      const u = new URL(e.name);
      if (e.name.endsWith('.js') || e.initiatorType === 'script') out.js.push(e.name);
      if (u.hostname.endsWith('perplexity.ai') &&
          (u.pathname.startsWith('/rest/') || u.pathname.startsWith('/api/')))
        out.rest.push(u.pathname + (u.search ? '?' : ''));
    } catch (_) {}
  }
  // also any <script src> the document declares
  for (const s of document.querySelectorAll('script[src]')) out.js.push(s.src);
  out.js = [...new Set(out.js)];
  out.rest = [...new Set(out.rest)];
  return out;  // agent-browser JSON-encodes object returns; parse once
})()
"""


def ab(*args: str, stdin: str | None = None, timeout: int = 90) -> str:
    """Run agent-browser, return stdout (stderr surfaced on failure)."""
    p = subprocess.run(
        ["agent-browser", *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,  # returncode handled below
    )
    if p.returncode != 0:
        sys.stderr.write(
            f"  ! agent-browser {args[0]} rc={p.returncode}: {p.stderr.strip()[:300]}\n"
        )
    return p.stdout


def ab_eval(js: str, timeout: int = 60) -> str:
    """eval via base64 (robust against quoting)."""
    b64 = base64.b64encode(js.encode()).decode()
    return ab("eval", "-b", b64, timeout=timeout)


def build_state(path: Path) -> int:
    raw = json.loads(COOKIES_FILE.read_text())
    cookies = [
        {
            "name": n,
            "value": v,
            "domain": ".perplexity.ai",
            "path": "/",
            "secure": True,
            "sameSite": "Lax",
        }
        for n, v in raw.items()
    ]
    path.write_text(json.dumps({"cookies": cookies, "origins": []}))
    return len(cookies)


def fetch_bundles(js_urls: list[str], cookies: dict) -> dict[str, str]:
    """Fetch loaded JS chunks (CDN hosts are usually not CF-gated) and grep later."""
    s = requests.Session(impersonate="chrome")
    for k, v in cookies.items():
        s.cookies.set(k, v, domain=".perplexity.ai")
    bodies: dict[str, str] = {}
    # cap to keep it polite; the big endpoint-bearing chunks are the app bundles
    for url in js_urls[:200]:
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 200 and r.text:
                bodies[url] = r.text
        except Exception as e:
            sys.stderr.write(f"  ! chunk {url[:80]}: {e}\n")
    return bodies


def grep_endpoints(bodies: dict[str, str]) -> set[str]:
    eps: set[str] = set()
    pats = [
        re.compile(r"""["'`](/rest/[a-zA-Z0-9_./{}$:-]+)["'`]"""),
        re.compile(r"""["'`](/api/[a-zA-Z0-9_./{}$:-]+)["'`]"""),
        re.compile(r"(/rest/[a-zA-Z0-9_/-]{3,})"),
    ]
    for body in bodies.values():
        for pat in pats:
            for m in pat.findall(body):
                eps.add(m)
    return eps


def parse_har(har_path: Path) -> dict[str, dict]:
    """Extract perplexity.ai /rest/ + /api/ calls: method, path, request-body keys."""
    har = json.loads(har_path.read_text())
    fired: dict[str, dict] = {}
    for ent in har.get("log", {}).get("entries", []):
        req = ent.get("request", {})
        url = req.get("url", "")
        m = re.match(r"https?://([^/]+)(/rest/[^?]*|/api/[^?]*)", url)
        if not m or "perplexity.ai" not in m.group(1):
            continue
        host, path = m.group(1), m.group(2)
        method = req.get("method", "GET")
        key = f"{method} {path}"
        body_keys: list[str] = []
        post = req.get("postData", {})
        if post.get("text"):
            try:
                obj = json.loads(post["text"])
                if isinstance(obj, dict):
                    body_keys = sorted(obj.keys())
            except Exception:
                pass
        status = ent.get("response", {}).get("status")
        rec = fired.setdefault(
            key,
            {
                "host": host,
                "method": method,
                "path": path,
                "body_keys": set(),
                "statuses": set(),
                "count": 0,
            },
        )
        rec["body_keys"].update(body_keys)
        if status is not None:
            rec["statuses"].add(status)
        rec["count"] += 1
    return fired


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-search", action="store_true", help="skip interactive search trigger")
    ap.add_argument("--keep-har", action="store_true", help="leave raw HAR in out dir")
    args = ap.parse_args()

    if not COOKIES_FILE.exists():
        sys.stderr.write(f"no cookies at {COOKIES_FILE}; run `pplx auth import` first\n")
        return 2
    raw_cookies = json.loads(COOKIES_FILE.read_text())
    OUTDIR.mkdir(parents=True, exist_ok=True)

    state_path = Path("/tmp") / f"pplx_state_{TS}.json"
    n = build_state(state_path)
    har_path = OUTDIR / "sweep.har"

    try:
        print("[1] closing stale agent-browser sessions", file=sys.stderr)
        ab("close", "--all")
        print(f"[2] launching headed Chromium + injecting {n} cookies", file=sys.stderr)
        ab("--state", str(state_path), "open", SURFACES[0][1], timeout=120)
        ab("wait", "--load", "networkidle", timeout=60)
        print("[3] HAR recording on", file=sys.stderr)
        ab("network", "har", "start")

        per_surface: dict[str, dict] = {}
        all_js: set[str] = set()
        for name, url in SURFACES:
            print(f"[surface] {name:14s} {url}", file=sys.stderr)
            ab("open", url, timeout=90)
            ab("wait", "--load", "networkidle", timeout=45)
            time.sleep(1.5)
            ab("screenshot", str(OUTDIR / f"{name}.png"))
            try:
                cap = json.loads(ab_eval(CAP_RESOURCE_JS) or "{}")
            except Exception:
                cap = {}
            per_surface[name] = {"url": url, "rest": cap.get("rest", [])}
            all_js.update(cap.get("js", []))

        if not args.no_search:
            print("[surface] search (interactive)", file=sys.stderr)
            ab("open", "https://www.perplexity.ai/", timeout=90)
            ab("wait", "--load", "networkidle", timeout=45)
            # best-effort: focus the composer and submit a query
            ab("find", "role", "textbox", "click")
            ab("keyboard", "type", "what is the perplexity api rate limit")
            ab("press", "Enter")
            ab("wait", "--load", "networkidle", timeout=60)
            time.sleep(4)
            ab("screenshot", str(OUTDIR / "search.png"))
            try:
                cap = json.loads(ab_eval(CAP_RESOURCE_JS) or "{}")
                per_surface["search"] = {"url": "(typed query)", "rest": cap.get("rest", [])}
                all_js.update(cap.get("js", []))
            except Exception:
                pass

        print(f"[4] HAR recording off -> {har_path}", file=sys.stderr)
        ab("network", "har", "stop", str(har_path))
    finally:
        state_path.unlink(missing_ok=True)

    # ---- analysis ----
    fired = parse_har(har_path) if har_path.exists() else {}
    js_urls = sorted(u for u in all_js if u.endswith(".js"))
    print(f"[5] fetching {len(js_urls)} JS chunks for endpoint grep", file=sys.stderr)
    bodies = fetch_bundles(js_urls, raw_cookies)
    referenced = grep_endpoints(bodies)

    fired_paths = {rec["path"] for rec in fired.values()}
    all_paths = fired_paths | referenced
    ns = defaultdict(list)
    for p in sorted(all_paths):
        ns["/".join(p.strip("/").split("/")[:2])].append(p)

    # write machine-readable summary
    summary = {
        "captured_at": TS,
        "surfaces": per_surface,
        "fired": {
            k: {**v, "body_keys": sorted(v["body_keys"]), "statuses": sorted(v["statuses"])}
            for k, v in fired.items()
        },
        "referenced_only": sorted(referenced - fired_paths),
        "js_chunks_fetched": len(bodies),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # markdown inventory
    lines = [
        f"# perplexity.ai endpoint inventory — {TS}",
        "",
        f"- JS chunks grepped: {len(bodies)}/{len(js_urls)}",
        f"- distinct endpoints (fired + referenced): {len(all_paths)}",
        f"- fired this sweep: {len(fired_paths)} · referenced-only (not fired): {len(referenced - fired_paths)}",
        "",
        "## NEW (not exposed and not in the May plan)",
        "",
    ]
    for p in sorted(all_paths):
        if p not in KNOWN:
            tag = "fired" if p in fired_paths else "ref-only"
            lines.append(f"- `{p}`  ({tag})")
    lines += ["", "## Known / already handled", ""]
    for p, why in KNOWN.items():
        seen = "fired" if p in fired_paths else ("ref" if p in referenced else "not-seen")
        lines.append(f"- `{p}` — {why}  [{seen}]")
    lines += ["", "## Fired endpoints with request-body key-shapes", ""]
    for key in sorted(fired):
        rec = fired[key]
        bk = ", ".join(sorted(rec["body_keys"])) or "—"
        st = ",".join(str(s) for s in sorted(rec["statuses"]))
        lines.append(f"- `{key}`  body=[{bk}]  status={st}  x{rec['count']}")
    lines += ["", "## By namespace", ""]
    for k in sorted(ns):
        lines.append(f"\n### {k} ({len(ns[k])})")
        lines += [f"- `{p}`" for p in sorted(ns[k])]
    (OUTDIR / "inventory.md").write_text("\n".join(lines) + "\n")

    if not args.keep_har:
        har_path.unlink(missing_ok=True)
        print("[6] raw HAR discarded (--keep-har to retain)", file=sys.stderr)

    print(
        f"\nDone. {len(all_paths)} endpoints. Inventory: {OUTDIR / 'inventory.md'}", file=sys.stderr
    )
    print(OUTDIR / "inventory.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
