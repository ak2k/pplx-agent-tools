#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.40"]
# ///
"""Drive Chromium via Playwright to capture Perplexity search XHRs.

Loads our session cookies, navigates to perplexity.ai, submits a query,
and records the full network log as HAR + a parsed XHR summary.
Headless by default; pass --headed to watch.

Usage:
  uv run scripts/re-capture.py "claude code"
  uv run scripts/re-capture.py "claude code" --type academic
  uv run scripts/re-capture.py --headed "claude code"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import Request, Response, async_playwright

REPO = Path(__file__).resolve().parent.parent
COOKIES_FILE = Path.home() / ".config/perplexity/default/cookies.json"
DEFAULT_OUTPUT_DIR = REPO / "re-fixtures"

# cf_clearance binds to UA + TLS fingerprint of the issuing browser; keep it
# and match Firefox UA. Other CF cookies are short-lived and CF re-issues.
SKIP_COOKIES: set[str] = set()

# Match Firefox UA on macOS so cf_clearance (issued via Firefox) validates.
FIREFOX_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.0; rv:128.0) Gecko/20100101 Firefox/128.0"


async def main(query: str, search_type: str, headed: bool, outdir: Path) -> int:
    if not COOKIES_FILE.exists():
        print(f"no cookies at {COOKIES_FILE}; run pplx-auth import first", file=sys.stderr)
        return 2

    raw_cookies = json.loads(COOKIES_FILE.read_text())
    cookies = [
        {
            "name": name,
            "value": value,
            "domain": ".perplexity.ai",
            "path": "/",
            "secure": True,
            "sameSite": "Lax",
        }
        for name, value in raw_cookies.items()
        if name not in SKIP_COOKIES
    ]
    print(
        f"injecting {len(cookies)} cookies (skipping {len(raw_cookies) - len(cookies)} CF-bound)",
        file=sys.stderr,
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    label = f"{search_type}-{query.replace(' ', '_')[:30]}-{ts}"
    capture_dir = outdir / "search-web"
    capture_dir.mkdir(parents=True, exist_ok=True)
    har_path = capture_dir / f"{label}.har"
    log_path = capture_dir / f"{label}.requests.jsonl"
    shot_dir = capture_dir / f"{label}.screenshots"
    shot_dir.mkdir(exist_ok=True)

    request_log: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed)
        ctx = await browser.new_context(
            record_har_path=str(har_path),
            record_har_content="embed",
            viewport={"width": 1400, "height": 900},
            user_agent=FIREFOX_UA,
        )
        await ctx.add_cookies(cookies)
        page = await ctx.new_page()

        def on_request(req: Request) -> None:
            request_log.append(
                {
                    "phase": "request",
                    "url": req.url,
                    "method": req.method,
                    "resource_type": req.resource_type,
                    "post_data_size": len(req.post_data or "") if req.post_data else 0,
                }
            )

        async def on_response(resp: Response) -> None:
            try:
                request_log.append(
                    {
                        "phase": "response",
                        "url": resp.url,
                        "status": resp.status,
                        "content_type": resp.headers.get("content-type", ""),
                        "size_hint": int(resp.headers.get("content-length", "0") or 0),
                    }
                )
            except Exception as e:
                request_log.append({"phase": "response_error", "url": resp.url, "error": str(e)})

        page.on("request", on_request)
        page.on("response", lambda r: asyncio.create_task(on_response(r)))

        print("→ navigating to perplexity.ai", file=sys.stderr)
        await page.goto("https://www.perplexity.ai/", wait_until="domcontentloaded", timeout=30000)
        # Cloudflare Turnstile may interstitially challenge — wait for the real
        # page DOM to render. Look for the textarea Perplexity uses for queries.
        # Up to 30s: passive CF challenges resolve in 3-10s for healthy browsers.
        for i in range(30):
            await asyncio.sleep(1)
            if await page.locator("textarea").count() > 0:
                print(f"→ DOM ready after {i + 1}s", file=sys.stderr)
                break
            if i in (0, 5, 15):
                await page.screenshot(path=str(shot_dir / f"01-loading-{i:02d}s.png"))
        else:
            print("→ DOM did not render in 30s; capturing failure shot", file=sys.stderr)
            await page.screenshot(path=str(shot_dir / "01-loaded.png"))
        await page.screenshot(path=str(shot_dir / "01-loaded.png"))

        # Find the query input. Perplexity uses a textarea for the main "Ask" box.
        candidates = [
            'textarea[placeholder*="Ask" i]',
            'textarea[placeholder*="anything" i]',
            "textarea",
            '[role="textbox"]',
            'input[type="text"]',
        ]
        input_el = None
        for sel in candidates:
            loc = page.locator(sel).first
            try:
                await loc.wait_for(state="visible", timeout=2000)
                input_el = loc
                print(f"→ found query input via selector: {sel}", file=sys.stderr)
                break
            except Exception:
                continue
        if input_el is None:
            print("could not find query input — capturing failure screenshot", file=sys.stderr)
            await page.screenshot(path=str(shot_dir / "02-no-input.png"))
            await ctx.close()
            await browser.close()
            return 1

        await input_el.fill(query)
        await page.screenshot(path=str(shot_dir / "02-typed.png"))
        print(f"→ submitting query: {query!r}", file=sys.stderr)
        await input_el.press("Enter")

        # Wait for the results UI to render rather than networkidle (which never
        # settles due to persistent telemetry). 20s should cover SSE completion.
        await asyncio.sleep(20)
        await page.screenshot(path=str(shot_dir / "03-results.png"), full_page=True)

        await ctx.close()
        await browser.close()

    log_path.write_text("\n".join(json.dumps(e) for e in request_log) + "\n")

    pplx_xhrs = [
        e
        for e in request_log
        if e.get("phase") == "request"
        and "perplexity.ai" in e.get("url", "")
        and e.get("resource_type") in ("xhr", "fetch", "eventsource")
    ]
    print(
        f"\ncaptured {len(request_log)} total events; {len(pplx_xhrs)} perplexity.ai XHR/fetch requests",
        file=sys.stderr,
    )
    print(
        f"\noutput:\n  har:  {har_path}\n  log:  {log_path}\n  shots: {shot_dir}", file=sys.stderr
    )
    print("\nfirst 20 perplexity.ai XHR URLs:", file=sys.stderr)
    seen = set()
    for e in pplx_xhrs:
        key = e["method"] + " " + e["url"].split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        print(f"  {e['method']:6s} {e['url'][:120]}", file=sys.stderr)
        if len(seen) >= 20:
            break

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query", help="search query")
    ap.add_argument("--type", default="web", help="search type (web, academic, ...)")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory (default: re-fixtures/)",
    )
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.query, args.type, args.headed, args.outdir)))
