#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.40"]
# ///
"""Capture the network traffic from pasting a URL into Perplexity's web UI.

Goal: find the endpoint Perplexity uses to fetch + classify a URL when the
user pastes one (potentially distinct from /rest/sse/perplexity_ask).
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import async_playwright

REPO = Path(__file__).resolve().parent.parent
COOKIES_FILE = Path.home() / ".config/perplexity/default/cookies.json"
OUT_DIR = REPO / "re-fixtures/fetch-url"
FIREFOX_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.0; rv:128.0) Gecko/20100101 Firefox/128.0"

TARGET_URL = "https://example.com"


async def main() -> int:
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

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    har_path = OUT_DIR / f"paste-{ts}.har"
    log_path = OUT_DIR / f"paste-{ts}.requests.jsonl"
    shot_dir = OUT_DIR / f"paste-{ts}.screenshots"
    shot_dir.mkdir(exist_ok=True)

    request_log: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            record_har_path=str(har_path),
            record_har_content="embed",
            user_agent=FIREFOX_UA,
            viewport={"width": 1400, "height": 900},
        )
        await ctx.add_cookies(cookies)
        page = await ctx.new_page()

        def on_request(req):
            entry = {
                "phase": "request",
                "url": req.url,
                "method": req.method,
                "resource_type": req.resource_type,
            }
            if req.post_data:
                entry["post_data"] = req.post_data[:8000]
            request_log.append(entry)

        page.on("request", on_request)
        page.on(
            "response",
            lambda r: request_log.append(
                {
                    "phase": "response",
                    "url": r.url,
                    "status": r.status,
                    "content_type": r.headers.get("content-type", ""),
                }
            ),
        )

        await page.goto("https://www.perplexity.ai/", wait_until="domcontentloaded", timeout=30000)
        # Wait for textbox
        for _ in range(30):
            await asyncio.sleep(1)
            if await page.locator('[role="textbox"]').count() > 0:
                break
        await page.screenshot(path=str(shot_dir / "01-loaded.png"))

        box = page.locator('[role="textbox"]').first
        # Type a URL — pasting via clipboard is tricky in Playwright; .fill() is equivalent
        # for testing what happens when a URL becomes input.
        await box.fill(TARGET_URL)
        await page.screenshot(path=str(shot_dir / "02-typed.png"))

        # Wait a bit — Perplexity may auto-detect URL and fire a preview request
        await asyncio.sleep(5)
        await page.screenshot(path=str(shot_dir / "03-after-typed.png"))

        # Submit
        await box.press("Enter")
        await asyncio.sleep(15)
        await page.screenshot(path=str(shot_dir / "04-submitted.png"), full_page=True)

        await ctx.close()
        await browser.close()

    log_path.write_text("\n".join(json.dumps(e) for e in request_log) + "\n")

    pplx = [
        e
        for e in request_log
        if e.get("phase") == "request"
        and "perplexity.ai" in e.get("url", "")
        and e.get("resource_type") in ("xhr", "fetch", "eventsource")
    ]
    print(f"\n{len(pplx)} perplexity.ai XHR/fetch requests", file=sys.stderr)
    seen = set()
    for e in pplx:
        key = e["method"] + " " + e["url"].split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        print(f"  {e['method']:6s} {e['url'][:120]}", file=sys.stderr)
    print(f"\nlog: {log_path}", file=sys.stderr)
    print(f"shots: {shot_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
