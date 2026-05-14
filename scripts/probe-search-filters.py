"""Probe whether country / domain_filter / excluded_domains are honored by
/rest/realtime/search-web.

Hits the endpoint directly (not via search_many) so we can pass arbitrary
body keys and observe the raw effect. For each flag we run a baseline and
a filtered variant, then compare result domain mixes. Verdict per flag:

  HONORED  — filtered result set clearly differs in the expected direction
  IGNORED  — filtered result set ~identical to baseline (server didn't filter)
  ERRORED  — server returned 4xx/5xx for the filtered call

Run:  uv run python scripts/probe-search-filters.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from typing import Any
from uuid import uuid4

from pplx_agent_tools.errors import PplxError
from pplx_agent_tools.verbs.search import ENDPOINT
from pplx_agent_tools.wire import Client


def call_raw(client: Client, queries: list[str], **extras: Any) -> list[dict[str, Any]]:
    """POST directly to the search-web endpoint with arbitrary extra body keys.

    Returns the raw `web_results` array — no dedup, no `is_*` filtering, so
    we can see exactly what the server sent back rather than what our verb
    chooses to surface.
    """
    body = {"session_id": str(uuid4()), "queries": list(queries), **extras}
    raw = client.post_json(ENDPOINT, body)
    if not isinstance(raw, dict):
        return []
    hits = raw.get("web_results") or []
    return hits if isinstance(hits, list) else []


def domain_mix(hits: list[dict[str, Any]]) -> Counter[str]:
    return Counter((h.get("domain") or "").lower() for h in hits if h.get("domain"))


def fmt_mix(c: Counter[str], top: int = 8) -> str:
    return ", ".join(f"{d} x{n}" for d, n in c.most_common(top))


def domain_matches(domain: str, base: str) -> bool:
    """True iff `domain` equals `base` or is a proper subdomain. Boundary-safe:
    `evilpython.org` does NOT match base `python.org`.
    """
    d = (domain or "").lower().strip(".")
    b = (base or "").lower().strip(".")
    return bool(b) and (d == b or d.endswith("." + b))


def probe_country(client: Client) -> None:
    q = "tagesschau news today"
    print(f"\n=== country probe (query: {q!r}) ===")
    base = call_raw(client, [q])
    time.sleep(1.5)
    de = call_raw(client, [q], country="DE")

    base_mix = domain_mix(base)
    de_mix = domain_mix(de)
    print(f"  US baseline: {fmt_mix(base_mix)}")
    print(f"  DE override: {fmt_mix(de_mix)}")

    base_urls = {h.get("url") for h in base}
    de_urls = {h.get("url") for h in de}
    overlap = len(base_urls & de_urls)
    total = max(len(base_urls), len(de_urls), 1)
    de_tld = sum(n for d, n in de_mix.items() if d.endswith(".de"))
    base_tld = sum(n for d, n in base_mix.items() if d.endswith(".de"))
    print(f"  URL overlap: {overlap}/{total}   .de hits: US={base_tld} DE={de_tld}")
    if overlap == total and de_tld == base_tld:
        print("  VERDICT: IGNORED (identical result sets)")
    elif de_tld > base_tld or overlap < total // 2:
        print("  VERDICT: HONORED")
    else:
        print("  VERDICT: AMBIGUOUS (mild difference; could be cache/noise)")


def probe_domains(client: Client) -> None:
    q = "asyncio tutorial"
    base_domain = "python.org"
    print(f"\n=== domain_filter probe (query: {q!r}, filter=[{base_domain!r}]) ===")
    base = call_raw(client, [q])
    time.sleep(1.5)
    try:
        filt = call_raw(client, [q], domain_filter=[base_domain])
    except PplxError as e:
        print(f"  VERDICT: ERRORED ({e})")
        return

    base_mix = domain_mix(base)
    filt_mix = domain_mix(filt)
    print(f"  baseline:        {fmt_mix(base_mix)}")
    print(f"  filtered:        {fmt_mix(filt_mix)}")

    off_domain = [
        h for h in filt if h.get("domain") and not domain_matches(h["domain"], base_domain)
    ]
    if not filt:
        print("  VERDICT: AMBIGUOUS (filtered call returned 0 hits)")
    elif not off_domain:
        print(f"  VERDICT: HONORED ({len(filt)}/{len(filt)} on {base_domain})")
    elif len(off_domain) == len(filt):
        print(f"  VERDICT: IGNORED (no hits on {base_domain})")
    else:
        on_domain = len(filt) - len(off_domain)
        print(f"  VERDICT: PARTIAL ({on_domain}/{len(filt)} on-domain)")


def probe_excluded(client: Client) -> None:
    q = "linux kernel"
    excl = ["wikipedia.org", "reddit.com"]
    print(f"\n=== excluded_domains probe (query: {q!r}, excl={excl}) ===")
    base = call_raw(client, [q])
    time.sleep(1.5)
    try:
        filt = call_raw(client, [q], excluded_domains=excl)
    except PplxError as e:
        print(f"  VERDICT: ERRORED ({e})")
        return

    base_mix = domain_mix(base)
    filt_mix = domain_mix(filt)
    print(f"  baseline:        {fmt_mix(base_mix)}")
    print(f"  excluded:        {fmt_mix(filt_mix)}")

    def hits_in_excluded(hits: list[dict[str, Any]]) -> int:
        return sum(1 for h in hits if any(domain_matches(h.get("domain") or "", e) for e in excl))

    base_excl_hits = hits_in_excluded(base)
    filt_excl_hits = hits_in_excluded(filt)
    print(f"  excluded-domain hits: baseline={base_excl_hits} filtered={filt_excl_hits}")
    if base_excl_hits > 0 and filt_excl_hits == 0:
        print("  VERDICT: HONORED")
    elif filt_excl_hits == base_excl_hits and filt_excl_hits > 0:
        print("  VERDICT: IGNORED")
    elif base_excl_hits == 0:
        print("  VERDICT: AMBIGUOUS (baseline had no excluded-domain hits to suppress)")
    else:
        print("  VERDICT: PARTIAL")


def main() -> int:
    try:
        client = Client.from_default_cookies()
    except PplxError as e:
        print(f"auth: {e}", file=sys.stderr)
        return 1

    probe_country(client)
    probe_domains(client)
    probe_excluded(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
