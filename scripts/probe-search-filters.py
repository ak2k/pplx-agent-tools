"""Probe whether --country, --domains, --excluded-domains actually filter.

Each filter is tested by running a baseline query and a filtered variant,
then comparing result domain mixes. Verdict per flag:

  HONORED  — filtered result set clearly differs in the expected direction
  IGNORED  — filtered result set ~identical to baseline (server didn't filter)
  ERRORED  — server returned 4xx/5xx for the filtered call

Run:  uv run python scripts/probe-search-filters.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter

from pplx_agent_tools.errors import PplxError
from pplx_agent_tools.verbs.search import search_many
from pplx_agent_tools.wire import Client


def domain_mix(hits) -> Counter[str]:
    return Counter((h.domain or "").lower() for h in hits)


def fmt_mix(c: Counter[str], top: int = 8) -> str:
    return ", ".join(f"{d}×{n}" for d, n in c.most_common(top))


def probe_country(client: Client) -> None:
    # A query whose top results tend to vary by locale.
    q = "tagesschau news today"
    print(f"\n=== --country probe (query: {q!r}) ===")
    base = search_many(client, [q], limit=10, country="US")
    time.sleep(1.5)
    de = search_many(client, [q], limit=10, country="DE")

    base_mix = domain_mix(base.hits)
    de_mix = domain_mix(de.hits)
    print(f"  US: {fmt_mix(base_mix)}")
    print(f"  DE: {fmt_mix(de_mix)}")

    base_urls = {h.url for h in base.hits}
    de_urls = {h.url for h in de.hits}
    overlap = len(base_urls & de_urls)
    total = max(len(base_urls), len(de_urls))
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
    # Pick a generic query where python.org has plenty to return.
    q = "asyncio tutorial"
    print(f"\n=== --domains probe (query: {q!r}, filter=['python.org']) ===")
    base = search_many(client, [q], limit=10)
    time.sleep(1.5)
    try:
        filt = search_many(client, [q], limit=10, domains=["python.org"])
    except PplxError as e:
        print(f"  VERDICT: ERRORED ({e})")
        return

    base_mix = domain_mix(base.hits)
    filt_mix = domain_mix(filt.hits)
    print(f"  baseline:           {fmt_mix(base_mix)}")
    print(f"  domains=python.org: {fmt_mix(filt_mix)}")

    off_domain = [
        h for h in filt.hits if h.domain and not h.domain.lower().endswith("python.org")
    ]
    if not filt.hits:
        print("  VERDICT: AMBIGUOUS (filtered call returned 0 hits)")
    elif not off_domain:
        print(f"  VERDICT: HONORED ({len(filt.hits)}/{len(filt.hits)} on python.org)")
    elif len(off_domain) == len(filt.hits):
        print("  VERDICT: IGNORED (no hits on python.org)")
    else:
        print(
            f"  VERDICT: PARTIAL ({len(filt.hits) - len(off_domain)}/{len(filt.hits)} on-domain)"
        )


def probe_excluded(client: Client) -> None:
    # Pick a query where wikipedia/reddit reliably dominate.
    q = "linux kernel"
    excl = ["wikipedia.org", "en.wikipedia.org", "reddit.com"]
    print(f"\n=== --excluded-domains probe (query: {q!r}, excl={excl}) ===")
    base = search_many(client, [q], limit=10)
    time.sleep(1.5)
    try:
        filt = search_many(client, [q], limit=10, excluded_domains=excl)
    except PplxError as e:
        print(f"  VERDICT: ERRORED ({e})")
        return

    base_mix = domain_mix(base.hits)
    filt_mix = domain_mix(filt.hits)
    print(f"  baseline:        {fmt_mix(base_mix)}")
    print(f"  excluded:        {fmt_mix(filt_mix)}")

    excl_lower = {e.lower() for e in excl}
    base_excl_hits = sum(1 for h in base.hits if (h.domain or "").lower() in excl_lower)
    filt_excl_hits = sum(1 for h in filt.hits if (h.domain or "").lower() in excl_lower)
    print(
        f"  excluded-domain hits: baseline={base_excl_hits} filtered={filt_excl_hits}"
    )
    if base_excl_hits > 0 and filt_excl_hits == 0:
        print("  VERDICT: HONORED")
    elif filt_excl_hits == base_excl_hits and filt_excl_hits > 0:
        print("  VERDICT: IGNORED")
    elif base_excl_hits == 0:
        print(
            "  VERDICT: AMBIGUOUS (baseline had no excluded-domain hits to suppress)"
        )
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
