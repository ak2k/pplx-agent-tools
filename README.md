# pplx-agent-tools

Shell-CLI agent toolkit for Perplexity, backed by your Pro subscription's web session cookies. Parallel to [`kagi-search`](https://github.com/Mic92/mics-skills/tree/main/skills/kagi-search) in shape and purpose.

**Status**: Phase 1 in progress. Three verbs ship today (`search`, `fetch`, `snippets`) plus auth. See the [design plan](https://github.com/ak2k/nix-config/blob/main/docs/plans/pplx-agent-tools.md) for surface, architecture, phasing, and risk register.

## Why

Perplexity exposes its agent verbs (raw search, URL fetch + LLM extraction, batched snippet extraction) internally but not via its public Sonar API. This project gives agents — Claude Code, Codex, anything that shells out — the same primitives, backed by an existing Pro subscription's web session cookies. No per-token billing, no separate API key.

## Verbs

Single binary, subcommand-style (`pplx <verb>`):

- `pplx search <query>...` — ranked web hits via `/rest/realtime/search-web`. Native multi-query. Each hit carries `title`, `snippet`, and a longer `summary` field.
- `pplx fetch <url>` — local fetch (`curl_cffi` chrome impersonation + `trafilatura`). With `--prompt`, routes URL+prompt through Perplexity's chat endpoint for LLM extraction in one round-trip.
- `pplx snippets <query> <url>...` — hybrid retrieval (FTS5 BM25 + `fastembed` semantic vectors, RRF-merged) over locally-fetched URLs. Per-URL and total token budgets.
- `pplx auth {check, refresh, import}` — cookie management. `import --browser <name>` lifts cookies from a local browser via `rookiepy` (Brave/Chrome/Chromium/Firefox/Safari/Edge/Arc/Vivaldi/Opera/LibreWolf/Zen).

Each verb: text output by default, `-j` for JSON. Stable exit codes for agent retry semantics (2 = auth, 3 = rate limit, 4 = network, 5 = anti-bot). Single-shot CLIs; no daemon (the daemon model is a deferred Phase 2).

## Install

```bash
# Install the CLI
uv tool install pplx-agent-tools  # or: pipx install pplx-agent-tools

# One-time: wire the agent skill so Claude Code (etc.) can find it
mkdir -p ~/.claude/skills/pplx-agent-tools
ln -sf "$(pplx skill-path)" ~/.claude/skills/pplx-agent-tools/SKILL.md

# Import cookies from your browser (only need to do once per ~30 days,
# or until you log out of perplexity.ai)
pplx auth import --browser firefox  # also: brave, chrome, safari, ...
pplx auth check                     # validate the session

# Optional: extend the session indefinitely by refreshing periodically
# (each refresh rotates the cookie with a fresh 30-day TTL)
pplx auth refresh                   # add to cron / launchd for hands-off
```

## Caveats

- **Unofficial**. Uses Perplexity's internal web endpoints, not the official Sonar API. Endpoints can change without notice.
- **Not affiliated with Perplexity AI.**
- **For your own subscription only**. Cookie pooling across users is an explicit anti-pattern.
- **Session cookies expire**. Reimport from the browser when `pplx auth check` fails.
- **No URL-fetch endpoint**: `pplx fetch` fetches locally (no Perplexity-backend paywall bypass / cache reuse). For LLM-extracted content, use `--prompt`.
- **`pplx search` is web-results only.** Variant search modes (academic / images / videos / shopping) and filter knobs (country, domain include/exclude) aren't supported by the realtime/search-web endpoint this verb uses — probed 2026-05-14 and found silently ignored. If we add them later they'll route through the ask-SSE endpoint instead.

## License

MIT — see [LICENSE](LICENSE).
