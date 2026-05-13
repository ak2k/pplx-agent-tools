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

## Caveats

- **Unofficial**. Uses Perplexity's internal web endpoints, not the official Sonar API. Endpoints can change without notice.
- **Not affiliated with Perplexity AI.**
- **For your own subscription only**. Cookie pooling across users is an explicit anti-pattern.
- **Session cookies expire**. Reimport from the browser when `pplx auth check` fails.
- **No URL-fetch endpoint**: `pplx fetch` fetches locally (no Perplexity-backend paywall bypass / cache reuse). For LLM-extracted content, use `--prompt`.
- **Variant searches deferred**: `-t academic/images/videos/shopping` are Phase 2 (no dedicated cookie-auth endpoints exist; the SPA renders those via chat-endpoint blocks).

## License

MIT — see [LICENSE](LICENSE).
