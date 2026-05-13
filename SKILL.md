---
name: pplx-agent-tools
description: Query Perplexity via your Pro subscription's web session. Use `pplx search` for ranked web hits with longer-form summaries, `pplx fetch URL --prompt "..."` for one-call URL-to-LLM-extracted-answer, or `pplx snippets QUERY URL...` for hybrid (keyword + semantic) excerpt extraction from N supplied URLs. Pair search → snippets for "find candidates, then dig into specific ones."
---

# When to reach for each verb

- **`pplx search <query>...`** — ranked web hits. Each hit carries `title`, `url`, `domain`, `snippet` (~200 chars), and `summary` (~1500 chars, agent-friendly extract). Multi-query is native — pass several queries, server merges/dedupes.
- **`pplx fetch <url>`** — local fetch + cleaned content extraction. With `--prompt`, routes to Perplexity's LLM which fetches the URL itself and answers your prompt in one round-trip.
- **`pplx snippets <query> <url>...`** — concurrent-fetch N URLs locally, return query-relevant paragraphs from each using hybrid retrieval (BM25 keyword + semantic vectors). Useful after `pplx search` narrows candidates.
- **`pplx auth check`** — validate cookies. Run if other verbs fail with exit code 2.

# vs `kagi-search`

- Prefer **`kagi-search`** for: small queries where the Quick Answer summary is enough; queries you'd rather route through Kagi than Perplexity.
- Prefer **`pplx search`** for: deeper extraction (the `summary` field is much longer than Kagi's), multi-query in one round-trip, when you want Perplexity's source-ranking specifically.
- Prefer **`pplx fetch --prompt`** over a "search + fetch + summarize" chain: Perplexity's LLM does fetch+extract in one call.
- Prefer **`pplx snippets`** over "fetch + grep" or "fetch + LLM-summarize each URL" pipelines — local hybrid retrieval is faster, free, and ranks by query relevance.

# Examples

```bash
# Ranked search, multi-query, server-side merge
pplx search "claude code agentic" "claude code installation" -n 5

# JSON output for parsing
pplx search "openssh persourcepenalties" -j | jq '.hits[0].summary'

# Plain URL fetch → cleaned markdown
pplx fetch "https://docs.anthropic.com/claude-code"

# LLM extraction in one round-trip (no fetch-then-feed-to-LLM chain)
pplx fetch "https://release.notes/perplexity-comet-1.2" \
  --prompt "What was added in this release? Bullet list."

# Hybrid retrieval over N URLs (BM25 + semantic via fastembed + sqlite-vec)
pplx snippets "TLS fingerprinting" \
  "https://github.com/lexiforest/curl_cffi" \
  "https://developers.cloudflare.com/turnstile/" \
  --max-tokens 1500 --max-tokens-per-page 600

# Validate session
pplx auth check
```

# Exit codes (stable contract for retry logic)

| Code | Meaning | Retry semantic |
|---|---|---|
| 0 | Success | n/a |
| 1 | Generic failure / bug | don't retry |
| 2 | Auth: cookies missing/expired/rejected | refresh cookies (`pplx auth import --browser <name>`) and retry |
| 3 | Rate limit (429) | exponential backoff |
| 4 | Network (DNS / timeout / TLS) | linear backoff |
| 5 | Anti-bot (Cloudflare challenge) | investigate, don't auto-retry |

Stdout is results only; stderr carries diagnostics. `2>/dev/null` gives clean parseable stdout.

# First-run notes

- `pplx snippets` downloads ~80 MB embedding model on first invocation (cached at `~/.cache/fastembed/`). Subsequent calls are 1–2 s for N≈5 URLs.
- `pplx auth import --browser <name>` pops a macOS keychain prompt the first time; click "Always Allow" so future runs are silent.

# Caveats

- Unofficial. Endpoints can change without notice — bug reports welcome at github.com/ak2k/pplx-agent-tools.
- `-t academic / images / videos / shopping` for `pplx search` are Phase 2 (raise `NotImplementedError` today; no cookie-auth endpoints exist for those variants).
- `pplx fetch` plain mode is a local fetch (no Perplexity-backend paywall bypass / cache reuse). Use `--prompt` for LLM-routed extraction when those features matter.
- Prompt-injection awareness: `pplx fetch --prompt` sends fetched page content to Perplexity's LLM. Adversarial pages can manipulate the extraction.
