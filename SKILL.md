---
name: pplx-agent-tools
description: Query Perplexity via your Pro subscription's web session. Use `pplx search` for ranked web hits (sources), `pplx ask "..." [--model X]` for one synthesized cited answer (Pro Search), `pplx research "..."` for deep multi-step cited research, `pplx fetch URL --prompt "..."` for one-call URL-to-LLM-extracted-answer, or `pplx snippets QUERY URL...` for hybrid (keyword + semantic) excerpt extraction from N supplied URLs. `pplx quota` shows rate-limit availability; `pplx models` lists models/modes. Pair search → snippets for "find candidates, then dig into specific ones."
---

# When to reach for each verb

- **`pplx search <query>...`** — ranked web hits (sources, no answer). Each hit carries `title`, `url`, `domain`, `snippet` (~200 chars), and `summary` (~1500 chars, agent-friendly extract). Multi-query is native — pass several queries, server merges/dedupes. Stateless (creates no thread).
- **`pplx ask <query>`** — the front door: ask a question, get one **synthesized, cited answer** (Pro Search). `search` returns sources; `ask` returns the answer. `--model <id>` picks the model (default `turbo` = "Best"; pass a thinking variant like `claude48opusthinking` for max reasoning — see `pplx models`). ~5–15 s. Session-creating but incognito + auto-cleanup.
- **`pplx research <query>`** — deep, multi-step, cited research (Perplexity's "Research" mode). Returns a long markdown report + a sources list. Takes ~10–60 s; far more thorough than `search`. This is the differentiated capability — reach for it when one search won't cut it. Session-creating but runs **incognito** (no history pollution) + auto-cleans the thread.
- **`pplx fetch <url>`** — local fetch + cleaned content extraction. With `--prompt`, routes to Perplexity's LLM which fetches the URL itself and answers your prompt in one round-trip.
- **`pplx snippets <query> <url>...`** — concurrent-fetch N URLs locally, return query-relevant paragraphs from each using hybrid retrieval (BM25 keyword + semantic vectors). Useful after `pplx search` narrows candidates.
- **`pplx quota`** — subscription rate-limit / availability per mode (`research`, `pro_search`, …) + per-source. Stateless GET; check before firing an expensive `research` call in a loop.
- **`pplx models`** — model catalog + mode catalog + default model per mode. Stateless GET; feeds `pplx research --mode`.
- **`pplx auth check`** — validate cookies. Run if other verbs fail with exit code 2.

# vs `kagi-search`

- Prefer **`kagi-search`** for: small queries where the Quick Answer summary is enough; queries you'd rather route through Kagi than Perplexity.
- Prefer **`pplx search`** for: deeper extraction (the `summary` field is much longer than Kagi's), multi-query in one round-trip, when you want Perplexity's source-ranking specifically.
- Prefer **`pplx research`** over `search`/`kagi-search` when the question needs synthesis across many sources, not a hit list — it runs multiple searches and writes a cited report. Slower + spends a research-quota unit, so use `search` for quick lookups and `research` for "go deep."
- Prefer **`pplx fetch --prompt`** over a "search + fetch + summarize" chain: Perplexity's LLM does fetch+extract in one call.
- Prefer **`pplx snippets`** over "fetch + grep" or "fetch + LLM-summarize each URL" pipelines — local hybrid retrieval is faster, free, and ranks by query relevance.

# Examples

```bash
# Ranked search, multi-query, server-side merge
pplx search "claude code agentic" "claude code installation" -n 5

# JSON output for parsing
pplx search "openssh persourcepenalties" -j | jq '.hits[0].summary'

# Ask a question → one synthesized cited answer (pick a model)
pplx ask "what changed in HTTP/3 vs HTTP/2 for CDNs?"
pplx ask "explain QUIC's 0-RTT security tradeoffs" --model claude48opusthinking

# Deep multi-step cited research (slower; returns a report + sources)
pplx research "Compare HTTP/3 adoption across major CDNs in 2026" --timeout 240
pplx research "..." -j | jq '.answer, .sources'

# Check availability before an expensive research loop; list models/modes
pplx quota
pplx models -j | jq '.default_models'

# Plain URL fetch → cleaned markdown
pplx fetch "https://docs.anthropic.com/claude-code"

# LLM extraction in one round-trip (no fetch-then-feed-to-LLM chain)
pplx fetch "https://release.notes/perplexity-comet-1.2" \
  --prompt "What was added in this release? Bullet list."

# Bound the wall-clock budget for slow prompts; on deadline, returns whatever
# the stream produced with stderr warning + "stream: incomplete" header marker.
pplx fetch "$URL" --prompt "..." --timeout 60

# Heartbeat dots to stderr (useful when backgrounding concurrent calls)
pplx fetch "$URL" --prompt "..." --progress

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
| 6 | Partial: stream incomplete (deadline tripped or server cut). Stdout still carries usable content. | accept partial OR bump `--timeout`; blind retry usually hits the same backend slowness |

Stdout is results only; stderr carries diagnostics. `2>/dev/null` gives clean parseable stdout.

# First-run notes

- `pplx snippets` downloads ~80 MB embedding model on first invocation (cached at `~/.cache/fastembed/`). Subsequent calls are 1–2 s for N≈5 URLs.
- `pplx auth import --browser <name>` pops a macOS keychain prompt the first time; click "Always Allow" so future runs are silent.

# Caveats

- Unofficial. Endpoints can change without notice — bug reports welcome at github.com/ak2k/pplx-agent-tools.
- `pplx search` is web-results only. Image/video/news "variant searches" are NOT standalone — they're entry-scoped (require an existing answer thread), so they aren't exposed. For deeper/synthesized results use `pplx research` (which selects `mode=research` on the ask endpoint).
- `pplx research` is session-creating (the only verb that is): it sends `is_incognito: true` so the thread never enters your Perplexity history, and best-effort-deletes it afterward (`--keep-thread` to retain). Default deadline 300 s (`--timeout N`, `$PPLX_RESEARCH_TIMEOUT`, or 0 to disable); on deadline trip you get the partial report + `stream: incomplete` marker + exit 6. Real Deep Research takes ~90–120 s (multi-round, ~40+ sources). `--model <id>` overrides the model (research accepts `pplx_alpha`/`o4mini`). `--mode council` (Model Council, `--council-models a,b,c`) is **experimental and currently unreliable** — observed to run >7 min without completing; prefer plain `research`.
- `pplx quota` / `pplx models` are stateless read-only GETs (no thread, no LLM cost).
- `pplx fetch` plain mode is a local fetch (no Perplexity-backend paywall bypass / cache reuse). Use `--prompt` for LLM-routed extraction when those features matter.
- Prompt-injection awareness: `pplx fetch --prompt` sends fetched page content to Perplexity's LLM. Adversarial pages can manipulate the extraction.
- `pplx fetch --prompt` defaults to a 180 s overall deadline (override with `--timeout N`, `$PPLX_FETCH_TIMEOUT`, or 0 to disable). On deadline trip you get any partial content + a `stream: incomplete` header marker + a stderr warning — check `stream_complete` in JSON output or grep stderr if your script can't tolerate a partial answer. 429s auto-retry up to 3 attempts honoring `retry-after`, bounded by the same deadline.
