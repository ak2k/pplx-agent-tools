---
name: pplx-agent-tools
description: Query Perplexity via your Pro subscription's web session. Use `pplx search` for ranked web hits (sources), `pplx ask "..." [--model X]` for one synthesized cited answer (Pro Search), `pplx research "..."` for deep multi-step cited research, `pplx fetch URL --prompt "..."` for one-call URL-to-LLM-extracted-answer, or `pplx snippets QUERY URL...` for hybrid (keyword + semantic) excerpt extraction from N supplied URLs. `pplx quota` shows rate-limit availability; `pplx models` lists models/modes. Pair search → snippets for "find candidates, then dig into specific ones."
---

# When to reach for each verb

- **`pplx search <query>...`** — ranked web hits (sources, no answer). Each hit carries `title`, `url`, `domain`, `snippet` (~200 chars), and `summary` (~1500 chars, agent-friendly extract). Multi-query is native — pass several queries, server merges/dedupes. Stateless (creates no thread).
- **`pplx ask <query>`** — the front door: ask a question, get one **synthesized, cited answer** + its sources (Pro Search). `search` returns sources; `ask` returns the answer (with `-j` also a `sources` list). `--model <id>` picks the model (default `turbo` = "Best"; pass a thinking variant like `claude55opusthinking` for max reasoning — see `pplx models`, ids rotate). ~5–15 s on `turbo`; a thinking variant measured 30–90 s. Session-creating but incognito + auto-cleanup. Check its figures against the cited sources before using them; see "Using `ask` answers" below.
- **`pplx research <query>`** — deep, multi-step, cited research (Perplexity's "Research" mode). Returns a long markdown report + a sources list — `answer` is the full report (cover note then body), not a summary of one. Takes ~90–120 s for a focused question, but runtime grows with the number of subjects: a 14-subject comparison measured 18 min and 560 sources. Far more thorough than `search`; see "Running `research`" below. This is the differentiated capability — reach for it when one search won't cut it. Session-creating but runs **incognito** (no history pollution) + auto-cleans the thread.
- **`pplx fetch <url>`** — local fetch + cleaned content extraction. With `--prompt`, routes to Perplexity's LLM which fetches the URL itself and answers your prompt in one round-trip (model-selectable via `--model`, default `turbo`; runs incognito + auto-cleanup like `ask`).
- **`pplx snippets <query> <url>...`** — concurrent-fetch N URLs locally, return query-relevant paragraphs from each using hybrid retrieval (BM25 keyword + semantic vectors). Useful after `pplx search` narrows candidates.
- **`pplx quota`** — subscription rate-limit / availability per mode (`research`, `pro_search`, …) + per-source. Stateless GET; check before firing an expensive `research` call in a loop. Validates the session first: an expired cookie exits 2 (same message as `pplx auth check`) instead of rendering the anonymous view, in which every mode reads `EXHAUSTED (0 remaining)`.
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

# Using `ask` answers

- **Check every figure against a source.** On a long compound question `ask` has returned a complete, plausible row of numbers that appeared in none of its sources, cited only to a site's landing page. Treat a figure or name as usable only when you can see it in a cited hit's `snippet`/`summary` or in a page you fetched. If the synthesized answer is the only place it appears, record it as unknown.
- **Citations drift even when the prose is right.** Numbered footnotes can point at the wrong document. Open the primary source before a claim goes anywhere someone else relies on it.
- **One fact per question, a few sentences per prompt.** Fabrication and timeouts both concentrate in long multi-part prompts (a ~250-word prompt has hit the 180 s deadline before the first token, or returned no answer). Ask several narrow questions in parallel instead; a failure then costs one fact, not the batch. For pure extraction, `search` (the `summary` field often carries the exact figure) or `search` → `snippets` is more reliable than `ask`.
- **Pin a stronger model.** `turbo` is the model that fabricated above. Set `$PPLX_MODEL` (or `$PPLX_ASK_MODEL` / `$PPLX_FETCH_MODEL`) to a current Opus thinking id from `pplx models` once instead of passing `--model` every call; `--model turbo` still downgrades a cheap query (~4× faster). The pin improves depth; it does not make citations trustworthy.
- **Fan-out.** `ask`, `research` and `fetch --prompt` share one endpoint and its rate limit (429, exit 3); `search` does not. Under many parallel agents, use `search` as the primary and `ask` opportunistically. Stacked ask calls running next to a `research` call have timed out before their first token while single-fact asks in parallel completed.

# Running `research`

- **Run it in the background from an agent loop.** A long run outlasts foreground tool-call caps (Claude Code's Bash tool stops at 10 min); background the command and read its output when it exits.
- **Split broad prompts.** More than 3–4 subjects in one prompt → several narrower `research` calls in parallel. Each finishes in minutes, and a failure costs one slice instead of a 20-minute run and its quota unit.
- **Leave `--timeout` alone for most runs.** The 1800 s default is a hard cap, not an estimate: a focused question finishes in ~2 min, a broad one can run 20+. A hung run is cut sooner by the stall guard (`--stall-timeout`, default 120 s without new data), which returns what arrived so far (exit 6). A deadline trip that returns hundreds of sources and no answer means the run was still working, not stuck: split the prompt or raise `--timeout`.

# Exit codes (stable contract for retry logic)

| Code | Meaning | Retry semantic |
|---|---|---|
| 0 | Success | n/a |
| 1 | Generic failure / bug | don't retry |
| 2 | Auth: cookies missing/expired/rejected | refresh cookies (`pplx auth import --browser <name>`) and retry |
| 3 | Rate limit (429) | exponential backoff |
| 4 | Network (DNS / timeout / TLS), or a deadline or stall before any content arrived | linear backoff |
| 5 | Anti-bot (Cloudflare challenge) | investigate, don't auto-retry |
| 6 | Partial: stream incomplete (deadline or stall tripped, or server cut; a warning names which), or the answer decoded shorter than an earlier frame (`content_shortfall: true` with `stream_complete: true`). Stdout still carries usable content. | incomplete stream → accept the partial or bump `--timeout` (blind retry usually hits the same backend slowness); `content_shortfall` with `stream_complete: true` → the stream finished, so a longer timeout cannot help: re-run once or accept the shorter answer |

Stdout is results only; stderr carries diagnostics. `2>/dev/null` gives clean parseable stdout.

# First-run notes

- `pplx snippets` downloads ~80 MB embedding model on first invocation (cached at `~/.cache/fastembed/`). Subsequent calls are 1–2 s for N≈5 URLs.
- `pplx snippets` needs SQLite 3.38 or newer; an older build is refused with a clear error rather than quietly returning no semantic matches.
- `pplx auth import --browser <name>` pops a macOS keychain prompt the first time; click "Always Allow" so future runs are silent.

# Caveats

- Unofficial. Endpoints can change without notice — bug reports welcome at github.com/ak2k/pplx-agent-tools.
- `pplx search` is web-results only. Image/video/news "variant searches" are NOT standalone — they're entry-scoped (require an existing answer thread), so they aren't exposed. For deeper/synthesized results use `pplx research` (which selects `mode=research` on the ask endpoint).
- `pplx research` and `pplx ask` are the session-creating verbs (both run incognito + auto-cleanup). `pplx ask` defaults to a 180 s deadline (`--timeout N`, `$PPLX_ASK_TIMEOUT`, or 0 to disable), sized to clear a thinking model's tail; same partial + `stream: incomplete` + exit 6 contract as the others. All three ask-family calls (`ask`, `research`, `fetch --prompt`) also cut a stream that sends no new data for 120 s (`--stall-timeout N`, `$PPLX_STALL_TIMEOUT`, or 0 to disable); server heartbeats don't count as data. `pplx research` sends `is_incognito: true` so the thread never enters your Perplexity history, and best-effort-deletes it afterward (`--keep-thread` to retain). Default deadline 1800 s, a hard cap (`--timeout N`, `$PPLX_RESEARCH_TIMEOUT`, or 0 to disable); on a deadline or stall trip you get the partial report + `stream: incomplete` marker + exit 6. A completed stream whose report is nonetheless shorter than an earlier frame's sets `content_shortfall: true` (+ exit 6) — treat that answer as truncated. Real Deep Research takes ~90–120 s for a focused question and far longer for a broad one. `--model <id>` overrides the model (research accepts `pplx_alpha`/`o4mini`). `--mode council` (Model Council: 3 frontier models cross-checked, ~80 s) — pick the trio with `--council-models a,b,c`; a sensible default trio is auto-sent (council stalls without one).
- `pplx quota` / `pplx models` are stateless read-only GETs (no thread, no LLM cost). Both endpoints answer 200 to an anonymous caller; `quota` pre-flights `/api/auth/session` so an all-`EXHAUSTED` table you actually see is real quota state, not a stale cookie.
- `pplx fetch` plain mode is a local fetch (no Perplexity-backend paywall bypass / cache reuse). Use `--prompt` for LLM-routed extraction when those features matter.
- Prompt-injection awareness: `pplx fetch --prompt` sends fetched page content to Perplexity's LLM. Adversarial pages can manipulate the extraction.
- `pplx fetch --prompt` defaults to a 180 s overall deadline (override with `--timeout N`, `$PPLX_FETCH_TIMEOUT`, or 0 to disable). On a deadline or stall trip you get any partial content + a `stream: incomplete` header marker + a stderr warning — check `stream_complete` in JSON output or grep stderr if your script can't tolerate a partial answer. 429s auto-retry up to 3 attempts honoring `retry-after`, bounded by the same deadline.
