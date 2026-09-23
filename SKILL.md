---
name: pplx-agent-tools
description: Query Perplexity via your Pro subscription's web session. Use `pplx search` for ranked web hits (sources), `pplx ask "..." [--model X]` for one synthesized cited answer (Pro Search), `pplx research "..."` for deep multi-step cited research, `pplx fetch URL --prompt "..."` for one-call URL-to-LLM-extracted-answer, or `pplx snippets QUERY URL...` for hybrid (keyword + semantic) excerpt extraction from N supplied URLs. `pplx quota` shows rate-limit availability; `pplx models` lists models/modes. Pair search → snippets for "find candidates, then dig into specific ones."
---

# When to reach for each verb

- **`pplx search <query>...`** — ranked web hits (sources, no answer). Each hit carries `title`, `url`, `domain`, `snippet` (~200 chars), and `summary` (~1500 chars, agent-friendly extract). Multi-query is native — pass several queries, server merges/dedupes. Stateless (creates no thread).
- **`pplx ask <query>`** — the front door: ask a question, get one **synthesized, cited answer** + its sources (Pro Search). `search` returns sources; `ask` returns the answer (with `-j` also a `sources` list). `--model <id>` picks the model (default `turbo` = "Best"; pass a thinking variant like `claude55opusthinking` for max reasoning — see `pplx models`, ids rotate). ~5–15 s on `turbo`; a thinking variant measured 30–90 s. Session-creating but incognito + auto-cleanup. Check its figures against the cited sources before using them; see "Using `ask` answers" below.
- **`pplx research <query>`** — deep, multi-step, cited research (Perplexity's "Research" mode). Returns a long markdown report + a sources list — `answer` is the full report (cover note then body), not a summary of one. Takes ~90–120 s for a focused question, but runtime grows with the number of subjects: a 14-subject comparison measured 18 min and 560 sources. Far more thorough than `search`; see "Running `research`" below. This is the differentiated capability — reach for it when one search won't cut it. Session-creating but runs **incognito** (no history pollution) + auto-cleans the thread.
- **`pplx fetch <url>`** — local fetch + cleaned content extraction; needs no Perplexity cookies. With `--prompt`, routes to Perplexity's LLM which fetches the URL itself and answers your prompt in one round-trip (model-selectable via `--model`, default `turbo`; runs incognito + auto-cleanup like `ask`).
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
- **Read the grounding check.** Every `ask` checks whether the answer's figures (normalized, so `$1.2M` = `1,200,000`) and multi-word names appear in any source's title or snippet. With `-j`: `grounded` is `true`, `false`, or `null` (nothing checkable, or `--no-grounded-check`); `checked_terms` counts the terms checked; `ungrounded_terms` lists those found in no source; `grounding_reasons` is a list of why (`every cited URL is a site root`, `no sources`, `N of M figures/names appear ...`). An ungrounded answer also gets a `grounded: no (...)` line on stdout and a `warning:` line on stderr; the exit code does not change. `grounded: true` means more than 1 in 5 checked terms appear in a cited source's title or snippet and not every cited URL is a site root; `false` means at most 1 in 5 do, every cited URL is a site root, or there are no sources. So `grounded: true` can still list `ungrounded_terms`, and it does not mean the answer is right; a figure outside the ~200-character snippet counts as unsupported.
- **`sources_complete: false`** means the answer is whole but the stream ended before its final sources frame, so `[n]` may not index `sources` (a `warnings` entry says so too); a cut stream (`stream_complete: false`) is also `sources_complete: false`.
- **Citations drift even when the prose is right.** Numbered footnotes can point at the wrong document. Open the primary source before a claim goes anywhere someone else relies on it.
- **One fact per question, a few sentences per prompt.** Fabrication and timeouts both concentrate in long multi-part prompts (a ~250-word prompt on a thinking model can take ~5 min, all of it before the first answer token). Ask several narrow questions in parallel instead; a failure then costs one fact, not the batch. For pure extraction, `search` (the `summary` field often carries the exact figure) or `search` → `snippets` is more reliable than `ask`.
- **Pin a stronger model.** `turbo` is the model that fabricated above. Set `$PPLX_MODEL` (or `$PPLX_ASK_MODEL` / `$PPLX_FETCH_MODEL`) to a current Opus thinking id from `pplx models` once instead of passing `--model` every call; `--model turbo` still downgrades a cheap query (~4× faster). The pin improves depth; it does not make citations trustworthy.
- **Fan-out.** `ask`, `research` and `fetch --prompt` share one endpoint and its rate limit (429, exit 3); `search` does not. Under many parallel agents, use `search` as the primary and `ask` opportunistically. Stacked ask calls running next to a `research` call have timed out before their first token while single-fact asks in parallel completed.

# Running `research`

- **Run it in the background from an agent loop.** A long run outlasts foreground tool-call caps (Claude Code's Bash tool stops at 10 min); background the command and read its output when it exits.
- **Split broad prompts.** More than 3–4 subjects in one prompt → several narrower `research` calls in parallel. Each finishes in minutes, and a failure costs one slice instead of a 30-minute run and its quota unit.
- **Leave `--timeout` alone for most runs.** The 3600 s default is a hard cap, not an estimate: a focused question finishes in ~2 min, a broad one can run 30+. A hung run is cut sooner by the stall guard (`--stall-timeout`, default 240 s without new content), which returns what arrived so far (exit 6). A deadline trip that returns hundreds of sources and no answer means the run was still working, not stuck: split the prompt or raise `--timeout`.

# Exit codes (stable contract for retry logic)

| Code | Meaning | Retry semantic |
|---|---|---|
| 0 | Success | n/a |
| 1 | Generic failure / bug, or a usage error (unknown flag, missing argument, out-of-range value such as `--limit 0` or `--timeout nan`). Under `--json`, a usage error prints an error envelope with `error.type: "UsageError"` and an unexpected exception one with `"InternalError"`. Also a refused `fetch` URL (`BlockedUrlError`: not http/https, no or malformed host, which includes an IPv4 host in any form but plain dotted decimal (octal, hex, short, integer, trailing dot) and a percent-encoded host, which includes the bad port that a password holding an unencoded `/`, `?` or `#` leaves (`user:pa/ss@host`); or a host that resolves to a private, loopback, link-local, CGNAT or other non-public address; redirects are checked the same way) and a fetched page that answers 4xx or redirects more than 5 times (`TargetHttpError`) | don't retry; fix the command line or pick another URL |
| 2 | Auth: cookies missing/malformed/expired/rejected (a cookie file whose names or values are not strings, or contain `;` or control characters, is refused; stderr names the source read) | refresh cookies (`pplx auth import --browser <name>`, which writes the profile file or `$PPLX_COOKIES_PATH`) and retry; if `$PPLX_COOKIES` is set, import refuses (exit 2): replace that variable instead |
| 3 | Rate limit (429 from Perplexity or from the page `fetch` requests) | exponential backoff |
| 4 | Network (DNS / timeout / TLS, or a fetched page answering 5xx or 408), or a deadline or stall before any content arrived | linear backoff |
| 5 | Anti-bot (Cloudflare challenge) | investigate, don't auto-retry |
| 6 | Partial: stream incomplete (deadline or stall tripped, or server cut; `cut_by` in JSON and the stdout marker name which), or the answer decoded shorter than an earlier frame (`content_shortfall: true` with `stream_complete: true`). Stdout still carries usable content. | deadline (`cut_by: "deadline"`, marker `stream: incomplete (deadline)`) → accept the partial or raise `--timeout` (blind retry usually hits the same backend slowness); stall (`cut_by: "stall"`, marker `stream: incomplete (stall: no new content)`) → accept the partial or raise `--stall-timeout` (a larger `--timeout` does not help); `content_shortfall` with `stream_complete: true` → the stream finished, so a longer timeout cannot help: re-run once or accept the shorter answer |

Stdout is results only; stderr carries diagnostics. `2>/dev/null` gives clean parseable stdout.

Counts (`-n/--limit`, `--max-chars`, `--max-tokens`, `--max-tokens-per-page`) must be at least 1; `--max-chars 0` is an error, so omit the flag for no cap. A timeout (`--timeout`, `--stall-timeout` and their env vars) of 0, a negative value, or `inf` disables it; `nan` is rejected. `pplx skill-path` takes no arguments.

# First-run notes

- `pplx snippets` downloads ~80 MB embedding model on first invocation (cached at `~/.cache/fastembed/`). Subsequent calls are 1–2 s for N≈5 URLs.
- `pplx snippets` needs SQLite 3.38 or newer; an older build is refused with a clear error rather than quietly returning no semantic matches.
- `pplx auth import --browser <name>` pops a macOS keychain prompt the first time; click "Always Allow" so future runs are silent.

# Caveats

- Unofficial. Endpoints can change without notice — bug reports welcome at github.com/ak2k/pplx-agent-tools.
- `pplx search` is web-results only. Image/video/news "variant searches" are NOT standalone — they're entry-scoped (require an existing answer thread), so they aren't exposed. For deeper/synthesized results use `pplx research` (which selects `mode=research` on the ask endpoint).
- `pplx research` and `pplx ask` are the session-creating verbs (both run incognito + auto-cleanup). `pplx ask` defaults to a 540 s hard cap (`--timeout N`, `$PPLX_ASK_TIMEOUT`, or 0 to disable): a thinking model on a long prompt can think for minutes and deliver the whole answer at the end, so a cut before that returns nothing (exit 4). Keep the call under your harness's foreground limit or run it in the background. All three ask-family calls also cut a stream that sends no new content (`--stall-timeout N`, `$PPLX_STALL_TIMEOUT`, or 0 to disable): after 480 s for `ask` and `fetch --prompt`, 240 s for `research`. Server heartbeats and frames repeating earlier content don't count. `pplx research` sends `is_incognito: true` so the thread never enters your Perplexity history, and best-effort-deletes it afterward (`--keep-thread` to retain). Default deadline 3600 s, a hard cap (`--timeout N`, `$PPLX_RESEARCH_TIMEOUT`, or 0 to disable); on a deadline or stall trip you get the partial report + `stream: incomplete (<cause>)` marker + `cut_by` in JSON + exit 6; a cut with no report and no sources yet exits 4 instead. A completed stream whose report is nonetheless shorter than an earlier frame's sets `content_shortfall: true` (+ exit 6) — treat that answer as truncated. Real Deep Research takes ~90–120 s for a focused question and far longer for a broad one. `--model <id>` overrides the model (research accepts `pplx_alpha`/`o4mini`). `--mode council` (Model Council: 3 frontier models cross-checked, ~80 s) — pick the trio with `--council-models a,b,c`; a sensible default trio is auto-sent (council stalls without one).
- `pplx quota` / `pplx models` are stateless read-only GETs (no thread, no LLM cost). Both endpoints answer 200 to an anonymous caller; `quota` pre-flights `/api/auth/session` so an all-`EXHAUSTED` table you actually see is real quota state, not a stale cookie.
- `pplx fetch` plain mode is a local fetch (no Perplexity-backend paywall bypass / cache reuse). Use `--prompt` for LLM-routed extraction when those features matter.
- Prompt-injection awareness: `pplx fetch --prompt` sends fetched page content to Perplexity's LLM. Adversarial pages can manipulate the extraction.
- `pplx fetch --prompt` removes `user:password@` from the URL before sending it to Perplexity and says so on stderr, so pages behind URL credentials cannot be reached that way. Both modes refuse a URL whose password holds an unencoded `/`, `?` or `#` and so leaves a malformed port (exit 1), rather than send it or print it. `pplx snippets` output never shows URL credentials. Plain `fetch` sends those credentials as HTTP basic auth.
- `pplx fetch --prompt` defaults to a 540 s hard cap (override with `--timeout N`, `$PPLX_FETCH_TIMEOUT`, or 0 to disable). On a deadline or stall trip you get any partial content + a `stream: incomplete (<cause>)` header marker + a stderr warning — check `stream_complete` and `cut_by` in JSON output or grep stderr if your script can't tolerate a partial answer. 429s auto-retry up to 3 attempts honoring `retry-after`, bounded by the same deadline.
