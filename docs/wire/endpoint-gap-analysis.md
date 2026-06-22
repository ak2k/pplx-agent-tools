# pplx-agent-tools — endpoint surface re-examination (2026-06-22)

Re-audit of Perplexity's web API surface against what the CLI exposes. Prompted
by "are there more capabilities than we've exposed?" — answer: **yes, a lot**,
and several "doesn't exist" conclusions from the May-2026 plan are now wrong.

Inventory artifacts: `docs/wire/endpoint-inventory/2026-06-22T15-32-48Z.{md,json}`.
Re-runnable via `scripts/enumerate-endpoints.py`.

## TL;DR

- **623 `/rest/*` endpoints across ~75 namespaces** referenced by the current
  frontend. The CLI exposes **3 verbs** (`search`, `fetch`, `snippets`) hitting
  2 endpoints. The May plan claimed "~120 endpoints / 30 namespaces" but never
  committed that inventory; today's surface is ~5× larger or the old walk was
  partial. Either way, there was no baseline to diff against — there is now.
- The biggest unlocks are **read-only GETs that need almost no RE**:
  `rate-limit/status` (quota), `models/config` + `models/modes` (model/mode
  catalog). All three return 200 today via the existing `curl_cffi` session
  (validated, see below).
- The richest deep surfaces are **finance (92 endpoints)**, the **agentic stack**
  (tasks/triggers/automations/workflows/computer/browser/memories/skills), and
  **search modes** (`research`, `agentic_research`, `study`, `document_review`).

## Method — and why CDP is the *wrong* tool here

The original ask assumed we'd need "a browser via CDP to explore further."
We tried it and it's a **dead end for perplexity.ai**:

- A Playwright/agent-browser Chromium hits Cloudflare's **managed-challenge
  loop** on every HTML page route — `GET /` → `403` → `cdn-cgi/challenge-platform`
  → still `403`. The browser presents automation signals + a TLS fingerprint that
  doesn't match the `cf_clearance` cookie (issued by the user's real browser), so
  CF never serves the document. `scripts/re-capture-cdp.py` reproduces this; kept
  as the documented reason CDP is not the path (and as a base if we ever drive
  Comet, Perplexity's own browser, which CF trusts).
- `curl_cffi` (chrome impersonation) **does** pass CF for this site — it's how
  `pplx search` already works. The block is specific to page documents.
- The frontend migrated **Next.js → Vite/Rolldown**: chunks moved from
  `_next/static/chunks/*.js` to `_spa/assets/*.js` on
  `pplx-next-static-public.perplexity.ai` (a static CDN, no challenge). The May
  walk targeted `_next/static`, which is why a naive grep finds zero today.

So the working technique is **static SPA bundle archaeology v2**: fetch the 12 KB
shell, seed from its `_spa/assets/*.js` refs, BFS the Rolldown chunk graph (800
chunks, ~32 MiB), grep for `/rest/` + `/api/` literals, infer method from the
call site. CF-free, no browser, no account-side actions.

## May-plan conclusions now falsified

The plan's "Resolved as unreachable" / "deferred — no dedicated endpoint" claims
do not hold against the current surface:

| May conclusion | Reality now |
|---|---|
| "no rate-limit visibility" (daemon trigger was "hitting 429s blind") | `GET /rest/rate-limit/status` returns per-mode + per-source quota — **validated 200 today** |
| "variant searches route only through the chat SSE; no dedicated endpoints" | `models/config.default_models` exposes modes `research / agentic_research / study / document_review` selectable on the ask endpoint — deep/variant *modes* are reachable (but via the ask/session flow, not statelessly) |
| variant search was "Phase 2, needs SSE block parsing" | `/rest/sse` has **23** endpoints incl. resume/terminate (`perplexity_ask/reconnect/{uuid}`, `perplexity_terminate`) — directly relevant to the `pplx fetch --prompt` partial-stream problem (exit 6) |

**Correction (verified 2026-06-22, supersedes an earlier draft of this doc):**
the image/video/news "variant search" endpoints are **NOT** stateless standalone
searches. `POST /rest/media/search-images-and-videos` requires `{query,
entry_uuid, read_write_token}` and `POST /rest/sources/search/news` requires
`{entry_uuid, limit, page}` — both return `422` without an `entry_uuid`. They are
SERP tabs that enrich an *existing* answer thread (downstream of an ask entry).
There is **no stateless multi-result image/video/news search**.
`/rest/realtime/query-video` takes a `{video_url}` ("analyze this video"), not a
search query. The only stateless search remains `/rest/realtime/search-web`
(+ single-best-match `search-youtube`). This kills the "stateless variant search"
verb idea below — offering images/videos/news means adopting the ask/entry flow.

## Validated live (2026-06-22)

Read-only GETs (stateless, 200):

```
GET /rest/rate-limit/status  -> modes{pro_search,research,agentic_research,labs} + per-source availability
GET /rest/models/modes       -> { modes: [...] }
GET /rest/models/config      -> models{turbo,pplx_pro,gpt5,gpt51_thinking,...},
                                 default_models{search,research,agentic_research,study,
                                                document_review,browser_agent,asi}
```

Statelessness probes (before/after `GET /rest/thread/list_recent`):

- `POST /rest/sources/search/news` and `POST /rest/media/search-images-and-videos`
  → **422, `entry_uuid` required** → entry-scoped, NOT stateless.
- `POST /rest/realtime/{search-web,query-video}` → no thread created (delta 0).
- `POST /rest/sse/perplexity_ask` with `params.is_incognito: true` → backend_uuid
  returned but **does not appear in `list_recent`**; with `is_incognito: false` it
  does. → incognito is the clean way to run the ask-family without history
  pollution (deletion becomes optional rather than load-bearing).

## Prioritized new-verb candidates

Tier 1 — cheap, read-only, high agent value, minimal RE:

| Verb idea | Endpoint(s) | Method | Notes |
|---|---|---|---|
| `pplx quota` / `pplx rate-limit` | `/rest/rate-limit/status` | GET | per-mode + per-source budget; lets agent loops self-throttle. Validated. |
| `pplx models` | `/rest/models/config`, `/rest/models/modes` | GET | model + mode catalog; feeds a `--model`/`--mode` flag. Validated. |
| `pplx suggest` | `/rest/autosuggest/list-autosuggest` | POST | query completion/expansion. |

Tier 2 — ask-family (session-creating; must use `is_incognito: true` + cleanup):

| Verb idea | Endpoint(s) | Method | Notes |
|---|---|---|---|
| `pplx research` (deep) | `/rest/sse/perplexity_ask` + `mode=research`/`agentic_research`; `/rest/deeper-research/export-asset` | SSE/POST | Pro/deep-research mode of the ask endpoint; far stronger than `search`. Heaviest: `agentic_research` may also spawn tasks/assets, not just a thread. |
| `pplx fetch --prompt` resume | `/rest/sse/perplexity_ask/reconnect/{uuid}`, `/rest/sse/perplexity_terminate` | GET/POST | fixes the exit-6 partial-stream blind-retry — resume instead. |
| ~~`pplx search -t images/videos/news`~~ | `/rest/media/*`, `/rest/sources/search/news` | POST | **Dropped as stateless** — verified entry-scoped (422 without `entry_uuid`). Only viable atop an ask entry; revisit only if `research` lands and we want its SERP tabs. |

Tier 3 — deep verticals / agentic (bigger RE, judge by real need):

| Area | Endpoints | Notes |
|---|---|---|
| finance | `/rest/finance/*` (92) | financials v2/v3, earnings + transcripts, analyst ratings, insider holders, history CSV, price alerts (`/rest/tasks/finance`). Directly relevant to qbo-analytics / sec-nport / tc-investment-analysis. |
| memories | `/rest/memories/{list,get,delete}` | persistent memory read/write. |
| skills | `/rest/skills*` | list/grant Perplexity "skills". |
| connectors / MCP | `/rest/sources/custom` (remote MCP), `/rest/connector-service/connectors/{id}/tools/{tool}/execute` | register custom remote MCP sources; execute connector tools. Agent-native. |
| tasks / automations | `/rest/tasks/*`, `/rest/triggers/event-subscriptions`, `/rest/workflows*` | recurring/background tasks + event triggers (the Comet "Computer" surface). |

## Caveats

- Methods are **inferred from call-site context**; `?` means unresolved (often
  built via a generic request helper). Confirm before relying on a method.
- **Request/response body shapes are NOT yet RE'd** for POST endpoints — the
  bundle gives the path + method, not the payload. That's the depth pass: read
  the call site in the named chunk, or (for read-only GETs) probe live.
- 9 endpoints are **dynamic** (`${...}` template fragments); paths are partial.
- Enumeration capped at 800 chunks (5 unfetched). Re-run with `--max-chunks` for
  full closure if a namespace looks truncated.
- Endpoints existing in the bundle ≠ callable with cookie auth — some are
  enterprise/org-gated. Probe before promising a verb.

## Recommended next step

Pick the verbs to build (Tier 1 is near-free and high-value), then a **depth pass
per chosen endpoint**: read the call site in its chunk for the request shape, do
one live probe, capture a fixture, implement. Don't bulk-probe POST endpoints
blindly — single-user human pace, read-only first.
