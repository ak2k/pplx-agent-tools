# pplx search web — wire format

Reverse-engineered 2026-05-12 against `www.perplexity.ai` frontend SPA build
`8e78ece`. The web SPA exposes two relevant search surfaces:

1. **`/rest/realtime/search-web`** — direct ranked-hit search (what we use)
2. `/rest/sse/perplexity_ask` — SSE-streamed chat with web_results as one of
   many response blocks (heavier; documented at end for reference / future
   `--prompt` use)

## Endpoint

```
POST https://www.perplexity.ai/rest/realtime/search-web
Content-Type: application/json
```

JSON request, JSON response, no streaming, no LLM cost. The endpoint takes
~5s on the wire — designed for fast hit retrieval.

## Request body

Minimum viable:

```json
{
  "session_id": "<uuid4>",
  "queries": ["<query>", "<query>", ...]
}
```

Required fields confirmed via 422 validation messages:
- `session_id` — per-call tracking UUID; no server-side state observed across
  calls, so we generate a fresh one each invocation
- `queries[]` — array of strings. **Multi-query is native** — pass N strings
  to get one merged/deduped result list back

Not yet probed against this endpoint (deliberately optimistic — may 422):
- `country` — country-code filter
- `domain_filter` / `excluded_domains` — domain include/exclude lists

If any of these 422, our code path forwards them only when explicitly set, so
the default-case call (no filters) is safe.

## Response

```json
{
  "media_items": [],
  "web_results": [
    { /* hit */ }, ...
  ]
}
```

`media_items` is non-empty when the query has video/image results (e.g.
"youtube travel videos"); see Step 9 for `-t images / videos` variants.

### Hit shape

Full field inventory (captured for query "claude code"):

| Field | Type | Notes |
|---|---|---|
| `url` | string | the link |
| `name` | string | page title |
| `domain` | string | bare domain (e.g. "anthropic.skilljar.com") |
| `snippet` | string | short description, ~200 chars typical |
| `summary` | string | longer extraction, ~1500 chars typical — agent-friendly |
| `timestamp` | ISO 8601 | publication or index date (varies by source) |
| `id` | string | per-hit UUID — likely stable across calls for cache keys |
| `language` | string | "en", "" if unknown |
| `meta_data` | object?\|null | `{images: [...]}` when image previews exist |
| `media_items` | array?\|null | nested media for this hit (mostly null) |
| `url_content` | string?\|null | always null in observed responses — possibly populated when a different param is set; unconfirmed |
| `relevance_score` / `score` | number?\|null | always null; ranking is implicit by array order |
| `tab_id`, `page_id`, `source`, `client`, `query`, `engine`, `request_source` | string? | metadata, mostly empty in observed responses |
| `connector_s3_key`, `file_metadata`, `finance_ticker_attributes` | various | non-web-result variants — null for web hits |
| `is_*` flags (15 booleans) | bool | type discriminators — see filter below |

### Filter flags

A `web_result` carries 15 `is_*` flags. We drop the hit if any of these are
true (they're not "web search hits" in the agent sense — they're widgets,
nav suggestions, etc.):

- `is_navigational` — URL-bar autosuggest
- `is_widget` — embedded UI cards (weather, finance, ...)
- `is_knowledge_card` — knowledge-graph panels
- `is_image` / `is_video` / `is_audio` — non-web media (use `-t <type>`)
- `is_map` — places preview
- `is_memory` / `is_conversation_history` / `is_conversation_summary` — user's prior threads
- `is_attachment` — user-uploaded files
- `is_extra_info` — metadata-only entries
- `is_pro_search_table` — pro-search structured tables

Kept (informational, no filtering):
- `is_entropy_visible_result` — true for normal web hits
- `is_code_interpreter`, `is_video_preview`, `is_scrubbed`, `is_truncated`, etc.

## Mapping to our Hit shape

| Our field | Source path | Notes |
|---|---|---|
| `url` | `.url` | direct |
| `title` | `.name` | renamed for consistency with kagi-search |
| `domain` | `.domain` | direct (no longer nested in meta_data) |
| `snippet` | `.snippet` | direct |
| `summary` | `.summary` | new field — ~10x snippet length, agent gold |
| `published_date` | `.timestamp` | optional; ISO-8601 |
| `images` | `.meta_data.images` | optional list |

## Pivot history

The initial Step-5 implementation used `/rest/sse/perplexity_ask` (the SSE
chat endpoint) and early-terminated after the first event with web_results.
That worked but cost a 3.9 MB stream per query and burned LLM tokens we
didn't use. Switched to `/rest/realtime/search-web` after broader RE pass
(2026-05-12) found this endpoint among the SPA's API client call sites.

## Reference: /rest/sse/perplexity_ask

Still relevant for the chat / `--prompt` use case (Step 7). Captured request
body and SSE event shape preserved in `re-fixtures/search-web/*.events.jsonl`
and `*.raw.sse`. Key facts:

- Body: ~30 params, see `scripts/re-replay-search.py` for canonical shape
- Response: SSE `text/event-stream` with CRLF line endings, ~150 events for
  a typical query, 3-4 MB total
- web_results land in `blocks[].web_result_block.web_results[]` at event ~2
- LLM-synthesized answer lands in `blocks[].markdown_block`
- Terminal event has `event: end_of_stream` and `status: COMPLETED`

## Captured fixtures

Working captures (gitignored):
- `re-fixtures/search-web/realtime-search-web-<ts>.json` — clean realtime response
- `re-fixtures/search-web/<query>-text_True-<ts>.events.jsonl` — parsed SSE events
- `re-fixtures/search-web/<query>-text_True-<ts>.raw.sse` — raw SSE bytes
- `re-fixtures/search-web/<query>-<ts>.har` — Playwright network capture

Sanitized fixtures intended for tests will live under `tests/fixtures/` per
the plan's test strategy section.
