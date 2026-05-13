# pplx-search web — wire format

Reverse-engineered 2026-05-12 against `www.perplexity.ai` frontend SPA build
`8e78ece`. Captured via `scripts/re-capture.py` (Playwright/CDP) + replayed
via `scripts/re-replay-search.py` for the SSE body (HAR can't record SSE).

## Endpoint

```
POST https://www.perplexity.ai/rest/sse/perplexity_ask
Content-Type: application/json
Accept: text/event-stream
```

The web frontend uses **one** endpoint for all queries (search, chat, deep
research). There is no separate ranked-hits endpoint. The `mode` and
`search_focus` params steer behavior.

## Request body

Captured production body has ~30 params, most UI-specific. The minimal viable
set for a web search verb:

```json
{
  "query_str": "<user query>",
  "params": {
    "query_source": "home",
    "prompt_source": "user",
    "source": "default",
    "version": "2.18",
    "language": "en-US",
    "timezone": "America/New_York",

    "search_focus": "internet",
    "sources": ["web"],
    "mode": "copilot",
    "model_preference": "turbo",

    "frontend_uuid": "<uuid4>",
    "frontend_context_uuid": "<uuid4>",
    "client_search_results_cache_key": "<same as frontend_uuid>",

    "use_schematized_api": true,
    "send_back_text_in_streaming_api": true,
    "skip_search_enabled": true,
    "is_nav_suggestions_disabled": false,
    "is_incognito": false,
    "is_related_query": false,
    "is_sponsored": false,
    "always_search_override": false,
    "override_no_search": false,
    "local_search_enabled": false,
    "extended_context": false,

    "attachments": [],
    "mentions": [],
    "client_coordinates": null,
    "dsl_query": "<same as query_str>"
  }
}
```

Fields we strip from the captured request:
- `time_from_first_type`, `rum_session_id` — UI telemetry
- `supported_block_use_cases`, `supported_features` — widget rendering hints
- `should_ask_for_mcp_tool_confirmation`, `browser_agent_*`, `force_*` — interactive UX

## Response: SSE stream

`Content-Type: text/event-stream; charset=utf-8`, CRLF line endings (RFC).
Two event types observed:

| Event | Count (per query) | Meaning |
|---|---|---|
| `message` | many (150+) | streaming progress + intermediate states + final answer |
| `end_of_stream` | 1 | terminal event with status `COMPLETED` |

Each `data:` payload is JSON. Top-level shape (only fields we care about):

```json
{
  "backend_uuid": "...",
  "status": "PENDING" | "COMPLETED",
  "text_completed": false | true,
  "blocks": [
    { "intended_usage": "...", "<block_kind>": { ... } },
    ...
  ],
  "telemetry_data": { "has_displayed_search_results": true, ... }
}
```

### web_results location

```
blocks[i].web_result_block.web_results[]
```

The block with `web_result_block` populated typically appears at event **index 2**
(very early in the stream — confirmed for `"claude code"` query). The same hit
list reappears in the final event (~151). The list is stable once populated:
no re-ranking observed across events.

### Hit shape

```json
{
  "name": "Claude Code overview - Claude Code Docs",
  "snippet": "Claude Code is an agentic coding tool that...",
  "url": "https://code.claude.com/docs/en/overview",
  "timestamp": "2026-05-11T00:00:00",
  "meta_data": {
    "citation_domain_name": "code.claude",
    "client": "web",
    "images": ["https://..."]
  },
  "is_attachment": false,
  "is_image": false,
  "is_code_interpreter": false,
  "is_knowledge_card": false,
  "is_navigational": false,
  "is_widget": false,
  "is_focused_web": false,
  "is_client_context": false,
  "is_memory": false,
  "is_conversation_history": false,
  "is_conversation_summary": false
}
```

## Mapping to our `hits[]` shape

| Our field | Source path | Notes |
|---|---|---|
| `url` | `.url` | direct |
| `title` | `.name` | renamed for consistency with kagi-search |
| `domain` | `.meta_data.citation_domain_name` | drop `meta_data` wrapper |
| `snippet` | `.snippet` | direct |
| `published_date` | `.timestamp` | optional; ISO-8601 |
| `images` | `.meta_data.images` | optional list of URLs |

## Filtering

Drop hits where any of the following are true (they're not "web search hits"
in the conventional sense the user wants from `pplx-search`):

- `is_navigational` — URL-bar autosuggest, "go to wikipedia.org"
- `is_widget` — embedded UI cards (weather, finance, etc.)
- `is_knowledge_card` — knowledge graph panels
- `is_image` — image-search-style results (use `-t images` for those)
- `is_memory` / `is_conversation_*` — user's prior threads
- `is_attachment` / `is_client_context` — user-uploaded context

## Early termination

We can abort the SSE connection as soon as we see the first event whose
`blocks[].web_result_block.web_results[]` is non-empty. This avoids streaming
the full LLM synthesis (the remaining ~95% of the response by bytes) which
we don't use. Saves bandwidth, latency, and the user's Pro-subscription
LLM-throughput budget.

## Multi-query

The plan describes positional multi-query: `pplx-search "q1" "q2" "q3"`.
The web endpoint takes one `query_str` per request. We fire N concurrent
POSTs and merge/dedupe by URL client-side. (Server-side merge is described
in the `pplx --help` text but not exposed via this endpoint.)

## Variants (`-t academic` / `-t images` / etc.)

Not RE'd yet. Likely changes:
- `sources` field: `["scholar"]` for academic, `["images"]` for images, etc.
- Possibly `search_focus`: `"scholar"`, `"images"`, etc.
- Hit shape may change for non-web sources.

Captured via `scripts/re-capture.py "..." --type <kind>` and document here as
each variant is RE'd.

## Captured fixtures

- `re-fixtures/search-web/<query>-<ts>.har` — full network capture (Playwright)
- `re-fixtures/search-web/<query>-<ts>.requests.jsonl` — request log
- `re-fixtures/search-web/<query>-text_True-<ts>.events.jsonl` — parsed SSE events
- `re-fixtures/search-web/<query>-text_True-<ts>.raw.sse` — raw SSE bytes

Cookie values are stripped via the .gitignore pattern; review before committing.
