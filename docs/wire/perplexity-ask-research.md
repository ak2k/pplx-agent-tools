# /rest/sse/perplexity_ask — research mode (depth-pass for `pplx research`)

Live-probed 2026-06-22 (re-verified after the mechanism was corrected). Same
endpoint as `fetch --prompt`; the mode is selected by **`model_preference`**.

## How the mode is actually selected — `model_preference`, NOT `params.mode`

The decisive finding (verified by sending each value and observing behavior):
**`params.mode` does not switch deep modes; `params.model_preference` does.**
Sending `params.mode="research"` with `model_preference="turbo"` yields a plain
copilot answer (1 search round, ~15 sources). It's the *model* that triggers the
behavior:

| Behavior | `model_preference` | observed | `params.mode` |
|---|---|---|---|
| Pro Search (copilot) | `turbo` | 1 SEARCH_WEB round, ~15 sources, ~20s | `copilot` |
| **Deep Research** | **`pplx_alpha`** | LOAD_SKILL + **3 SEARCH_WEB rounds** + THOUGHT steps, **~43 sources**, ~90–115s | `copilot` |
| **Model Council** | **`pplx_agentic_research`** | `INITIAL_QUERY` → `COUNCIL_RESEARCH` → `FINAL`, ~80s | `copilot` |
| Computer / ASI | `pplx_asi` | echoes mode `ASI`; separate runtime (needs `/rest/realtime/v2/computer/session`) | `copilot` |

**Model Council gotcha (verified 2026-06-23):** `pplx_agentic_research` STALLS
forever unless `params.compare_model_preferences` is set (the web always sends the
3-model trio). With the trio it completes in ~80s and returns a normal `FINAL`
block (`content.answer` JSON-wrapped like Deep Research, so the same decoder
works). `pplx research --mode council` therefore auto-sends a default trio
(`gpt55_thinking, claude48opusthinking, gemini31pro_high`) unless `--council-models`
overrides it.

So `pplx research` keeps `params.mode = "copilot"` (coarse; the server derives the
real mode from the model) and maps its user-facing `--mode` to the driving model
(`research`→`pplx_alpha`, `agentic_research`/`council`→`pplx_agentic_research`).
See `verbs/research.py:_MODE_MODEL`. (The SPA's `Xe`/`Ze` enum-map of UI mode →
`search`/`research`/`model-council`/`computer` is a *route/label* slug, not the
ask-body field — an earlier draft of this doc wrongly assumed it was `params.mode`.)

## Request

`POST /rest/sse/perplexity_ask`, body identical to the `copilot` chat body
(`pplx_agent_tools/verbs/fetch.py:_build_chat_body`) except:

- `params.model_preference = "pplx_alpha"` (Deep Research) / `"pplx_agentic_research"` (Council)
- `params.is_incognito = true` — **required discipline**: an incognito ask never
  enters `list_recent`/history (verified); deletion becomes belt-and-suspenders
  rather than load-bearing. See `CLAUDE.md` → "Endpoint selection principle".

## Response (schematized SSE)

With `use_schematized_api: true`, frames are JSON objects; the answer accumulates
in the `text` field. Top-level frame keys observed:

```
backend_uuid, context_uuid, uuid, frontend_context_uuid, text, display_model,
mode, search_focus, source, attachments, thread_url_slug, gpt4, text_completed,
message_mode, answer_modes, reconnectable, cursor, search_mode
```

- `text` is a **JSON string** → decodes to a **list of blocks**, each
  `{step_type, content, uuid}`. Observed `step_type`s for Deep Research:
  `INITIAL_QUERY`, `LOAD_SKILL`, `LOAD_SKILL_RESPONSE`, `SEARCH_WEB` (×N rounds),
  `SEARCH_RESULTS` (×N), `THOUGHT` (×N reasoning steps), `FINAL`. Model Council
  emits a `COUNCIL_RESEARCH` block instead. The verb walks blocks by `step_type`:
  the **`FINAL` block's `content.answer` is itself a JSON string** wrapping
  `{answer: <markdown>, web_results: [<cited>], chunks, structured_answer}` — so
  decode unwraps that and prefers `FINAL.web_results` over the intermediate
  `SEARCH_RESULTS` rounds. (Deep Research answers sometimes carry a short
  thinking preamble like "Now I have comprehensive data… Let me compose…" at the
  head — it's the reasoning model's output, not a parse artifact.) A `FAILED`
  frame (incompatible model↔mode) still carries `text`, so check `status` first.
  This differs from the non-schematized `markdown_block` chunks `fetch --prompt`
  consumes, so `research` has its own parser.
- `text_completed: true` + a final `status: "COMPLETED"` frame mark the end.
- `reconnectable: true` + `cursor` tie into
  `GET /rest/sse/perplexity_ask/reconnect/{resume_entry_uuid}` — the basis for
  fixing the exit-6 partial-stream blind-retry (resume from cursor instead).

## Observations

- **Latency:** ~11 s (simple query) to ~54 s (multi-part query). Minutes-scale
  only expected for `model-council`/agentic. Bound the verb with the existing
  `--timeout` + partial-return (exit 6) machinery; deep research is the prime
  user of stream-resume.
- **Answer size:** 48–76 KB of schematized blocks — research returns a long
  cited report, the actual value over `search`.
- **Statelessness:** creates a thread (`backend_uuid`), but `is_incognito: true`
  keeps it out of history; `DELETE /rest/thread/delete_thread_by_entry_uuid`
  returns 200 as secondary cleanup. No `list_recent` pollution.
- **No clarifying-question stall** for specific queries. Vague queries may trigger
  `/rest/sse/handle_perplexity_research_clarifying_answers` — handle or avoid by
  passing specific prompts.

## Implementation sketch (`pplx research`)

1. `verbs/research.py` — `research(client, query, *, mode="research",
   keep_thread=False, timeout=...) -> ResearchResult`. Reuse `_build_chat_body`
   with `mode` + `is_incognito=True`; reuse `_consume_one_stream` to capture
   `backend_uuid`/`read_write_token`; add a block decoder for the schematized
   `text` (json.loads → walk `step_type` blocks → answer + sources).
2. `render.py` — `render_research_{text,json}`; JSON surfaces the cited report +
   sources array.
3. `cli_research.py` + register in `cli.VERBS`.
4. Carry over `fetch`'s deadline/partial (exit 6) + 429 retry; wire `reconnect`
   for resume as a follow-up.
