# /rest/sse/perplexity_ask — research mode (depth-pass for `pplx research`)

Live-probed 2026-06-22 to de-risk the flagship deep-research verb. Same endpoint
as `fetch --prompt`, different `mode` + response shape.

## Mode taxonomy (from the SPA bundle)

The UI mode enum maps to internal `mode` strings:

| UI mode | `params.mode` value | notes |
|---|---|---|
| Search | `search` | |
| Pro Search | `copilot` | what `fetch --prompt` sends today |
| Research | `research` | **deep research — the target** |
| Agentic Research | `model-council` | Model Council (parallel GPT/Claude/Gemini); heavier/slower |
| Study | `learn-files-and-apps` | |
| Browser Agent | `control-browser` | Comet agent |
| ASI | `computer` | Comet "Computer" |

## Request

`POST /rest/sse/perplexity_ask`, body identical to the `copilot` chat body
(`pplx_agent_tools/verbs/fetch.py:_build_chat_body`) except:

- `params.mode = "research"`
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
  `{step_type, content, uuid}`. The verb parser must `json.loads(text)` and walk
  blocks by `step_type` (final answer + sources/citations live in `content`;
  exact `step_type` taxonomy to be catalogued at implementation — only a couple
  of probes run so far). This differs from the non-schematized `markdown_block`
  chunks `fetch --prompt` consumes today, so `research` needs its own parser, not
  a flag on the existing path.
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
