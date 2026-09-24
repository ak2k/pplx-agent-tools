# /rest/sse/perplexity_ask in diff mode: wire facts (U1)

Live-probed 2026-09-24 for the transport plan (epic work-o30uy, unit U1). This
file answers the "Settles" column of plan §8 for P1, P2, P3 and P5, and records
the values U5a, U6 and U7 take from them.

## Method

- Probe scripts and logs live outside the repo, in
  `$HOME/src/ak2k/pplx-agent-tools-wt/cpu-artifacts/probes/u1/`. Log file names
  below are relative to that directory. Raw captures are in `captures/` (mode 600,
  directory 700) and are never committed.
- Transport: the b5r3 direct-drive prototype (`AsyncCurl`, `allow_redirects=False`,
  `SUPPRESS_CONNECT_HEADERS=1`), so a deliberate drop closes the stream within
  milliseconds. curl_cffi 0.15.0, CPython 3.12.
- Request: `base_ask_params` plus `send_back_text_in_streaming_api: false`,
  `supported_block_use_cases` (the 29-entry web list unless stated),
  `should_ask_for_mcp_tool_confirmation: false`, `supports_tool_approval_modal: false`,
  `is_incognito: true`.
- Offline analysis (`analyze.py`) replays each capture through an RFC 6902 applier
  (all six ops) keyed by `(intended_usage, field)`.
- Every run was incognito and its thread was deleted in a `finally`
  (`delete_thread ok=True` on all 17 runs). `GET /rest/thread/list_recent`
  returned 20 threads before and after, with the same hashed id set, and none of
  the probes' `backend_uuid` or `context_uuid` hashes (`preflight.log`).
- Spend: 5 of 5 Deep Research runs, 12 of 12 ask/fetch runs (`budget.jsonl`).
  `pro_search.remaining_detail.kind` was `not_provided` before and after.

## P5 static: terminate shape (`p5_static.log`)

Anonymous walk of the SPA bundle (1787 chunks, 58.9 MB).

- `POST /rest/sse/perplexity_terminate`, JSON body
  `{entry_uuid, context_uuid, model_preference, terminate_requested_at_ms}`. The web
  client fills them from the entry: `entry_uuid` is the `backend_uuid`,
  `context_uuid` and `model_preference` (`display_model`) come from the stream's
  envelope. No `read_write_token`.
- Every client call sends `X-Perplexity-Request-Reason: <reason>`. The stop button
  uses `thread-floating-footer`.
- No clarifying-question skip parameter exists in the bundle.
- Verdict: live P5 A/B was possible.

## P1: Deep Research, 29-entry list (A) vs minimal list (B)

Concurrent, same prompt, read to the final frame. Minimal list:
`["diff_blocks", "workflow_steps", "preserve_latex"]`. Log: `p1.log` (the
re-analysis lines at the end are authoritative for (e) and progress).

| | A (29 entries) | B (minimal) |
|---|---|---|
| status, outcome | 200, COMPLETED | 200, COMPLETED |
| wall time | 97.7 s | 74.3 s |
| frames / bytes | 1146 / 10.92 MB | 587 / 4.46 MB |
| largest frame | 171 KB | 346 KB (terminal) |
| terminal `text` | 69 KB, carries RESEARCH_ANSWER (body 13,136 chars) | 175 KB, carries RESEARCH_ANSWER (body 8,656 chars) |
| first progress | 0.79 s | 0.63 s |
| max inter-byte gap | 8.14 s | 5.55 s |
| max progress gap (§5.1 rule, report growth counted) | 8.14 s | 6.45 s |
| applier errors / desync | 0 | 0 |

**(a) G1, where the mid-stream report body lives.** The body streams before the
terminal frame, so G1 does not fire.

- A: `unified_assets/unified_assets_block` at `/assets/N/research_report/source_content`.
  Half the final body was present at 45.2 s, all of it at 78.5 s (terminal at 97.7 s).
- B: `workflow_root/workflow_block` at
  `/steps/N/items/N/payload/sources_payload/assets/N/research_report/source_content`
  (half at 46.5 s, all at 63.7 s). B also sent the whole body once in
  `answer_assets_preview/inline_entity_block` at 69.1 s.
- In both, `ask_text` carries only the cover note (1,891 and 481 chars), equal to
  the FINAL answer in the terminal `text`.

**(b) `unified_assets` shape; is the minimal list safe?** The body grows by
whole-string `replace` of `source_content`: A sent 1,021 such replaces, 7.03 MB of
values for a 13 KB body (about 13 chars of growth per replace, up to 13.4 KB per
op). This is the quadratic term from plan §5. The minimal list is safe (B completed
and its terminal `text` still carried RESEARCH_ANSWER), but it does not remove the
term: B sent the same whole-string replaces inside `workflow_block` (489 replaces,
2.05 MB of values for an 8.7 KB body).

**(c) First op per field.** Every diff field opened with `replace ""` carrying the
whole block. `pending_followups_block` arrived materialized. Later ops:
`add /chunks/N` (markdown), `replace /eta_seconds_remaining`, `/pct_complete`,
`add /goals/N` (plan), `replace|add|remove /web_results/N/...` (web results and
sources mode), `add /steps/N`, `replace /steps/N/status` (workflow). No dotted
`field` was seen.

**(d) Max inter-byte gap.** 8.14 s (A), 5.55 s (B). See the `silence_s` derivation below.

**(e) Accumulated vs terminal materialized.** Not byte-equal on any run, but equal
at every projection a verb reads:

- Equal: `workflow_block`, `unified_assets_block`, `answer_tabs_block`,
  `canvas_block`, `pending_followups_block`, `inline_entity_block`.
- `ask_text` markdown: the chunk join is equal. The terminal adds `answer` (equal to
  the join here) and sets `progress` from `IN_PROGRESS` to `DONE`.
- `ask_text_0_markdown`: the terminal coalesces the chunks into one (same join).
- `web_result_block`: same URLs in the same order, only `progress` differs.
- `plan_block`: the terminal finalizes goals (`final` true, descriptions rewritten,
  more goals than the diffs delivered).
- `sources_mode_block` (A): same URL set, reordered rows with changed metadata.

**(f) UI-wait flags.** Both flags were accepted: 200 and COMPLETED on all 17 runs.

**(g) Downgrade rule.** `display_model` equaled the requested `model_preference`
on every frame of every run (`pplx_alpha`, `turbo`, `claude48opusthinking`). Rule:
downgraded when `display_model` differs from the requested model. The negative
case was not observed (plan §8, not probed).

**(h) First-progress time.** 0.63 to 0.79 s for research.

## P2: asks and fetch in diff mode

5 asks each on `turbo` and `claude48opusthinking`, one `fetch --prompt` (turbo),
29-entry list, each read to its final frame. Logs: `p2.log` (live) and
`p2-analysis.log` (final analyzer; authoritative).

| model | text_completed → COMPLETED (s) | max | max inter-byte gap (s) | first progress (s) | wall (s) |
|---|---|---|---|---|---|
| turbo | 0.495, 0.196, 0.229, 0.408, 0.251 | 0.495 | 0.84, 1.57, 1.15, 0.77, 0.84 | 4.54, 2.93, 1.67, 1.88, 2.95 | 16.6 to 20.5 |
| claude48opusthinking | 1.100, 1.353, 1.276, 1.023, 1.011 | 1.353 | 4.08, 4.24, 4.49, 3.92, 3.96 | 0.49, 0.54, 0.90, 0.50, 0.52 | 21.8 to 27.1 |
| fetch (turbo) | 0.189 | | 1.04 | 1.88 | 6.3 |

Turbo's first progress equals its time to first byte: the server sends nothing
for 1.6 to 4.5 s.

**Where the answer lives depends on the model and the list.**

- `turbo` and `fetch --prompt` with the 29-entry list send **no `ask_text` block
  and no terminal `text` field**. The answer streams in `workflow_root/workflow_block`
  at `/steps/N/items/N/payload/text_payload/chunks/N`, and the same step also gets
  `text_payload.text` and `is_streaming: false`. At `text_completed` the chunk join
  already equals `text_payload.text`, and the terminal workflow block equals the
  accumulated one (5 of 5 turbo runs, and the fetch run).
- With `workflow_steps` and `workflow_widgets` removed from the list, turbo streams
  the answer in `ask_text` again (P3 ask, 190 chunks; n = 1).
- `claude48opusthinking` streams in `ask_text` with the 29-entry list and sends a
  terminal `text` (47 to 69 KB).

**`ask_text` parity (thinking model, 5 runs).** The accumulated chunk join at
`text_completed` equals the terminal `ask_text` chunk join on 5 of 5 runs. The
terminal `ask_text.answer`, `ask_text_0_markdown` and the FINAL answer in `text`
are equal to each other, but on 2 of 5 runs they differ from the chunk join in two
citation numbers each (`[n]` markers renumbered, same length, no other change).
The chunk join is therefore not the final answer. The terminal `answer` is.

**Terminal offset semantics.** The terminal `ask_text` block has
`chunk_starting_offset: 0` and the same chunks as the accumulated state (97 to 197
chunks). Nothing is repainted. Keys: `answer`, `chunk_starting_offset`, `chunks`,
`progress`.

**Terminal `web_results` order.** The accumulated `web_result_block` at
`text_completed`, the terminal block, and the FINAL `web_results` in `text` (thinking
model) are the same URLs in the same order on all 11 runs. Only `progress` changes.

**Max inter-byte gap for the thinking model.** 4.49 s, well under 20 s, so ask does
not need `silence_s = stall_s`.

## P3: drop and reconnect

`POST /rest/sse/perplexity_ask/reconnect/{backend_uuid}` with
`{"reconnectInitialSnapshot": true}`. Log: `p3.log` (last line is the probe summary).

**Research, dropped at 30.24 s, reconnected 2 s later.**

- First data frame 0.18 s after the POST: 137 KB, 7 materialized blocks, no diffs,
  no `text`, `status` PENDING, `reconnectable` true. Its `cursor` equals the last
  cursor before the drop.
- **Snapshot == accumulated:** equal on 7 of 7 keys.
- The stream then continued with diffs to COMPLETED (541 data frames, 68.7 s, no
  applier errors). Projections at the terminal frame matched the accumulated state
  as in P1(e). One more difference: the accumulated `sources_mode_block` had 0 rows
  and the terminal had 79.
- **Completed while away:** a reconnect 5 s after COMPLETED returned 200 and one
  frame: 607 KB, COMPLETED, `reconnectable` false, `final_sse_message` true, with
  `text`. Its materialized blocks are not byte-equal to the live terminal frame's.

**Ask (turbo), dropped at 2.04 s after one frame, reconnected 2 s later.**

- The snapshot (44 KB, PENDING) held `web_results` and `sources_mode` blocks that
  the dropped stream never received. The stream then ran to COMPLETED (29.2 s).
- **Copilot continues server-side after a drop.** Ask and fetch reconnect need not
  be `Off` in U6 for this reason.
- A reconnect after COMPLETED returned one COMPLETED frame equal to the terminal.

**The `Gone` status code is 403**, with `content-type: application/json` and the
body `{}` (2 bytes). It came back for a deleted research thread, a deleted ask
thread, the P5 B thread, and a random uuid (`smoke.log`). Sending
`x-perplexity-request-reason: reconnect-stream` did not change it. No 404 or 410
was seen.

## P5 live: terminate vs delete (`p5live.log`)

Two short research runs, concurrent.

- **A, terminate at 20 s.** The connection was closed at 21.4 s, then
  `perplexity_terminate` was sent with the static shape, **without** a
  `read_write_token`: 200 in 0.17 s, body
  `{status: "completed", _response_type: "COMPLETED_RESPONSE", thread_url_slug}`.
  Reconnect 3 s later: one 249 KB frame, COMPLETED, `reconnectable` false,
  `final_sse_message` true, `text_completed` false, no further progress. A reconnect
  60 s after the terminate returned the same frame (same materialized hash). The
  run stopped, and the stopped run reports COMPLETED with `text_completed` false.
- **B, delete only.** Closed at 30.1 s, `delete_thread` ok, reconnect 3 s later:
  403 `{}`. After a delete nothing can observe the run, so a delete does not show
  that the run stopped.
- **Verdict for U7:** send terminate (no token needed), then delete. Delete alone
  gives no evidence of a stop.

**P5 live C was not run.** It would have been the 13th ask/fetch run against a
ceiling of 12. Plan §8 gives U1 13 ask runs (P2 11, P3 1, P5 C 1). The P5 C spend
check (§3.5 first row, no Terminate after `SettledWithoutTerminal`) is therefore
still unmeasured. Some evidence exists from other runs, but it is not the probe:
P3 showed that a copilot run dropped at 2 s continues to COMPLETED on its own, and
P3's post-COMPLETED reconnects showed a finished run that stays COMPLETED and not
reconnectable.

## Derived values

| value | result | from |
|---|---|---|
| `settle_s` | `max(15, ⌈2 × 1.353⌉)` = **15 s**. Largest gap: turbo 0.495 s, thinking 1.353 s, fetch 0.189 s | P2 |
| `silence_s` | at least 3 × 8.14 s, so **25 s** for all three verbs. Gaps stayed far below `stall_s/3` (160 s ask, 80 s research), so the silence check stays on | P1 A, P2 |
| G1 | the report body streams mid-run (in `unified_assets_block`, or in `workflow_block` with the minimal list); not terminal-only | P1 |
| copilot after a drop | continues server-side and completes | P3 |
| `Gone` | 403 with body `{}`; no 404 or 410 seen | P3, P5 B, smoke |
| terminate | works without the token; stops the run; U7 uses terminate, then delete | P5 A/B |
| first content | first progress within 5 s on every run (0.49 to 4.54 s), so `FirstContentWithin(90)` is supported | P1, P2 |
| P5 C spend check | not run (budget); spend after `text_completed` not measured | none |

## Plan assumptions these probes contradict

1. **§5 "Answer: project from `ask_text` only" and "the terminal frame carries the full
   `text`".** With the 29-entry list, turbo and `fetch --prompt` have no `ask_text`
   and no terminal `text`. The answer is in `workflow_block` `text_payload`. U3's
   projection, grounded's FINAL-based ask sources, and U5b's request list are
   affected. Dropping `workflow_steps` and `workflow_widgets` brings `ask_text` back
   for turbo (n = 1).
2. **§8 P1(e), the U3 terminal-equality oracle, and the §11 canary's "no field
   `Desynced`" rule.** Terminal materialized blocks are never byte-equal to the
   accumulated state (`progress`, `answer`, coalesced `ask_text_0_markdown`,
   finalized `plan`, reordered or refilled `sources_mode`). Equality holds only at
   the projection level, so a whole-block comparison would flag every run.
3. **§5 / grounded: the chunk join is the answer.** On the thinking model the
   terminal `answer` renumbers citations (2 of 5 runs). Ask must read the terminal
   `answer`.
4. **§6 bounds ("404, 410 or another 4xx: fall back"; "on a 403, add the
   request-reason header") and §1 decision 14 / T10 ("an unrecognized 403 on a
   reconnect is `AuthError`, cookies expired").** A gone thread answers 403 `{}`, so
   decision 14 would report a deleted or expired thread as expired cookies, and the
   header retry does not help.
5. **§10 "use the minimal use-case list" as the fix for quadratic growth.** The
   minimal list moves the whole-string report replaces into `workflow_block`, which
   §5.1 classifies as `content` and §2.4.1 tracks.
6. **§8 budget.** U1's rows need 13 ask runs, not 12, so P5 C did not run.
