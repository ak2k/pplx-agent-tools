# pplx-agent-tools

Shell-CLI agent toolkit for Perplexity, backed by your Pro subscription's web session cookies. Parallel to [`kagi-search`](https://github.com/Mic92/mics-skills/tree/main/skills/kagi-search) in shape and purpose.

**Status**: planning. No working code yet — this is a scaffold. See the [design plan](https://github.com/ak2k/nix-config/blob/main/docs/plans/pplx-agent-tools.md) for surface, architecture, phasing, and risk register.

## Why

Perplexity exposes its agent verbs (raw search, URL fetch + LLM extraction, batched snippet extraction) internally but not via its public Sonar API. This project gives agents — Claude Code, Codex, anything that shells out — the same primitives, backed by an existing Pro subscription's web session cookies. No per-token billing, no separate API key.

The novel value over existing Perplexity wrappers: `pplx-fetch --prompt` (URL → LLM-extracted answer in one call) and `pplx-snippets` (query + N URLs → relevant excerpts per URL). Neither has a public client today.

## Planned verbs

- `pplx-search` — web / academic / images / videos / shopping search; multi-query and domain filters on web
- `pplx-fetch` — URL → cleaned page content, or LLM-extracted answer with `--prompt`
- `pplx-snippets` — query + N URLs → relevance-ranked excerpts per URL
- `pplx-auth` — cookie management: import from browser, validate, keepalive

Each verb: text output by default, `-j` for JSON. Stable exit codes for agent retry semantics. Single-shot CLIs; no daemon. (Daemon-pooled architecture is documented as a deferred Phase 2 — only if v1 usage shows real cross-invocation coordination needs.)

## Caveats

- **Unofficial**. Uses Perplexity's internal web endpoints, not the official Sonar API. Endpoints can change without notice.
- **Not affiliated with Perplexity AI.**
- **For your own subscription only**. Cookie pooling across users is an explicit anti-pattern.
- **Session cookies expire**. Reimport from the browser when `pplx-auth check` fails.

## License

MIT — see [LICENSE](LICENSE).
