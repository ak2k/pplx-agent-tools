# pplx-agent-tools

Shell CLI that gives agents (Claude Code, Codex, anything that shells out) Perplexity's search, answers and Research mode through your Pro subscription's web session cookies. Perplexity's public Sonar API needs its own key and bills by usage; this needs neither. Parallel to [`kagi-search`](https://github.com/Mic92/mics-skills/tree/main/skills/kagi-search) in shape and purpose.

Pre-1.0: flags and JSON shapes can change in a minor release.

## Verbs

- `pplx search <query>...`: ranked web hits with snippets (`-j` adds a longer summary per hit). Several queries return one merged list.
- `pplx ask <query>`: one synthesized answer with cited sources (Pro Search). `--model` picks the model.
- `pplx research <query>`: a long cited report from Perplexity's Research mode. Takes minutes and spends a research-quota unit.
- `pplx resume --last` (or `<uuid>`): gets the report of a research run whose connection dropped or whose process died. The run keeps going on the server.
- `pplx fetch <url>`: fetches the page locally and returns its cleaned text; needs no cookies. With `--prompt`, Perplexity's model fetches the URL and answers the prompt in one call.
- `pplx snippets <query> <url>...`: query-relevant excerpts from several URLs, ranked locally by keyword (BM25) and semantic (`fastembed`) search. Needs SQLite 3.38 or newer.
- `pplx quota`: remaining rate limit per mode.
- `pplx models`: available models and modes.
- `pplx auth {check,refresh,import}`: cookie management.
- `pplx skill-path`: path of the bundled agent skill, [SKILL.md](SKILL.md).

`ask`, `research` and `fetch --prompt` run incognito, so they never appear in your Perplexity library, and delete their thread afterward. A research run whose connection drops or whose process is killed keeps its thread (about 24 hours) so `pplx resume` can get the report.

Verbs print text, or JSON with `-j`; diagnostics go to stderr. Exit codes are stable so agents can choose a retry: 1 usage or other error, 2 auth, 3 rate limit, 4 network, 5 anti-bot, 6 partial result (stdout still usable). SKILL.md is the full reference for flags, JSON fields and what to do on each exit code.

## Install

Not on PyPI. Install a [release](https://github.com/ak2k/pplx-agent-tools/releases) from GitHub; the commands below use v0.10.0.

```bash
uv tool install git+https://github.com/ak2k/pplx-agent-tools@v0.10.0
# or with Nix
nix profile install github:ak2k/pplx-agent-tools/v0.10.0

# Claude Code: install the skill
mkdir -p ~/.claude/skills/pplx-agent-tools
ln -sf "$(pplx skill-path)" ~/.claude/skills/pplx-agent-tools/SKILL.md
# (Nix: link ~/.nix-profile/share/skills/pplx-agent-tools/SKILL.md instead,
#  which survives upgrades)

# Import cookies from a browser where you are logged in to perplexity.ai
pplx auth import --browser firefox  # also: brave, chrome, edge, safari, librewolf, zen, ...
pplx auth check

# Optional: each refresh resets the cookie's 30-day expiry
pplx auth refresh                   # run from cron / launchd
```

Cookies are stored in `~/.config/perplexity/<profile>/cookies.json`. Pass `--profile` or set `$PPLX_PROFILE` to use more than one account.

`auth import` reads the browser profile whose cookies changed last, which may be another account's. `--browser-profile` picks one by profile directory name (e.g. `Default`, `Profile 1`, `xxxx.default-release`) or path; for Safari, give the path of a `Cookies.binarycookies` file.
Arc is not supported: export its perplexity.ai cookies as JSON with the Cookie-Editor extension to the cookie file above, mode 600.

## Caveats

- **Unofficial and not affiliated with Perplexity AI.** It uses internal web endpoints, not the Sonar API, and they can change without notice.
- **For your own subscription only.** Don't pool cookies across users.
- **Session cookies expire.** Re-import from the browser when `pplx auth check` fails.
- **`pplx search` returns web results only:** no academic, image, video or shopping mode, and no country or domain filter.

## License

MIT. See [LICENSE](LICENSE).
