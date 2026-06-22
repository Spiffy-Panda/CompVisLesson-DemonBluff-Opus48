# knowledge-base/ — game knowledge + learned lessons (human-facing)

Two kinds of knowledge live here, both feeding the pipeline and the course:

1. **Cached Demon Bluff facts** — transcribed/transformed from the wiki (<https://demonbluff.wiki.gg/>), pulled **once** and cached so we never re-fetch. Roles, mechanics, and per-townee data (villager / minion / outcast / demon).
2. **Our own learned lessons** — what worked, what didn't, and the gotchas we hit building the CV pipeline. The institutional memory the course is distilled from.

The agent-facing spec is **[KNOWLEDGE-BASE.md](KNOWLEDGE-BASE.md)**.

## Layout

| Path | What | Tracked? |
|------|------|----------|
| `wiki/*.md` | Transcribed/transformed wiki pages (roles, mechanics) | yes — small, transformative |
| `wiki/_raw_cache/` | Verbatim fetched pages (HTML/JSON), the "fetch once" cache | **no** (gitignored, Rule 6 — bulk verbatim) |
| `card-art/` | Downloaded townee card images for recognition templates | **no** (gitignored, Rule 6 — third-party art) |
| `lessons/*.md` | Our learned lessons, incl. the reusable CV-project playbook | yes |

## Rules that apply here

- **Fetch once.** Anything pulled from the wiki is cached under `wiki/_raw_cache/` so we hit the site a single time per page. Transcriptions are derived from the cache, not from re-fetching.
- **Public-surface gate (Rule 6).** The cached wiki text and card art are *reference inputs for the pipeline, not redistributable content.* They never go into a public bundle; only small transformative excerpts may surface in the course.
- **Demon Bluff facts only.** Outside knowledge (CV techniques, libraries) belongs in `research/`, not here.
