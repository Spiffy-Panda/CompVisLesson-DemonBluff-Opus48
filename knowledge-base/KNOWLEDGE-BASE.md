# KNOWLEDGE-BASE.md — game knowledge + lessons (agent-facing)

Agent-facing spec for `knowledge-base/`. Tells you how the cached game knowledge and our learned lessons are organized, so you can find a fact or record one without opening every file.

## Two halves

### 1. Cached Demon Bluff facts (`wiki/`)

- **`wiki/_raw_cache/`** (gitignored) — the verbatim "fetch once" cache. One file per wiki page, named after the page. Hitting the wiki for a page that already exists here is a bug.
- **`wiki/*.md`** (tracked) — transcribed/transformed pages. Keep these *transformative*: structured summaries, extracted data tables, our own phrasing — not verbatim copies (Rule 6). One `.md` per meaningful page or per townee group.
- **`wiki/townees/`** (created during harvest) — currently holds only `ROSTER.md`, the transformative roster of all 44 roles by class (villager / minion / outcast / demon). Per-townee mechanics transcriptions are deferred (write on demand — see the inventory note below); raw per-role data stays in `_raw_cache/`.

### 2. Learned lessons (`lessons/`)

- **`lessons/*.md`** (tracked) — one lesson per file or a small set of themed files. What worked, what didn't, the surprising gotcha. These are the raw material the lesson plan is distilled from.
- **`lessons/cv-project-playbook.md`** — the **reusable cross-project playbook**: how to set up and run a CV project like this one, so the next one doesn't start from a blank page. Append to it whenever a reusable lesson emerges.

## Conventions

- **Fetch once, cache always.** Wiki content is pulled a single time into `_raw_cache/`; everything else derives from the cache.
- **Transcriptions are transformative**, not verbatim — keep tables, summaries, and extracted fields; drop prose bulk.
- **Card art lives in `card-art/`** (gitignored), referenced by the townee `.md`, never embedded.
- **Versioned for art swaps.** Note the art-set version with any card-art reference so a swap is traceable.

## Inventory

| Area | Status |
|------|--------|
| `wiki/_raw_cache/<class>/` | **44 role pages cached** (25 villager, 9 minion, 7 outcast, 3 demon) via `scrap_scripts/python/01_wiki_harvest.py` (2026-06-21) |
| `wiki/_raw_cache/knowledge/` | 27 mechanics pages cached (Gameplay 6, Relics 8, Unused Roles 13) — `Delusion` added on the 2026-07-29 re-harvest |
| `wiki/townees/ROSTER.md` | **transcribed** — transformative roster of all 44 roles by class. **Re-verified against the wiki 2026-07-29: unchanged, 44/44.** Carries the *Footage-version drift* section — the dataset is ≤ v0.389 while the wiki is v0.762a; read it before building any recognizer or name vocabulary |
| `wiki/harvest-manifest.json` | tracked manifest: class → role → cache path + art files (67 art files total) |
| `card-art/<class>/<role>/` | **downloaded** (gitignored) — 67 art files; reference gallery, versioned per art set |
| `lessons/cv-project-playbook.md` | seeded during bootstrap; appended over time |
| `../models/` (repo root) | **pointer** — the fine-tuned embedding backbone (ONNX/`.pt`, gitignored, regenerable) that consumes `card-art/` to build the in-memory embedding gallery at runtime; documented in `models/README.md` |
| `lessons/observed-board-layout.md` | **written** — ground-truth board layout from inspecting sample frames |

Per-role mechanics transcriptions remain deferred (write on demand, keep transformative — Rule 6).
