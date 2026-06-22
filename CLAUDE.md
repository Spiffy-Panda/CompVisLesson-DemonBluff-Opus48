# CLAUDE.md — agent entry point

This is the canonical entry point for any LLM agent working in this repo. Read it first, then follow the entry-point chain in Rule 4.

## What this repo is

A **web-based course** that teaches modern computer-vision practice by building one real system end to end: extracting live **game state** from *Demon Bluff* using **video frames only** (no audio, no memory injection, no game files) and serving that state over a **REST API**.

Two things ship from here, and they are weighted:

1. **The lesson plan** (primary deliverable) — a course that explains *why* each technique was chosen, grounded in professional/academic recommendations, not in a wish to "cover the syllabus."
2. **The working pipeline** (the worked example the course is built around) — the CV code, models, and REST service that actually read the game.

If the two ever conflict, the lesson plan wins: this is a teaching artifact first.

### Demon Bluff in one paragraph

A social-deduction game in the family of *One Night Ultimate Werewolf* / Mafia. Players are **townees**, divided into **villager / minion / outcast / demon** roles, each shown as a **card**. Mechanics are described by the wiki (<https://demonbluff.wiki.gg/>); we cache it locally rather than re-fetch. The first engineering stage is the computer vision, driven by the sample footage in `dataset/`.

## Hard project constraints (these gate design decisions)

- **Runtime target: a mid-grade gaming PC.** The heaviest models are off-limits *at runtime*. They are fine for dataset creation, labeling, debugging, and offline development. Any custom model that runs at inference time must be trainable on a single Titan XP or a free Google Colab tier.
- **Never open the sample videos directly into context.** They are ~370 MB each, ~1 h long. Write a frame-extraction / frame-selection script (`scrap_scripts/` → promote to `utils/`) and inspect only the selected frames.
- **Never bake in a resolution.** Both samples happen to share one resolution; read it from the media at runtime. No hard-coded width/height/crop pixels anywhere.
- **Research before deciding.** Before committing to any non-trivial CV technique, search for professional and academic guidance and log it in `research/RESEARCH.md` (source, authority/trust, why you looked, abstract of findings). Do not add a technique merely to make the course look complete.
- **Tooling follows the field, not a default.** Don't force Python/PowerShell because they're familiar — use what professionals use for the specific kind of CV at hand (this will usually be Python + the standard CV/DL stack, but justify it).
- **Card art can change** to a limited alternate set. Anything that recognizes a card must be **cheap to retrain** — assume the art will be swapped and the recognizer re-fit.

---

## Rule set (verbatim — order matters)

### Rule 1 — No inline interpreter calls (but shell one-liners are fine)

Hard rule for **interpreters**: no `python -c`, `python3 -c`, `py -c`, `node -e`, etc.

Trigger: if `import` (or `require`, `using`, `#include`) appears in a command line you are about to send to a shell, **stop**. Create `scrap_scripts/<lang>/<NN>_<slug>.<ext>` and run that file.

Shell is different. Short one-liners in bash / PowerShell / cmd are allowed (`git status`, a single `grep`, `ls | head`). Escalate to a file the moment the one-liner grows loops, variables, conditionals, or more than a couple of pipes — then it goes in `scrap_scripts/<lang>/`.

Every script — scrap or util — must anchor to the repo root so it runs from any CWD: `Path(__file__).resolve().parents[N]` (Python) or the language equivalent. Never assume the invocation directory.

Promotion: the moment a scrap file is depended on by anything other than (a) a human at the CLI or (b) an LLM agent — i.e. it produces a build artifact, regenerates tracked content, or gets run often enough to justify a stable name — move it to `utils/<lang>/<descriptive_name>.<ext>`, drop the `NN_` prefix, give it a human name and a header comment, **and add a row to `utils/README.md`**.

Pass this rule, **verbatim and with a stern warning**, into every subagent prompt. Sonnet-tier models have ignored it before.

### Rule 2 — Production hierarchy

| Tier | Where | Pattern | Purpose |
|------|-------|---------|---------|
| Plan | `plans/` | `PLAN-<slug>.md`, indexed by `PLAN.md` | Forward-looking. Chat dump. |
| Design | `PROJECT-PITCH.md` | single narrative + decisions table | Why it is built this way. |
| Code-doc | `CodeDocs/` | mirrors `src/` with `.md`, indexed by `CODE-DESIGN.md` | Signatures + line numbers. Read **before** the matching code. |
| Deliverable | `<folder>/` | `README.md` + `<FOLDER-NAME>.md` pair | LLM-authored deliverables (`research/`, `Lesson-Plan/`, `knowledge-base/`). |
| Src | `src/` | code | Reality. |

`PLAN.md` is an *index only* — one line per slug.

### Rule 3 — Sync discipline

Touching one tier means updating the matching files in the others.

- Change `src/` → update the matching `CodeDocs/sources/...` overview (signatures, line numbers, status, who-uses-it), tick off matching items in the `PLAN-<slug>.md`. If you changed a generated/consumed file format, update `CodeDocs/io/inputs.md` or `outputs.md`.
- Change `PROJECT-PITCH.md` → if a decision conflicts with existing `src/`, flag it in `DEV-LOG.md`.
- Change `PLAN-<slug>.md` → update root `PLAN.md` if the change adds or completes a top-level item.

**Read before navigating code.** Before opening any source file, read `CODE-DESIGN.md` and `CodeDocs/00_PROJECT.md` — they tell you whether an overview already answers your question without opening the source.

**On noticed desync, full audit.** If you read an overview and it's wrong, don't just patch that one line — sweep the whole `CodeDocs/sources/` and `CodeDocs/io/` against current code and re-sync everything that drifted. Flag it to the user, say docs need a resync, and re-raise at the start of every subsequent phase until handled. Force a sync check before any `git push`.

### Rule 4 — Entry-point convention

- LLM: `CLAUDE.md` → `PLAN.md` / `PROJECT-PITCH.md` → per-slug → `CODE-DESIGN.md` → `CodeDocs/` → `src/`.
- Human: `README.md` → same downstream chain.

### Rule 5 — DEV-LOG vs git commits

Git commits = *what changed*. `DEV-LOG.md` = *why we chose this, what we tried first, what would surprise the next person*. Append-only, newest on top, absolute dates. **Write an entry before every commit** — minimum one line; a short paragraph for any non-obvious change.

### Rule 6 — Public-surface gate

Anything copied into a published bundle (a `build-public/` folder, a Pages deploy, a released course, a public knowledge base) becomes publicly redistributed content. Before pushing changes that touch a public surface, run a content review:

- **Third-party material** (game data, wiki prose, **card art**, decompiled assets): only small, transformative excerpts. No verbatim bulk. The cached wiki text and the downloaded card art in this repo are **reference inputs for the pipeline, not redistributable content** — they stay out of any public bundle. For anything creative that must ship, walk the four fair-use factors and note which you weighed.
- **Identity / private data:** no dead names, real last names, private absolute paths, secrets, or local-only material in anything public. See Rule 7.

If a deliverable drifts toward "comprehensive reproduction of the source," pull it back before pushing.

### Rule 7 — Identity & naming rules live in CLAUDE.local.md, never committed

Identity rules that gate public output live in the **gitignored** `CLAUDE.local.md`, not here. This file (committed) must never contain the user's dead name, real last name, or private absolute paths.

### Rule 8 — Chat-local enumeration prefixes

When you enumerate items in a list **in chat** (not in committed files), give the list a short mnemonic prefix and prepend `_` to mark it chat-local (ephemeral — a conversational handle, not a tracked identifier). Reference items as `_<PREFIX>.<n>`. The prefix abbreviates the list's purpose so references across separate lists stay unambiguous.

Example: a "Done this session" list → `_D.1`, `_D.2`; a "needs your **C**all" list → `_C.1`, `_C.2`. Then `_D.1` and `_C.1` are distinct — no bare "1." that could mean either list.

The leading `_` denotes chat-local scope; permanent/committed enumerations (the pipeline stages, plan items, the rules above) keep their own durable numbering, no `_`.

---

## Where to look first

- `PLAN.md` — index of active/completed plan slugs.
- `PROJECT-PITCH.md` — the long-arc design narrative + decisions table.
- `CODE-DESIGN.md` → `CodeDocs/00_PROJECT.md` — code map; read before opening `src/`.
- `research/RESEARCH.md` — every non-Demon-Bluff thing we researched, with trust ratings and abstracts. **Consult before choosing a technique.**
- `Lesson-Plan/LESSON-PLAN.md` — the course spec (the primary deliverable).
- `knowledge-base/KNOWLEDGE-BASE.md` — cached wiki facts + our own learned lessons.
- `CLAUDE.local.md` — machine paths, tooling locations, identity rules (gitignored).
- `README.md` — human entry point.

## Briefing subagents

Every subagent prompt must carry: (1) Rule 1 verbatim with a stern warning; (2) a pointer to this file; (3) the exact tier(s) it may touch and the tiers it must keep in sync; (4) an instruction to flag — not silently fix — any desync; (5) every load-bearing convention its task intersects (research-before-deciding, the public-surface gate, the deliverable-pairing rule, the no-resolution-baking and never-open-the-videos constraints).
