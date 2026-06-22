# DEV-LOG

Append-only decision log. **Newest entry on top.** Absolute dates. Git commits record *what changed*; this records *why we chose it, what we tried first, and what would surprise the next person*. Write an entry before every commit (Rule 5).

### Entry template

```
## YYYY-MM-DD — <short title>

**Context:** what prompted this.
**Options considered:** A / B / C.
**Choice:** what we picked.
**Why:** the deciding reason(s).
**Notes / risks:** anything the next person should know.
```

---

## 2026-06-21 — git/GitHub init, local server, Pages deploy, landing site

**Context:** Put the repo under version control and stand up a web surface. Motivation for the local server: the Claude Code desktop preview window can't render complex plain local HTML — it needs a real HTTP origin.

**What landed:**
- **`utils/python/serve_site.py`** — stdlib static server, binds `0.0.0.0`, no-cache headers, prints the LAN URL for the preview window. **Local only.**
- **`.github/workflows/deploy-pages.yml`** — GitHub Pages via Actions (`configure-pages` with `enablement: true` → `upload-pages-artifact` path `site/` → `deploy-pages`). Same `site/` content as the local server; Pages uses its own static stack. Project Pages URL is a subdirectory: `https://spiffy-panda.github.io/CompVisLesson-DemonBluff-Opus48/`.
- **`site/`** — `index.html` landing page listing pages; `pages/notes.html` (scratch/misc, first entry = rough plan outline); `assets/style.css`. **All links relative** so the site is byte-identical at `/` (local) and `/<repo>/` (Pages) — the one subtlety of project-Pages subdirectory hosting.

**Options considered / why:**
- *Local server tech:* stdlib `http.server` over FastAPI — the site is static; no need to couple the dev-preview server to the (future) lesson REST API. Kept them separate.
- *Pages mechanism:* Actions artifact deploy over "serve /docs from branch" — the user asked for an Action, and the artifact flow lets the source dir be `site/` and adds no Jekyll surprises.
- *Repo visibility:* **public** — free GitHub Pages requires it, and the user explicitly asked to host on Pages. Ran the Rule 6/7 name gate first (below).

**Public-surface gate (Rule 6/7):** grep for the dead name / real last name across the repo matched **only `CLAUDE.local.md`**, which is gitignored — confirmed excluded from the commit. Published `site/` is our own prose; no third-party bulk, no identity leakage. Commit identity is the handle `Spiffy-Panda` / handle email, not a real name.

**Notes / risks:** First push triggers the workflow; `enablement: true` should turn Pages on automatically — verified the run after pushing. Pages can take a minute to go live on first deploy. The harvester/probe/sampler remain in gitignored `scrap_scripts/` (not committed); the harvester is still a `utils/` promotion candidate if we want it tracked.

## 2026-06-21 — Wiki harvest + first research pass + frame tooling

**Context:** Same session as bootstrap, continued into the chosen scope (scaffold + harvest + first research). 

**What landed:**
- **Harvest** via `scrap_scripts/python/01_wiki_harvest.py` (MediaWiki API, stdlib urllib, fetch-once, polite UA + delay): 44 role pages (25/9/7/3), 26 mechanics pages, 67 card-art files. Raw cache + art gitignored; `wiki/harvest-manifest.json` and a transcribed `wiki/townees/ROSTER.md` tracked.
- **First research pass**: six A/B-sourced `research/RESEARCH.md` entries (frame selection, localization, identification, OCR, REST serving, compute budget), produced by six parallel background subagents, each briefed with Rule 1 verbatim + the project constraints.
- **Frame tooling**: `02_probe_video_meta.py` (ffprobe; both samples are 1920×1080 h264 60 fps, ~48 m / ~62 m — read from media, *not* baked in) and `03_sample_frames.py` (fast-seek uniform sampler → `dataset/frames/`). Inspected a spread of frames *without opening the raw video*.

**Key findings driving design:**
- Localization and identification are **separable**; the UI **layout is stable across art swaps** (radial card ring + numbered position badges), so localization is a **geometry problem, not a learned detector** — classical wins on speed, labels, and art-swap robustness.
- Identification should be **embedding-NN over a per-art gallery** (re-fit with reference images, zero training) with **name-label OCR** as a cross-check; a trained classifier is explicitly rejected for its retrain cost.
- On-card/HUD text is a **closed glyph set** → tiny custom recognizer beats general OCR; PaddleOCR-mobile (ONNX) is the narrow fallback.
- Runtime budget anchor: **≤~30 M params / ≤~100 MB**; SAM/Grounding-DINO/large ViT are **dev-only**. Train FP32 on Titan XP/Colab; no mixed-precision/INT8 on Pascal.
- **Real-world wrinkle:** sample footage has **streamer overlays** ("Benji") and transient **modals that occlude the board** → a board/menu/modal state-gate and overlay tolerance are mandatory. Logged in `knowledge-base/lessons/observed-board-layout.md`.

**Options considered / why:** Background subagents over inline research — keeps six independent web investigations off the main context and runs them concurrently with the harvest. MediaWiki API over HTML scraping — cleaner page/category/image enumeration and a natural "fetch once" key.

**Sync:** `PROJECT-PITCH.md` decisions table + direction updated; `KNOWLEDGE-BASE.md` inventory updated; `CodeDocs/io/*` already describe the intended contracts (no code yet, so no `sources/` overviews). No desync found.

**Notes / risks:** Harvester writes the tracked manifest, so it's a **promotion candidate** (`utils/`) once we re-run it for an art swap — left in `scrap_scripts/` for now. Card art may include base/mechanic pages (`Minion`, `Puppet`, `Twin Minion`) that aren't distinct faces — verify when building the gallery. RESEARCH directions are **provisional** (no code yet); supersede via the decisions table as the pipeline proves them.

## 2026-06-21 — Bootstrap

**Context:** New repo for a web-based course on modern computer vision, worked through one system: extracting *Demon Bluff* game state from video frames and serving it via REST. Two ~370 MB sample videos were the only contents of the root.

**Options considered:**
- *Repo shape:* code-only vs. prose-only vs. **mixed**. The deliverable is a course (prose) but it is built around a real CV pipeline (code).
- *Sample-video placement:* in-repo gitignored dataset folder vs. out-of-tree referenced by path.
- *First-session scope:* scaffold only vs. scaffold + wiki/townee harvest vs. scaffold + harvest + first research pass.

**Choice:** Mixed repo (both the code-doc tier and the deliverable-pairing tier are live). Videos moved to `dataset/raw-video/` and gitignored. First-session scope = scaffold + harvest + first research pass (user-selected).

**Why:** The course can't be written without the pipeline existing, and the pipeline can't be reasoned about without the cached game knowledge — so both tiers earn their keep. In-repo gitignored keeps the project self-contained while guaranteeing the 740 MB never lands in history.

**Notes / risks:**
- Added four non-skill folders beyond the standard scaffold: `research/` (RESEARCH.md log of every non-Demon-Bluff thing researched), `Lesson-Plan/` (the course), `knowledge-base/` (cached wiki + learned lessons), `dataset/` (large files).
- Card art and verbatim wiki text are third-party reference inputs — gitignored where bulk/verbatim (raw caches, `card-art/`), kept where small and transformative (transcribed `.md`). Public-surface gate (Rule 6) applies before any publish.
- Git not initialized (skill says wait until asked). `.gitignore` written ahead of time so the dataset, scrap, local file, and bulk third-party caches are covered the moment git appears.
- Constraints encoded in CLAUDE.md: runtime on a mid-grade gaming PC (heavy models dev-only; runtime models must train on Titan XP / Colab), never open the sample videos directly, never bake in a resolution, research-before-deciding, card recognition must be cheap to retrain.
