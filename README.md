# DemonBluff CV Lesson

A web-based course that teaches **modern computer-vision practice** by building one real system end to end: reading live *Demon Bluff* **game state from video frames alone** (no audio, no game files) and serving it over a **REST API**.

The course is the product. The pipeline is the worked example the course is built around. When the two pull in different directions, the course wins.

## Start here

- **Agents:** read [CLAUDE.md](CLAUDE.md) first — it carries the full rule set and the constraints that gate every design decision.
- **Humans:** this file, then the design narrative in [PROJECT-PITCH.md](PROJECT-PITCH.md), then the plan index in [PLAN.md](PLAN.md), then the running [DEV-LOG.md](DEV-LOG.md).

## Where things live

| Path | What | Read me |
|------|------|---------|
| [`Lesson-Plan/`](Lesson-Plan/README.md) | The course itself — primary deliverable | [README](Lesson-Plan/README.md) |
| [`research/`](research/README.md) | Every non-Demon-Bluff thing we researched, rated for trust | [README](research/README.md) |
| [`knowledge-base/`](knowledge-base/README.md) | Cached wiki facts + our own learned lessons | [README](knowledge-base/README.md) |
| [`dataset/`](dataset/README.md) | Sample footage and derived frames (large files, gitignored) | [README](dataset/README.md) |
| `site/` | The web surface — landing page + notes. Served locally and on GitHub Pages | [index.html](site/index.html) |
| `src/` | The CV pipeline + REST service (code) | [CODE-DESIGN.md](CODE-DESIGN.md) |
| `CodeDocs/` | Per-file code overviews — read before the code | [00_PROJECT.md](CodeDocs/00_PROJECT.md) |
| `plans/` | Per-slug plan files | [PLAN.md](PLAN.md) |
| `utils/` | Durable promoted tooling | [README](utils/README.md) |
| `scrap_scripts/` | Throwaway exploration scripts (gitignored) | [README](scrap_scripts/README.md) |

## Working conventions (plain English)

- **Research before deciding.** No non-trivial CV technique goes in without a professional/academic source logged in `research/RESEARCH.md` first. We don't add techniques just to look thorough.
- **The samples are huge — never open them directly.** Frame-selection scripts pull the few frames worth looking at.
- **No hard-coded resolution, ever.** Read it from the media.
- **Runtime runs on a mid-grade gaming PC.** Heavy models are for dataset-building and debugging only; anything at inference time must train on a Titan XP or free Colab.
- **Card art may be swapped** for an alternate set, so card recognition must be cheap to retrain.
- **Two scratch tiers:** throwaway code in `scrap_scripts/<lang>/`, promoted tooling in `utils/<lang>/` (cataloged). No inline `python -c`-style interpreter calls — see Rule 1 in [CLAUDE.md](CLAUDE.md).

## Web surface

- **Live (GitHub Pages):** <https://spiffy-panda.github.io/CompVisLesson-DemonBluff-Opus48/> — auto-deployed from `site/` by [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml) on push to `main`.
- **Local preview:** `python utils/python/serve_site.py` → serves the same `site/` on `0.0.0.0:8000` (use the printed network URL in the Claude Code preview window). Links are relative, so the site is identical locally and on Pages.

## Status

Bootstrapped + harvested + first research pass done; web surface live. The CV pipeline is not built yet — see [PLAN.md](PLAN.md) and [plans/PLAN-pipeline.md](plans/PLAN-pipeline.md).
