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
| `models/` | ONNX/weights artifacts (gitignored, regenerable) | [README](models/README.md) |
| `tests/` | pytest suite for `src/dbcv/` (110 tests) | — |
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

## Fresh-machine bootstrap

Everything heavy is gitignored and regenerable; a clone needs these steps to become a working dev box:

1. **Python 3.12** + venv: `python -m venv .venv`, then `.venv/Scripts/pip install -r requirements.txt` (torch pins are `+cu118` CUDA builds; on a CPU-only box install the matching CPU wheels instead).
2. **ffmpeg/ffprobe on PATH** (frame extraction + media probing fall back to OpenCV, but ffprobe gives authoritative fps).
3. **Card-art gallery** (gitignored): `.venv/Scripts/python.exe utils/python/wiki_harvest.py` — fetch-once harvest of the wiki roster pages + card art into `knowledge-base/`. Without it the gallery build, the REST server, and part of the test suite have nothing to load.
4. **Served model** (gitignored): either copy `models/*.onnx`/`*.pt` from the old machine, or regenerate with `.venv/Scripts/python.exe utils/python/finetune_embedding.py` (needs torch + a GPU; ~minutes on a Titan Xp-class card; seeded, so results reproduce). Missing model = REST falls back to the classical identifier; `tests/test_embed.py` skips.
5. **Footage** (gitignored): place capture files in `dataset/raw-video/`. The two v0.389-era samples stay on the old machine unless copied; new current-version capture supersedes them for wave 2 (see `DEV-LOG.md` 2026-07-29 on footage-version drift).
6. **Machine-local file:** create a fresh gitignored `CLAUDE.local.md` (Rule 7 — machine paths, tooling locations, identity rules; never committed).
7. Verify: `.venv/Scripts/python.exe -m pytest tests/ -q` → expect all green (or embed-tests skipped if step 4 was deferred).

## Web surface

- **Live (GitHub Pages):** <https://spiffy-panda.github.io/CompVisLesson-DemonBluff-Opus48/> — auto-deployed from `site/` by [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml) on push to `main`.
- **Local preview:** `python utils/python/serve_site.py` → serves the same `site/` on `0.0.0.0:8000` (use the printed network URL in the Claude Code preview window). Links are relative, so the site is identical locally and on Pages.

## Status

Stages 0–2 of the pipeline plus the REST service are shipped in `src/dbcv/` (frame selection, classical localization, identification via a fine-tuned embedding backbone with a classical fallback) — **110 tests green** (~27 s, re-verified 2026-07-28). Lesson plan is 8/10 modules authored. Remaining: Stage 3 on-card OCR, Stage 4 temporal assembly, lesson modules 06–07. Details in [plans/PLAN-pipeline.md](plans/PLAN-pipeline.md); index at [PLAN.md](PLAN.md).
