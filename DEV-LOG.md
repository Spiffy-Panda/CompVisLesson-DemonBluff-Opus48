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

## 2026-06-22 — Lesson modules 01 (framing) + 09 (staying alive) → 8/10 authored

Authored the two "bookend" modules: `Lesson-Plan/modules/01_framing.md` (the course's front door — why this project teaches CV through one genuinely-constrained real system; the constraints-as-characters from `PROJECT-PITCH.md`; a preview of the pipeline arc; cites the compute-budget research entry) and `09_staying-alive.md` (the art-swap-cheap thesis as shipped — localization HSV re-tune, gallery rebuild with zero training, font re-render for the future OCR; why the trained classifier was rejected for production; honest that drift/health monitoring is design-not-code). **Lesson plan now 8/10 authored** (00–05, 08, 09); only **06 (on-card OCR)** and **07 (state assembly/temporal)** remain, and both correctly await their unbuilt pipeline stages. Citations all trace to existing RESEARCH entries / PROJECT-PITCH / shipped-code docstrings; no new research or desync.

## 2026-06-22 — Lesson modules 03 (resolution geometry) + 08 (REST) + Module 04 reconciliation

Authored `Lesson-Plan/modules/03_resolution-agnostic-geometry.md` (the "never bake a resolution" principle as enforced in shipped code — `Resolution` from `image.shape`, `bbox_rel` fractions, thresholds relative to `min(W,H)`, relative→pixel only at the edge via `crop_relative`; the self-correcting resolution test; the 1280×720-vs-1920×1080 sampler-downscale story) and `08_rest-serving.md` (lifespan load-once onto `app.state`, plain-`def`-in-threadpool vs event-loop-blocking `async def`, versioned `GameStateSnapshot` + `schema_version`, the 0.1.0→0.2.0 bump as versioning-in-action; cites RESEARCH entry 5). **Inventory now 6/10 authored** (00, 02, 03, 04, 05, 08; planned: 01, 06, 07, 09 — 06/07 await the unbuilt OCR/temporal stages).

**Reconciled a desync I introduced:** Module 04 was authored in parallel with the Stage 0 gate, so it claimed the state gate "is weak / no reliable gate" and that the `observed-board-layout.md` badge caveat was "not yet corrected." Both became false within the same round (the gate shipped; I added the caveat). Updated Module 04's failure-modes section to reflect that the gate is now handled upstream (`frame_state.py`) and the KB caveat is in place. Lesson: parallel authoring + implementation in one round can self-contradict — reconcile at the round's commit.

## 2026-06-22 — Lesson modules 02 (frame selection) + 05 (identification)

Authored two more course modules, each grounded in shipped+tested code and matching the established skeleton/voice: `Lesson-Plan/modules/02_frame-selection.md` (the frame-selection design space + the shipped board/modal/menu gate, including the honest "absolute brightness failed → the center-vs-ring *ratio* is the invariant" debugging story; forward-references the still-owed stride/dedup) and `05_card-identification.md` (the four identification families on the retrain-cost axis, the shipped classical baseline, the honest ~40–60% face-up result, the `compareHist(zeros,*)==1.0` bug, and the deliberately-deferred embedding-NN upgrade). Cites existing RESEARCH entries 1 and 3 (no new research needed). `LESSON-PLAN.md` inventory now **4/10 authored** (00, 02, 04, 05). No desync.

## 2026-06-22 — End-to-end CLI runner (capstone): the pipeline reads the board

Promoted a durable runner to `utils/python/run_pipeline.py` (Rule 1 promotion: descriptive name, repo-root-anchored, **row added to `utils/README.md`**). It runs the full pipeline offline (no HTTP) on sampled PNGs: builds the gallery once, then per frame does gate → localize (board only) → identify, prints a readable summary (frame_state + per-card identity@confidence), writes snapshot JSON, and (with `--overlay`) saves annotated PNGs to the gitignored `dataset/pipeline-out/`. Flags: `--frames`, `--out`, `--overlay`, `--limit`, `--no-gallery`.

**Verified end-to-end (I viewed an overlay myself):** board frames render boxes + identity labels + a `board` banner; modal frames (`Sample1_000`, `Sample2_000`) gate to `modal` with **zero** boxes; the partial-modal `Sample1_006` is correctly treated as `board` (peripheral cards still found). Real identities surface on face-up cards — `Wretch@0.80`, `Baa@0.69/0.70`, `Confessor`, `Hunter`, `Druid`, `Scout`, `Fortune_Teller`, `Doppelganger` — with face-down cards correctly `unknown`, consistent with the honest ~40–60% face-up baseline. This is the reproducible hands-on artifact the lesson modules point at.

**Notes:** 14 light CLI unit tests (helpers only, 0.31s — deliberately no gallery build, so the slow-suite issue isn't worsened). Fixed a Windows cp1252 crash (a `→` in the argparse help string; module docstring keeps the arrows since argparse doesn't print it). `dataset/pipeline-out/` added to `.gitignore`.

## 2026-06-22 — Stage 2: classical identification baseline (embedding-NN deferred)

**Context:** Cards were being localized but not named (`identify` was a stub). Built the classical identification baseline per the conservative directive — opencv+numpy only, **no torch/onnxruntime, no model download** — explicitly deferring the research-preferred embedding-NN to a later, heavier round.

**What landed:** `src/dbcv/gallery.py` — `build_gallery()` walks `knowledge-base/card-art/<class>/<role>/*.png` and builds an **in-memory** gallery (no persisted artifact → card art stays a gitignored input, Rule 6): **43 townees / 67 references** (24 have skin variants, all loaded). Directory = label (`class`→role_class, dir→identity); `Twin_Minion`→`Minion` aliased. `src/dbcv/identify.py` rewritten: `classify_crop` matches a card crop by **2-D HSV (hue×sat) histogram correlation** (primary — value excluded to survive state-tinting) with an **ORB-feature tiebreaker** when top-2 are within 0.05; confidence = clamped Pearson correlation, threshold 0.40 → else "unknown". Gallery built once in the API `lifespan` onto `app.state` (load-once pattern). **46/46 pytest green.**

**Honest result (the pedagogical point):** ~**40–60% on face-up cards**, **100% correct "unknown" on face-down cards** (a uniform card back matches no character art → low confidence, which is the right answer). A `Scout` was predicted twice in one frame (impossible in-game) — a real false match. **Verdict: classical histograms are a useful, honest *lower bound* but insufficient for production → the embedding-NN upgrade is warranted.** This is exactly the worked example for the (future) identification lesson module.

**Teaching bug found:** `cv2.compareHist(zeros, anything)` returns **1.0** (Pearson 0/0 → clamped), so a degenerate all-black crop "perfectly matched" everything. Fixed with a zero-sum guard. Good "always test degenerate inputs" lesson material.

**Known issue (tracked separately, not a blocker):** the suite now runs ~73s (was ~1.4s) because the gallery rebuilds ~7× across test files + per-frame matching in API tests. A session-scoped shared-gallery fixture (and injecting the gallery into the app for tests) would fix it. Deferred so as not to perturb the just-validated identifier under this run's budget.

**Deferred (logged per conservative directive):** embedding-NN identification — a small frozen backbone (e.g. MobileNetV3) exported to ONNX + nearest-neighbour over the gallery, served on CPU via onnxruntime. Needs `onnxruntime` (+ a one-time model export/download) — the first genuinely "heavier" dependency. Left for when Panda is back to approve the dep, or a later round.

## 2026-06-22 — Deepening round 1: foundation lesson modules + Stage 0 state gate

Two parallel sub-agents on disjoint tiers (lessons vs `src/`), both light/classical (conservative path).

**Lessons (primary deliverable, first modules authored):** `Lesson-Plan/modules/00_python-environments.md` (the user-requested **venv vs virtualenv vs pip vs conda vs uv** module — teaches the interpreter/venv/installer split, compares all five honestly, justifies this project's venv+pip choice, no fabricated benchmarks) and `Lesson-Plan/modules/04_card-localization.md` (classical vs trained-detector vs foundation-model, worked example = our real spike results, with the honest caveats — art-tuned HSV, badge-blob failure, weak state gate). `LESSON-PLAN.md` inventory populated (Modules 00–09; 2 authored / 8 planned). New `research/RESEARCH.md` entry on env tooling (official docs, trust A). Module files use `NN_slug` per the LESSON-PLAN skeleton. I read Module 00 end-to-end — accurate and well-pitched.

**Stage 0 — frame-state gate (the spike's known gap):** `src/dbcv/frame_state.py` — `classify_frame_state(image) -> "board"|"modal"|"menu"`. The winning discriminator is a **center-vs-ring brightness ratio**: a modal's bright panel sits on the *same dark starfield* as the board, so absolute center-brightness failed (the spike's 0/3), but the panel is ~3–6× brighter than the dark ring around it, where a board is ~1.0–1.1. Threshold 2.0 sits in a clean 3× gap. **7/7 labelled board+modal frames correct**; the partial-modal (`Sample1_006`, peripheral cards visible) is deliberately called "board" so the localizer can still read it. Schema → **v0.2.0** with a `frame_state` field; the pipeline now runs the gate first and **skips localization on non-board frames** (returns `cards=[]`). **24/24 pytest green** (verified by me). CodeDocs + `CODE-DESIGN.md`/`00_PROJECT.md` synced; the stale localize.py "gate held back" TODO removed.

**Also:** resolved the twice-flagged desync — `knowledge-base/lessons/observed-board-layout.md` now carries the badge implementation caveat (blob detection aliases on clue text → badges are for ordering, not anchoring) and notes the implemented state gate.

**Notes / risks:** the gate threshold (2.0) is validated on 3 modal types; a future modal with a very small bright panel could approach it (`Sample1_000` already the closest at 3.1). Frame *selection* proper (stride decode + perceptual-hash dedup) was intentionally skipped this round to keep the gate focused — still owed in Stage 0.

## 2026-06-22 — Classical localizer promoted into `src/`; approach confirmed

Integrated the spike algorithm into `src/dbcv/localize.py` as `classical_localize` (5 stages: relative HUD-exclusion → HSV segmentation → morphology → contour/geometry filter → IoU-NMS), now the pipeline/API default; `stub_localize` retained as the teaching "before" baseline. Made the API test deterministic on a known board frame (`Sample1_003`, validated 8/8) and added a direct localizer unit test. **12/12 pytest green**; I verified the suite and eyeballed the overlay myself — boxes sit cleanly on all 8 ring cards, HUD + "Benji" overlay excluded. Recorded localization + the REST contract + the env choice as **confirmed** rows in the `PROJECT-PITCH.md` decisions table (superseding the provisional localization entry). This closes the "confirm the approach works first" milestone; next is the step-back into deepening (lesson modules for the validated foundation, then Stage 0's board/modal gate and Stage 2 identification).

## 2026-06-22 — Vertical slice lands + classical localization validated

**Context:** First code in `src/`. Two parallel sub-agents: one built the end-to-end REST skeleton (stub localizer/identifier), one ran a classical-localization spike on the real sampled frames to test the project's central architectural bet.

**What landed (skeleton):** `src/dbcv/` package — `schema.py` (Pydantic `GameStateSnapshot` v0.1.0, all coords relative, resolution read from media), `localize.py`/`identify.py` (pluggable interfaces + stubs), `pipeline.py`, `assemble.py`, `config.py`, `api.py` (FastAPI, `lifespan` "load once onto app.state" pattern, `POST /v1/snapshot` accepting an uploaded frame, **plain `def`** so CPU-bound inference runs in the threadpool per RESEARCH entry 5). `tests/` + repo-root `conftest.py` (puts `src/` on path). **11/11 pytest green.** CodeDocs synced: `sources/dbcv/*.md` overviews + `io/inputs.md`/`io/outputs.md` reconciled (schema bumped 0.0.0→0.1.0, `confidence` bounded [0,1], `role_class` a validated Literal).

**What the spike found (the important part):** **Classical, layout-based localization is viable** — confidence ~0.80. On clean board frames it hit **8/8 and 9/9 cards exact, zero false positives**, using HSV colour segmentation of card regions → morphology → external contours filtered by area/aspect → relative HUD-exclusion zones → IoU-NMS. All thresholds expressed relative to `min(W,H)` (no baked resolution). This validates the decision to NOT train a detector.

**What did NOT work / open risks:**
- **Board-vs-modal state gate is weak** (0/3 on modal frames). The game's modals are *dark*-backgrounded with bright text/art, so a center-brightness threshold misreads them as "board." Needs a better signal (pentagram-absence or modal-header detection). This is Stage 0's real problem, now concrete.
- **Numbered position badges are NOT usable as primary anchors via blob detection** — card clue/ability text panels produce indistinguishable bright blobs (badge blob detection over-fired 30–60/frame and was demoted). Badges may still work for *ordering* detected boxes via targeted `#`-glyph OCR. **Flag:** `knowledge-base/lessons/observed-board-layout.md` calls badges "ideal landmarks" — true for geometry, but the implementation note that *raw blob detection on them fails* should be added when we deepen Stage 1. (Not edited yet — flagged per Rule 3.)
- HSV hue ranges are tuned to this art set; an art swap = re-tune ranges (cheap, no training) — consistent with the "cheap to re-fit" thesis, but worth teaching as the honest caveat of the classical path.

**Choice:** Commit the skeleton as a rewindable checkpoint; next integrate the spike's `localize()` into `src/dbcv/localize.py` (replacing the stub) and make the API test deterministic on a known board frame.

**Notes / risks:** Sampled frames are **1280×720** (the frame sampler downscaled the 1920×1080 source) — code reads resolution from the image, so this is transparent. Spike artifacts (script + overlays) live in gitignored `scrap_scripts/`; the real `localize()` is promoted into `src/`.

## 2026-06-22 — Repo-local venv + start of the pipeline build (overseer run)

**Context:** Long unattended "overseer" session: spawn sub-agents to build the pipeline (Stages 0–5) toward the functional + teaching goals, committing as we go, no push. User set three guardrails up front: (1) **repo-local env**, and *teach* uv vs conda vs pip as a course module; (2) **confirm the approach works end-to-end first, then step back and deepen**; (3) at forks, **prefer the conservative/lighter/classical path** and log it.

**Choice — environment:** `python -m venv .venv` at repo root (gitignored) + pinned `requirements.txt`. Installed numpy 2.5, pillow 12.2, **opencv-python-headless 4.13**, fastapi 0.138, uvicorn 0.49, pydantic 2.13, httpx 0.28, pytest 9.1. Smoke-imported all via a scrap script (Rule 1 — never `python -c`). Standard interpreter for every script/agent from here: `.venv/Scripts/python.exe`.

**Why venv+pip over uv/conda:** zero-install, universally reproducible baseline — every Python ships `venv`; a learner can follow without first installing a tool. `opencv-python-headless` (not `opencv-python`) because the pipeline is server/batch, no GUI. uv (speed) and conda (binary deps) become the *alternatives* in the owed env-management lesson module, not a runtime requirement. Did **not** install onnxruntime/torch/imagehash yet — deferred until a stage's research justifies them (conservative path).

**Build plan:** thin **vertical slice** next — load an existing sampled frame → classical localization (the riskiest assumption) → placeholder identity → `GameStateSnapshot` → `POST /v1/snapshot` — to validate the architecture, schema, and REST contract on real frames before deepening any single stage. Then reassess. Recorded in `PLAN-pipeline.md` ("Build approach").

**Notes / risks:** A fresh venv is isolated, so the global numpy/fastapi/etc. do **not** carry in — `requirements.txt` is the source of truth. The env-management lesson module is owed (tracked in `PLAN-pipeline.md` cross-cutting). Localization viability on our footage is still unproven; the slice exists to find out early.

## 2026-06-22 — Claude launcher + townee clarification

- Added `.claude/launch.json` — the Claude Code desktop launcher (`local-server` → `python utils/python/serve_site.py --port 8000`). Verified via the preview MCP that it drives the running server and the site renders. Gitignored `.claude/settings.local.json` defensively (public repo). Backed out a `.vscode/launch.json` written from a first-pass misread of the request.
- Recorded game knowledge on the three ambiguous minion entries (**source: project player; not yet cross-checked against the wiki text**): `Minion` and `Twin Minion` are functionally identical (a lone one is *usually* `Minion` — not a hard rule, due to card/mode interactions) → collapse to **one recognition class** for CV; `Puppet` is **created by the `Puppeteer`** card (distinctions live on the cached Puppeteer page). Synced into `ROSTER.md`, `PROJECT-PITCH` (still-open → clarified), `PLAN-pipeline`, and the public `site/pages/notes.html` (verified rendered).

## 2026-06-21 — serve_site.py: flush startup banner

One-line follow-up: flush the local server's startup banner so the URL prints immediately even when stdout is captured/redirected (the Claude Code preview-window case). Also confirmed the first Pages run's `startup_failure` was Pages-not-yet-enabled; after enabling the Actions source the `workflow_dispatch` run deployed cleanly and the site returns 200 at the project subdirectory.

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
