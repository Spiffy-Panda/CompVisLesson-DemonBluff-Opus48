# PLAN-pipeline — video → game state → REST

Forward plan for the CV pipeline (the worked example the course is built around). Direction is set by `research/RESEARCH.md` (2026-06-21) and `knowledge-base/lessons/observed-board-layout.md`. Provisional until code proves each stage.

Stages are ordered, but each is also a **lesson-plan module** (`Lesson-Plan/LESSON-PLAN.md`). Check items off and keep `CodeDocs/` in sync as code lands (Rule 3).

## Build approach (2026-06-22 run)

**Confirm the approach end-to-end first, then step back and deepen.** Before investing in any one stage:
1. **Stage -1 — Infrastructure** *(done 2026-06-22)*: repo-local `.venv` (stdlib `venv` + pinned `requirements.txt`); CV+REST stack imports clean. A course module on env management (venv vs **uv** vs **conda** vs **pip**) is owed — user-requested.
2. **Vertical slice**: a thin path through all stages — load an existing sampled frame → **classical localization** (the single riskiest assumption) → placeholder identity → assemble `GameStateSnapshot` → `POST /v1/snapshot`. Proves the architecture + REST contract + schema on real frames, leaning classical/conservative (no heavy deps).
3. **Step back**: assess what the slice reveals (esp. whether classical localization is viable on our footage), then deepen each stage with its research-backed method + full lesson module + `CodeDocs/` sync, committing per teachable unit.

## Stage 0 — Frame selection (no runtime model)
- [ ] Promote `03_sample_frames.py` patterns into a real selector in `src/`: low fixed-stride decode (fps from media) → perceptual-hash dedup → board/menu/modal gate.
- [ ] Board-gate via `matchTemplate`/HSV on stable UI anchors; detect occluding modals + streamer overlays.
- [ ] Dev-only: PySceneDetect segmentation of the long samples for dataset building.

## Stage 1 — Localization (classical, layout-based)
- [ ] Detect art-independent landmarks: the radial card ring, numbered position badges, panel/UI chrome.
- [ ] Derive card slots relative to landmarks, scaled by measured resolution; handle **variable card count**.
- [ ] Output resolution-relative bboxes (`CodeDocs/io/outputs.md` schema).

## Stage 2 — Identification (embedding-NN gallery + OCR cross-check)
- [ ] Build the reference gallery from `knowledge-base/card-art/` (per art set, versioned).
- [ ] Small frozen embedding backbone → NN over gallery; prototypical averaging of references.
- [ ] Name-label OCR cross-check; reconcile visual + text identity with confidences.
- [ ] Retrain story: art swap = re-embed references, **zero training**. Verify.

## Stage 3 — On-card / HUD reading (closed-vocab recognizer)
- [ ] Tiny custom recognizer for role names + 1–2-digit counts (rendered-crop training set).
- [ ] PaddleOCR-mobile (ONNX) fallback for free-form text.
- [ ] Preprocess: locate-region → upscale → grayscale → contrast-normalize (no global binarization).

## Stage 4 — State assembly
- [ ] Merge per-card reads into the `GameStateSnapshot`; temporal smoothing across frames (handle modal occlusion).

## Stage 5 — REST service
- [ ] FastAPI: models loaded once in `lifespan` onto `app.state`; inference in plain `def`.
- [ ] Versioned Pydantic `GameStateSnapshot` (`schema_version`, `resolution` from media); `POST /v1/snapshot`.
- [ ] Serve small models via ONNX Runtime (CPU); no batching.

## Cross-cutting
- [x] Repo-local `.venv` + pinned `requirements.txt` (2026-06-22). Standard interpreter for all scripts/agents: `.venv/Scripts/python.exe`.
- [ ] Author the env-management lesson module: **venv vs uv vs conda vs pip** — when/why each, why this project chose venv+pip (user-requested).
- [ ] Keep every runtime model ≤~30 M params / ≤~100 MB; foundation models dev-only.
- [ ] Every stage that adopts a technique cites its `research/RESEARCH.md` entry and gets a lesson module.
- [ ] Resolve open questions: temporal logic depth; course delivery stack. (`Minion`/`Twin Minion` → one recognition class; `Puppet` is `Puppeteer`-created — see [ROSTER](../knowledge-base/wiki/townees/ROSTER.md).)
