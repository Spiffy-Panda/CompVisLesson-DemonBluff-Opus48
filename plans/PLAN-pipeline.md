# PLAN-pipeline — video → game state → REST

Forward plan for the CV pipeline (the worked example the course is built around). Direction is set by `research/RESEARCH.md` (2026-06-21) and `knowledge-base/lessons/observed-board-layout.md`. Provisional until code proves each stage.

Stages are ordered, but each is also a **lesson-plan module** (`Lesson-Plan/LESSON-PLAN.md`). Check items off and keep `CodeDocs/` in sync as code lands (Rule 3).

## Build approach (2026-06-22 run)

**Confirm the approach end-to-end first, then step back and deepen.** Before investing in any one stage:
1. **Stage -1 — Infrastructure** *(done 2026-06-22)*: repo-local `.venv` (stdlib `venv` + pinned `requirements.txt`); CV+REST stack imports clean. A course module on env management (venv vs **uv** vs **conda** vs **pip**) is owed — user-requested.
2. **Vertical slice**: a thin path through all stages — load an existing sampled frame → **classical localization** (the single riskiest assumption) → placeholder identity → assemble `GameStateSnapshot` → `POST /v1/snapshot`. Proves the architecture + REST contract + schema on real frames, leaning classical/conservative (no heavy deps).
3. **Step back**: assess what the slice reveals (esp. whether classical localization is viable on our footage), then deepen each stage with its research-backed method + full lesson module + `CodeDocs/` sync, committing per teachable unit.

## Open decisions & next steps (handoff, 2026-06-22)

**Where it stands:** Stages 0, 1, 2 + the REST service + a CLI runner are shipped (classical/CPU, plus a torch-trained-but-*frozen* embedding backbone served via ONNX-CPU). **81 tests pass (~22 s).** `torch 2.7.1+cu118` + `onnxruntime` installed; GPU verified on the Titan Xp. Authoritative history is in `DEV-LOG.md` (newest first); decisions in `PROJECT-PITCH.md`; lesson plan is **8/10 modules** authored.

**Two decisions await the user (do NOT resolve autonomously):**
1. **Default identifier.** Embedding-NN is *currently wired as the default*, but the frozen-ImageNet backbone is **not more accurate** than the classical matcher (it over-identifies — see Stage 2). Recommendation: flip the default to the conservative **classical** identifier until the backbone is fine-tuned. Both live on `app.state` (`identifier`, `classical_identifier`) — a one-line change. It's a product call.
2. **Fine-tuning round.** The real fix for the collapsed embedding space: a light metric-learning fine-tune on (synthetically-augmented, possibly real-mined) card crops → re-export to ONNX → re-run the head-to-head. Feasible locally on the Titan Xp. Awaiting a go + approach (head-only vs full backbone; synthetic-only vs also mining real board crops).

**Independent work ready to pick up (no decision needed):** Stage 0 stride-decode + perceptual-hash dedup; Stage 3 on-card OCR (closed-vocab recognizer); Stage 4 temporal smoothing; lesson Modules 05 (update once fine-tuned), 06 (OCR), 07 (assembly).

## Stage 0 — Frame selection (no runtime model)
- [ ] Promote `03_sample_frames.py` patterns into a real selector in `src/`: low fixed-stride decode (fps from media) → perceptual-hash dedup → board/menu/modal gate. *(partial 2026-06-22: the gate is done in `src/dbcv/frame_state.py`; stride-decode + pHash dedup still owed)*
- [x] Board-gate detecting occluding modals + menus *(2026-06-22 — `src/dbcv/frame_state.py`, **center-vs-ring brightness ratio**; more robust than matchTemplate against the dark-starfield modal background. Streamer overlays tolerated by the localizer's relative HUD-exclusion zones.)*
- [ ] Dev-only: PySceneDetect segmentation of the long samples for dataset building.

## Stage 1 — Localization (classical, layout-based) — **shipped**
> **Validated + integrated 2026-06-22 (`src/dbcv/localize.py`, `classical_localize`):** 8/8 & 9/9 cards exact on clean board frames via HSV colour-segmentation → morphology → contour filter (area/aspect) → relative HUD-exclusion → IoU-NMS. **Badge blob-detection failed** (clue-text aliases as badges) — badges are for *ordering*, not anchoring. Optional deepening left: skew-robustness, ring-geometry sanity check, art-swap hue re-tuning.
- [x] Detect art-independent landmarks (colour/contour over the card ring; badges demoted to ordering).
- [x] Derive card slots relative to landmarks; handle **variable card count** (8/9/10 seen).
- [x] Output resolution-relative bboxes (`bbox_rel`, `CodeDocs/io/outputs.md`).

## Stage 2 — Identification (embedding-NN gallery + OCR cross-check) — **both identifiers shipped**
> **2026-06-22.** (1) **Classical** (`gallery.py` + `identify.py`): in-memory gallery (43 townees / 67 refs incl. skins), HSV-histogram + ORB; ~40–60% on face-up cards, honest "unknown" on face-down. (2) **Embedding-NN** (`embed.py` + `identify.py`): frozen MobileNetV3-Small → **ONNX → onnxruntime-CPU** (torch↔onnx parity 1.7e-6; serving stays torch-free) → cosine-NN over a re-embeddable 576-d **prototypical** gallery, loaded once in `lifespan`. **Honest finding:** the *frozen ImageNet* backbone collapses the 43 cartoon characters into one cluster (inter-prototype cosine 0.65–0.94) → it names ~100% of slots but **is not more accurate than classical** (it over-identifies; classical's conservative unknowns give better precision). Architecture is correct (re-fit-cheap, ONNX-CPU); the fix is **light domain fine-tuning** — see *Open decisions* above.
- [x] Build the reference gallery from `knowledge-base/card-art/` *(in-memory; rebuild = the versioning story; used by both identifiers)*.
- [x] Small frozen embedding backbone → NN over gallery; prototypical mean per townee *(built — frozen ImageNet insufficient → fine-tune next)*.
- [ ] Name-label OCR cross-check; reconcile visual + text identity with confidences *(blocked on Stage 3 OCR)*.
- [x] Retrain story: art swap = **re-embed** references (`build_embedding_gallery`) / re-fit the classical gallery, **zero training**.

## Stage 3 — On-card / HUD reading (closed-vocab recognizer)
- [ ] Tiny custom recognizer for role names + 1–2-digit counts (rendered-crop training set).
- [ ] PaddleOCR-mobile (ONNX) fallback for free-form text.
- [ ] Preprocess: locate-region → upscale → grayscale → contrast-normalize (no global binarization).

## Stage 4 — State assembly
- [ ] Merge per-card reads into the `GameStateSnapshot`; temporal smoothing across frames (handle modal occlusion).

## Stage 5 — REST service — **shipped**
- [x] FastAPI: models loaded once in `lifespan` onto `app.state`; inference in plain `def`. *(2026-06-22 — `src/dbcv/api.py`; the ONNX embedder + both identifiers + the gallery load once in lifespan)*
- [x] Versioned Pydantic `GameStateSnapshot` (`schema_version` 0.2.0, `resolution` from media, `frame_state`); `POST /v1/snapshot`. *(2026-06-22 — accepts an uploaded frame; resolution read from it)*
- [x] Serve small models via ONNX Runtime (CPU); no batching. *(2026-06-22 — MobileNetV3 embedding backbone served on CPU via `src/dbcv/embed.py`; ONNX is gitignored + regenerable via `utils/python/export_backbone.py`, documented in `models/README.md`)*

## Cross-cutting
- [x] Repo-local `.venv` + pinned `requirements.txt` (2026-06-22). Standard interpreter for all scripts/agents: `.venv/Scripts/python.exe`.
- [x] Author the env-management lesson module: **venv vs uv vs conda vs pip** *(2026-06-22 — `Lesson-Plan/modules/00_python-environments.md`; cited by new `research/RESEARCH.md` env entry)*.
- [ ] Keep every runtime model ≤~30 M params / ≤~100 MB; foundation models dev-only.
- [ ] Every stage that adopts a technique cites its `research/RESEARCH.md` entry and gets a lesson module.
- [ ] Resolve open questions: temporal logic depth; course delivery stack. (`Minion`/`Twin Minion` → one recognition class; `Puppet` is `Puppeteer`-created — see [ROSTER](../knowledge-base/wiki/townees/ROSTER.md).)
