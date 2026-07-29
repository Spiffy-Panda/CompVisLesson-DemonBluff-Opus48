# PLAN-pipeline — video → game state → REST

Forward plan for the CV pipeline (the worked example the course is built around). Direction is set by `research/RESEARCH.md` (2026-06-21) and `knowledge-base/lessons/observed-board-layout.md`. Provisional until code proves each stage.

Stages are ordered, but each is also a **lesson-plan module** (`Lesson-Plan/LESSON-PLAN.md`). Check items off and keep `CodeDocs/` in sync as code lands (Rule 3).

## Build approach (2026-06-22 run)

**Confirm the approach end-to-end first, then step back and deepen.** Before investing in any one stage:
1. **Stage -1 — Infrastructure** *(done 2026-06-22)*: repo-local `.venv` (stdlib `venv` + pinned `requirements.txt`); CV+REST stack imports clean. A course module on env management (venv vs **uv** vs **conda** vs **pip**) is owed — user-requested.
2. **Vertical slice**: a thin path through all stages — load an existing sampled frame → **classical localization** (the single riskiest assumption) → placeholder identity → assemble `GameStateSnapshot` → `POST /v1/snapshot`. Proves the architecture + REST contract + schema on real frames, leaning classical/conservative (no heavy deps).
3. **Step back**: assess what the slice reveals (esp. whether classical localization is viable on our footage), then deepen each stage with its research-backed method + full lesson module + `CodeDocs/` sync, committing per teachable unit.

## Open decisions & next steps (handoff, 2026-06-22)

**Where it stands:** Stages 0, 1, 2 + the REST service + a CLI runner (`utils/python/run_pipeline.py` — the end-to-end offline runner, which wires the *classical* identifier; distinct from the Stage-0 frame-selection batch CLI, which was deliberately *not* promoted) are shipped (classical/CPU, plus a **domain-fine-tuned** embedding backbone served via ONNX-CPU). **110 tests pass (~25 s).** `torch 2.7.1+cu118` + `onnxruntime` installed; GPU verified on the Titan Xp. The Stage-2 embedding fine-tune (round 1) is **done and adopted as the default identifier** (Proxy-Anchor LP-FT fixed the collapse: inter-prototype cosine 0.85 → 0.41; abstains on a top1−top2 margin). Authoritative history is in `DEV-LOG.md` (newest first); decisions in `PROJECT-PITCH.md`; lesson plan is **8/10 modules** authored.

**Both prior decisions are now resolved (2026-06-22):**
1. **Default identifier — RESOLVED → fine-tuned embedding-NN + margin gate.** Round 1 fixed the collapse, so the served model is the fine-tuned backbone and `classify_crop_embedding` abstains on the **top1−top2 cosine margin** (`_EMBED_MARGIN_THRESHOLD = 0.12`, provisional) instead of an absolute cosine. Real-frame behaviour: 24% confident IDs, honest "unknown" on the rest; classical↔embedding agreement 27 → 90. See `DEV-LOG.md`.
2. **Fine-tuning round — RESOLVED → round 1 done, stopped there.** Proxy-Anchor LP-FT with synthetic augmentation (`utils/python/finetune_embedding.py`). Leak-proof metric: inter-prototype cosine mean 0.850 → 0.409. Round 2 (real *mined + labeled* board crops) is the documented next lever **if** real-frame accuracy beyond the confident few proves insufficient — it needs labeled data to even measure, so it folds into dataset-building, not a quick retrain.

**Independent work ready to pick up (no decision needed):** Stage 3 on-card OCR (closed-vocab recognizer); Stage 4 temporal smoothing; lesson Modules 06 (OCR), 07 (assembly). **Round-2 dataset-building** (mine + label real board crops) is the natural follow-on to deepen Stage 2; dev-only PySceneDetect segmentation (Stage 0) remains optional.

## Stage 0 — Frame selection (no runtime model)
- [x] Promote `03_sample_frames.py` patterns into a real selector in `src/`: low fixed-stride decode (fps from media) → perceptual-hash dedup → board/menu/modal gate. *(done 2026-06-22 — `src/dbcv/frame_select.py`: stride from the media's real fps → **dHash** dedup (Hamming ≤ 8 vs last kept) → reuse `classify_frame_state`; dev/batch only, off the REST path; +29 tests)*
- [x] Board-gate detecting occluding modals + menus *(2026-06-22 — `src/dbcv/frame_state.py`, **center-vs-ring brightness ratio**; more robust than matchTemplate against the dark-starfield modal background. Streamer overlays tolerated by the localizer's relative HUD-exclusion zones.)*
- [ ] Dev-only: PySceneDetect segmentation of the long samples for dataset building.

## Stage 1 — Localization (classical, layout-based) — **shipped**
> **Validated + integrated 2026-06-22 (`src/dbcv/localize.py`, `classical_localize`):** 8/8 & 9/9 cards exact on clean board frames via HSV colour-segmentation → morphology → contour filter (area/aspect) → relative HUD-exclusion → IoU-NMS. **Badge blob-detection failed** (clue-text aliases as badges) — badges are for *ordering*, not anchoring. Optional deepening left: skew-robustness, ring-geometry sanity check, art-swap hue re-tuning.
- [x] Detect art-independent landmarks (colour/contour over the card ring; badges demoted to ordering).
- [x] Derive card slots relative to landmarks; handle **variable card count** (8/9/10 seen).
- [x] Output resolution-relative bboxes (`bbox_rel`, `CodeDocs/io/outputs.md`).

## Stage 2 — Identification — **both visual identifiers (classical + embedding) shipped; OCR cross-check awaits Stage 3**
> **2026-06-22 (updated).** (1) **Classical** (`gallery.py` + `identify.py`): in-memory gallery (43 townees / 67 refs incl. skins), HSV-histogram + ORB; ~35% identify rate, honest "unknown" otherwise. (2) **Embedding-NN — now domain-fine-tuned + default** (`embed.py` + `identify.py`): MobileNetV3-Small → **ONNX → onnxruntime-CPU** (torch↔onnx parity ~5e-6; serving stays torch-free) → cosine-NN over a re-embeddable 576-d **prototypical** gallery, loaded once in `lifespan`. The original *frozen ImageNet* backbone collapsed the 43 characters (inter-prototype cosine 0.65–0.94) and over-identified; **round-1 Proxy-Anchor LP-FT fine-tuning fixed it** (cosine mean 0.85 → 0.41), and abstention switched to the **top1−top2 margin**. Served model = the fine-tuned backbone (`utils/python/finetune_embedding.py`); the frozen baseline (`export_backbone.py` → `_frozen.onnx`) is kept only for the head-to-head.
- [x] Build the reference gallery from `knowledge-base/card-art/` *(in-memory; rebuild = the versioning story; used by both identifiers)*.
- [x] Small embedding backbone → NN over gallery; prototypical mean per townee *(frozen ImageNet insufficient → **domain-fine-tuned, round 1, 2026-06-22**; adopted as default with a margin gate)*.
- [ ] Name-label OCR cross-check; reconcile visual + text identity with confidences *(blocked on Stage 3 OCR)*.
- [x] Retrain story: art swap = re-fit classical (zero training) + re-embed references; **note (2026-06-22):** the now-fine-tuned backbone is best **re-fine-tuned** on a new art set (`finetune_embedding.py`, ~min on Titan Xp) for full separation — the accuracy↔retrain-cost tradeoff (see DEV-LOG).

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
- [ ] Resolve open question: temporal logic depth. *(Course delivery stack settled in practice — static `site/` + GitHub Pages auto-deploy; recorded in `PROJECT-PITCH.md` 2026-07-28. `Minion`/`Twin Minion` → one recognition class; `Puppet` is `Puppeteer`-created — see [ROSTER](../knowledge-base/wiki/townees/ROSTER.md).)*

### Standing invariants (never "done" — apply to every stage)
- Keep every runtime model ≤~30 M params / ≤~100 MB; foundation models dev-only.
- Every stage that adopts a technique cites its `research/RESEARCH.md` entry and gets a lesson module.
