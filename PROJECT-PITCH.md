# PROJECT-PITCH — design narrative

The single long-arc design document. Why this project exists, the shape it is taking, and a running decisions table. Supersede entries through the table rather than rewriting history.

## Why this exists

Most computer-vision teaching uses toy datasets and classroom-clean examples that hide the decisions that actually matter in practice: which technique to reach for, when a cheap method beats a fancy one, how to stay within a real compute budget, and how to keep a recognizer alive when the inputs shift under you. This project refuses that artificiality. It teaches modern CV by building **one genuinely constrained system** and showing the reasoning at every fork.

The system: read *Demon Bluff* **game state from video frames only** — no audio, no save files, no API into the game — and expose that state over a **REST API**. *Demon Bluff* is a social-deduction game (think *One Night Ultimate Werewolf* / Mafia) whose board is a layout of role **cards** (villager / minion / outcast / demon). Reading the board means: find the cards, identify each one, read the surrounding text/number/state, and assemble a structured snapshot.

## The shape it is taking

- **Primary deliverable: the lesson plan.** A web-based course. Every module is anchored to a real decision made while building the pipeline, with the supporting research cited from `research/RESEARCH.md`. Techniques appear because the problem demanded them, never to round out a curriculum.
- **The pipeline is the worked example.** It is real code in `src/`, documented in `CodeDocs/`, planned in `plans/`. Its job is to be correct *and* to be explainable.
- **Knowledge is cached, not re-derived.** Game facts come from the wiki, pulled once into `knowledge-base/`. Lessons we learn (what worked, what didn't, gotchas) live alongside them.
- **Research gates technique choice.** Before any non-trivial method is adopted, the professional/academic basis is logged with a trust rating.

## Constraints that shape the design

| Constraint | Design consequence |
|------------|--------------------|
| Runtime = mid-grade gaming PC | Inference-time models must be small/fast; train on a single Titan XP or free Colab. Heavy models are dataset/debug-only. |
| Video only (no audio/state) | Everything is inferred from pixels in frames; temporal cues across frames are fair game, audio is not. |
| Samples are huge (~370 MB, ~1 h) | A frame-selection stage is mandatory; nothing downstream sees raw video, only chosen frames. |
| No baked-in resolution | Geometry is derived from the media; the pipeline is resolution-agnostic. |
| Card art may be swapped | Card recognition is built to be re-fit cheaply when the art set changes. |
| Teaching-first | Explainability and honest trade-off discussion outrank cleverness. |

## Research-backed direction (provisional, 2026-06-21)

The first research pass (`research/RESEARCH.md`, six A/B-sourced entries) plus inspection of the sample frames resolved most of the open forks. Leanings, not yet code:

- **Frame selection** → cheap classical CPU cascade, **no runtime model**: low fixed-stride decode (fps read from media) → perceptual-hash dedup → template/HSV "board vs. menu/modal" gate. Scene-detection/VLM scoring is dev-only.
- **Card localization** → **classical, layout-driven** off art-independent landmarks (the radial ring, numbered position badges, UI chrome). Immune to art swaps by construction; ~10 ms CPU; **no detector trained** unless footage proves the layout unparseable.
- **Card identification** → small embedding backbone + nearest-neighbor over a per-art reference gallery (collapses to prototypical); name-label **OCR cross-check** corroborates. The trained classifier is the approach we *reject* for production, for its retrain cost. **Updated 2026-06-22 (see decisions table):** a *frozen* ImageNet backbone collapsed the stylised characters and over-identified, so the adopted identifier **domain-fine-tunes** the same backbone (Proxy-Anchor LP-FT) and abstains on a **cosine margin**. The art-swap cost rose from "zero-gradient re-embed" to "re-fine-tune (~min on Titan Xp)" for best separation — the accuracy ↔ retrain-cost tradeoff.
- **On-card / HUD reading** → **tiny custom recognizer over the closed glyph set** (role names, 1–2-digit counts), trained on rendered crops; PaddleOCR-mobile (ONNX) as a narrow fallback. General OCR (Tesseract/EasyOCR) is dev-only.
- **Serving** → **FastAPI**, models loaded once in `lifespan` onto `app.state`, inference in plain `def` (threadpool), versioned Pydantic `GameStateSnapshot` with `resolution` from media; **ONNX Runtime on CPU**; no batching.
- **Compute budget anchor** → runtime models ≤~30 M params / ≤~100 MB (nano-detector + small-classifier class); foundation models (SAM/Grounding-DINO/large ViT) are dev-only. Train FP32 on Titan XP / Colab T4; no mixed-precision/INT8 on Pascal.

### Still open

- **Temporal logic**: how much state to recover by tracking across frames vs. re-reading each frame cold (sample shows transient modals that occlude — argues for temporal smoothing).
- **Course delivery** stack: how the web-based lesson is built and (if ever) published — gated by Rule 6.
- ~~Whether `Minion` / `Puppet` / `Twin Minion` are distinct faces~~ **clarified:** `Minion` and `Twin Minion` are functionally identical (a lone one is *usually* `Minion` — not a hard rule, due to card/mode interactions), so they collapse to one recognition class; `Puppet` is created by the `Puppeteer` card. Residual CV detail only: confirm whether Minion/Twin Minion share art. See `knowledge-base/wiki/townees/ROSTER.md`.

## Decisions table

| Date | Decision | Why | Supersedes |
|------|----------|-----|------------|
| 2026-06-21 | Mixed repo: code-doc tier and deliverable-pairing tier both live | Course needs the pipeline; pipeline needs cached game knowledge | — |
| 2026-06-21 | Sample videos in-repo under gitignored `dataset/raw-video/` | Self-contained without risking 740 MB in git history | — |
| 2026-06-21 | Lesson plan is the primary deliverable; pipeline is its worked example | Teaching-first mandate | — |
| 2026-06-21 | Localization is classical/layout-based, not a trained detector (provisional) | UI layout stable across art swaps; classical ~12 ms vs YOLO ~19 ms and needs no labels; survives art swaps by construction (`research/RESEARCH.md` localization) | — |
| 2026-06-21 | Identification is embedding-NN over a per-art reference gallery, not a trained classifier (provisional) | Art swap re-fits with new reference images and zero training; classifier would need relabel+retrain (`research/RESEARCH.md` identification) | — |
| 2026-06-21 | On-card/HUD text via a tiny closed-vocabulary recognizer; general OCR dev-only (provisional) | Cards have a known fixed glyph set; custom CRNN/CNN >95%, sub-ms, retrains by re-rendering the font (`research/RESEARCH.md` OCR) | — |
| 2026-06-21 | Runtime model budget ≤~30 M params / ≤~100 MB; foundation models dev-only | Nano-detector/small-classifier class runs real-time on a 3060 and 10–30 FPS CPU; SAM/G-DINO violate the budget (`research/RESEARCH.md` compute budget) | — |
| 2026-06-22 | Repo-local env via stdlib `venv` + pinned `requirements.txt` (not uv/conda) | Zero-install, universally reproducible baseline for a course; uv/conda taught as alternatives in the env-management module | — |
| 2026-06-22 | **Localization confirmed** (no longer provisional): HSV colour-segmentation → morphology → contour/geometry filter → relative HUD-exclusion → IoU-NMS, in `src/dbcv/localize.py` | Spike on real footage hit **8/8 & 9/9 cards exact, 0 false positives** on clean board frames (confidence ~0.80). Caveat: HSV hue ranges are art-tuned → **re-tune (not retrain)** on an art swap — still label-free, still cheap. Numbered badges are for *ordering*, not primary anchoring (blob detection aliases on clue text). | 2026-06-21 localization (provisional) |
| 2026-06-22 | REST contract realized: `POST /v1/snapshot` (upload a frame) → versioned `GameStateSnapshot` v0.1.0; FastAPI `lifespan` + plain-`def` inference | Vertical slice proved the schema + serving shape end-to-end (11→12 tests green); resolution always read from the decoded image (`research/RESEARCH.md` REST) | — |
| 2026-06-22 | Stage 0 **frame-state gate** added (`src/dbcv/frame_state.py`); schema → **v0.2.0** with a `frame_state` field; pipeline skips localization on non-board frames | A **center-vs-ring brightness ratio** cleanly separates board (~1.0–1.1) from modal (~3–6) where absolute brightness failed (modals sit on the same dark starfield); 7/7 labelled frames correct | — |
| 2026-06-22 | Ship a **classical identification baseline first** (HSV-histogram + ORB over an in-memory gallery), embedding-NN deferred | Conservative path: opencv-only, zero new deps, gallery rebuilds with zero training. Measured ~40–60% on face-up cards (correct "unknown" on face-down) — an honest lower bound that *motivates* the embedding upgrade rather than assuming it | precedes (does not supersede) the 2026-06-21 embedding-NN target |
| 2026-06-22 | Adopt **torch + ONNX**; local GPU dev on the **Titan Xp** (cu118, Pascal sm_61, GPU op verified); un-defer the embedding-NN identifier | Panda green-lit it. Clarified the weight classes: `onnxruntime`-CPU is light (the runtime serving dep); `torch` is dev-only (export/train), heavy by ~3 GB wheel-size not by the system CUDA toolkit (which pip torch doesn't use). Runtime stays ONNX-on-CPU per RESEARCH entry 5 | supersedes the 2026-06-22 "embedding-NN deferred" note |
| 2026-06-22 | Embedding-NN identifier built (frozen MobileNetV3-Small → ONNX/CPU, cosine-NN over a re-embeddable gallery) — but **a frozen ImageNet backbone does NOT beat classical** here | The backbone collapses cartoon characters into one cluster (cosine 0.65–0.94); it overidentifies rather than improving correctness. The architecture is right; **fine-tuning on card crops is the real fix** (now feasible on the Titan Xp). Default identifier flagged for Panda (recommend conservative classical until fine-tuned) | refines the 2026-06-21 embedding-NN target (frozen alone is insufficient) |
| 2026-06-22 | **Identifier = fine-tuned embedding-NN + margin gate, adopted as default.** Domain-fine-tune the *same* MobileNetV3-Small (Proxy-Anchor loss inline; LP-FT — proxy warm-up frozen, then top-4 blocks unfrozen, ~736k params; synthetic augmentation from the reference art), re-export to ONNX-on-CPU. Abstain on the **top1−top2 cosine margin** (`_EMBED_MARGIN_THRESHOLD = 0.12`), not an absolute cosine. | Fine-tuning separates the characters (inter-prototype cosine **0.85→0.41**, synthetic top-1 **79.9%→100%**, torch↔ONNX parity 5.5e-6) and fixes the over-identification — but it **compressed the absolute cosine scale**, so the old 0.60 cutoff over-identified 125/125 real cards; the margin is the honest abstention signal. Adopted real-frame behaviour: embedding 30/125 confident (abstains on 95) vs classical 44/125, agreement 27→90. Correct diagnosis was **domain shift, not "neural collapse"** (`research/RESEARCH.md`, 2026-06-22). Runtime path unchanged (ONNX/CPU). **Tradeoff:** an art swap now wants **re-fine-tuning** (~min on Titan Xp) for best separation, not just a zero-gradient re-embed. Round-2 lever: real-frame generalisation beyond the confident few. | supersedes the 2026-06-22 "frozen does NOT beat classical" recommendation; realizes the 2026-06-21 embedding-NN target |
