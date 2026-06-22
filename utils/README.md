# utils — durable tooling catalog

Promoted, human-named scripts that something *depends on* (they build artifacts, regenerate tracked content, or run often enough to deserve a stable name). The opposite of `scrap_scripts/` (throwaway, gitignored).

**Promotion rule (Rule 1):** a scrap script graduates here the moment anything other than a human-at-the-CLI or an LLM agent depends on it. On promotion: drop the `NN_` prefix, give it a descriptive name and a header comment, anchor it to the repo root, and **add a row to the table below**.

Split by language: `utils/python/`, `utils/powershell/`, etc.

## Catalog

| Script | Language | Purpose | Depended on by |
|--------|----------|---------|----------------|
| [`python/serve_site.py`](python/serve_site.py) | Python (stdlib) | Local static web server for `site/` on `0.0.0.0` (no-cache, prints LAN URL). For previewing the site locally; GitHub Pages serves the same `site/` via Actions. | Local preview workflow; `.github/workflows/deploy-pages.yml` mirrors what it serves |
| [`python/run_pipeline.py`](python/run_pipeline.py) | Python (cv2, numpy, stdlib) | Offline CLI runner for the full CV pipeline (gate → localize → identify) on sampled PNGs in `dataset/frames/`. Builds the card-art gallery once, runs `run_pipeline` on each frame, prints per-frame summary (frame_state, card identities + confidences), writes snapshot JSON, and optionally saves annotated overlay PNGs to `dataset/pipeline-out/`. No HTTP server required. | Lesson plan hands-on demos; `tests/test_cli.py` tests its pure helpers |
| [`python/export_backbone.py`](python/export_backbone.py) | Python (torch, torchvision, onnxruntime, numpy) | **DEV-ONLY.** Exports MobileNetV3-Small (ImageNet-pretrained, classifier head stripped) to ONNX at `models/mobilenetv3_small_embed.onnx`. Validates torch↔onnxruntime parity (must be < 1e-4 max abs diff). Requires torch + torchvision (not in the runtime requirements). Re-run after any model change or on a fresh clone before running the Stage 3 tests. | `src/dbcv/embed.py` loads the ONNX output at runtime; `tests/test_embed.py` skips if file absent |
