# models/ — ONNX runtime artifacts

This directory holds binary model artifacts consumed by the runtime pipeline (`src/dbcv/`).
All `.onnx` files are **gitignored** (`*.onnx` in `.gitignore`) — they are regenerable
build artifacts, not source. They must be rebuilt before running the server or tests that
exercise the embedding identifier.

## Artifacts

| File | Size | Generator | How to regenerate |
|------|------|-----------|-------------------|
| `mobilenetv3_small_embed.onnx` | ~3.6 MB | `utils/python/export_backbone.py` | `.venv\Scripts\python.exe utils\python\export_backbone.py` |

## mobilenetv3_small_embed.onnx

**What it is:** MobileNetV3-Small (ImageNet-pretrained) with the classifier head removed.
Outputs a 576-dimensional pooled feature vector per image.

**Inputs / outputs:**

| Port | Name | Shape | dtype | Notes |
|------|------|-------|-------|-------|
| Input | `image` | [N, 3, 224, 224] | float32 | NCHW, ImageNet-normalized, dynamic batch N |
| Output | `embedding` | [N, 576] | float32 | Raw embedding (L2-normalized at runtime in `OnnxEmbedder.embed`) |

**Expected preprocessing** (applied in `src/dbcv/embed.py` `OnnxEmbedder.preprocess`):

1. BGR to RGB conversion (OpenCV loads as BGR).
2. Resize to 224 x 224 (bilinear).
3. Normalize with ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225].
4. Layout NCHW, dtype float32, shape [1, 3, 224, 224].

**Why frozen / ImageNet pretrained?**
An art swap requires only re-embedding the new reference images — zero gradient steps.
See research/RESEARCH.md entry 3 (identification, 2026-06-21) for the full rationale.

**ONNX opset:** 13 (stable, well-supported by onnxruntime >= 1.10).

**Parity:** torch vs onnxruntime max absolute difference = 1.73e-06 (well within 1e-4).

**Runtime provider:** CPU (`CPUExecutionProvider`). torch is NOT required at runtime.
