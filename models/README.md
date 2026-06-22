# models/ — ONNX runtime artifacts

This directory holds binary model artifacts consumed by the runtime pipeline (`src/dbcv/`).
All `.onnx` and `.pt` files are **gitignored** — they are regenerable build artifacts, not
source. They must be rebuilt before running the server or tests that exercise the embedding
identifier (Stage 2).

## Artifacts

| File | Size | Generator | How to regenerate |
|------|------|-----------|-------------------|
| `mobilenetv3_small_embed.onnx` | ~3.7 MB | `utils/python/finetune_embedding.py` | `.venv\Scripts\python.exe utils\python\finetune_embedding.py` |
| `mobilenetv3_small_embed.pt` | ~3.8 MB | `utils/python/finetune_embedding.py` | (written alongside the `.onnx` by the same run) |
| `mobilenetv3_small_embed_frozen.onnx` | ~3.7 MB | `utils/python/export_backbone.py` | `.venv\Scripts\python.exe utils\python\export_backbone.py` |

The **served** model is `mobilenetv3_small_embed.onnx` — the **fine-tuned** backbone.
`mobilenetv3_small_embed_frozen.onnx` is the **frozen-ImageNet baseline**, kept only for the
head-to-head comparison; it is not loaded at runtime.

## mobilenetv3_small_embed.onnx — the served (fine-tuned) model

**What it is:** MobileNetV3-Small with the classifier head removed, **domain-fine-tuned** on the
43 Demon Bluff card characters. Outputs a 576-dimensional pooled feature vector per image. This is
the model `src/dbcv/embed.py` loads at runtime (Stage 2 identification).

**Why fine-tuned (not frozen).** The frozen-ImageNet backbone (see baseline below) collapsed the
43 stylised characters into one cluster — inter-prototype cosine 0.65–0.94 — so nearest-neighbour
over its embeddings over-identified instead of improving correctness. This is **domain shift of
frozen ImageNet features to a stylised, fine-grained domain** (not "neural collapse"); the fix is
to adapt the features. See `research/RESEARCH.md`, "Why a frozen-ImageNet embedding + NN gallery
collapses on stylized cards, and how to fix it — 2026-06-22."

**How it was fine-tuned (`utils/python/finetune_embedding.py`):**

- **Loss:** Proxy-Anchor (implemented inline) — enforces an inter-class cosine margin, converges
  fast, and tolerates few examples per class (one learnable proxy per identity).
- **Schedule (LP-FT):** Phase A — 250 steps proxy warm-up with the backbone frozen; Phase B —
  600 steps with the top-4 feature blocks unfrozen (~736k trainable params). Warming the proxies
  first means fine-tuning perturbs the trunk less (Kumar et al., ICLR 2022).
- **Augmentation:** synthetic crops generated from the clean reference art via
  `torchvision.transforms.v2` — RandomResizedCrop + perspective + ≤6° rotation + RandAugment +
  GaussianBlur + RandomErasing. **No horizontal flip** (card art is not flip-invariant).
- **Hardware:** trained on the Titan Xp in ~minutes (FP32).

**Leak-proof result** (clean references, off-diagonal cosine; synthetic held-out retrieval):

| Metric | Frozen baseline | Fine-tuned (served) |
|--------|-----------------|---------------------|
| Inter-prototype cosine, mean | 0.850 | **0.409** |
| Inter-prototype cosine, max | 0.939 | **0.536** |
| Synthetic top-1 retrieval | 79.9% | **100%** |
| top1−top2 margin (synthetic) | 0.031 | **0.405** |
| torch↔ONNX parity (max abs diff) | — | **5.5e-6** |

(The synthetic eval is optimistic — it scores augmented reference art, not real board crops. Real-frame
generalisation beyond the confident few is the open round-2 lever; see the lesson-plan module 05.)

**Inputs / outputs:**

| Port | Name | Shape | dtype | Notes |
|------|------|-------|-------|-------|
| Input | `image` | [N, 3, 224, 224] | float32 | NCHW, ImageNet-normalized, dynamic batch N |
| Output | `embedding` | [N, 576] | float32 | Raw embedding (L2-normalized at runtime in `OnnxEmbedder.embed`) |

**Expected preprocessing** (unchanged from the frozen export — applied in `src/dbcv/embed.py`
`OnnxEmbedder.preprocess`):

1. BGR to RGB conversion (OpenCV loads as BGR).
2. Resize to 224 x 224 (bilinear).
3. Normalize with ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225].
4. Layout NCHW, dtype float32, shape [1, 3, 224, 224].

**Abstention is now margin-based.** Fine-tuning compressed the absolute cosine scale (a correct
match now sits ~0.6, an unrelated prototype ~0.4), so the old absolute-cosine threshold no longer
discriminates. `src/dbcv/identify.py` now abstains on the **top1−top2 cosine margin**
(`_EMBED_MARGIN_THRESHOLD = 0.12`, provisional), and the snapshot `confidence` field for the
embedding identifier **is that margin** (decisiveness), not the old `(cos+1)/2` remap. Real-frame
behaviour after adoption: embedding identifies **30/125 (24%)** confident cards and abstains on 95;
the classical baseline identifies 44/125 (35%); classical↔embedding agreement rose 27 → 90.

**Art-swap cost (changed — teachable tradeoff).** Previously, with a frozen backbone, an art swap
was *re-embed only, zero training*. Now the served backbone is fine-tuned to the **current** 43
characters, so a **new** art set is best handled by **re-fine-tuning** (`finetune_embedding.py`,
~minutes on the Titan Xp) and then rebuilding the gallery. A quick re-embed against the existing
fine-tuned backbone still works but won't separate the new art as well — this is the accuracy ↔
retrain-cost tradeoff the design now makes explicitly.

**ONNX opset:** 13 (stable, well-supported by onnxruntime >= 1.10).

**Runtime provider:** CPU (`CPUExecutionProvider`). torch is NOT required at runtime — only at
fine-tune/export time.

## mobilenetv3_small_embed_frozen.onnx — the frozen-ImageNet baseline

**What it is:** the same MobileNetV3-Small architecture but with **frozen ImageNet-pretrained**
weights (classifier head stripped), exported by `utils/python/export_backbone.py`. It is kept
**only** as the comparison arm for the fine-tune head-to-head documented above and in the lesson
plan. The runtime never loads it.

**Parity:** torch vs onnxruntime max absolute difference ≈ 1.7e-06 at export (well within 1e-4).

**Preprocessing / I/O:** identical to the served model (same architecture, same input contract).
