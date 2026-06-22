"""
utils/python/export_backbone.py — Export MobileNetV3-Small embedding backbone to ONNX.

DEV-ONLY — this tool imports torch and torchvision.
The RUNTIME (src/dbcv/) MUST NOT import torch; it uses onnxruntime only.

What this does
--------------
1. Loads MobileNetV3-Small (torchvision, ImageNet-pretrained weights).
2. Strips the classifier head: keeps features → adaptive avg pool → flatten.
   This produces a 576-dim L2-normalized embedding vector per image.
3. Exports to ONNX (opset 13) at models/mobilenetv3_small_embed.onnx.
   Input: [N, 3, 224, 224] float32 (dynamic N batch axis).
   Output: [N, 576] float32 (raw embedding, NOT yet L2-normalized here —
           normalization is applied at runtime in OnnxEmbedder.embed()).
4. Validates parity: runs the same 224×224 input through the torch model and
   through onnxruntime, asserts the max absolute difference < 1e-4.

Expected preprocessing (must match OnnxEmbedder.preprocess in src/dbcv/embed.py)
----------------------------------------------------------------------------------
  - Resize to 224×224 (bilinear).
  - Convert BGR → RGB.
  - Normalize with ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
  - Shape: NCHW float32.

Why MobileNetV3-Small?
-----------------------
  See research/RESEARCH.md entry 3 (identification, 2026-06-21).
  Key facts: ~2.5M parameters; fits comfortably on Titan XP / free Colab tier;
  ImageNet-pretrained weights transfer well to cartoon art (colour, shape, texture
  priors carry over). Using it frozen means zero gradient steps on an art swap.
  The 576-dim embedding is rich enough for 44-class nearest-neighbor lookup but
  cheap to store (44 × 576 × 4 bytes ≈ 100 KB gallery).

Why export to ONNX and run on CPU at inference?
------------------------------------------------
  See research/RESEARCH.md entry 6 (compute budget).
  onnxruntime-cpu is a single pip install (~5 MB), no CUDA, no 3 GB torch.
  For a 44-class gallery nearest-neighbor this is perfectly fast:
  one 224×224 forward pass ≈ 5–15 ms on CPU.

Why NOT keep torch in the runtime?
-----------------------------------
  torch + CUDA takes ~3 GB of disk and requires matching CUDA libraries.
  A learner running the REST server should not need to install a 3 GB deep
  learning stack just to serve the pipeline. ONNX + onnxruntime separates
  the dev-side training tooling from the runtime deployment artifact.

Rule 1 compliance
-----------------
  No inline interpreter calls. No `python -c`. This file is run as a script:
      .venv\\Scripts\\python.exe utils/python/export_backbone.py

Anchoring
---------
  All paths are computed from Path(__file__).resolve().parents[2] (repo root).
  Never assumes CWD.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path anchoring — all relative to repo root, never CWD
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
# utils/python/export_backbone.py → parents[0]=utils/python, parents[1]=utils, parents[2]=repo root
_REPO_ROOT = _HERE.parents[2]
_MODELS_DIR = _REPO_ROOT / "models"
_ONNX_PATH = _MODELS_DIR / "mobilenetv3_small_embed.onnx"

# ---------------------------------------------------------------------------
# Torch imports (DEV ONLY — never imported in src/dbcv/)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
except ImportError as exc:
    print(
        "ERROR: torch / torchvision not installed in this environment.\n"
        f"       {exc}\n"
        "       Install with: pip install torch torchvision\n"
        "       This tool is dev-only; the runtime (src/dbcv/) uses onnxruntime."
    )
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError as exc:
    print(
        "ERROR: onnxruntime not installed.\n"
        f"       {exc}\n"
        "       Install with: pip install onnxruntime"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Embedding module: features → adaptive avg pool → flatten
# ---------------------------------------------------------------------------


class MobileNetV3SmallEmbedder(nn.Module):
    """MobileNetV3-Small with the classifier head removed.

    Outputs the pooled feature vector (576-dim) before any classifier layer.
    This vector is the 'embedding' we use for nearest-neighbor lookup.

    Architecture note
    -----------------
    torchvision.models.mobilenet_v3_small has two top-level sub-modules:
      - model.features: the convolutional backbone
      - model.avgpool: AdaptiveAvgPool2d(1)
      - model.classifier: the final FC layers (dropped here)

    After avgpool the tensor is [N, 576, 1, 1]; flatten gives [N, 576].
    The 576 dimension is the MobileNetV3-Small penultimate feature size.
    """

    def __init__(self, pretrained_model: tv_models.MobileNetV3) -> None:
        super().__init__()
        # Keep only the feature extractor and global pooling; drop the classifier.
        self.features = pretrained_model.features
        self.avgpool = pretrained_model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, 224, 224] float32
        x = self.features(x)      # [N, 576, 7, 7] (for 224×224 input)
        x = self.avgpool(x)       # [N, 576, 1, 1]
        x = torch.flatten(x, 1)   # [N, 576]
        return x


# ---------------------------------------------------------------------------
# Build and export
# ---------------------------------------------------------------------------


def build_embedder() -> MobileNetV3SmallEmbedder:
    """Load ImageNet-pretrained MobileNetV3-Small, strip classifier, set eval."""
    weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
    base = tv_models.mobilenet_v3_small(weights=weights)
    embedder = MobileNetV3SmallEmbedder(base)
    embedder.eval()
    return embedder


def export_to_onnx(embedder: MobileNetV3SmallEmbedder) -> Path:
    """Export the embedder to ONNX at models/mobilenetv3_small_embed.onnx.

    ONNX opset 13 is well-supported by onnxruntime ≥ 1.10 and is stable.
    The batch axis (N) is dynamic so single-image inference and batch inference
    both work without a model re-export.

    Returns the path to the written .onnx file.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Dummy input in the expected shape: [1, 3, 224, 224] float32.
    # The actual value doesn't matter for the export graph; any random tensor works.
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"Exporting to {_ONNX_PATH} …")
    with torch.no_grad():
        torch.onnx.export(
            embedder,
            dummy_input,
            str(_ONNX_PATH),
            opset_version=13,
            input_names=["image"],
            output_names=["embedding"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "embedding": {0: "batch_size"},
            },
            do_constant_folding=True,  # fuse BN into conv for faster inference
        )

    print(f"  Written: {_ONNX_PATH} ({_ONNX_PATH.stat().st_size // 1024} KB)")
    return _ONNX_PATH


# ---------------------------------------------------------------------------
# Parity validation: torch output vs onnxruntime output
# ---------------------------------------------------------------------------


def validate_parity(embedder: MobileNetV3SmallEmbedder, onnx_path: Path) -> float:
    """Run the same input through torch and onnxruntime; assert embeddings match.

    Parameters
    ----------
    embedder:
        The torch model (eval mode, no grad).
    onnx_path:
        Path to the .onnx file just exported.

    Returns
    -------
    float
        Maximum absolute difference between torch and onnxruntime outputs.
        Should be < 1e-4 for a lossless export (float32 numerics).

    Raises
    ------
    AssertionError
        If max abs diff >= 1e-4 (indicates an export bug).
    """
    print("Validating torch ↔ onnxruntime parity …")

    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    test_input_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np)

    # --- Torch output ---
    with torch.no_grad():
        torch_out = embedder(test_input_torch).numpy()  # [1, 576]

    # --- OnnxRuntime output ---
    # Use CPU provider — this is exactly how the runtime will run it.
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_out = session.run(["embedding"], {"image": test_input_np})[0]  # [1, 576]

    # --- Compare ---
    max_abs_diff = float(np.abs(torch_out - ort_out).max())
    print(f"  Max absolute diff torch vs onnxruntime: {max_abs_diff:.2e}")

    assert max_abs_diff < 1e-4, (
        f"Parity check FAILED: max abs diff = {max_abs_diff:.2e} (threshold 1e-4).\n"
        "This indicates an export bug or a dtype mismatch."
    )

    print("  Parity OK ✓")
    return max_abs_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("export_backbone.py -- MobileNetV3-Small to ONNX")
    print(f"Repo root   : {_REPO_ROOT}")
    print(f"Output path : {_ONNX_PATH}")
    print("=" * 60)

    # 1. Build the embedder (loads ImageNet weights)
    print("Loading MobileNetV3-Small (ImageNet pretrained) …")
    embedder = build_embedder()
    print(f"  Model loaded. Output dim: 576 (pooled MobileNetV3-Small features)")

    # 2. Export to ONNX
    onnx_path = export_to_onnx(embedder)

    # 3. Validate parity
    max_diff = validate_parity(embedder, onnx_path)

    print()
    print("=" * 60)
    print("Export complete.")
    print(f"  ONNX file : {onnx_path}")
    print(f"  Size      : {onnx_path.stat().st_size // 1024} KB")
    print(f"  Max diff  : {max_diff:.2e}")
    print()
    print("Preprocessing expected by the runtime OnnxEmbedder (src/dbcv/embed.py):")
    print("  1. BGR → RGB")
    print("  2. Resize to 224×224 (bilinear)")
    print("  3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
    print("  4. Layout: NCHW float32 (shape [1, 3, 224, 224])")
    print()
    print("To regenerate after an art swap: just re-run this script.")
    print("The ONNX file is gitignored (models/*.onnx); see models/README.md.")
    print("=" * 60)


if __name__ == "__main__":
    main()
