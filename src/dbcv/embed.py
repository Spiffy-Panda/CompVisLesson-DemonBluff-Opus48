"""
dbcv/embed.py — ONNX-based image embedder for card identification.

RUNTIME-ONLY module: no torch, no torchvision, no CUDA.
Depends only on onnxruntime, numpy, and opencv (cv2).

This module implements the Stage 2 embedding backbone at INFERENCE TIME.
The served ONNX model is the domain-fine-tuned backbone produced by
utils/python/finetune_embedding.py (dev-only, requires torch + a GPU is
strongly preferred).  utils/python/export_backbone.py produces the *frozen*
ImageNet baseline (models/..._frozen.onnx) used only for the head-to-head.
At runtime, onnxruntime runs the same computation on CPU.

Architecture
------------
  Input  : BGR crop (H x W x 3, any size)
  Output : L2-normalised 576-dim float32 embedding vector

Preprocessing matches the ImageNet standard used when exporting the backbone:
  1. BGR -> RGB
  2. Resize to 224 x 224 (bilinear)
  3. Normalise with ImageNet mean / std
  4. Layout NCHW float32

Why L2-normalise the output?
-----------------------------
When the embedding vectors are L2-normalised to the unit sphere,
cosine similarity = dot product.  This makes nearest-neighbour search
cheaper (just matmul) and keeps the similarity values in [-1, 1],
which we then map to a confidence score in [0, 1].

Why use cosine similarity for gallery lookup?
----------------------------------------------
  See research/RESEARCH.md entry 3 (identification, 2026-06-21).
  Wu et al. (ECCV 2018) show that nearest-neighbor over learned embeddings
  with cosine distance can match or beat a softmax classifier.
  For 44 gallery classes the lookup is a 44-element dot product -- trivial
  cost, interpretable result, no gradient steps on an art swap.

Why one OnnxEmbedder instance per process?
-------------------------------------------
  onnxruntime.InferenceSession loads the ONNX graph once into memory
  and initialises operator kernels.  Creating it per-request would add
  ~50 ms overhead and waste memory.  The FastAPI lifespan builds it once
  (Phase 4) and stores it on app.state.embedder.

Rule 1 compliance
-----------------
  This module never executes inline interpreter calls.
  It contains no 'import torch', 'import torchvision', or any GPU library.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# onnxruntime is CPU-only at runtime (no GPU required, no torch required).
try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError(
        "onnxruntime is required by src/dbcv/embed.py but is not installed.\n"
        "Install with: pip install onnxruntime"
    ) from exc

# ---------------------------------------------------------------------------
# Constants — preprocessing parameters
# ---------------------------------------------------------------------------

# The backbone was exported expecting 224x224 input (MobileNetV3-Small default).
_INPUT_SIZE: int = 224

# ImageNet normalisation constants.
# These MUST match what the backbone was trained with; deviating here will
# corrupt the embedding and break cosine similarity comparisons.
# Source: torchvision ImageNet transforms (official torchvision docs).
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)  # RGB order
_IMAGENET_STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)  # RGB order

# The MobileNetV3-Small backbone (with classifier dropped) outputs 576 dims.
# This is used for type-checking assertions in tests, not to gate behaviour.
EMBEDDING_DIM: int = 576

# Model path: anchored to repo root via __file__ (never CWD).
# __file__ = src/dbcv/embed.py -> parents[0]=dbcv, parents[1]=src, parents[2]=repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ONNX_PATH = _REPO_ROOT / "models" / "mobilenetv3_small_embed.onnx"


# ---------------------------------------------------------------------------
# Process-level cache for OnnxEmbedder instances
# ---------------------------------------------------------------------------
# Keyed by the resolved ONNX model path.  Loading an ONNX InferenceSession
# takes ~50 ms and allocates the operator kernels once; reusing the same
# instance across all callers (test fixtures, lifespan, helper functions) is
# correct because OnnxEmbedder is stateless after construction.
#
# This cache is module-level so it persists for the lifetime of the process.
# A model swap requires a process restart, which clears the cache automatically.
_EMBEDDER_CACHE: dict[Path, "OnnxEmbedder"] = {}


def get_onnx_embedder(onnx_path: "Path | str | None" = None) -> "OnnxEmbedder":
    """Return a cached OnnxEmbedder for ``onnx_path``, constructing it once.

    Preferred over calling ``OnnxEmbedder()`` directly when the same model
    file will be used multiple times in one process (e.g. test suite, server).

    Parameters
    ----------
    onnx_path:
        Path to the ONNX model file.  Defaults to the canonical path used by
        OnnxEmbedder (models/mobilenetv3_small_embed.onnx, repo-root anchored).
    """
    resolved = Path(onnx_path).resolve() if onnx_path is not None else _DEFAULT_ONNX_PATH
    if resolved not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[resolved] = OnnxEmbedder(resolved)
    return _EMBEDDER_CACHE[resolved]


# ---------------------------------------------------------------------------
# OnnxEmbedder
# ---------------------------------------------------------------------------


class OnnxEmbedder:
    """Wrap the exported MobileNetV3-Small ONNX model for runtime inference.

    Usage
    -----
        embedder = OnnxEmbedder()          # loads the model once
        vec = embedder.embed(bgr_crop)     # [576] float32, unit norm

    The embedder is stateless after construction (no mutable internal state
    during inference), so it is safe to share across threads in the FastAPI
    threadpool.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX file.  Defaults to models/mobilenetv3_small_embed.onnx
        anchored from the repo root.
    """

    def __init__(self, onnx_path: Path | str | None = None) -> None:
        if onnx_path is None:
            onnx_path = _DEFAULT_ONNX_PATH
        onnx_path = Path(onnx_path)

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "The served model is the domain-fine-tuned backbone. Regenerate with:\n"
                "    .venv\\Scripts\\python.exe utils\\python\\finetune_embedding.py\n"
                "(utils/python/export_backbone.py produces the frozen baseline only.)"
            )

        # Store the resolved path so gallery.py can use it as a cache key.
        # This enables build_embedding_gallery to distinguish galleries built
        # from different ONNX models without importing OnnxEmbedder directly.
        self._onnx_path: Path = onnx_path

        # Load the ONNX session once — CPU-only provider.
        # Using CPUExecutionProvider explicitly (no CUDA required at runtime).
        # Thread count is left at default (os.cpu_count()) — adequate for our
        # single-request-at-a-time use case and avoids overcommitting.
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        # Cache input/output names for clarity
        self._input_name: str = self._session.get_inputs()[0].name    # "image"
        self._output_name: str = self._session.get_outputs()[0].name  # "embedding"

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Convert a BGR crop to a normalised NCHW float32 tensor.

        Steps (must match the preprocessing baked into the export):
          1. BGR -> RGB  (OpenCV convention vs PyTorch/ImageNet convention)
          2. Resize to 224x224 bilinear
          3. float32 /255 -> [0, 1]
          4. Subtract ImageNet mean, divide by ImageNet std
          5. Transpose HWC -> CHW, add batch dim -> [1, 3, 224, 224]

        Parameters
        ----------
        bgr_crop:
            BGR image, any spatial size, uint8 or float32.

        Returns
        -------
        np.ndarray
            Shape [1, 3, 224, 224], float32, ready for onnxruntime.
        """
        # 1. BGR -> RGB
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

        # 2. Resize to model input size (bilinear as in standard ImageNet preprocessing)
        if rgb.shape[0] != _INPUT_SIZE or rgb.shape[1] != _INPUT_SIZE:
            rgb = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        # 3. Scale to [0, 1]
        x = rgb.astype(np.float32) / 255.0

        # 4. ImageNet normalisation: (pixel - mean) / std
        # Using numpy broadcasting over the channel axis.
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)  # [3]
        std  = np.array(_IMAGENET_STD,  dtype=np.float32)  # [3]
        x = (x - mean) / std  # shape [224, 224, 3]

        # 5. HWC -> CHW, add batch dimension
        x = x.transpose(2, 0, 1)   # [3, 224, 224]
        x = x[np.newaxis, ...]      # [1, 3, 224, 224]

        return x

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Embed a BGR crop into a unit-norm vector.

        Runs preprocess -> ONNX forward pass -> L2 normalise.

        Parameters
        ----------
        bgr_crop:
            BGR image (H x W x 3), any size, uint8 or float32.

        Returns
        -------
        np.ndarray
            Shape [576], float32, L2-normalised (unit norm).
            If the crop is zero-size or the forward pass returns a degenerate
            vector, a zero vector is returned (will produce low cosine similarity
            against any gallery entry).

        Notes on L2 normalisation
        -------------------------
        The ONNX model outputs the raw pooled feature vector.
        We normalise here so that:
            cosine_similarity(a, b) = dot(a, b)   (fast matmul, no sqrt needed)
        Gallery embeddings are also L2-normalised when the embedding gallery
        is built (see gallery.py, build_embedding_gallery).
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Preprocess: BGRcrop -> [1, 3, 224, 224] float32
        x = self.preprocess(bgr_crop)

        # Forward pass
        raw: np.ndarray = self._session.run(
            [self._output_name],
            {self._input_name: x},
        )[0]  # [1, 576]

        vec = raw[0]  # [576]

        # L2 normalise onto the unit sphere
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            # Degenerate output (e.g., a completely uniform crop produces near-zero activations).
            # Return a zero vector; cosine similarity against any gallery entry will be ~0.
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        return (vec / norm).astype(np.float32)
