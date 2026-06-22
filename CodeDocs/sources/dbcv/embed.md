# CodeDocs/sources/dbcv/embed.py

**Status:** active — Stage 3 runtime embedder (onnxruntime only; no torch in this module).
`get_onnx_embedder()` factory added 2026-06-22 for process-level session reuse.

**Purpose:** Wraps the exported MobileNetV3-Small ONNX backbone for CPU inference at runtime.
Given a BGR card crop, produces an L2-normalised 576-dim embedding vector for nearest-neighbor
identification. Used by `build_embedding_gallery` (gallery.py) and `classify_crop_embedding`
(identify.py).

**Who uses it:**
- `dbcv/gallery.py` — `build_embedding_gallery()` calls `OnnxEmbedder.embed()` to embed ~67 reference PNGs
- `dbcv/identify.py` — `classify_crop_embedding()` and `make_embedding_identifier()` accept an `OnnxEmbedder`
- `dbcv/api.py` — lifespan builds `OnnxEmbedder()` once, stored on `app.state.embedder`
- `tests/test_embed.py` — tests the embedder and embedding gallery

**Critical constraint:** This module MUST NOT import torch or torchvision.
The ONNX model is loaded and run via `onnxruntime` (CPU provider) only.
Dev-time export (which does require torch) lives in `utils/python/export_backbone.py`.

---

## Constants (module-level)

| Constant | Value | Purpose |
|----------|-------|---------|
| `_INPUT_SIZE` | 224 | Model input spatial size (MobileNetV3-Small default) |
| `_IMAGENET_MEAN` | (0.485, 0.456, 0.406) | ImageNet normalisation mean (RGB order) |
| `_IMAGENET_STD` | (0.229, 0.224, 0.225) | ImageNet normalisation std (RGB order) |
| `EMBEDDING_DIM` | 576 | MobileNetV3-Small pooled feature dim (exported) |
| `_DEFAULT_ONNX_PATH` | `models/mobilenetv3_small_embed.onnx` (repo-root anchored) | Default model path |

---

## Key signatures

### `get_onnx_embedder(onnx_path=None) -> OnnxEmbedder` — module-level factory (line ~103)
```python
def get_onnx_embedder(onnx_path: Path | str | None = None) -> OnnxEmbedder:
```
Returns a **cached** `OnnxEmbedder` for `onnx_path` (key = resolved path).  Constructs
it once per process via `_EMBEDDER_CACHE`; subsequent calls are ~0 ms.

**Preferred over `OnnxEmbedder()` directly** when the same model will be used multiple
times in one process (test suite, server lifespan).  `api.py` lifespan uses this.

---

### `OnnxEmbedder` — class (line ~126)

```python
class OnnxEmbedder:
    def __init__(self, onnx_path: Path | str | None = None) -> None: ...
    def preprocess(self, bgr_crop: np.ndarray) -> np.ndarray: ...
    def embed(self, bgr_crop: np.ndarray) -> np.ndarray: ...
```

#### `__init__(onnx_path=None)`
Loads the ONNX session once (CPU provider). Raises `FileNotFoundError` with a clear
message if the ONNX file is absent (tells user to run `export_backbone.py`).

Anchors default path to repo root via `Path(__file__).resolve().parents[2]`.

Stores `self._onnx_path: Path` (resolved) for use as a cache key by
`build_embedding_gallery` in `gallery.py`.

#### `preprocess(bgr_crop) -> np.ndarray` — returns shape `[1, 3, 224, 224]` float32
Steps:
1. BGR → RGB (OpenCV loads BGR; ImageNet/PyTorch expects RGB)
2. Resize to 224×224 bilinear
3. Scale to [0, 1] (÷255)
4. Subtract ImageNet mean, divide by std (per-channel, RGB order)
5. Transpose HWC → CHW, add batch dim

These steps **must match** the preprocessing used when exporting the backbone.
See `utils/python/export_backbone.py` for the export-side documentation.

#### `embed(bgr_crop) -> np.ndarray` — returns shape `[576]` float32, unit L2 norm
1. `preprocess(bgr_crop)` → `[1, 3, 224, 224]`
2. ONNX forward pass → `[1, 576]` raw embedding
3. L2-normalise → `[576]` unit vector

Returns a zero vector (`[576]` of 0.0) if:
- Input is None or zero-size
- The forward pass produces a near-zero vector (norm < 1e-9)

The zero vector produces cosine similarity ≈ 0 against all gallery entries.
Face-down cards embed to a non-zero but distant vector (cosine ≈ -0.23 against
any character prototype, confidence ≈ 0.38) — well below the threshold (0.60).

---

## Error conditions

- `FileNotFoundError` at construction if ONNX file is missing.
  Clear message: "Run `.venv\Scripts\python.exe utils\python\export_backbone.py`".
- `ImportError` if `onnxruntime` is not installed (with install hint).

---

## Performance notes

- ONNX session loads once at lifespan startup (~50 ms).
- Per-inference cost: ~5–15 ms on CPU (224×224, MobileNetV3-Small).
- Thread-safe for concurrent requests (onnxruntime sessions are thread-safe by default).
- Session is kept alive for the process lifetime; no per-request session creation.
