# CodeDocs/sources/dbcv/gallery.py

**Status:** active — Stage 2 classical gallery + **Stage 3 embedding gallery** (updated 2026-06-22).
Loads ~67 PNGs, precomputes HSV histograms + ORB descriptors + thumbnails (classical),
and now also builds per-identity prototype embeddings (embedding-NN).

**Purpose:** Builds and holds the reference galleries for card identification:
- Classical (`Gallery`): HSV histograms + ORB descriptors. Called by `classify_crop`.
- Embedding (`EmbeddingGallery`): L2-normalised prototype embeddings via `OnnxEmbedder`.
  Called by `classify_crop_embedding`. Art swap = re-run both builders; zero gradient steps.

**Who uses it:**
- `dbcv/api.py` — calls `build_gallery()` and `build_embedding_gallery()` in `lifespan()`;
  results stored on `app.state.gallery`, `app.state.embed_gallery`
- `dbcv/identify.py` — imports `_compute_hsv_hist`, `_compute_orb` (classical helpers);
  `classify_crop` and `make_gallery_identifier` accept a `Gallery` object;
  `classify_crop_embedding` and `make_embedding_identifier` accept an `EmbeddingGallery`
- `dbcv/embed.py` — `build_embedding_gallery` calls `OnnxEmbedder.embed` (no circular import)
- `tests/test_gallery.py` — tests classical gallery structure and invariants
- `tests/test_identify.py` — builds a gallery fixture for classifier tests
- `tests/test_embed.py` — builds both gallery fixtures for embedding tests

---

## Key types

### `GalleryEntry` — `NamedTuple` (line ~56)
```python
class GalleryEntry(NamedTuple):
    identity: str          # e.g. "Alchemist" (alias-resolved)
    role_class: str        # "villager" | "minion" | "outcast" | "demon"
    file_stem: str         # original filename stem (for debugging)
    thumb: np.ndarray      # BGR thumbnail (64×64)
    hsv_hist: np.ndarray   # 2-D Hue×Saturation histogram (32×16), L1-normalised
    orb_kp: list           # list of cv2.KeyPoint objects
    orb_desc: DescriptorArray  # (N, 32) uint8 ORB descriptors, or None
```

### `Gallery` — `dataclass` (line ~78)
```python
@dataclass
class Gallery:
    entries: list[GalleryEntry]
    townee_names: list[str]    # sorted unique identities (alias-resolved)
    role_classes: list[str]    # sorted unique role classes
    n_townees: int             # property: len(townee_names)
    n_references: int          # property: len(entries)
```

---

## Key signatures

### `build_gallery(art_root=None) -> Gallery`
```python
def build_gallery(art_root: Path | str | None = None) -> Gallery:
```
Walks `art_root/<role_class>/<identity>/<file>.png`, loads every PNG, and
precomputes matchable representations for each.

**Default `art_root`:** `<repo_root>/knowledge-base/card-art/` (anchored via
`Path(__file__).resolve().parents[2]` — never assumes CWD).

**Alias rule:** `Twin_Minion/` directory → `identity = "Minion"` (game fact:
functionally identical; see `_IDENTITY_ALIASES` dict).

**Returns:** Gallery with 43 townees and 67 references (current art tree).

**Raises:** `FileNotFoundError` if `art_root` doesn't exist; `ValueError` if
no PNGs found.

### Internal helpers (module-private, used by `identify.py`)

#### `_compute_hsv_hist(bgr) -> np.ndarray`
Computes a 2-D Hue×Saturation histogram (32×16 bins), masking out near-black
and near-white pixels (card background / transparency fill), and L1-normalises
to sum to 1.0.  Returns a zero array for empty/degenerate images.

**Key edge case:** a zero-sum histogram (all pixels masked) compared via
`cv2.compareHist(HISTCMP_CORREL)` returns 1.0 (Pearson division by zero).
`classify_crop` checks for this before comparing.

#### `_compute_orb(bgr) -> (keypoints, descriptors | None)`
Computes ORB keypoints and descriptors.  Returns `([], None)` if fewer than
2 keypoints found.

#### `_load_reference_image(path) -> np.ndarray | None`
Loads a PNG (BGRA → composited on white background), crops to top 80% of the
reference image height (the art region), returns BGR array or None on failure.

---

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `_THUMB_SIZE` | 64 | Gallery thumbnail side (px) |
| `_HIST_BINS_HUE` | 32 | Hue bins for 2-D HSV histogram |
| `_HIST_BINS_SAT` | 16 | Saturation bins |
| `_ORB_N_FEATURES` | 200 | Max ORB keypoints per image |
| `_ART_CROP_BOTTOM` | 0.80 | Crop reference to top 80% (removes bottom margins) |
| `_IDENTITY_ALIASES` | `{"Twin_Minion": "Minion"}` | Directory-name → identity aliases |

---

---

## Stage 3 additions — Embedding gallery

### `EmbeddingGalleryEntry` — `NamedTuple` (line ~370)
```python
class EmbeddingGalleryEntry(NamedTuple):
    identity: str       # e.g. "Alchemist" (alias-resolved)
    role_class: str     # "villager" | "minion" | "outcast" | "demon"
    embedding: np.ndarray  # [576] float32, unit L2 norm (prototypical mean)
```

### `EmbeddingGallery` — `dataclass` (line ~395)
```python
@dataclass
class EmbeddingGallery:
    entries: list[EmbeddingGalleryEntry]
    townee_names: list[str]       # sorted unique identity strings
    role_classes: list[str]       # sorted unique role class strings
    embeddings: np.ndarray        # [K, 576] float32, all rows unit-norm
    n_classes: int                # property: len(entries)
```

### `build_embedding_gallery(classical_gallery, embedder, art_root=None) -> EmbeddingGallery`
```python
def build_embedding_gallery(
    classical_gallery: Gallery,
    embedder: OnnxEmbedder,
    art_root: Path | str | None = None,
) -> EmbeddingGallery:
```
Embeds every reference PNG via `embedder.embed()`, then computes a
**prototypical mean** per identity (mean of all reference embeddings for that
identity, re-normalised). Returns an `EmbeddingGallery` with K entries
(one per distinct identity/role_class pair) and a pre-stacked `[K, 576]`
embeddings matrix for fast cosine-similarity via single matmul.

**Prototypical mean rationale:** Snell et al. (NeurIPS 2017) — averaging
multiple reference views per class and re-normalising consistently outperforms
single-embedding lookup. For ~44 classes with 1-2 references each the gain
is modest but the computation is free.

**Does NOT import torch.** Uses only OnnxEmbedder (onnxruntime) + numpy.

---

## Art-swap contract

To retrain after an art swap (both classical and embedding):
1. Replace PNGs in `knowledge-base/card-art/`.
2. Restart the server (or call both `build_gallery()` and `build_embedding_gallery()` in the lifespan).
3. No gradient steps. No model file changes. ONNX backbone stays the same.

The ONNX backbone file is NOT retrained on an art swap — only the reference
embeddings in the gallery are recomputed (seconds of compute).

This satisfies the "cheap to retrain" constraint from `CLAUDE.md`.
