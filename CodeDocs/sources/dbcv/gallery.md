# CodeDocs/sources/dbcv/gallery.py

**Status:** active — Stage 2 classical gallery builder. Loads ~67 PNGs,
precomputes HSV histograms + ORB descriptors + thumbnails entirely in-memory.

**Purpose:** Builds and holds the reference gallery for classical card
identification.  Called once at startup via the FastAPI lifespan; stored on
`app.state.gallery`.  On an art swap, re-run `build_gallery()` — zero training.

**Who uses it:**
- `dbcv/api.py` — calls `build_gallery()` in `lifespan()`; result stored on `app.state.gallery`
- `dbcv/identify.py` — imports `_compute_hsv_hist`, `_compute_orb` (internal helpers);
  `classify_crop` and `make_gallery_identifier` accept a `Gallery` object
- `tests/test_gallery.py` — tests gallery structure and invariants
- `tests/test_identify.py` — builds a gallery fixture for classifier tests

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

## Art-swap contract

To retrain after an art swap:
1. Replace PNGs in `knowledge-base/card-art/`.
2. Restart the server (or call `build_gallery()` again in the lifespan).
3. No gradient steps. No stored model files. Zero training.

This satisfies the "cheap to retrain" constraint from `CLAUDE.md`.
