# CodeDocs/sources/dbcv/pipeline.py

**Status:** slice/active — end-to-end orchestration with stub localizer + identifier.

**Purpose:** Orchestrates the frame → GameStateSnapshot pipeline.  Reads
resolution from `image.shape`, calls the localizer, crops each box,
calls the identifier, and assembles the snapshot.

**Who uses it:**
- `dbcv/api.py` — calls `run_pipeline(image, source)`
- `tests/test_api.py` — indirectly (via the FastAPI TestClient)

---

## Key signatures (with line numbers)

### `crop_relative(image, bbox_rel) -> np.ndarray` — line 57
```python
def crop_relative(image: np.ndarray, bbox_rel: BboxRel) -> np.ndarray:
```
**The single point of relative → pixel conversion in the pipeline.**
Reads `image.shape[:2]` for (h, w), computes pixel coords from fractions,
clamps to valid range, and returns the cropped sub-array.

Conversion logic (line 79):
```python
x0 = round(x_rel * w_img)
y0 = round(y_rel * h_img)
x1 = round((x_rel + w_rel) * w_img)
y1 = round((y_rel + h_rel) * h_img)
```
Clamping (line 84): `max(0, min(coord, dimension))` on all four values.

### `run_pipeline(image, source, localizer, identifier) -> GameStateSnapshot` — line 96
```python
def run_pipeline(
    image: np.ndarray,
    source: Source,
    localizer: LocalizerCallable = stub_localize,
    identifier: Callable[[np.ndarray], tuple[str, str, float]] = identify,
) -> GameStateSnapshot:
```
**Parameters:**
- `image` — decoded frame (H x W x C, typically BGR from cv2)
- `source` — provenance metadata
- `localizer` — defaults to `stub_localize`; accepts any `LocalizerCallable`
- `identifier` — defaults to `identify` (stub); accepts any matching callable

**Steps (lines 121–143):**
1. Measure resolution from `image.shape[:2]` → `Resolution(w, h)`.
2. Call `localizer(image, resolution)` → list of `BboxRel`.
3. For each box: `crop_relative(image, bbox_rel)` → call `identifier(crop)`.
   Zero-size crops (degenerate boxes) → `("unknown", "unknown", 0.0)`.
4. Call `assemble(source, resolution, boxes, identities)` → `GameStateSnapshot`.

---

## Design notes

- `run_pipeline` does **no I/O** — it is a pure function over numpy arrays.
  Image loading (and format-specific decoding) happens in `api.py`.
- The zero-size crop guard (line 138) prevents crashes from stub boxes that
  accidentally collapse to zero pixels on very small images.
- The `localizer` and `identifier` parameters are the injection points for
  real implementations; the defaults keep the slice self-contained.
