# CodeDocs/sources/dbcv/pipeline.md

**Status:** active — Stage 0 gate wired in (frame_state); default localizer
is `classical_localize`; schema version bumped to 0.2.0.

**Purpose:** Orchestrates the frame → GameStateSnapshot pipeline.  Runs
the frame-state gate first; non-board frames skip localisation and return
`cards=[]` immediately.  For board frames: reads resolution from `image.shape`,
calls the localizer, crops each box, calls the identifier, and assembles the
snapshot.

**Who uses it:**
- `dbcv/api.py` — calls `run_pipeline(image, source)`
- `tests/test_api.py` — indirectly (via the FastAPI TestClient)
- `tests/test_frame_state.py` — indirectly (API integration test for modal frames)

---

## Key signatures (with line numbers)

### `crop_relative(image, bbox_rel) -> np.ndarray` — line 57
```python
def crop_relative(image: np.ndarray, bbox_rel: BboxRel) -> np.ndarray:
```
**The single point of relative → pixel conversion in the pipeline.**
Reads `image.shape[:2]` for (h, w), computes pixel coords from fractions,
clamps to valid range, and returns the cropped sub-array.

Conversion logic (lines 90–93):
```python
x0 = round(x_rel * w_img)
y0 = round(y_rel * h_img)
x1 = round((x_rel + w_rel) * w_img)
y1 = round((y_rel + h_rel) * h_img)
```
Clamping (lines 96–99): `max(0, min(coord, dimension))` on all four values.

### `run_pipeline(image, source, localizer, identifier, frame_state_fn) -> GameStateSnapshot` — line 109
```python
def run_pipeline(
    image: np.ndarray,
    source: Source,
    localizer: LocalizerCallable = classical_localize,
    identifier: Callable[[np.ndarray], tuple[str, str, float]] = identify,
    frame_state_fn: Callable[[np.ndarray], FrameState] = classify_frame_state,
) -> GameStateSnapshot:
```
**Parameters:**
- `image` — decoded frame (H x W x C, typically BGR from cv2)
- `source` — provenance metadata
- `localizer` — defaults to `classical_localize` (validated implementation);
  pass `stub_localize` explicitly to use the teaching baseline
- `identifier` — defaults to `identify` (stub); accepts any matching callable.
  Pass `make_gallery_identifier(gallery)` from `dbcv.identify` to use the
  classical gallery matcher (Stage 2). The API does this via `app.state.identifier`.
- `frame_state_fn` — defaults to `classify_frame_state` (Stage 0 gate);
  inject `lambda _: "board"` in tests to skip the gate

**Steps:**
0. Run `frame_state_fn(image)` → `FrameState` ("board"/"modal"/"menu").
   If not "board": return `assemble(source, resolution, [], [])` with
   `frame_state` set, **skipping Steps 2–4**.
1. Measure resolution from `image.shape[:2]` → `Resolution(w, h)`.
2. Call `localizer(image, resolution)` → list of `BboxRel`.
3. For each box: `crop_relative(image, bbox_rel)` → call `identifier(crop)`.
   Zero-size crops (degenerate boxes) → `("unknown", "unknown", 0.0)`.
4. Call `assemble(source, resolution, boxes, identities)` → `GameStateSnapshot`.
   Apply `model_copy(update={"frame_state": state})` to stamp the gate result.

---

## Stage 0 gate (added 2026-06-22)

The `frame_state_fn` parameter (defaults to `classify_frame_state`) runs
first, before any expensive CV.  Non-board frames (modal/menu) skip the
localizer entirely, preventing false card detections on overlay frames.

The gate is injectable so tests can decouple it from the localizer:
```python
run_pipeline(image, source, frame_state_fn=lambda _: "board")
```

See `CodeDocs/sources/dbcv/frame_state.md` for the algorithm and thresholds.

---

## Default localizer (from 2026-06-22)

The `localizer` parameter default is `classical_localize` (validated: 8/8
and 9/9 board cards exact on sample frames, 0 false positives).

`stub_localize` is still available for tests needing predictable output:
```python
run_pipeline(image, source, localizer=stub_localize)
```

---

## Design notes

- `run_pipeline` does **no I/O** — it is a pure function over numpy arrays.
  Image loading (and format-specific decoding) happens in `api.py`.
- The zero-size crop guard prevents crashes from degenerate stub boxes.
- `model_copy(update=...)` is used instead of attribute mutation to keep
  Pydantic v2 happy (avoids any `model_config` dependency).
- The `frame_state` field on the returned snapshot is always set by the
  pipeline; only test fixtures constructed directly via `GameStateSnapshot(...)`
  will show the default "unknown".
