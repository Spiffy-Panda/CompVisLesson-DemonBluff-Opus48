# CodeDocs/sources/dbcv/localize.py

**Status:** slice/stub — real classical localizer not yet implemented.

**Purpose:** Defines the localizer interface and a stub that returns fixed
relative boxes so the rest of the pipeline can run end-to-end.

**Who uses it:**
- `dbcv/pipeline.py` — imports `stub_localize` as the default `localizer`
  argument, and imports `localize` as the named single-dispatch entry point
- `dbcv/__init__.py` — re-exported in public-surface documentation
- Future: the classical localizer will be a drop-in replacement callable

---

## Key signatures (with line numbers)

### Type alias `BboxRel` — line 47
```python
BboxRel = tuple[float, float, float, float]
# (x, y, w, h) in [0, 1], relative to frame width/height, origin top-left
```

### `class LocalizerCallable(Protocol)` — line 55
```python
class LocalizerCallable(Protocol):
    def __call__(
        self, image: np.ndarray, resolution: Resolution
    ) -> list[BboxRel]: ...
```
Structural (Protocol) typing so any callable with this signature qualifies —
no inheritance required.

### `stub_localize(image, resolution) -> list[BboxRel]` — line 82
```python
def stub_localize(image: np.ndarray, resolution: Resolution) -> list[BboxRel]:
```
Returns 3 hard-coded approximate card positions.  Ignores `image` entirely.
Must be replaced by the real classical localizer before any production use.

**Stub boxes (line 113):**
```python
[
    (0.08, 0.30, 0.18, 0.30),   # left-side card slot
    (0.40, 0.05, 0.18, 0.30),   # top-centre card slot
    (0.72, 0.30, 0.18, 0.30),   # right-side card slot
]
```

### `localize(image, resolution) -> list[BboxRel]` — line 121
```python
def localize(image: np.ndarray, resolution: Resolution) -> list[BboxRel]:
```
Named dispatch entry point.  Currently calls `stub_localize`.  Update this
function (not its callers) when the real localizer is ready.

---

## Replacement guide

The real classical localizer (research/RESEARCH.md entry 2) will:
1. Detect art-independent UI landmarks via contour/edge/HoughLines.
2. Locate numbered position badges (template match on UI chrome, not card art).
3. Derive card slot boxes relative to those landmarks, scaled by
   `resolution.w` / `resolution.h`.

Pass it as `localizer=my_real_localizer` to `run_pipeline`, or update
`localize()` to call it directly.  The `LocalizerCallable` Protocol is the
contract it must satisfy.
