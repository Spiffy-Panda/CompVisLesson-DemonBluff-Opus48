# CodeDocs/sources/dbcv/localize.py

**Status:** integrated / validated — classical localizer promoted from spike.

**Purpose:** Defines the localizer interface, the stub teaching baseline, and
the classical implementation validated on real Demon Bluff sample frames.

**Who uses it:**
- `dbcv/pipeline.py` — imports `classical_localize` as the default `localizer`
  argument, and imports `localize` as the named single-dispatch entry point
- `dbcv/__init__.py` — re-exported in public-surface documentation
- `tests/test_api.py` — imports `classical_localize` directly for the unit test

---

## Key signatures (with line numbers)

### Type alias `BboxRel` — line 62
```python
BboxRel = tuple[float, float, float, float]
# (x, y, w, h) in [0, 1], relative to frame width/height, origin top-left
```

### `class LocalizerCallable(Protocol)` — line 72
```python
class LocalizerCallable(Protocol):
    def __call__(
        self, image: np.ndarray, resolution: Resolution
    ) -> list[BboxRel]: ...
```
Structural (Protocol) typing so any callable with this signature qualifies —
no inheritance required.

### `stub_localize(image, resolution) -> list[BboxRel]` — line ~110
```python
def stub_localize(image: np.ndarray, resolution: Resolution) -> list[BboxRel]:
```
**TEACHING BASELINE ("the before").**  Returns 3 hard-coded approximate card
positions.  Ignores `image` entirely.  Retained so the lesson plan can show the
delta between "no vision" and "real detection".  Pass explicitly as
`localizer=stub_localize` to `run_pipeline` to use it.

**Stub boxes:**
```python
[
    (0.08, 0.30, 0.18, 0.30),   # left-side card slot (approximate)
    (0.40, 0.05, 0.18, 0.30),   # top-centre card slot (approximate)
    (0.72, 0.30, 0.18, 0.30),   # right-side card slot (approximate)
]
```

### `classical_localize(image, resolution) -> list[BboxRel]` — line ~150
```python
def classical_localize(image: np.ndarray, resolution: Resolution) -> list[BboxRel]:
```
**Validated implementation.**  Five-stage classical pipeline, all CPU, ~10 ms:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | HUD-strip zeroing (relative fractions) | Exclude objective bar, score panels, name labels, edge icons |
| 2 | HSV colour segmentation | Mask purple / orange / red / bright-saturated card pixels |
| 3 | Morphological close + open (size ∝ min(w,h)) | Bridge intra-card gaps; remove speckle noise |
| 4 | Contour → bbox, filtered by area / aspect / HUD zone overlap | Reject non-card contours |
| 5 | Greedy IoU-NMS (threshold 0.30, largest-first) | One box per card |

**Validation results (spike, 2026-06-21):**
- Sample1 board frames: 8/8 cards detected, 0 false positives.
- Sample2 board frames: 9/9 cards detected, 0 false positives.
- Modal frames: correctly returns ~0–2 boxes (no false board parse).

**Art-swap note:** HSV thresholds are tuned to the current art palette.  On an
art swap, re-tune these ranges (15–30 min with an HSV visualiser); morphology
sizes and contour-filter ratios are geometry-derived and art-independent.

**Research grounding:** research/RESEARCH.md entry 2 (Card/region localization
robust to art swaps — 2026-06-21).

### `localize` — module-level alias (bottom of file)
```python
localize = classical_localize
```
Named dispatch entry point.  **Now points at `classical_localize`** (previously
pointed at `stub_localize`).  Import `localize` for the current best-available
implementation; import `stub_localize` explicitly to use the teaching baseline.

---

## Teaching baseline vs. production comparison

| | `stub_localize` | `classical_localize` |
|---|---|---|
| Vision? | No | Yes — 5-stage OpenCV pipeline |
| Cards found | Always 3 (hardcoded) | ~8 on a real board frame |
| Frame-dependent? | No | Yes |
| Art-swap safe? | N/A | Re-tune HSV (~15–30 min) |
| CPU cost | < 0.1 ms | ~10 ms |
| Use in production | No — teaching baseline only | Yes |

---

## TODO (future work)

- `classify_frame_state` (board-vs-modal gate): spiked but only 0/3 reliable on
  modal frames at time of spike.  Held back pending more modal sample coverage.
  See scrap_scripts/ for the spike.  This is Stage-0 work.
