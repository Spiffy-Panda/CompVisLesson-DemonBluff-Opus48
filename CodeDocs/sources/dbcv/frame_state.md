# CodeDocs/sources/dbcv/frame_state.md

**Status:** active — Stage 0 frame-state gate, implemented and tested.
**2026-07-29:** added `measure_red_shift`/`is_red_tint`, a Kill-Mode
red-tint detector (plans/PLAN-live-capture.md Fix 1).

**Purpose:** Classifies a decoded game frame as one of `"board"`, `"modal"`,
or `"menu"` using a classical, CPU-only signal.  The pipeline calls this
function first; non-board frames skip localisation entirely (returning `cards=[]`).
Also provides a separate, independent whole-frame tint signal (see below).

**Who uses it:**
- `dbcv/pipeline.py` — calls `classify_frame_state` as the first step in
  `run_pipeline`; result is stored in `snapshot.frame_state`. Also imports
  `is_red_tint` as the default `tint_fn` for the Kill-Mode confidence discount.
- `tests/test_frame_state.py` — unit-tests the gate on the labeled frame set
  and the red-tint detector on synthetic frames; also integration-tests the
  full pipeline via the API

---

## Key signatures (with line numbers)

### `FrameState` type alias — line 81
```python
FrameState = Literal["board", "modal", "menu"]
```
Used as the return type of `classify_frame_state` and as the type of the
`frame_state_fn` parameter injected into `run_pipeline`.

### `classify_frame_state(image) -> FrameState` — line 120
```python
def classify_frame_state(image: np.ndarray) -> FrameState:
```
**Parameters:**
- `image` — decoded BGR frame (H x W x C numpy array).  Also accepts
  grayscale (H x W) via a fallback path.  Resolution-agnostic; all geometry
  is relative to `image.shape[:2]`.

**Returns:** one of `"board"`, `"modal"`, or `"menu"`.

**Algorithm (three stages):**
1. Menu check (line ~180): if `mean(gray) >= 160`, return `"menu"`.
2. Center-vs-ring ratio (line ~222): compute mean brightness in the inner
   center box (30–70 % x 30–70 %) and in the surrounding ring (10–90 %
   minus the center).  If `center / ring >= 2.0`, return `"modal"`.
3. Default (line ~227): return `"board"`.

---

## Why center-vs-ring ratio (and why brightness alone fails)

The naive approach (mean brightness of the center 40 % × 40 %) scored 0/3
on modal frames because **Demon Bluff modals have a dark outer background**.
The modal dialog is bright, but surrounded by the same dark star-field seen
in non-modal frames.  The overall center mean is only mildly elevated.

The center-vs-ring ratio removes the absolute brightness dependence:

| Label   | Frame          | Ratio  |
|---------|---------------|--------|
| board   | Sample1_003   | 1.063  |
| board   | Sample1_018   | 1.021  |
| board   | Sample2_012   | 1.109  |
| board   | Sample2_023   | 1.097  |
| modal   | Sample1_000   | 3.099  |
| modal   | Sample2_000   | 5.878  |
| modal   | Sample2_006   | 4.121  |
| partial | Sample1_006   | 0.943  |

Threshold: **2.0** (midpoint of the ~3x gap: board max 1.11, modal min 3.10).

---

## Threshold constants (all at module level, lines 89–112)

| Constant | Value | Purpose |
|----------|-------|---------|
| `_CENTER_Y0/_Y1` | 0.30 / 0.70 | Inner box vertical bounds (fraction of H) |
| `_CENTER_X0/_X1` | 0.30 / 0.70 | Inner box horizontal bounds (fraction of W) |
| `_RING_Y0/_Y1` | 0.10 / 0.90 | Outer ring vertical bounds |
| `_RING_X0/_X1` | 0.10 / 0.90 | Outer ring horizontal bounds |
| `_MODAL_RATIO_THRESHOLD` | 2.0 | center/ring ratio above which frame → "modal" |
| `_MENU_BRIGHTNESS_THRESHOLD` | 160.0 | Whole-frame mean above which frame → "menu" |

All geometry is in fractional units — no pixel value is hard-coded.

---

## Partial-modal handling

`Sample1_006` (a "Pick 3 characters" dialog with peripheral cards visible)
scores 0.94 on the ratio — below the 2.0 threshold — and is therefore
classified as `"board"`.  This is the correct production decision: the
localizer CAN still find the peripheral cards.  The decision is documented
and tested in `tests/test_frame_state.py::test_partial_modal_documented_handling`.

---

## Design notes

- The gate is injectable in the pipeline (`frame_state_fn` parameter) so tests
  can stub it out to decouple from localiser tests.
- The gate is resolution-agnostic: all regions are expressed as fractions of
  `image.shape[:2]` measured at call time.
- Adding a new state class (e.g. "voting_screen") only requires adding a new
  branch here and extending the `FrameState` Literal in `schema.py`.

---

## Kill-Mode red-tint detection (2026-07-29, plans/PLAN-live-capture.md Fix 1)

### `measure_red_shift(image) -> float` — line 263
```python
def measure_red_shift(image: np.ndarray) -> float:
```
Returns `mean(R - max(G, B))` over every pixel (BGR channel order). Not a
gate itself — a continuous signal for `is_red_tint` and any future caller
that wants the raw score.

### `is_red_tint(image, threshold=_RED_TINT_THRESHOLD) -> bool` — line 292
```python
def is_red_tint(image: np.ndarray, threshold: float = _RED_TINT_THRESHOLD) -> bool:
```
Thin wrapper: `measure_red_shift(image) >= threshold`. Passed as the default
`tint_fn` to `dbcv.pipeline.run_pipeline`, which discounts and re-abstains
low-confidence identifications while active (see `pipeline.md`).

**Calibration:** measured on 147 frames across `collect_01` + `collect_02`
(2026-07-29 live evals) — tinted frames scored 14.4-32.2; every other frame
scored -16.9 to +0.5. `_RED_TINT_THRESHOLD = 10.0` (line 140) sits in the
middle of that gap with no observed overlap.

**Scope:** this only addresses the *identification*-reliability drop under
Kill-Mode tint. A separate, unfixed issue — the localizer's red HSV mask
firing on the center demon-altar decoration under tint, producing a spurious
card box — is documented but not fixed this wave (see `localize.md`).

Tests: `tests/test_frame_state.py` (synthetic frames; neutral/green/red
cases + threshold override).
