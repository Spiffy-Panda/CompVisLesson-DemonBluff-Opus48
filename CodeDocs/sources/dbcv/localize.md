# CodeDocs/sources/dbcv/localize.py

**Status:** integrated / validated — classical localizer promoted from spike;
HUD-zone fix landed 2026-07-29 (plans/PLAN-live-capture.md, Fix 1) from the
two live-frame evals.

**Purpose:** Defines the localizer interface, the stub teaching baseline, and
the classical implementation validated on real Demon Bluff sample frames.

**Who uses it:**
- `dbcv/pipeline.py` — imports `BboxRel`, `LocalizerCallable`, and
  `classical_localize` (the default `localizer` argument)
- `dbcv/__init__.py` — listed in the public-surface docstring
  (`classical_localize`, `stub_localize`)
- `tests/test_api.py` — imports `classical_localize` directly for the unit test

---

## Key signatures (with line numbers)

### Type alias `BboxRel` — line 74
```python
BboxRel = tuple[float, float, float, float]
# (x, y, w, h) in [0, 1], relative to frame width/height, origin top-left
```

### `class LocalizerCallable(Protocol)` — line 83
```python
class LocalizerCallable(Protocol):
    def __call__(
        self, image: np.ndarray, resolution: Resolution
    ) -> list[BboxRel]: ...
```
Structural (Protocol) typing so any callable with this signature qualifies —
no inheritance required.

### `stub_localize(image, resolution) -> list[BboxRel]` — line 117
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

### `classical_localize(image, resolution) -> list[BboxRel]` — line 188
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

---

## HUD-zone fix (2026-07-29, plans/PLAN-live-capture.md Fix 1)

Two live-frame evals (`collect_01`/`eval_01`, `collect_02`/`eval_02`) found the
original HUD strips (top 9% full-width, left 13% full-height) too thin to
cover two real HUD elements. Both are added as **corner-only** zones — not a
widened full-width band, which was checked and found to clip the top-center
card slot (its top edge can sit as high as y≈0.076).

`HUD_ZONES` (inside `classical_localize`, both Stage 1 pixel-zeroing and
Stage 4 bbox-overlap rejection) gained two entries:

| Zone | Fractions (x, y, w, h) | Covers |
|------|------------------------|--------|
| top-left objective-text block | `(0.00, 0.00, 0.27, 0.27)` | Title + minion/demon counts + Evils-killed/Village/Ascension/Score lines — the "Hunter@0.42-0.50" FP that fired on 73-83% of live board frames before this fix |
| top-right revealed-evils badge strip | `(0.86, 0.00, 0.14, 0.36)` | Small character-art thumbnails of already-executed evils — the second-largest live FP cluster (55 confident hits in eval_02, up to 0.94 confidence) |

Both zones were checked against every confident real-card detection in both
evals; no real card intrudes on either zone (left-column cards start x≥0.29;
right-column cards stay ≤0.76 outside Kill-Mode-tinted frames).

**Investigated, no zone added — "Poisoner@0.42-0.44" recurring FP.** Traced
to a genuine, moving board card (a revealed Hunter card the classical
identifier confuses for Poisoner), not a fixed background element. A HUD
zone here would create false negatives on real cards; left as a known
classical-identifier weakness (see `identify.py`'s ensemble section).

**Kill-Mode red-tint.** The localizer-side false positive this causes (a
spurious box around the center demon-altar decoration, from the red HSV mask
firing on non-card red content) is identified but **not fixed** in
`localize.py` this wave — see `frame_state.py`'s `is_red_tint` /
`measure_red_shift` and `pipeline.py`'s tint-discount wiring, which address
the *identification*-side reliability drop instead.

Tests: `tests/test_localize.py` (synthetic-frame HUD-zone masking + recall
regression checks, resolution-agnostic).

### Removed: the `localize` module-level alias
The former `localize = classical_localize` alias (bottom of file) was removed
2026-07-28 — it had zero call sites.  Import `classical_localize` for the
current best-available implementation; import `stub_localize` explicitly to
use the teaching baseline.

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
