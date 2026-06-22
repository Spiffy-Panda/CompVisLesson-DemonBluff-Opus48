# CodeDocs/sources/dbcv/identify.py

**Status:** Stage 2 — classical HSV histogram matcher + ORB tiebreaker.
Stub retained for teaching comparison. Gallery-aware `classify_crop` is the
real implementation; `identify()` still returns stub for backward compatibility.

**Purpose:** Given a localized card crop, identifies the townee, role class,
and confidence.  Primary method: 2-D HSV colour histogram correlation.
Tiebreaker: ORB feature match count when top-2 scores are within 0.05.

**Who uses it:**
- `dbcv/pipeline.py` — imports `identify` as the default `identifier` argument
- `dbcv/api.py` — calls `make_gallery_identifier(gallery)` in lifespan to get the real identifier
- `tests/test_identify.py` — tests all exported functions

---

## Key signatures

### `classify_crop(card_crop, gallery) -> (identity, role_class, confidence)`
```python
def classify_crop(
    card_crop: np.ndarray,
    gallery: Gallery,
) -> tuple[str, str, float]:
```
The real classical matcher.  Returns `("unknown", "unknown", low_conf)` when:
- The crop is zero-size or the gallery is empty.
- The crop histogram sums to zero (all pixels too dark/desaturated — guards
  against `cv2.compareHist(zeros, X) = 1.0` edge case).
- The best histogram score is below `_CONFIDENCE_THRESHOLD` (0.40).

**Algorithm:**
1. Extract the art sub-band (top 12%–62% of crop height, 8%–92% width).
2. Compute 2-D HSV histogram for the art band.
3. Zero-sum guard: if histogram sum < 1e-6 → return "unknown".
4. Score all gallery entries by Pearson histogram correlation.
5. If top-2 scores differ by < 0.05 → ORB tiebreaker (BFMatcher + ratio test).
6. Return winner identity/role_class + score as confidence.

**Confidence definition:**
`confidence = cv2.compareHist(HISTCMP_CORREL)` clamped to [0, 1].
Represents the Pearson correlation between the crop's and reference's
HSV histograms.  A score ≥ 0.40 signals a credible match.

### `make_gallery_identifier(gallery) -> Callable[[np.ndarray], tuple[str, str, float]]`
```python
def make_gallery_identifier(gallery: Gallery) -> Callable[...]:
```
Bridges `classify_crop` into the 1-argument pipeline interface.  Returns a
closure `fn(card_crop)` that calls `classify_crop(card_crop, gallery)`.

Usage:
```python
gallery = build_gallery()
identifier = make_gallery_identifier(gallery)
snapshot = run_pipeline(image, source, identifier=identifier)
```

### `stub_identify(card_crop) -> ("unknown", "unknown", 0.0)`
Always returns the triple above.  Preserved for teaching comparison ("before")
and for tests that need deterministic stub output.

### `identify(card_crop) -> ("unknown", "unknown", 0.0)`
Legacy name imported by `pipeline.py` as the default.  Delegates to
`stub_identify`.  The pipeline falls back to this when no gallery identifier
is passed (i.e., in bare `run_pipeline` calls without the gallery).

---

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `_CONFIDENCE_THRESHOLD` | 0.40 | Minimum histogram score to report a match |
| `_HIST_TIEBREAK_DELTA` | 0.05 | How close top-2 scores must be to trigger ORB |
| `_ORB_MIN_GOOD_MATCHES` | 4 | Minimum ORB inliers for tiebreaker to override |
| `_ART_BAND_TOP` | 0.12 | Top fraction of crop to skip (badge / corner) |
| `_ART_BAND_BOTTOM` | 0.62 | Bottom of art region (below: name label + text) |
| `_ART_BAND_LEFT` | 0.08 | Skip left border |
| `_ART_BAND_RIGHT` | 0.92 | Skip right border |

---

## Honest performance assessment (from scrap_scripts/python/07_identify_probe.py)

Evaluated on 3 board frames (25 card slots total).

**Key gap:** Most cards are face-down (showing a uniform card back pattern).
Face-down cards correctly return "unknown" with conf 0.02–0.39 (below threshold).

**Face-up identifications observed:**
- Frame 003: 1 card positively identified (Scout, conf=0.723)
- Frame 005: 1 card positively identified (Doppelganger, conf=0.461)
- Frame 008: 3 cards positively identified (Fortune_Teller 0.668, Scout 0.478, Wretch 0.650)
  plus 1 disputed (Scout conf=0.596 in a second slot — possible confusion or repeated char)

**Candid accuracy:** Classical matching can identify some face-up cards when
their colour palette is distinctive (warm pink = Fortune_Teller; green/brown =
Scout, etc.) but fails for many face-up cards because:
1. The board crop includes borders, labels, and tinting not in the reference.
2. Many game cards have similar brown/dark colour palettes that wash out
   the hue histogram signal.
3. The art sub-band heuristic (12%–62% of crop height) is approximate; the
   actual art region varies by card layout.

**Verdict:** Classical is a useful *lower bound* and correctly handles face-down
cards. The embedding-NN upgrade (Stage 3) is warranted: it would learn to be
invariant to the crop-vs-reference gap from training examples.

---

## Pipeline wiring

`pipeline.py` `run_pipeline` still has `identifier=identify` as its default,
so bare calls (without gallery) behave as before (stub).

`api.py` lifespan now calls `make_gallery_identifier(gallery)` and stores
the result on `app.state.identifier`, then passes it to `run_pipeline`.

The upgrade path (Stage 3 embedding-NN) fits here: swap `make_gallery_identifier`
with an embedding-based closure and the pipeline is unchanged.
