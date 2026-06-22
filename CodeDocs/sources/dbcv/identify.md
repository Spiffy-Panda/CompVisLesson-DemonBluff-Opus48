# CodeDocs/sources/dbcv/identify.py

**Status:** Stage 2 (classical) + Stage 3 (embedding-NN) — updated 2026-06-22.
Both classifiers live here. Classical is retained as a selectable baseline.
Embedding-NN (`classify_crop_embedding`) is the new default identifier wired
into the API lifespan via `make_embedding_identifier`.

**Purpose:** Given a localized card crop, identifies the townee, role class,
and confidence. Two implementations:
- Classical: 2-D HSV histogram correlation + ORB tiebreaker.
- Embedding-NN: ONNX cosine nearest-neighbor over prototype embeddings (Stage 3).

**Who uses it:**
- `dbcv/pipeline.py` — imports `identify` as the legacy default `identifier` argument
- `dbcv/api.py` — lifespan calls `make_embedding_identifier(embedder, embed_gallery)` for the default;
  `make_gallery_identifier(gallery)` kept as the classical fallback/baseline
- `tests/test_identify.py` — tests classical functions
- `tests/test_embed.py` — tests embedding functions

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

---

## Stage 3 additions — Embedding-NN identifier

### `classify_crop_embedding(card_crop, embedder, embed_gallery) -> (identity, role_class, confidence)` (line ~400)
```python
def classify_crop_embedding(
    card_crop: np.ndarray,
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
) -> tuple[str, str, float]:
```
Cosine nearest-neighbor over the embedding gallery.

**Algorithm:**
1. `embedder.embed(card_crop)` → [576] unit-norm vector.
2. `embed_gallery.embeddings @ crop_vec` → [K] cosine similarity scores.
3. Pick `argmax` → nearest prototype.
4. Map cosine [-1,1] → confidence [0,1]: `(cosine + 1) / 2`.
5. Return "unknown" if confidence < `_EMBED_CONFIDENCE_THRESHOLD` (0.60).

**Threshold calibration:** Blank/face-down crops embed to cosine ≈ -0.23
(confidence ≈ 0.38). Face-up board crops score 0.70-0.91 confidence.
Threshold of 0.60 sits cleanly in the gap, rejecting degenerate inputs
without suppressing real character crops.

**Known limitation:** MobileNetV3-Small (ImageNet-pretrained, frozen)
maps all cartoon game characters to a tight cluster in embedding space
(inter-prototype cosines 0.65-0.94). The nearest-neighbor result is
determined by small within-cluster differences; some characters with
similar colour/texture statistics are confused. The model is a useful
first baseline and correctly separates face-down from face-up; further
improvement would require fine-tuning on labelled board crops.

### `make_embedding_identifier(embedder, embed_gallery) -> Callable` (line ~480)
Bridges `classify_crop_embedding` into the 1-arg pipeline interface.
Used by the API lifespan to make the default identifier.

### Constants (Stage 3)

| Constant | Value | Purpose |
|----------|-------|---------|
| `_EMBED_CONFIDENCE_THRESHOLD` | 0.60 | Minimum cosine→conf to report a match |

---

## Pipeline wiring

`pipeline.py` `run_pipeline` still has `identifier=identify` (stub) as its default,
so bare calls (without gallery) behave as before.

`api.py` lifespan (Stage 3) now calls `make_embedding_identifier(embedder, embed_gallery)`
and stores it on `app.state.identifier` as the **default identifier**.
The classical identifier is stored on `app.state.classical_identifier` as a fallback/baseline.

If the ONNX file is absent at startup, the lifespan warns and falls back to the
classical identifier automatically — the server still starts and serves requests.
