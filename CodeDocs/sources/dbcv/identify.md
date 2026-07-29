# CodeDocs/sources/dbcv/identify.py

**Status:** Stage 2 identification — classical baseline + embedding-NN — updated 2026-06-22.
Both classifiers live here. Classical is retained as a selectable baseline.
Embedding-NN (`classify_crop_embedding`) is the **adopted default** identifier wired
into the API lifespan via `make_embedding_identifier`. (Stage 3 = OCR, not built yet.)
**2026-07-29:** a third, opt-in ensemble identifier (`combine_identifications` /
`make_ensemble_identifier`) was added — plans/PLAN-live-capture.md Fix 3.

**Purpose:** Given a localized card crop, identifies the townee, role class,
and confidence. Three implementations:
- Classical: 2-D HSV histogram correlation + ORB tiebreaker.
- Embedding-NN: ONNX cosine nearest-neighbor over prototype embeddings, using the
  **domain-fine-tuned** backbone (Proxy-Anchor LP-FT, 2026-06-22).
- Ensemble (opt-in): a pure composition layer over the two above — see below.

**Who uses it:**
- `dbcv/pipeline.py` — imports `identify` as the legacy default `identifier` argument
- `dbcv/api.py` — lifespan calls `make_embedding_identifier(embedder, embed_gallery)` for the
  default, or `make_ensemble_identifier(...)` when `Settings.identifier == "ensemble"`;
  `make_gallery_identifier(gallery)` kept as the classical fallback/baseline
- `tests/test_identify.py` — tests classical + ensemble functions
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
cards. The embedding-NN upgrade (still Stage 2 — identification) is warranted: it learns to be
invariant to the crop-vs-reference gap from training examples. A frozen ImageNet backbone alone
was *not* enough (it over-identified); the adopted fix was to domain-fine-tune it (see below).

---

---

## Embedding-NN identifier (Stage 2, adopted default)

### `classify_crop_embedding(card_crop, embedder, embed_gallery) -> (identity, role_class, confidence)` (line 419)
```python
def classify_crop_embedding(
    card_crop: np.ndarray,
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
) -> tuple[str, str, float]:
```
Cosine nearest-neighbor over the embedding gallery, using the **fine-tuned** backbone.

**Algorithm:**
1. `embedder.embed(card_crop)` → [576] unit-norm vector.
2. `embed_gallery.embeddings @ crop_vec` → [K] cosine similarity scores.
3. Take the two nearest prototypes (top-1 and top-2).
4. `confidence` = the **top1−top2 cosine margin**, clipped to [0, 1].
5. Return "unknown" if that margin is below `_EMBED_MARGIN_THRESHOLD` (0.12).

**Why a margin, not an absolute cosine (changed 2026-06-22).** The served backbone is
domain-fine-tuned (Proxy-Anchor LP-FT), which **compressed the absolute cosine scale** — a
correct match now sits ~0.6 and an unrelated prototype ~0.4 — so the since-removed absolute
cutoff (`_EMBED_CONFIDENCE_THRESHOLD = 0.60`; no longer in the code — do not grep for it)
no longer separated matches from non-matches and
over-identified **125/125** real cards. The top1−top2 margin does separate them: a decisive
match pulls clearly ahead of the runner-up, while a face-down / ambiguous crop sits roughly
equidistant from several prototypes (small margin) → "unknown". The returned `confidence`
field **is** that margin now (decisiveness), not the old `(cos+1)/2` remap. See
`research/RESEARCH.md` (2026-06-22 entry).

**Margin calibration (provisional):** confident real-frame cards score margin ≥ ~0.11;
ambiguous / face-down crops < ~0.06. `_EMBED_MARGIN_THRESHOLD = 0.12` was calibrated on
round-1 real-frame margins (`scrap_scripts/python/11_ft_abstain_probe.py`); refine once
labelled board crops and face-down samples exist.

**Adopted real-frame behaviour:** the fine-tuned embedding identifier confidently identifies
**30/125 (24%)** cards and abstains on 95; the classical baseline identifies 44/125 (35%);
classical↔embedding agreement rose 27 → 90 after the fine-tune. The synthetic retrieval eval
is optimistic (augmented reference art, not real crops); real-frame generalisation beyond the
confident few is the open round-2 lever.

### `make_embedding_identifier(embedder, embed_gallery) -> Callable` (line 500)
Bridges `classify_crop_embedding` into the 1-arg pipeline interface.
Used by the API lifespan to make the default identifier.

### Constants (embedding identifier)

| Constant | Value | Purpose |
|----------|-------|---------|
| `_EMBED_MARGIN_THRESHOLD` | 0.12 | Minimum top1−top2 cosine margin to report a match (provisional); below this → "unknown" |

---

## Ensemble identifier (2026-07-29, plans/PLAN-live-capture.md Fix 3)

Motivated by `eval_02`'s complementarity finding: an IoU-matched pass over
917 card-slot pairs (classical vs embedding on the same crop) found only
78 agreements and 20 disagreements, but **334** cases where exactly one
identifier abstained and the other answered — the biggest lever by far.

### `combine_identifications(classical, embedding) -> (identity, role_class, confidence, source)` (line 556)
```python
def combine_identifications(
    classical: tuple[str, str, float],
    embedding: tuple[str, str, float],
) -> tuple[str, str, float, EnsembleSource]:
```
Pure function, no gallery/embedder needed — the tested core (see
`tests/test_identify.py`'s stub-output tests). Combination rules, in order:

| Case | Result | `source` tag |
|------|--------|---------------|
| both "unknown" | `("unknown", "unknown", 0.0)` | `"both_unknown"` |
| same non-"unknown" identity on both | that identity; `confidence = min(1.0, max(c_conf, e_conf) + 0.15)` | `"agree"` |
| classical "unknown", embedding answers | embedding's result, unchanged | `"embedding_only"` |
| embedding "unknown", classical answers | classical's result, unchanged | `"classical_only"` |
| different non-"unknown" identities | `("unknown", "unknown", 0.0)` — **abstain, not "prefer higher confidence"** | `"disagree_abstain"` |

**Why disagreement abstains rather than picking the higher raw confidence:**
every disagreement case recorded in `eval_02` has classical's raw confidence
(0.40-0.90 histogram-correlation scale) numerically exceeding embedding's raw
confidence (0.12-0.40 top1-top2-margin scale) — yet the concrete example
(`collect_02/018.png`: classical `Poisoner@0.47`, embedding `Hunter@0.20`,
visually confirmed ground truth `Hunter`) shows classical is the wrong one.
The two confidence scales are not calibrated against each other and there is
no labeled live-crop set to fit a fair normalization without calibrating on
the same frames being evaluated (same reason `_EMBED_MARGIN_THRESHOLD`
recalibration — Fix 4 — is deferred).

### `make_ensemble_identifier(classical_identifier, embedding_identifier) -> Callable` (line 635)
Bridges `combine_identifications` into the 1-arg pipeline interface — calls
both identifiers, combines, and **drops the `source` tag** (the pipeline's
identifier contract and `CardRead` schema only carry 3-tuple
identity/role_class/confidence; no provenance field exists yet — noted as an
open item in plans/PLAN-live-capture.md rather than forcing a schema bump
alongside two other fixes).

### Constants (ensemble)

| Constant | Value | Purpose |
|----------|-------|---------|
| `_ENSEMBLE_AGREEMENT_BOOST` | 0.15 | Confidence bonus added when both identifiers agree |

**Wired as opt-in, not default:** `Settings.identifier` (dbcv/config.py,
`DBCV_IDENTIFIER` env var) selects `"embedding"` (default, unchanged),
`"classical"`, or `"ensemble"` in `api.py`'s lifespan.

---

## Pipeline wiring

`pipeline.py` `run_pipeline` still has `identifier=identify` (stub) as its default,
so bare calls (without gallery) behave as before.

`api.py` lifespan now calls `make_embedding_identifier(embedder, embed_gallery)`
and stores it on `app.state.identifier` as the **default identifier** (the adopted
fine-tuned embedding-NN).
The classical identifier is stored on `app.state.classical_identifier` as a fallback/baseline.

If the ONNX file is absent at startup, the lifespan warns and falls back to the
classical identifier automatically — the server still starts and serves requests.
