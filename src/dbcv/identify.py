"""
dbcv/identify.py — Card-identification interface, stub, and classical matcher.

Given a cropped card image (already localized), the identifier returns the
townee name, role class, and a confidence score in [0, 1].

Interface
---------
    classify_crop(card_crop, gallery) -> (identity, role_class, confidence)

    - identity   : townee name (e.g. "Alchemist") or "unknown"
    - role_class : one of "villager" | "minion" | "outcast" | "demon" | "unknown"
    - confidence : float in [0.0, 1.0]

    The legacy zero-argument ``identify(card_crop)`` stub is preserved below for
    backward compatibility with existing tests.  Pipeline.py now accepts an
    ``identifier`` callable that may close over a Gallery (see
    ``make_gallery_identifier`` below).

Key design choices and their teaching rationale
-----------------------------------------------
**Method chosen: HSV colour histogram correlation (primary), with ORB match
count as a tiebreaker when the top-2 histogram scores are within 0.05 of each
other.**

Why colour histograms over template NCC?
  The reference art is a clean cartoon illustration with a white/transparent
  background.  A board crop contains: the card frame/border, a numbered badge,
  the name label, possible clue/ability text, and state-tinting applied by the
  game.  Many cards are also FACE-DOWN (uniform card-back pattern, no character
  art) — those should return "unknown".  Pixel-level NCC requires accurate
  spatial alignment between reference and crop; without knowing the exact crop
  geometry relative to the art sub-region, NCC is too brittle.

  Colour histograms are translation- and moderate-scale-invariant: the
  distribution of hues in a crop should roughly match the distribution in the
  reference even when cropped slightly differently.

Why add ORB as a tiebreaker?
  For townees whose colour palettes are similar (e.g., two characters that are
  both predominantly brown + beige), histogram correlation alone may fail to
  discriminate.  ORB feature point matching adds a structural signal when the
  top histogram scores are close.

Why is this a CLASSICAL baseline, not the final approach?
  The fundamental gap: the reference gallery contains clean, full-character
  illustrations.  Board crops are partial, bordered, tinted, and sometimes
  occluded.  Classical similarity metrics can only approximate the art sub-region
  heuristically (we crop the upper-central portion of the board crop).  An
  embedding NN (MobileNetV3-Small with cosine nearest-neighbour lookup) learns to
  be invariant to the crop-vs-reference gap.  That is now the SERVED default
  (domain-fine-tuned, 2026-06-22; see classify_crop_embedding below); this
  classical matcher is kept as the honest baseline + fallback.

  This baseline measures *how far* classical gets — honestly — so the lesson plan
  can quantify the gap and motivate the (now-shipped) embedding upgrade.

Confidence definition
---------------------
  confidence = histogram_correlation_score (Pearson-like, from cv2.HISTCMP_CORREL)
               in [0, 1], where 1.0 = identical histograms.

  A score below _CONFIDENCE_THRESHOLD (0.40) is treated as "no match" and
  returns ("unknown", "unknown", low_score).  Face-down cards have uniform
  brown/pattern histograms that fail to match any character; they correctly
  return low confidence and "unknown".

Art-region heuristic
--------------------
  Board crops are roughly:
    - Top ~15-25%:  rounded card top / small badge area
    - Middle ~40%:  character art (what we want to match)
    - Bottom ~30%:  name label + ability/clue text + role-colour band
  We sample the upper-middle 40% of the crop height, centred horizontally,
  as the "art sub-region" for matching.  This is an approximation; the
  embedding-NN approach would learn the correct mapping.

Research grounding
------------------
- HSV histogram comparison via cv2.compareHist (HISTCMP_CORREL):
  correlation is bounded [0,1] for non-negative histograms; robust to
  brightness and moderate geometric variation.  (OpenCV docs, compareHist.)
- ORB for coarse structural tiebreaker: Rublee et al. (2011), implemented
  in OpenCV without licence restrictions.
- "Why classical now, NN later": see research/RESEARCH.md entry 3 for the
  deferred embedding-NN approach motivation.

Ensemble identifier (2026-07-29, plans/PLAN-live-capture.md)
--------------------------------------------------------------
A third identifier, ``combine_identifications`` / ``make_ensemble_identifier``
(bottom of this module), composes ``classify_crop`` and
``classify_crop_embedding`` without changing either.  It is opt-in (not the
default) and motivated directly by the collect_02 live eval's finding that
the two identifiers catch different cards more often than they agree or
both miss -- see the ensemble section's own docstring for the combination
rules and the eval evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

if TYPE_CHECKING:
    from dbcv.gallery import Gallery

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Minimum histogram correlation score to report a match.
# Below this, the crop is too different from any reference → "unknown".
# Face-down cards score ~0.10–0.20 (uniform brown pattern vs. character art).
_CONFIDENCE_THRESHOLD: float = 0.40

# How much better the top match must be vs. the runner-up before we skip ORB.
# If top_score - second_score < this, we run ORB and let it break the tie.
_HIST_TIEBREAK_DELTA: float = 0.05

# Minimum ORB inlier ratio to accept the ORB tiebreaker suggestion.
# If ORB finds no reliable matches at all, we fall back to histogram winner.
_ORB_MIN_GOOD_MATCHES: int = 4

# Art sub-region: which vertical fraction of the board crop to use for matching.
# Values are relative to the crop height.
# Top 15% is the card frame top / badge; bottom 40% is name label + text.
# We sample the middle band.
_ART_BAND_TOP: float = 0.12      # skip top 12% (badge / corner rounding)
_ART_BAND_BOTTOM: float = 0.62   # use down to 62% of crop height
_ART_BAND_LEFT: float = 0.08     # skip narrow side borders
_ART_BAND_RIGHT: float = 0.92    # up to 92% width


# ---------------------------------------------------------------------------
# Art-region extraction
# ---------------------------------------------------------------------------


def _extract_art_band(card_crop: np.ndarray) -> np.ndarray:
    """Extract the character-art sub-region from a board card crop.

    Board crops contain card frame, badge, character art, name label, and
    ability text.  This function returns the middle band that most likely
    contains the character illustration.

    Parameters
    ----------
    card_crop:
        BGR numpy array (H×W×C) — the full localized card crop.

    Returns
    -------
    np.ndarray
        The art sub-region, or the whole crop if the crop is too small.
    """
    if card_crop.size == 0:
        return card_crop

    h, w = card_crop.shape[:2]
    y0 = max(0, int(h * _ART_BAND_TOP))
    y1 = min(h, int(h * _ART_BAND_BOTTOM))
    x0 = max(0, int(w * _ART_BAND_LEFT))
    x1 = min(w, int(w * _ART_BAND_RIGHT))

    if y1 <= y0 or x1 <= x0:
        return card_crop   # crop too small; use whole thing

    return card_crop[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Histogram-based matching
# ---------------------------------------------------------------------------


def _hist_score(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Return Pearson-correlation between two L1-normalised HSV histograms.

    cv2.HISTCMP_CORREL returns values in [-1, 1]; we clamp to [0, 1]
    because negative correlation has no useful meaning for our matching task
    (two completely different characters) — we treat anything <= 0 as 0.0.

    Important edge case: cv2.compareHist(zeros, anything) = 1.0 due to a
    division-by-zero in the Pearson formula (both means are zero).  Callers
    must guard against a zero-sum histogram before calling this function;
    see ``classify_crop`` for the explicit zero-sum check.
    """
    raw = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return float(np.clip(raw, 0.0, 1.0))


def _compute_crop_hist(art_band: np.ndarray) -> np.ndarray:
    """Compute the HSV colour histogram for the extracted art band.

    Reuses the same procedure as gallery.py so the histograms are comparable.
    """
    # Import here to avoid a circular dependency at module level.
    from dbcv.gallery import _compute_hsv_hist  # noqa: PLC0415
    return _compute_hsv_hist(art_band)


# ---------------------------------------------------------------------------
# ORB tiebreaker
# ---------------------------------------------------------------------------


def _orb_match_count(
    crop_desc: np.ndarray | None,
    ref_desc: np.ndarray | None,
) -> int:
    """Return the number of 'good' ORB matches between two descriptor sets.

    Uses brute-force Hamming matching with a ratio test (Lowe's ratio = 0.75).
    Returns 0 if either descriptor set is None or has fewer than 2 descriptors.
    """
    if crop_desc is None or ref_desc is None:
        return 0
    if len(crop_desc) < 2 or len(ref_desc) < 2:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(crop_desc, ref_desc, k=2)

    good = 0
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good += 1

    return good


# ---------------------------------------------------------------------------
# Public: classical classifier
# ---------------------------------------------------------------------------


def classify_crop(
    card_crop: np.ndarray,
    gallery: "Gallery",
) -> tuple[str, str, float]:
    """Classify a localized card crop against the reference gallery.

    Primary method: HSV colour histogram correlation.
    Tiebreaker: ORB feature match count when the top two histogram scores
    are within ``_HIST_TIEBREAK_DELTA`` (0.05) of each other.

    Parameters
    ----------
    card_crop:
        BGR numpy array (H×W×C) from the pipeline's crop_relative call.
    gallery:
        Pre-built Gallery object (from ``build_gallery()``).

    Returns
    -------
    (identity, role_class, confidence)
        identity    — townee name, or "unknown" if confidence < threshold
        role_class  — role family, or "unknown"
        confidence  — float in [0.0, 1.0] (the best histogram score found)

    Face-down cards
    ---------------
    Face-down cards show a uniform card-back pattern.  Their HSV histograms
    are dominated by brown/orange hues from the card back, which do not match
    any character's art histogram well.  They correctly score below the
    threshold and return ("unknown", "unknown", low_confidence).
    """
    if card_crop.size == 0 or len(gallery.entries) == 0:
        return ("unknown", "unknown", 0.0)

    # Extract the art sub-region from the board crop
    art_band = _extract_art_band(card_crop)

    # Compute the crop's colour histogram
    crop_hist = _compute_crop_hist(art_band)

    # Guard: if the crop histogram sums to zero (all pixels were black/white
    # background, or the crop is too dark/desaturated to have any colour signal),
    # cv2.compareHist(zeros, anything) = 1.0 due to a division-by-zero in the
    # Pearson correlation formula.  A structureless crop (uniform card back,
    # black frame, dark tinting) should return "unknown", not a spurious match.
    if crop_hist.sum() < 1e-6:
        return ("unknown", "unknown", 0.0)

    # Compute ORB descriptors for the crop (used only if tiebreak needed)
    from dbcv.gallery import _compute_orb  # noqa: PLC0415
    _, crop_desc = _compute_orb(art_band)

    # Score every gallery entry
    scored: list[tuple[float, int]] = []   # (hist_score, entry_index)
    for i, entry in enumerate(gallery.entries):
        hs = _hist_score(crop_hist, entry.hsv_hist)
        scored.append((hs, i))

    # Sort descending by histogram score
    scored.sort(key=lambda t: t[0], reverse=True)

    best_score, best_idx = scored[0]

    # If the top score is below the confidence threshold → unknown
    if best_score < _CONFIDENCE_THRESHOLD:
        return ("unknown", "unknown", round(best_score, 4))

    # Check if a tiebreaker is needed
    if len(scored) >= 2:
        second_score, second_idx = scored[1]
        delta = best_score - second_score

        if delta < _HIST_TIEBREAK_DELTA:
            # Scores are very close — use ORB to break the tie
            best_entry = gallery.entries[best_idx]
            second_entry = gallery.entries[second_idx]

            orb_best = _orb_match_count(crop_desc, best_entry.orb_desc)
            orb_second = _orb_match_count(crop_desc, second_entry.orb_desc)

            if orb_second > orb_best and orb_second >= _ORB_MIN_GOOD_MATCHES:
                # ORB favours the runner-up — use it, but keep the hist score
                best_idx = second_idx
                best_score = second_score

    winner = gallery.entries[best_idx]
    return (winner.identity, winner.role_class, round(best_score, 4))


# ---------------------------------------------------------------------------
# Convenience factory: close a gallery into the legacy 1-arg signature
# ---------------------------------------------------------------------------


def make_gallery_identifier(
    gallery: "Gallery",
) -> Callable[[np.ndarray], tuple[str, str, float]]:
    """Return a single-argument callable that wraps ``classify_crop``.

    This bridges the Gallery-aware classifier into the existing pipeline
    interface (``identifier: Callable[[np.ndarray], tuple[str, str, float]]``).
    The returned function can be passed directly as ``run_pipeline(...,
    identifier=make_gallery_identifier(gallery))``.

    Parameters
    ----------
    gallery:
        Pre-built Gallery object from ``build_gallery()``.

    Returns
    -------
    Callable[[np.ndarray], tuple[str, str, float]]
        A closure that calls ``classify_crop(card_crop, gallery)``.

    Usage
    -----
    ::

        gallery = build_gallery()
        identifier = make_gallery_identifier(gallery)
        snapshot = run_pipeline(image, source, identifier=identifier)
    """
    def _identify(card_crop: np.ndarray) -> tuple[str, str, float]:
        return classify_crop(card_crop, gallery)

    return _identify


# ---------------------------------------------------------------------------
# Legacy stub — preserved for backward compatibility and teaching
# ---------------------------------------------------------------------------


def stub_identify(card_crop: np.ndarray) -> tuple[str, str, float]:
    """TEACHING BASELINE stub — always returns ("unknown", "unknown", 0.0).

    Preserved so that existing tests and lesson-plan discussions that pass
    ``identifier=stub_identify`` continue to work unchanged.

    Compare to ``classify_crop`` (the classical matcher) to see the full delta
    in complexity — and to understand what the classical approach achieves and
    where it falls short before the embedding-NN upgrade.
    """
    _ = card_crop   # explicitly discarded
    return ("unknown", "unknown", 0.0)


def identify(card_crop: np.ndarray) -> tuple[str, str, float]:
    """Legacy default identifier — now delegates to the stub.

    This name is imported by pipeline.py as the default ``identifier``
    argument to ``run_pipeline``.  It still returns the stub result so that
    the pipeline works when no gallery is supplied (the API lifespan will
    build the gallery and pass a real identifier).

    To use the classical matcher, call ``make_gallery_identifier(gallery)``
    and pass the result as the ``identifier`` argument to ``run_pipeline``.
    """
    return stub_identify(card_crop)


# ---------------------------------------------------------------------------
# Stage 2 — Embedding-NN identifier
# ---------------------------------------------------------------------------

# Abstention is decided by the top1-top2 cosine MARGIN, not an absolute cosine.
# The served backbone is domain-fine-tuned (Proxy-Anchor LP-FT, 2026-06-22), which
# compressed the absolute cosine scale (a correct match now sits ~0.6, an unrelated
# prototype ~0.4), so the old absolute threshold no longer separates matches from
# non-matches.  The *margin* to the runner-up is the honest signal: confident
# real-frame cards score margin >= ~0.11, ambiguous/face-down ones < ~0.06.
# Provisional value, calibrated on round-1 real-frame margins
# (scrap_scripts/python/11_ft_abstain_probe.py); refine once labeled board crops
# and face-down samples exist.  research/RESEARCH.md (2026-06-22) prescribes a
# margin criterion for exactly this reason.
_EMBED_MARGIN_THRESHOLD: float = 0.12


def classify_crop_embedding(
    card_crop: np.ndarray,
    embedder: "object",          # OnnxEmbedder — avoids circular import
    embed_gallery: "object",     # EmbeddingGallery
) -> tuple[str, str, float]:
    """Classify a card crop using embedding nearest-neighbour lookup.

    The approach:
      1. Embed the crop via the OnnxEmbedder (one ONNX forward pass).
      2. Compute cosine similarities against all gallery prototypes (one matmul).
      3. Take the two nearest prototypes (top-1 and top-2).
      4. confidence = the top1-top2 cosine MARGIN (clipped to [0, 1]).
      5. Return "unknown" if that margin is below _EMBED_MARGIN_THRESHOLD.

    Why cosine instead of Euclidean?
    ---------------------------------
    Both the crop embedding and gallery prototypes are L2-normalised, so
    cosine similarity = dot product -- cheap (O(K * D) single matmul) and
    invariant to embedding vector scale, which varies with input brightness.

    Why a margin instead of an absolute cosine?
    -------------------------------------------
    The served backbone is domain-fine-tuned, which compressed the absolute
    cosine scale (a correct match sits ~0.6, an unrelated prototype ~0.4), so a
    fixed cosine cutoff no longer separates matches from non-matches.  The
    top1-top2 margin does: a decisive match pulls clearly ahead of the runner-up,
    while a face-down / ambiguous crop sits roughly equidistant from several
    prototypes (small margin) -> "unknown".  See research/RESEARCH.md (2026-06-22).

    Parameters
    ----------
    card_crop:
        BGR numpy array (H x W x C) from pipeline's crop_relative call.
    embedder:
        A pre-loaded OnnxEmbedder instance.
    embed_gallery:
        Pre-built EmbeddingGallery (from build_embedding_gallery()).

    Returns
    -------
    (identity, role_class, confidence)
        identity   -- townee name or "unknown"
        role_class -- role family or "unknown"
        confidence -- float in [0, 1]
    """
    if card_crop is None or card_crop.size == 0:
        return ("unknown", "unknown", 0.0)

    # Embed the crop -- [576] float32, unit norm
    crop_vec = embedder.embed(card_crop)

    # Degenerate embedding (e.g., pure black crop) -> unknown
    if crop_vec is None or float(np.linalg.norm(crop_vec)) < 1e-9:
        return ("unknown", "unknown", 0.0)

    # Cosine similarity against all gallery prototypes
    # embed_gallery.embeddings: [K, 576] (each row already unit-norm)
    # crop_vec: [576] unit-norm
    # dot product of unit vecs = cosine similarity in [-1, 1]
    cosine_scores: np.ndarray = embed_gallery.embeddings @ crop_vec  # [K]

    # Abstain on the top1-top2 MARGIN rather than an absolute cosine (see the
    # _EMBED_MARGIN_THRESHOLD note above).  The reported confidence IS the margin
    # (clipped to [0, 1]) -- the honest "how decisively did this beat the
    # runner-up" signal -- so downstream can rank by decisiveness.
    order = np.argsort(cosine_scores)  # ascending
    best_idx = int(order[-1])
    margin = (
        float(cosine_scores[best_idx] - cosine_scores[int(order[-2])])
        if cosine_scores.shape[0] >= 2
        else 0.0
    )
    confidence = float(np.clip(margin, 0.0, 1.0))

    if margin < _EMBED_MARGIN_THRESHOLD:
        return ("unknown", "unknown", round(confidence, 4))

    winner = embed_gallery.entries[best_idx]
    return (winner.identity, winner.role_class, round(confidence, 4))


def make_embedding_identifier(
    embedder: "object",       # OnnxEmbedder
    embed_gallery: "object",  # EmbeddingGallery
) -> Callable[[np.ndarray], tuple[str, str, float]]:
    """Return a single-argument callable that wraps classify_crop_embedding.

    Bridges the embedding classifier into the same 1-argument pipeline interface
    used by make_gallery_identifier.

    Usage
    -----
        embedder = OnnxEmbedder()
        embed_gallery = build_embedding_gallery(classical_gallery, embedder)
        identifier = make_embedding_identifier(embedder, embed_gallery)
        snapshot = run_pipeline(image, source, identifier=identifier)
    """
    def _embedding_identify(card_crop: np.ndarray) -> tuple[str, str, float]:
        return classify_crop_embedding(card_crop, embedder, embed_gallery)

    return _embedding_identify


# ---------------------------------------------------------------------------
# Ensemble — classical + embedding composition layer (2026-07-29 live-eval fix)
# ---------------------------------------------------------------------------
#
# Motivation (plans/PLAN-live-capture.md, "Fix 3"): the collect_02 live eval
# found the two identifiers catch genuinely DIFFERENT cards, not just
# agreeing-or-not on the same ones.  On the new roles that session surfaced:
# classical alone got Wretch and Jester right; embedding alone got Judge and
# Slayer right; both missed Empress, Knight, and Witch.  An IoU-matched pass
# over eval_02's full_classical/full_embedding JSONs (917 matched card-slot
# pairs) found:
#
#     agree (same identity, both non-unknown)        78
#     classical abstains, embedding answers          235
#     embedding abstains, classical answers            99
#     disagree (different identity, both non-unknown)  20
#     both abstain                                    485
#
# This module keeps classify_crop / classify_crop_embedding completely
# unchanged -- the ensemble is a pure composition layer that calls both and
# combines their independent results.  It does NOT require a fine-tune, a
# new model, or a schema change; it is opt-in via Settings.identifier
# (dbcv/config.py) / DBCV_IDENTIFIER=ensemble.

# Confidence boost added when both identifiers agree (their independent
# raw confidence scales are NOT comparable -- see the disagreement note
# below -- so agreement combines via max(), not an average, then adds a
# fixed boost as the "two independent methods agree" bonus).
_ENSEMBLE_AGREEMENT_BOOST: float = 0.15

# Ensemble outcome tags returned as the 4th element of combine_identifications.
EnsembleSource = str  # "agree" | "classical_only" | "embedding_only" | "disagree_abstain" | "both_unknown"


def combine_identifications(
    classical: tuple[str, str, float],
    embedding: tuple[str, str, float],
) -> tuple[str, str, float, EnsembleSource]:
    """Combine one classical result and one embedding result for the SAME crop.

    This is the tested, source-tagged core of the ensemble.  It is a pure
    function of two independent (identity, role_class, confidence) tuples --
    it does not call either identifier itself, which is what makes it easy
    to unit-test with stub outputs (see tests/test_identify.py).

    Combination rules (in order)
    -----------------------------
    1. **Both abstain** ("unknown" identity on both) -> ("unknown", "unknown",
       0.0, "both_unknown").
    2. **Agreement** (same non-"unknown" identity on both) -> that identity,
       confidence = ``min(1.0, max(classical_conf, embedding_conf) +
       _ENSEMBLE_AGREEMENT_BOOST)``, source "agree".  ``max()`` rather than
       an average because the two confidence scales are not comparable (see
       rule 4) -- boosting the stronger of the two independent signals is
       the only combination that doesn't require a cross-scale calibration
       this project does not have yet.
    3. **One abstains, the other doesn't** -> adopt the non-abstaining
       result's identity/role_class/confidence UNCHANGED, source
       "classical_only" or "embedding_only".  This is the biggest lever in
       the eval_02 evidence above (334 of 917 matched pairs).
    4. **Disagreement** (different non-"unknown" identities) -> abstain:
       ("unknown", "unknown", 0.0, "disagree_abstain").  NOT "prefer the
       higher confidence" -- eval_02's disagreement cases (e.g.
       `collect_02/018.png`: classical says Poisoner@0.47, embedding says
       Hunter@0.20; visually confirmed ground truth is Hunter) show
       classical's raw confidence is numerically higher in every recorded
       disagreement, yet classical is the one that's wrong.  Classical's
       histogram-correlation scale (typically 0.40-0.90) and embedding's
       top1-top2-margin scale (typically 0.12-0.40) are not calibrated
       against each other, and there is no labeled live-crop set to fit a
       fair normalization without calibrating on the same frames this
       ensemble is being evaluated against.  Abstaining is the honest
       choice until that data exists (see PLAN-live-capture.md, Fix 4).

    Parameters
    ----------
    classical:
        (identity, role_class, confidence) from ``classify_crop``.
    embedding:
        (identity, role_class, confidence) from ``classify_crop_embedding``.
        Must be the result for the SAME crop as ``classical`` -- this
        function does no spatial matching itself.

    Returns
    -------
    (identity, role_class, confidence, source)
        ``source`` is one of "agree", "classical_only", "embedding_only",
        "disagree_abstain", "both_unknown" -- see ``make_ensemble_identifier``
        for why this 4th element is dropped before reaching the pipeline.
    """
    c_identity, c_role, c_conf = classical
    e_identity, e_role, e_conf = embedding

    c_known = c_identity != "unknown"
    e_known = e_identity != "unknown"

    if not c_known and not e_known:
        return ("unknown", "unknown", 0.0, "both_unknown")

    if c_known and e_known:
        if c_identity == e_identity:
            boosted = min(1.0, max(c_conf, e_conf) + _ENSEMBLE_AGREEMENT_BOOST)
            return (c_identity, c_role, round(boosted, 4), "agree")
        # Disagreement: abstain rather than guess via an uncalibrated
        # cross-scale confidence comparison (see docstring rule 4).
        return ("unknown", "unknown", 0.0, "disagree_abstain")

    if c_known:
        return (c_identity, c_role, c_conf, "classical_only")

    return (e_identity, e_role, e_conf, "embedding_only")


def make_ensemble_identifier(
    classical_identifier: Callable[[np.ndarray], tuple[str, str, float]],
    embedding_identifier: Callable[[np.ndarray], tuple[str, str, float]],
) -> Callable[[np.ndarray], tuple[str, str, float]]:
    """Return a single-argument callable combining classical + embedding.

    Calls both identifiers on the same crop and combines their results via
    ``combine_identifications``.  Matches the same 1-argument pipeline
    interface as ``make_gallery_identifier`` / ``make_embedding_identifier``,
    so it is a drop-in ``identifier=`` for ``run_pipeline`` / the API.

    Note on the dropped ``source`` tag
    -----------------------------------
    ``combine_identifications`` returns a 4-tuple tagging *how* the result
    was produced (agreement / one-abstained / disagreement).  The pipeline's
    identifier contract and ``CardRead`` (dbcv/schema.py) only carry
    (identity, role_class, confidence) -- there is no provenance field yet.
    This wrapper drops the tag rather than force a schema bump alongside two
    other fixes in the same wave; callers that want the tag should call
    ``combine_identifications`` directly (as the tests do).  Surfacing
    provenance end-to-end is noted as an open item in
    plans/PLAN-live-capture.md.

    Parameters
    ----------
    classical_identifier:
        A 1-argument callable, e.g. ``make_gallery_identifier(gallery)``.
    embedding_identifier:
        A 1-argument callable, e.g.
        ``make_embedding_identifier(embedder, embed_gallery)``.

    Returns
    -------
    Callable[[np.ndarray], tuple[str, str, float]]
        A closure that runs both identifiers and combines their results.

    Usage
    -----
        classical_id = make_gallery_identifier(gallery)
        embedding_id = make_embedding_identifier(embedder, embed_gallery)
        ensemble_id = make_ensemble_identifier(classical_id, embedding_id)
        snapshot = run_pipeline(image, source, identifier=ensemble_id)
    """
    def _ensemble_identify(card_crop: np.ndarray) -> tuple[str, str, float]:
        classical_result = classical_identifier(card_crop)
        embedding_result = embedding_identifier(card_crop)
        identity, role_class, confidence, _source = combine_identifications(
            classical_result, embedding_result
        )
        return (identity, role_class, confidence)

    return _ensemble_identify
