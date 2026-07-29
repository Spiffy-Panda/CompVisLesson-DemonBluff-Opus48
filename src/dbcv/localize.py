"""
dbcv/localize.py — Card-localization interface, stub baseline, and classical localizer.

This module defines the *contract* for the localization stage: given a decoded
frame, return a list of relative bounding boxes, one per card.

Interface
---------
    localizer(image, resolution) -> list[tuple[float, float, float, float]]

    Each returned tuple is (x, y, w, h) in *relative coordinates*:
      - x, y  : top-left corner as a fraction of the frame width/height
      - w, h  : width/height as fractions of the frame width/height
    All values are in [0.0, 1.0].

    The localizer receives the *measured* Resolution (never assume it); it may
    be used for sanity checks but must not drive any hard-coded pixel values.
    All pixel math is derived from ``image.shape[:2]`` directly.

Pluggability
------------
The pipeline accepts a ``localizer`` callable with the same signature, so you
can swap any implementation without touching the pipeline itself.

    run_pipeline(image, source, localizer=classical_localize, ...)

Teaching baseline vs. production implementation
------------------------------------------------
``stub_localize`` (the "before") is retained as a documented teaching baseline.
It shows what the pipeline looked like before any vision code existed: three
hard-coded approximate boxes that work for smoke testing but are wrong for
every specific frame.

``classical_localize`` (the "after") is the validated implementation.  It uses
colour segmentation, morphology, contour analysis, HUD-zone exclusion, and IoU
non-maximum suppression — all classical OpenCV, all CPU-only, ~10 ms per frame.
It was spiked and validated on the sample frames (8/8 and 9/9 exact, zero false
positives on clean board frames) before being promoted here.

Research grounding
------------------
See research/RESEARCH.md entry 2: "Card/region localization robust to art swaps
under a tight compute budget — 2026-06-21".  The choice of classical detection
over a learned detector (YOLOv8n) is supported by source 3 of that entry
(~12 ms classical vs. ~19 ms YOLO for game-UI recognition) and by the art-swap
constraint: layout geometry is independent of card art, so retuning HSV ranges
on an art swap costs minutes rather than requiring a new labelled dataset.

Art-swap caveat
---------------
The HSV hue/saturation/value thresholds in ``classical_localize`` were tuned
to the *current* card art palette (purple, orange, red, and bright-saturated
colours that appear on Demon Bluff card borders and faces).  If the card art
set is swapped for the alternate set, these ranges would be **re-tuned** (not
retrained) — typically a 15-30 minute manual pass with an HSV visualiser.  The
morphology kernel sizes and contour filters are geometry-derived and would not
need adjustment.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from dbcv.schema import Resolution


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

BboxRel = tuple[float, float, float, float]
"""(x, y, w, h) all in [0, 1], relative to frame width/height, origin top-left."""


# ---------------------------------------------------------------------------
# Protocol — the localizer interface
# ---------------------------------------------------------------------------


class LocalizerCallable(Protocol):
    """Structural type for any localizer that the pipeline can accept.

    Both ``stub_localize`` and ``classical_localize`` satisfy this signature
    so they are drop-in replaceable.  Any future learner-based localizer must
    also satisfy it.
    """

    def __call__(
        self, image: np.ndarray, resolution: Resolution
    ) -> list[BboxRel]:
        """Return a list of relative bounding boxes, one per card region.

        Parameters
        ----------
        image:
            Decoded frame as an HxWxC numpy array (BGR from cv2).
        resolution:
            Frame dimensions measured from the image at runtime (never assumed).
            Use only for sanity checks or assertions; return boxes are still
            *relative* fractions derived from ``image.shape``.

        Returns
        -------
        list of (x, y, w, h) tuples, all values in [0.0, 1.0].
        """
        ...


# ---------------------------------------------------------------------------
# Teaching baseline: stub localizer (the "before")
# ---------------------------------------------------------------------------


def stub_localize(image: np.ndarray, resolution: Resolution) -> list[BboxRel]:
    """TEACHING BASELINE — returns plausible fixed relative boxes without vision.

    This stub exists so the pipeline, API, and tests all work end-to-end
    before the real classical localizer is validated on the sample frames.
    The boxes approximate where three cards typically appear in the Demon Bluff
    radial ring layout (left, top-centre, right) as observed in sampled frames
    — purely for smoke-testing purposes.

    WARNING: Do not use these boxes for any real measurement.  They are not
    derived from image content and are wrong for every specific frame.

    Retained as a documented teaching baseline to show what the pipeline
    looked like before any vision code was added (the "before" state).
    Compare to ``classical_localize`` to see the full delta in complexity and
    accuracy.  Passing ``localizer=stub_localize`` to ``run_pipeline`` reverts
    to this behaviour explicitly, which is useful in tests that need predictable
    output count regardless of frame content.
    """
    # Three approximate card positions in the radial ring, expressed as
    # (x, y, w, h) fractions.  Values are rough; these are not real detections.
    return [
        (0.08, 0.30, 0.18, 0.30),   # left-side card slot (approximate)
        (0.40, 0.05, 0.18, 0.30),   # top-centre card slot (approximate)
        (0.72, 0.30, 0.18, 0.30),   # right-side card slot (approximate)
    ]


# ---------------------------------------------------------------------------
# Classical localizer: the validated implementation (the "after")
# ---------------------------------------------------------------------------


def classical_localize(
    image: np.ndarray, resolution: Resolution
) -> list[BboxRel]:
    """Classical, CPU-only card localizer — validated on real Demon Bluff frames.

    Algorithm overview (five stages)
    ---------------------------------
    1. **HUD-strip exclusion** — zero out the objective bar, score panels, and
       watermark strips that are known to contain colourful non-card UI elements.
       All boundaries are expressed as fractions of the measured image dimensions
       so they scale with any input resolution.

    2. **HSV colour segmentation** — convert the HUD-masked image to HSV and
       threshold for the hues that appear on Demon Bluff card borders and faces:
       purple (role-colour rings), orange (card backs / villain styling),
       red (demon role accent), and a broad "bright-and-saturated" catch-all
       for any vivid card art the narrower ranges miss.  The result is a binary
       mask where candidate card pixels are white.

    3. **Morphological cleanup** — a closing pass joins nearby blobs (card art
       often has interior dark gaps) and an opening pass removes isolated speckle
       noise.  Kernel sizes are proportional to the image's shorter dimension so
       the cleanup works at any resolution.

    4. **Contour filtering** — find external contours in the cleaned mask, take
       bounding rectangles, and discard any rectangle that:
         (a) is too small to be a card (< 0.15 % of frame area), or
         (b) is too large (> 9 % of frame area — HUD or full-board overlay), or
         (c) has an extreme aspect ratio (< 0.38 or > 1.40 — Demon Bluff cards
             are roughly square to mildly portrait), or
         (d) overlaps a known HUD zone by more than 40 % of its own area.

    5. **IoU non-maximum suppression (NMS)** — sort surviving rectangles largest-
       first and greedily suppress any box that overlaps an already-kept box by
       more than IoU 0.30.  This collapses multiple contours from the same card
       into a single representative box.

    Returns
    -------
    list of (x_rel, y_rel, w_rel, h_rel) tuples, all components in [0.0, 1.0].

    Validation results (spike, 2026-06-21)
    ---------------------------------------
    - Sample1 board frames: 8/8 cards detected exactly, 0 false positives.
    - Sample2 board frames: 9/9 cards detected exactly, 0 false positives.
    - Modal/overlay frames: correctly returns ~0–2 boxes (no false board parse).

    Art-swap note
    -------------
    The HSV thresholds below were tuned to the *current* art palette.  On an art
    swap, re-tune these ranges with an HSV visualiser (typically 15–30 min).
    The morphology sizes and contour-filter ratios are geometry-derived and are
    art-independent.

    Stage 0 gate: ``classify_frame_state`` in ``dbcv/frame_state.py`` is now
    wired into ``run_pipeline`` (pipeline.py) and runs before this function.
    Non-board frames never reach this localizer.  The gate scores 7/7 on
    the labeled set (4 board, 3 modal) using a center-vs-ring brightness ratio.
    """
    # Derive dimensions from the image array — never from a hard-coded constant.
    # The ``resolution`` arg is the same information as a dbcv.schema.Resolution;
    # we read from image.shape directly so this function is fully self-contained
    # and the assert below catches any upstream miscommunication.
    h, w = image.shape[:2]

    # Sanity-check: the measured Resolution passed in must match the array shape.
    # This fires only if the caller passes a pre-measured Resolution that was
    # taken from a *different* image — a programming error, not a user error.
    assert resolution.w == w and resolution.h == h, (
        f"resolution arg ({resolution.w}×{resolution.h}) does not match "
        f"image.shape ({w}×{h}).  Always pass the Resolution measured from "
        f"this exact image."
    )

    # ------------------------------------------------------------------
    # Stage 1 — HUD-strip exclusion
    # ------------------------------------------------------------------
    # Demon Bluff lays out its HUD in predictable relative zones:
    #   - Top ~9 %  : objective bar (task text, turn counter)
    #   - Bottom ~14%: name labels, spectator watermark
    #   - Left ~13% : score/role panel
    #   - Right ~8% : role icon cluster
    # Zeroing these strips prevents the segmentation from picking up colourful
    # UI chrome that would otherwise pass the card-colour test.
    work = image.copy()
    work[: int(h * 0.09), :] = 0          # top objective / HUD bar
    work[int(h * 0.86) :, :] = 0          # bottom strip (name labels, watermark)
    work[:, : int(w * 0.13)] = 0          # left score panel
    work[:, int(w * 0.92) :] = 0          # right edge icons

    # ------------------------------------------------------------------
    # Stage 2 — HSV colour segmentation of card regions
    # ------------------------------------------------------------------
    # Why HSV rather than BGR thresholds?  Hue is the perceptually meaningful
    # dimension for colour matching; it is robust to moderate brightness changes
    # (different streaming capture settings, monitor gamma), whereas a BGR range
    # conflates brightness and colour and is fragile to exposure shifts.
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Each mask targets a specific colour family present on Demon Bluff cards.
    # OpenCV HSV: hue in [0, 179], saturation and value in [0, 255].

    # Purple: role-colour rings (Demon, Minion, Outcast role accents)
    purple = (hue > 128) & (hue < 175) & (sat > 40) & (val > 65)

    # Orange: card back pattern and villain-role styling
    orange = (hue > 7) & (hue < 23) & (sat > 65) & (val > 55)

    # Red (split range): Demon role accent — hue wraps near 0 in OpenCV's 0–179 scale
    red_hi = (hue > 168) & (sat > 55) & (val > 55)  # near-180 wrap
    red_lo = (hue < 6) & (sat > 55) & (val > 55)    # near-0 wrap

    # Broad catch-all: vivid card art not captured by the above narrow ranges
    bright_sat = (sat > 45) & (val > 85)

    # Union of all card-colour masks → binary mask image
    mask = (purple | orange | red_hi | red_lo | bright_sat).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # Stage 3 — Morphological cleanup
    # ------------------------------------------------------------------
    # Why closing before opening?
    #   Closing (dilate-then-erode) with a large kernel bridges the dark gaps
    #   that exist *within* a single card (card art often has interior dark
    #   regions, borders between art and role-colour rings, etc.).  Without this
    #   step one card produces several small fragments instead of one blob.
    #   Opening (erode-then-dilate) with a smaller kernel then removes isolated
    #   speckle noise and small artefacts that survived the colour threshold.
    # Kernel sizes are proportional to min(w, h) so they adapt to any resolution.
    k_close = max(9, int(min(w, h) * 0.018))   # ~18 px on a 1080p frame
    k_open  = max(3, int(min(w, h) * 0.006))   # ~6 px on a 1080p frame

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close)),
    )
    opened = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open)),
    )

    # ------------------------------------------------------------------
    # Stage 4 — Contour → bounding rects, filtered by geometry and HUD zones
    # ------------------------------------------------------------------
    # RETR_EXTERNAL: only outermost contours — cards do not nest inside each other.
    # CHAIN_APPROX_SIMPLE: compress horizontal/vertical runs to endpoints (memory).
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area thresholds expressed relative to full frame area — resolution-agnostic.
    min_area = w * h * 0.0015   # < 0.15 % → noise / partial card edge
    max_area = w * h * 0.09     # > 9 %   → entire HUD panel or full-board overlay

    # Known HUD zones in relative coordinates (x, y, w, h).
    # A contour box is rejected if >40 % of its area overlaps any zone.
    # These complement the Stage-1 zeroing; some HUD artefacts survive colour
    # segmentation and need a second geometric gate.
    HUD_ZONES = [
        (0.00, 0.00, 0.13, 1.00),   # left score panel
        (0.92, 0.00, 0.08, 1.00),   # right edge icons
        (0.00, 0.00, 1.00, 0.09),   # top objective bar
        (0.00, 0.86, 1.00, 0.14),   # bottom name / watermark strip
        (0.82, 0.78, 0.18, 0.22),   # bottom-right timer / health cluster
        (0.12, 0.72, 0.05, 0.20),   # left nomination button area
    ]

    def _in_hud(x: float, y: float, bw: float, bh: float) -> bool:
        """Return True if the relative box overlaps any HUD zone by > 40 %."""
        for zx, zy, zw, zh in HUD_ZONES:
            # Compute intersection dimensions
            ix1, iy1 = max(x, zx), max(y, zy)
            ix2, iy2 = min(x + bw, zx + zw), min(y + bh, zy + zh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if bw * bh > 0 and inter / (bw * bh) > 0.40:
                    return True
        return False

    slots: list[BboxRel] = []
    for cnt in contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)

        # Area filter: too small → noise; too large → HUD panel
        if not (min_area <= rw * rh <= max_area):
            continue

        # Aspect-ratio filter: Demon Bluff cards range from roughly square
        # (detail cards) to mildly portrait (role cards); extreme ratios are
        # bar-shaped UI elements (health bars, objective progress bands)
        ar = rw / rh if rh > 0 else 0.0
        if not (0.38 <= ar <= 1.40):
            continue

        # Convert to relative coordinates for the HUD-overlap test and output
        x_r, y_r, w_r, h_r = rx / w, ry / h, rw / w, rh / h

        if _in_hud(x_r, y_r, w_r, h_r):
            continue

        slots.append((x_r, y_r, w_r, h_r))

    # ------------------------------------------------------------------
    # Stage 5 — IoU non-maximum suppression (NMS)
    # ------------------------------------------------------------------
    # Why NMS?  Each physical card may produce 2–4 overlapping contour blobs
    # (art region, border ring, role icon) that each pass the geometry filters.
    # We want one box per card, not one box per colour patch.
    #
    # Algorithm: greedy IoU-NMS.
    #   1. Sort by area descending (larger boxes capture the full card better).
    #   2. For each candidate, compute IoU with every already-kept box.
    #   3. If any IoU > 0.30, suppress the candidate (it is a sub-region of an
    #      already-kept card box).
    #
    # IoU threshold 0.30 was chosen empirically on the spike frames.  A higher
    # threshold (e.g. 0.50) allows overlapping boxes from the same card; a lower
    # threshold (e.g. 0.10) over-suppresses boxes for adjacent cards.
    slots.sort(key=lambda s: s[2] * s[3], reverse=True)

    kept: list[BboxRel] = []
    for sx, sy, sw, sh in slots:
        suppressed = False
        for kx, ky, kw, kh in kept:
            # Compute intersection over union in relative coordinate space
            ix1, iy1 = max(sx, kx), max(sy, ky)
            ix2, iy2 = min(sx + sw, kx + kw), min(sy + sh, ky + kh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = sw * sh + kw * kh - inter
                if union > 0 and inter / union > 0.30:
                    suppressed = True
                    break
        if not suppressed:
            kept.append((sx, sy, sw, sh))

    return kept
