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

Live-capture HUD-zone fix (2026-07-29, ``plans/PLAN-live-capture.md``)
------------------------------------------------------------------------
Two live-frame evals (`collect_01`, `collect_02` — see ``DEV-LOG.md``,
2026-07-29) found the original HUD strips (top 9% full-width, left 13% full
height) were too thin to cover two real HUD elements, both measured directly
off live frames:

1. The **objective-text block** (title + minion/demon counts + "Evils
   killed" + "Village" + "Ascension" + "Score" lines) sits **top-left**,
   measured at up to x≈0.25, y≈0.25 of frame size — wider and taller than
   either the old top strip or the old left panel covered alone.  This was
   the dominant live-frame false positive: a "Hunter@0.42-0.50" card hit
   fired in 73-83% of board frames across both evals.
2. The **top-right "revealed evils" thumbnail strip** (small badge portraits
   of already-executed evil characters) is genuine character art in a HUD
   summary widget, not a board card — it produced the second-largest false
   positive cluster in `eval_02` (55 confident hits, some >0.9 confidence,
   because the badge genuinely resembles the real character).

**Why not just widen the existing full-width top strip to ~20%?**  Verified
harmful on real frames: the top-center card slot's top edge can sit as high
as y≈0.076 (e.g. `collect_02/050.png` card #7, at x≈0.48-0.53) — well inside
a naive 0-20%-height full-width band.  The fix is two **narrow, corner-only**
zones (see ``HUD_ZONES`` below) instead of widening the full-width strip.

Both new zones were checked against every confident real-card detection in
both evals (classical confidence >= 0.5, or any non-"unknown" embedding hit):
no real card intrudes into either zone with margin to spare (left column
real cards start at x>=0.29; right column real cards stay under x<=0.76 in
every non-Kill-Mode frame sampled).

A third recurring false positive ("Poisoner@0.42-0.44") was investigated and
found to be a **genuine board card**, not a fixed background element — the
classical HSV matcher has a Hunter->Poisoner confusion that recurs wherever a
revealed Hunter card happens to sit.  No HUD zone or guard was added for it;
doing so would blind the localizer to real cards.  See
``plans/PLAN-live-capture.md`` for the frame-by-frame evidence.

Background-totem phantom-box fix (2026-07-29, ``scrap_scripts/python/out/eval_04``)
------------------------------------------------------------------------------------
`eval_04` (352 board frames, ``collect_03``) found two fixed **background-art
props** — skull/bone totem pillars painted into the board background, not
cards — being localized as card-shaped contours in the large majority of
frames:

1. **Left totem** (bottom-left, ~x 0.13-0.26, ~y 0.50-0.77 of frame) — hit in
   93.3% of the 344 board frames scanned (321/344). Usually abstains
   ("unknown"), but misidentifies as Hunter (4x, conf 0.52-0.55) or Wretch
   (7x, conf 0.40-0.43) often enough to pollute the identification stream.
2. **Right totem** (top-right, ~x 0.72-0.86, ~y 0.14-0.41 of frame) — hit in
   95.3% of frames (328/344), and misidentifies far more often: Hunter
   (26x, conf 0.50-0.67) and Rambler (9x) — the largest single confusion
   cluster in the eval.

Both totems are most visible (highest colour saturation, easiest to confirm
by eye) in Kill-Mode frames, where the whole board gets a red tint that
happens to push the stone/skull totem art into the same HSV range as real
card borders — but the phantom boxes fire on ordinary non-Kill-Mode frames
too, just slightly less saturated. Confirmed visually against
``full_classical/043_overlay.png`` (Kill Mode, both totems clearly visible
as red skull-topped pillars under the false boxes) and
``full_classical/140_overlay.png`` / ``019_overlay.png`` (decagon and
octagon boards, ordinary lighting, same props visible at the same
positions).

**Zones added** (see ``HUD_ZONES`` below, Stage 4 only — see the "why
geometric-only" note in Stage 1's comments):
``(0.15, 0.56, 0.07, 0.16)`` (left totem) and ``(0.78, 0.18, 0.08, 0.22)``
(right totem). Verified against every localizer box produced across all 344
``eval_04`` board frames (not just the 29 hand-labeled ground-truth frames):
**zero** non-totem boxes overlap either zone at all — the closest is a
0.4%-of-its-own-area graze from the (unrelated, out-of-scope) Kill-Mode
timer/health HUD blob near the right zone. Cross-checked against
``ground_truth.json``'s hand-built slot coordinates across every board shape
including the two largest (nonagon, decagon): the real-card centre-x column
never goes below 0.305 (left column) or above 0.692 (right column) in any
shape, leaving wide margin either side of both new zones. A closer look at
one nonagon frame (099) also found a *real* card box (a Twin-Minion reveal
with a blood-splatter execution effect, frame 196) whose splatter-inflated
bounding box reaches left edge x≈0.236 — only 0.016 clear of the left zone's
x=0.22 edge, the tightest margin found anywhere in this fix; this is why the
zone was kept at its originally-recommended width rather than widened
further (see the fragmentation note below for why it didn't need to be).
Coverage: the two zones (via the standard >40%-of-box-area overlap rule)
suppress 319/321 (99.4%) left-totem hits and 325/328 (99.1%) right-totem
hits *as measured against the original, unfragmented blob shapes* — see
below for why the *shipped* fix does even better than that (0.6% / 0.9%
residual).

**Why Stage-4-only, not Stage-1 pixel zeroing (a fragmentation gotcha)**:
an earlier version of this fix also zeroed the two totem rectangles in
Stage 1 (like the corner HUD zones do). Re-running
``scrap_scripts/python/13_eval_collect03b_classical.py`` against that
version showed the residual left-totem hit rate only dropped to 14.8% (not
near-zero) and the *misidentified* count went **up** (11→26) — worse than
doing nothing on that axis. Cause: a totem sitting away from the frame edge
is one isolated colour blob; zeroing only the zone's sub-rectangle out of
the middle of it **splits** the blob into two smaller side slivers (the
part of the totem to the left of the zone and the part to the right), and
each sliver individually has *less* than 40% overlap with the (narrower)
zone, so *neither* gets excluded — a strictly worse outcome than leaving
the totem as one whole blob and letting the single resulting box fail the
40% overlap test as a whole. The corner HUD zones (top-left, top-right)
don't hit this failure mode because they sit flush against a frame edge:
trimming a frame-edge strip can only shrink a blob, never split it into two
disconnected pieces. The fix: leave the totem's full blob intact through
Stage 1 (no pixel zeroing) and rely on Stage 4's geometric exclusion only,
which is how the zone was originally validated (single whole-blob box vs.
zone) and matches the >99% coverage measured above.

**Shipped result** (``scrap_scripts/python/13_eval_collect03b_classical.py``,
full re-run of all 344 board frames, classical arm): left-totem hit rate
93.3%→0.6% (321/344→2/344), right-totem hit rate 95.3%→0.9% (328/344→3/344),
**zero** misidentified totem hits remaining on either side (down from 11 and
35) — the 5 residual hits left across both zones are all confidence ≤0.09
("unknown" or near-zero), i.e. noise that would abstain regardless. Mean
(predicted − ground-truth) card count per frame, averaged across board
shapes: 2.20 → 0.16 (decagon 2.50→0.50, heptagon 2.17→0.17, hexagon
2.25→0.25, nonagon 2.00→−0.25, octagon 2.08→0.15). The nonagon frames'
small negative post-fix mean (−0.25) is **not a regression**: frame-by-frame
inspection (``scrap_scripts/python/15_inspect_shape_deltas.py``) confirmed
it unmasks a pre-existing, unrelated localizer miss (frame 109 was already
missing 2 real cards before this fix; its previous pred==gt count was a
coincidence — 2 real misses cancelled by the 2 totem phantom boxes).

**Top-right badge-zone widened** 0.86 → 0.84 (both the Stage-1 pixel mask
and the matching ``HUD_ZONES`` entry) while re-verifying under the same
eval: no non-totem box with y < 0.36 (the badge strip's height) ever reaches
a left edge past x=0.695 — 0.145 of clear headroom past the new 0.84
boundary, so the widen is safe with margin to spare, not just "trivially"
safe.

Reproduce: ``scrap_scripts/python/12_totem_zone_check.py`` (localization-only
zone/coverage/safety checks against the pre-fix ``eval_04`` data) and
``scrap_scripts/python/13_eval_collect03b_classical.py`` (full before/after
re-run, writes ``scrap_scripts/python/out/eval_04b/``). See ``DEV-LOG.md``
(2026-07-29 entry) for the full narrative including the fragmentation bug.
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

    Live-frame validation (2026-07-29, see plans/PLAN-live-capture.md)
    ---------------------------------------------------------------------
    The sample-video spike above did not catch the HUD false positives live
    capture exposed (73-83% of live board frames spuriously reported a
    "Hunter" hit from the objective-text HUD block).  The two new corner-only
    HUD_ZONES entries fix that without regressing recall: every confident
    real-card detection in both live evals (eval_01, eval_02) stays clear of
    both new zones with margin.  Re-run scrap_scripts/python/08_eval_collect02.py
    (extended with a full_ensemble arm) after this change for updated numbers.

    Background-totem phantom-box fix (2026-07-29, eval_04, see module docstring)
    -------------------------------------------------------------------------------
    `eval_04` (352 collect_03 board frames) found the localizer was still
    reporting two fixed background-art skull/bone totem props as card boxes
    in the large majority of frames (93.3% / 95.3%), sometimes with a
    confident-but-wrong identity (Hunter up to 0.67, Wretch up to 0.55). Two
    new HUD_ZONES entries (geometric-only -- deliberately NOT also
    pixel-zeroed in Stage 1; see the Stage-1 comments and module docstring
    for the blob-fragmentation gotcha that motivated this) fixed it:
    post-fix, left/right totem hit rate 93.3%/95.3% -> 0.6%/0.9%, zero
    misidentified totem hits remaining (down from 11 and 35), mean
    predicted-minus-ground-truth card count per frame 2.20 -> 0.16 averaged
    across board shapes (full numbers, incl. per-shape breakdown, in the
    module docstring and DEV-LOG.md). Re-run
    scrap_scripts/python/13_eval_collect03b_classical.py (classical-only,
    fastest) for a fresh before/after comparison, or
    scrap_scripts/python/11_eval_collect03.py for the full three-arm
    re-score.

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
    #   - Top-left corner (0-27% w, 0-27% h): the FULL objective-text block
    #     (title + minion/demon counts + Evils-killed/Village/Ascension/Score
    #     lines) — taller AND wider than the plain top/left strips above.
    #     Corner-only (not full-width/full-height) so the top-center and
    #     left-column card slots are untouched — see the module docstring's
    #     "Live-capture HUD-zone fix" section.
    #   - Top-right corner (84-100% w, 0-36% h): the "revealed evils" badge
    #     thumbnail strip — genuine character-art HUD widget, not a card.
    #     Widened from 86% to 84% (2026-07-29, eval_04) after confirming no
    #     real card box ever reaches past x=0.695 in that y-band — see the
    #     module docstring's "Background-totem phantom-box fix" section.
    # NOTE on the two skull/bone totem props (2026-07-29, eval_04): unlike
    # the corner HUD elements above, these are NOT pixel-zeroed here.  A
    # totem sits away from the frame edge as one isolated colour blob, and
    # zeroing only a sub-rectangle of it (rather than a frame-edge strip)
    # was found to SPLIT that blob into two smaller side slivers, each of
    # which then dodges the Stage-4 HUD_ZONES 40%-overlap-of-own-area rule
    # individually — a regression discovered by re-running
    # scrap_scripts/python/13_eval_collect03b_classical.py after an earlier
    # version of this fix zeroed the totems here too (residual hit rate
    # dropped but did not go to ~0, and misidentified-hit count went UP).
    # Leaving the totem's full blob intact through Stage 1 and excluding it
    # geometrically in Stage 4 (see HUD_ZONES below) matches how the
    # zone was originally validated (single whole-blob box vs. zone,
    # >99% suppression) and avoids the fragmentation failure mode.  See the
    # module docstring's "Background-totem phantom-box fix" section.
    # Zeroing these strips prevents the segmentation from picking up colourful
    # UI chrome that would otherwise pass the card-colour test.
    work = image.copy()
    work[: int(h * 0.09), :] = 0          # top objective / HUD bar
    work[int(h * 0.86) :, :] = 0          # bottom strip (name labels, watermark)
    work[:, : int(w * 0.13)] = 0          # left score panel
    work[:, int(w * 0.92) :] = 0          # right edge icons
    work[: int(h * 0.27), : int(w * 0.27)] = 0   # top-left objective-text block (corner-only)
    work[: int(h * 0.36), int(w * 0.84) :] = 0   # top-right revealed-evils badge strip (corner-only; widened 0.86->0.84, eval_04)

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
        (0.00, 0.00, 0.27, 0.27),   # top-left objective-text block (corner-only; 2026-07-29 live-eval fix)
        (0.84, 0.00, 0.16, 0.36),   # top-right revealed-evils badge strip (corner-only; widened 0.86->0.84, 2026-07-29 eval_04 fix)
        (0.15, 0.56, 0.07, 0.16),   # left skull/bone totem prop (background, not a card; 2026-07-29 eval_04 fix)
        (0.78, 0.18, 0.08, 0.22),   # right skull/bone totem prop (background, not a card; 2026-07-29 eval_04 fix)
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
