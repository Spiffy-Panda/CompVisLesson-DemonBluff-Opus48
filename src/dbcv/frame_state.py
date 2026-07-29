"""
dbcv/frame_state.py  --  Stage 0: frame-state gate.

Classifies a decoded game frame as one of:
    "board"  -- the radial card ring around the central pentagram is visible;
                the localizer should run.
    "modal"  -- a dialog / deck-viewer panel is overlaid on the board, hiding
                most of the playfield; skip localisation.
    "menu"   -- a full-screen menu / loading screen (no board content at all);
                skip localisation.

Why the prior center-brightness approach failed
------------------------------------------------
The naive approach from the first spike used mean brightness in the central
40 % x 40 % region.  It scored 0/3 on modal frames because *Demon Bluff
modals are dark-background dialogs*: the outer ring of the modal frame is the
same dark board we see in non-modal frames.  The modal panel itself is bright,
but its bright pixels are surrounded by a large dark perimeter, so the overall
center-region mean brightness is not much higher than a board frame's.

The winning discriminator: center-brightness RATIO
---------------------------------------------------
The key geometric insight (confirmed empirically on 7 labeled frames with
scrap_scripts/python/06_frame_state_probe.py):

   modal frames:  center is MUCH brighter than the surrounding ring
   board frames:  center and ring are similar in brightness (cards are
                  distributed around the ring; the center pentagram is dark)

Measured center-vs-ring ratios on the labeled set (all 1280x720):
    board  : 1.063, 1.021, 1.109, 1.097  (mean 1.07, max 1.11)
    modal  : 3.099, 5.878, 4.121         (mean 4.37, min 3.10)
    partial: 0.943                        (correctly classified as board)

A threshold of 2.0 cleanly separates boards (max 1.11) from full modals
(min 3.10) with a 3x gap.  No modal frame is misclassified; no board frame
is misclassified.

Partial modals (e.g. Sample1_006 -- "Pick 3 characters" dialog with
peripheral cards still visible) land at 0.94, well below the threshold.
They are classified as "board" -- the correct production decision because
the localizer CAN still find the peripheral cards and the game state
is partially readable.

Algorithm
---------
1.  Compute mean brightness in the inner center box (30-70% x 30-70%).
2.  Compute mean brightness in the surrounding ring (10-90% minus the center).
3.  Ratio = center / ring.  If >= MODAL_THRESHOLD --> "modal".
4.  Full-screen bright check for menu screens (mean brightness of the whole
    frame exceeds MENU_BRIGHTNESS_THRESHOLD -- high overall brightness is
    diagnostic of a white/light loading screen, which never occurs on a board
    or modal frame in these samples).
5.  Otherwise --> "board".

All geometry is expressed as fractions of image dimensions.
No pixel values are hard-coded.

Research grounding
------------------
The center-vs-ring brightness ratio approach is a classical texture/layout
discrimination strategy described in the computer-vision UI-detection
literature (see research/RESEARCH.md -- add entry if promoted to curriculum).
The technique is analogous to "integral image" region comparisons used in
face detection (Viola-Jones, 2001) and modern saliency models: when a modal
window is present, the center-of-mass of bright pixels shifts dramatically
toward the image center.

Kill-Mode red-tint detection (2026-07-29, plans/PLAN-live-capture.md)
--------------------------------------------------------------------
The `collect_02` live eval surfaced a UI variant not present in the sample
videos: a full-frame red colour grade ("Kill Mode") that (a) causes the
classical localizer's red HSV mask to fire on non-card red decoration (the
center demon-altar statue) and (b) degrades identification reliability on
every card in the frame.  ``measure_red_shift`` / ``is_red_tint`` below give
a cheap, whole-frame signal for (b) -- see ``dbcv.pipeline.run_pipeline``'s
``tint_fn`` parameter, which discounts and re-abstains low-confidence
identifications while tint is active.  (a) is a deeper localizer-mask fix,
not addressed by this signal -- see PLAN-live-capture.md's open items.

The discriminator is the global mean of ``R - max(G, B)`` per pixel (BGR
channels from cv2): a red-graded frame has every pixel's red channel pushed
above green/blue, while a normal frame's red channel is not systematically
dominant (this UI's normal palette leans green/teal/purple).  Measured on
all 147 sampled frames across `collect_01` + `collect_02`: red-tinted frames
scored 14.4 to 32.2; every other frame scored -16.9 to +0.5 -- a clean gap
with no overlap.  ``_RED_TINT_THRESHOLD = 10.0`` sits in the middle of that
gap.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Type alias for the return value
# ---------------------------------------------------------------------------

FrameState = Literal["board", "modal", "menu"]

# ---------------------------------------------------------------------------
# Tuning constants
# All geometry expressed as fractions; no pixel value is hard-coded.
# ---------------------------------------------------------------------------

# CENTER BOX: inner rectangle tested for brightness (fractions of H and W)
_CENTER_Y0 = 0.30
_CENTER_Y1 = 0.70
_CENTER_X0 = 0.30
_CENTER_X1 = 0.70

# RING BOX: outer rectangle that surrounds the center box
# The ring is the area of the outer box minus the center box.
_RING_Y0 = 0.10
_RING_Y1 = 0.90
_RING_X0 = 0.10
_RING_X1 = 0.90

# Threshold for the center-vs-ring brightness ratio.
# - Board frames score between 0.94 and 1.11 (probe results).
# - Modal frames score between 3.10 and 5.88 (probe results).
# - 2.0 is the midpoint of the gap; we choose 2.0 to give margin in both
#   directions.  There is a 3x gap on the labeled set so this is robust.
_MODAL_RATIO_THRESHOLD: float = 2.0

# Threshold for "menu": full-frame mean brightness.
# A loading screen / main menu is typically light-coloured.
# None of the current labeled frames approach this threshold, but we gate
# on 160 / 255 (very bright) to avoid false positives on modal frames.
_MENU_BRIGHTNESS_THRESHOLD: float = 160.0

# Threshold for the global red-shift (Kill-Mode tint) score, mean(R - max(G,B))
# over the whole frame.  Measured on 147 live-captured frames (collect_01 +
# collect_02, 2026-07-29): tinted frames scored 14.4-32.2, all other frames
# scored -16.9 to +0.5.  10.0 sits in the middle of that gap with margin on
# both sides.  See "Kill-Mode red-tint detection" in the module docstring.
_RED_TINT_THRESHOLD: float = 10.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_frame_state(image: np.ndarray) -> FrameState:
    """Classify a decoded game frame as 'board', 'modal', or 'menu'.

    Parameters
    ----------
    image:
        Decoded frame as a numpy array with shape (H, W, C) in any channel
        order (BGR or RGB -- only brightness is used here, which is channel-
        order-independent after conversion to grayscale via the mean).
        The image may be any resolution; all geometry is relative.

    Returns
    -------
    FrameState
        One of:
        - ``"board"``  -- radial card ring is visible; run the localizer.
        - ``"modal"``  -- a dialog / overlay is occluding most of the board;
                          skip localization; return empty cards list.
        - ``"menu"``   -- full-screen menu or loading screen;
                          skip localization; return empty cards list.

    Notes
    -----
    The discriminating signal is the ratio of mean luminance in the center
    region (30-70 % x 30-70 %) to mean luminance in the surrounding ring
    (10-90 % minus the center box).  On labeled Demon Bluff frames:

    - Board frames: ratio 0.94 to 1.11  (center ~ ring; cards distributed radially)
    - Full-modal frames: ratio 3.10 to 5.88  (bright panel dominates center)
    - Partial-modal frame (Sample1_006): ratio 0.94 (classified as board, which
      is correct -- the localizer can still find peripheral cards)

    This approach is robust against the specific failure mode of the naive
    center-brightness baseline: dark-background modals with only a centered
    bright panel still have a high ratio even when the absolute brightness of
    the center is moderate, because the RING is even darker.
    """
    h, w = image.shape[:2]

    # Convert to single-channel brightness.
    # cv2.cvtColor handles both BGR and grayscale inputs; if the image is
    # already 1-channel this branch would need adjustment, but the pipeline
    # always passes BGR from cv2.imdecode (IMREAD_COLOR).
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        # Fallback for unexpected shapes: collapse to first channel
        gray = image[:, :, 0].astype(np.float32)

    # ------------------------------------------------------------------
    # Gate 1 — menu / full-screen light UI
    # ------------------------------------------------------------------
    # A loading screen or main menu is typically much brighter overall than
    # any in-game frame (board or modal).  Check the whole-frame mean first
    # to catch this case before the ratio test, which would be meaningless
    # on a uniformly bright image.
    full_mean = float(np.mean(gray))
    if full_mean >= _MENU_BRIGHTNESS_THRESHOLD:
        return "menu"

    # ------------------------------------------------------------------
    # Gate 2 — center-vs-ring brightness ratio (the key discriminator)
    # ------------------------------------------------------------------
    # Pixel boundary coordinates derived from the measured image shape.
    # All fractions above → pixel indices here.
    cy0 = int(h * _CENTER_Y0)
    cy1 = int(h * _CENTER_Y1)
    cx0 = int(w * _CENTER_X0)
    cx1 = int(w * _CENTER_X1)

    ry0 = int(h * _RING_Y0)
    ry1 = int(h * _RING_Y1)
    rx0 = int(w * _RING_X0)
    rx1 = int(w * _RING_X1)

    # Center box mean brightness
    center_mean = float(np.mean(gray[cy0:cy1, cx0:cx1]))

    # Ring = outer_box minus center_box.
    # We extract the outer region and mask out the center rows/columns
    # to get the pure ring.  Using np.nan for masked values lets us use
    # nanmean cleanly.
    ring_patch = gray[ry0:ry1, rx0:rx1].copy()

    # Offset the center coordinates into the ring_patch coordinate system
    rc_y0 = cy0 - ry0
    rc_y1 = cy1 - ry0
    rc_x0 = cx0 - rx0
    rc_x1 = cx1 - rx0

    ring_patch[rc_y0:rc_y1, rc_x0:rc_x1] = np.nan  # mask out center
    ring_mean = float(np.nanmean(ring_patch))

    # Avoid division by zero on pathological (all-black) images
    if ring_mean < 1.0:
        ring_mean = 1.0

    ratio = center_mean / ring_mean

    if ratio >= _MODAL_RATIO_THRESHOLD:
        return "modal"

    # ------------------------------------------------------------------
    # Default: board
    # ------------------------------------------------------------------
    return "board"


# ---------------------------------------------------------------------------
# Kill-Mode red-tint detection (2026-07-29, plans/PLAN-live-capture.md)
# ---------------------------------------------------------------------------


def measure_red_shift(image: np.ndarray) -> float:
    """Return a cheap whole-frame red-tint score.

    Computes ``mean(R - max(G, B))`` over every pixel in the frame (BGR
    channel order, as produced by ``cv2.imread``/``cv2.imdecode``).  A frame
    with no systematic red bias scores near zero or negative (this game's
    normal palette leans green/teal/purple); a full-frame red colour grade
    ("Kill Mode") pushes every pixel's red channel above its green/blue
    channels, producing a large positive score.

    Parameters
    ----------
    image:
        Decoded frame, BGR, any resolution.  All pixels are used; there is
        no resolution-specific geometry here (unlike the localizer/gate).

    Returns
    -------
    float
        The red-shift score.  See ``is_red_tint`` for the calibrated
        threshold and the measured value ranges on real frames.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return 0.0
    bgr = image[:, :, :3].astype(np.float32)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return float(np.mean(r - np.maximum(g, b)))


def is_red_tint(image: np.ndarray, threshold: float = _RED_TINT_THRESHOLD) -> bool:
    """Return True if the frame shows the Kill-Mode red colour grade.

    Thin wrapper around ``measure_red_shift`` with the calibrated default
    threshold (10.0 -- see the module docstring's "Kill-Mode red-tint
    detection" section for the measured gap).  Passed as the default
    ``tint_fn`` to ``dbcv.pipeline.run_pipeline``.

    Parameters
    ----------
    image:
        Decoded frame, BGR, any resolution.
    threshold:
        Red-shift score at or above which the frame is considered tinted.
        Defaults to the calibrated ``_RED_TINT_THRESHOLD``.

    Returns
    -------
    bool
        True if ``measure_red_shift(image) >= threshold``.
    """
    return measure_red_shift(image) >= threshold
