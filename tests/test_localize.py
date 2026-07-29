"""
tests/test_localize.py — Unit tests for the classical card localizer.

Covers two things not exercised elsewhere:
  1. The classical/stub interface contract (basic shape/format checks).
  2. The 2026-07-29 live-eval HUD-zone fix (plans/PLAN-live-capture.md,
     Fix 1) — synthetic frames with a colourful blob dropped at a known
     position verify that:
       - a blob inside either NEW HUD zone (top-left objective-text block,
         top-right revealed-evils badge strip) is masked out;
       - a blob at a plausible real card position — including the top-center
         slot, whose top edge can sit high in frame (the exact case the
         "don't just widen the full-width top band" warning is about) — is
         still found.

All frames here are synthetic (drawn with cv2, never loaded from the sample
videos or dataset/), so these tests run unconditionally — no skip needed.

Rule 1 compliance: no inline interpreter calls.  No ``import`` on the command
line.  All code runs through pytest.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from dbcv.localize import classical_localize, stub_localize
from dbcv.schema import Resolution

# ---------------------------------------------------------------------------
# Synthetic-frame helpers
# ---------------------------------------------------------------------------

# A colour that reliably satisfies classical_localize's "bright_sat" catch-all
# mask (sat > 45, val > 85) regardless of the specific hue ranges, so these
# tests aren't coupled to the exact purple/orange/red thresholds.
_BLOB_COLOR_BGR = (0, 255, 0)   # pure green: HSV sat=255, val=255


def _blank_frame(w: int = 1280, h: int = 720) -> np.ndarray:
    """A black frame of the given size — the board background is dark."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _draw_blob_rel(
    image: np.ndarray, x: float, y: float, w: float, h: float
) -> np.ndarray:
    """Draw a filled, card-coloured rectangle at a relative (x, y, w, h) box.

    Returns the same image (mutated in place) for convenient chaining.
    """
    img_h, img_w = image.shape[:2]
    x0, y0 = int(x * img_w), int(y * img_h)
    x1, y1 = int((x + w) * img_w), int((y + h) * img_h)
    cv2.rectangle(image, (x0, y0), (x1, y1), _BLOB_COLOR_BGR, thickness=-1)
    return image


def _run(image: np.ndarray) -> list[tuple[float, float, float, float]]:
    h, w = image.shape[:2]
    return classical_localize(image, Resolution(w=w, h=h))


def _any_box_overlaps(
    boxes: list[tuple[float, float, float, float]],
    target: tuple[float, float, float, float],
    min_iou: float = 0.3,
) -> bool:
    """True if any detected box has IoU >= min_iou with the target box."""
    tx, ty, tw, th = target
    for bx, by, bw, bh in boxes:
        ix1, iy1 = max(bx, tx), max(by, ty)
        ix2, iy2 = min(bx + bw, tx + tw), min(by + bh, ty + th)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = bw * bh + tw * th - inter
        if union > 0 and inter / union >= min_iou:
            return True
    return False


# ---------------------------------------------------------------------------
# Basic interface contract
# ---------------------------------------------------------------------------


def test_stub_localize_returns_three_boxes() -> None:
    """stub_localize returns the three documented approximate boxes."""
    img = _blank_frame()
    boxes = stub_localize(img, Resolution(w=1280, h=720))
    assert len(boxes) == 3
    for x, y, w, h in boxes:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_classical_localize_on_blank_frame_finds_nothing() -> None:
    """An all-black frame (no card-coloured pixels) yields zero boxes."""
    img = _blank_frame()
    boxes = _run(img)
    assert boxes == []


def test_classical_localize_returns_relative_boxes_in_range() -> None:
    """Every returned box component must be a fraction in [0, 1]."""
    img = _blank_frame()
    _draw_blob_rel(img, 0.30, 0.30, 0.10, 0.15)
    for x, y, w, h in _run(img):
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert 0.0 < w <= 1.0
        assert 0.0 < h <= 1.0


# ---------------------------------------------------------------------------
# HUD-zone fix (2026-07-29): new zones mask text-like blobs
# ---------------------------------------------------------------------------


def test_top_left_objective_text_blob_is_masked() -> None:
    """A blob inside the new top-left HUD zone (0.00-0.27, 0.00-0.27) is dropped.

    This stands in for the objective-text HUD block that produced the
    "Hunter@0.42-0.50" false positive on 73-83% of live board frames before
    this fix (see plans/PLAN-live-capture.md, Fix 1).
    """
    img = _blank_frame()
    target = (0.05, 0.05, 0.10, 0.10)   # fully inside (0, 0, 0.27, 0.27)
    _draw_blob_rel(img, *target)
    boxes = _run(img)
    assert not _any_box_overlaps(boxes, target), (
        f"Blob inside the top-left HUD zone was NOT masked: {boxes}"
    )


def test_top_right_badge_strip_blob_is_masked() -> None:
    """A blob inside the new top-right HUD zone (0.86-1.00, 0.00-0.36) is dropped.

    Stands in for the "revealed evils" thumbnail badge strip — genuine
    character-art content in a HUD summary widget, not a board card (the
    second-largest false-positive cluster found in eval_02).
    """
    img = _blank_frame()
    target = (0.90, 0.10, 0.08, 0.12)   # fully inside (0.86, 0, 0.14, 0.36)
    _draw_blob_rel(img, *target)
    boxes = _run(img)
    assert not _any_box_overlaps(boxes, target), (
        f"Blob inside the top-right HUD zone was NOT masked: {boxes}"
    )


# ---------------------------------------------------------------------------
# HUD-zone fix (2026-07-29): recall on real card positions is NOT regressed
# ---------------------------------------------------------------------------


def test_left_column_card_below_hud_zone_is_still_found() -> None:
    """A card-shaped blob just past the new top-left zone's right edge is found.

    Real left-column card slots in the live evals never appeared with
    x < 0.29 (measured across eval_01 + eval_02); this places the blob at
    x=0.30, clear of the new zone's x=0.27 edge, and expects it to survive.
    """
    img = _blank_frame()
    target = (0.30, 0.30, 0.12, 0.16)   # aspect ratio 0.75, ~3.3% of frame area
    _draw_blob_rel(img, *target)
    boxes = _run(img)
    assert _any_box_overlaps(boxes, target), (
        f"Card-shaped blob just outside the top-left HUD zone was masked: {boxes}"
    )


def test_right_column_card_below_badge_zone_is_still_found() -> None:
    """A card-shaped blob clear of the new top-right zone's left edge is found.

    Real right-column card slots stayed under x<=0.76 in every non-Kill-Mode
    frame checked; this places the blob at x=0.65, clear of the new zone's
    x=0.86 edge.
    """
    img = _blank_frame()
    target = (0.65, 0.25, 0.12, 0.16)
    _draw_blob_rel(img, *target)
    boxes = _run(img)
    assert _any_box_overlaps(boxes, target), (
        f"Card-shaped blob clear of the top-right HUD zone was masked: {boxes}"
    )


def test_top_center_card_with_high_top_edge_is_still_found() -> None:
    """A card-shaped blob whose top edge sits high in frame (y~0.05) survives.

    This is the exact scenario the "do not just widen the full-width top
    band to 20%" warning is about: collect_02/050.png's top-center card #7
    has its top edge at y~0.076, x~0.48-0.53.  A naive full-width 0-20%-height
    band would have masked it; the corner-only zones used here must not.
    """
    img = _blank_frame()
    # Full card-height blob starting near the measured real top-edge (y~0.076);
    # tall enough that clipping the top-9% HUD strip off the top still leaves
    # a plausible card aspect ratio (0.38-1.40) for the surviving contour.
    target = (0.46, 0.06, 0.12, 0.20)   # x is far from both corner zones
    _draw_blob_rel(img, *target)
    boxes = _run(img)
    assert _any_box_overlaps(boxes, target, min_iou=0.15), (
        "Top-center card with a high top edge was masked by the HUD-zone fix "
        f"(regression on the exact case the fix was designed to avoid): {boxes}"
    )


def test_hud_zone_fix_is_resolution_agnostic() -> None:
    """The same masked-vs-found behaviour holds at a different resolution.

    Guards against any hard-coded pixel creeping into the new zones — they
    must be expressed as fractions, per CLAUDE.md's never-bake-a-resolution
    constraint.
    """
    img = _blank_frame(w=640, h=360)
    masked_target = (0.05, 0.05, 0.10, 0.10)
    found_target = (0.30, 0.30, 0.12, 0.16)
    _draw_blob_rel(img, *masked_target)
    _draw_blob_rel(img, *found_target)
    boxes = _run(img)
    assert not _any_box_overlaps(boxes, masked_target)
    assert _any_box_overlaps(boxes, found_target)


# ---------------------------------------------------------------------------
# Poisoner@0.42-0.44 investigation (documented, no code change)
# ---------------------------------------------------------------------------


def test_a_hunter_shaped_card_at_a_mid_board_slot_is_still_found() -> None:
    """A card-shaped blob at a mid-board slot (not a HUD corner) is found.

    plans/PLAN-live-capture.md documents that the recurring "Poisoner@0.42-0.44"
    false identification was traced to a genuine, moving board card (a
    revealed Hunter card misclassified by the classical HSV *identifier*, not
    a localization/HUD problem) — so no HUD zone was added for it.  This test
    is the localizer-side half of that decision: a card at a typical
    mid-board slot position (neither corner zone) must still be localized,
    since identification-layer confusion is out of scope for localize.py.
    """
    img = _blank_frame()
    target = (0.46, 0.25, 0.10, 0.15)   # a typical non-corner ring slot
    _draw_blob_rel(img, *target)
    boxes = _run(img)
    assert _any_box_overlaps(boxes, target), (
        f"Mid-board card slot was unexpectedly masked: {boxes}"
    )


@pytest.mark.parametrize("w,h", [(1280, 720), (1920, 1080), (854, 480)])
def test_classical_localize_never_asserts_on_matching_resolution(w: int, h: int) -> None:
    """classical_localize accepts any resolution when Resolution matches image.shape."""
    img = _blank_frame(w=w, h=h)
    boxes = classical_localize(img, Resolution(w=w, h=h))
    assert isinstance(boxes, list)
