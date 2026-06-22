"""
dbcv/pipeline.py — End-to-end frame → GameStateSnapshot pipeline.

This is the orchestration layer.  It:
  1. Reads the frame's dimensions from image.shape — never from a constant.
  2. Calls the localizer to find card bounding boxes (relative coords).
  3. For each box, converts relative → pixel coords, crops the region, and
     calls the identifier.
  4. Passes everything to assemble() for packaging into a GameStateSnapshot.

The localizer and identifier are injectable callables, so tests and the
lesson plan can swap in stubs or real implementations without touching this
module.

Teaching note on crop_relative
--------------------------------
Converting from relative → pixel coordinates is a deliberate teaching moment:
  pixel_x = round(rel_x * width)
This computation must happen *after* reading the resolution from the image,
not from a stored constant — which is why ``crop_relative`` accepts the image
itself rather than a pre-measured size.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from dbcv.assemble import assemble
from dbcv.identify import identify
from dbcv.localize import BboxRel, LocalizerCallable, stub_localize
from dbcv.schema import GameStateSnapshot, Resolution, Source


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------


def crop_relative(image: np.ndarray, bbox_rel: BboxRel) -> np.ndarray:
    """Crop a region from an image using relative (fractional) coordinates.

    Parameters
    ----------
    image:
        The full frame as a numpy array with shape (H, W, C) or (H, W).
        Dimensions are read from the array — never assumed.
    bbox_rel:
        (x, y, w, h) where each value is a fraction in [0.0, 1.0].
        x, y are the top-left corner; w, h are the box width and height.
        Origin is the top-left of the frame.

    Returns
    -------
    np.ndarray
        The cropped sub-image.  If the box extends past the image boundary
        (e.g. a stub box that slightly overshoots), it is silently clamped —
        a real localizer should produce valid boxes, but clamping prevents hard
        crashes during prototyping.

    Teaching note
    -------------
    The conversion ``pixel = round(fraction * dimension)`` is the only place
    in the whole pipeline where a relative coordinate becomes a pixel coordinate.
    Everything upstream stays in fractions so that the code never encodes an
    assumption about resolution.
    """
    h_img, w_img = image.shape[:2]   # read from the array — never a constant

    x_rel, y_rel, w_rel, h_rel = bbox_rel

    # Convert fractions → pixels
    x0 = round(x_rel * w_img)
    y0 = round(y_rel * h_img)
    x1 = round((x_rel + w_rel) * w_img)
    y1 = round((y_rel + h_rel) * h_img)

    # Clamp to valid range (protects against stub overreach or float rounding)
    x0 = max(0, min(x0, w_img))
    y0 = max(0, min(y0, h_img))
    x1 = max(0, min(x1, w_img))
    y1 = max(0, min(y1, h_img))

    return image[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    image: np.ndarray,
    source: Source,
    localizer: LocalizerCallable | Callable[..., list[BboxRel]] = stub_localize,
    identifier: Callable[[np.ndarray], tuple[str, str, float]] = identify,
) -> GameStateSnapshot:
    """Run the full frame → GameStateSnapshot pipeline.

    Parameters
    ----------
    image:
        Decoded frame as a numpy array (H x W x C).  Dimensions are read from
        the array — never from a stored constant or hard-coded value.
    source:
        Provenance metadata: which video, which frame index, what timestamp.
    localizer:
        Callable matching ``LocalizerCallable``.  Defaults to ``stub_localize``.
        The real classical localizer will be injected here once validated.
    identifier:
        Callable ``(card_crop: np.ndarray) -> (identity, role_class, confidence)``.
        Defaults to the stub identifier.

    Returns
    -------
    GameStateSnapshot
        A fully-formed, versioned snapshot ready to be returned by the API.

    Notes
    -----
    - Resolution is measured once from ``image.shape`` and stored in the
      snapshot; the same measured value is passed to the localizer.
    - The relative-to-pixel conversion happens inside ``crop_relative``,
      which is the single, documented place that converts fractions → pixels.
    """
    # Step 1: Measure the frame — never assume a resolution
    h_img, w_img = image.shape[:2]
    resolution = Resolution(w=w_img, h=h_img)

    # Step 2: Localize — find card regions as relative bounding boxes
    boxes: list[BboxRel] = localizer(image, resolution)

    # Step 3: Identify — for each box, crop and classify
    identities: list[tuple[str, str, float]] = []
    for bbox_rel in boxes:
        crop = crop_relative(image, bbox_rel)
        # Guard against a zero-size crop (degenerate box from a stub or edge case)
        if crop.size == 0:
            identities.append(("unknown", "unknown", 0.0))
        else:
            identities.append(identifier(crop))

    # Step 4: Assemble — package everything into the versioned snapshot
    return assemble(source, resolution, boxes, identities)
