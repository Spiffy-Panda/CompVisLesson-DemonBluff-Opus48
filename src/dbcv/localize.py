"""
dbcv/localize.py — Card-localization interface and stub implementation.

This module defines the *contract* for the localization stage: given a decoded
frame, return a list of relative bounding boxes, one per card.  A stub
implementation lets the rest of the pipeline run end-to-end today; the real
implementation will replace it once a classical localizer is validated on the
sample frames.

Interface
---------
    localize(image, resolution) -> list[tuple[float, float, float, float]]

    Each returned tuple is (x, y, w, h) in *relative coordinates*:
      - x, y  : top-left corner as a fraction of the frame width/height
      - w, h  : width/height as fractions of the frame width/height
    All values are in [0.0, 1.0].

    The localizer receives the *measured* Resolution (never assume it); the
    stub ignores it because its boxes are already relative — the real
    implementation should use it too only for sanity-checks, not hard-coding.

Pluggability
------------
The pipeline accepts a ``localizer`` callable with the same signature, so you
can swap stub_localize for a real implementation (or a mock in tests) without
touching the pipeline itself.

    run_pipeline(image, source, localizer=stub_localize, ...)

Research grounding
------------------
The real localizer will use classical, layout-driven localization: detect
art-independent UI landmarks (radial card ring, numbered position badges,
panel chrome) via contour/edge/HoughLines + small template matches, then
derive card slots relative to those landmarks scaled by the measured resolution.
This approach is:
  - ~10 ms CPU, deterministic
  - Requires zero training labels
  - Immune to card-art swaps (geometry is separate from appearance)
See research/RESEARCH.md: "Card/region localization robust to art swaps under
a tight compute budget — 2026-06-21".

A learned detector (YOLOv8n/YOLO11n) is the fallback only if footage proves
the layout isn't reliably parseable (skewed captures, animated panel reflow).
"""

from __future__ import annotations

from typing import Protocol

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

    Both ``stub_localize`` and the future classical localizer must satisfy
    this signature so they are drop-in replaceable.
    """

    def __call__(
        self, image: np.ndarray, resolution: Resolution
    ) -> list[BboxRel]:
        """Return a list of relative bounding boxes, one per card region.

        Parameters
        ----------
        image:
            Decoded frame as an HxWxC numpy array (channels = BGR from cv2 or
            RGB from PIL — the localizer should be agnostic to channel order
            for geometry tasks).
        resolution:
            Frame dimensions measured from the image at runtime (never assumed).
            Use ``resolution.w`` and ``resolution.h`` only for sanity checks or
            landmark detection; the returned boxes must still be *relative*.

        Returns
        -------
        list of (x, y, w, h) tuples, all values in [0.0, 1.0].
        """
        ...


# ---------------------------------------------------------------------------
# Stub implementation
# ---------------------------------------------------------------------------


def stub_localize(image: np.ndarray, resolution: Resolution) -> list[BboxRel]:
    """Stub localizer — returns plausible fixed relative boxes without vision.

    This stub exists so the pipeline, API, and tests all work end-to-end
    before the real classical localizer is validated on the sample frames.
    The boxes approximate where three cards typically appear in the Demon Bluff
    radial ring layout (left, top-centre, right) as observed in the sampled
    frames — purely for smoke-testing purposes.

    WARNING: Do not use these boxes for any real measurement.  They are not
    derived from image content and are wrong for every specific frame.

    Replacement path
    ----------------
    Drop a real localizer into ``pipeline.run_pipeline`` via the ``localizer``
    parameter.  The real implementation will:
      1. Detect the card-ring landmarks with HoughLines / contour analysis.
      2. Locate numbered position badges (resolution-agnostic template match
         on UI chrome, not card art).
      3. Derive card slot boxes relative to those landmarks, scaled by
         ``resolution.w`` / ``resolution.h``.
    """
    # Three approximate card positions in the radial ring, expressed as
    # (x, y, w, h) fractions.  Values are rough; replace with real geometry.
    return [
        (0.08, 0.30, 0.18, 0.30),   # left-side card slot
        (0.40, 0.05, 0.18, 0.30),   # top-centre card slot
        (0.72, 0.30, 0.18, 0.30),   # right-side card slot
    ]


# ---------------------------------------------------------------------------
# Public interface helper (mirrors the Protocol; plain function for direct use)
# ---------------------------------------------------------------------------


def localize(image: np.ndarray, resolution: Resolution) -> list[BboxRel]:
    """Default-dispatching localizer — currently delegates to stub_localize.

    When the classical localizer is ready, update this function to call it
    instead.  Keeping a single named entry point here means callers that
    import ``localize`` directly (rather than passing a callable) get the real
    implementation automatically.
    """
    return stub_localize(image, resolution)
