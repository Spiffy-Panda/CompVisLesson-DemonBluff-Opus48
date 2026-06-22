"""
dbcv/identify.py — Card-identification interface and stub implementation.

Given a cropped card image (already localized), the identifier returns the
townee name, role class, and a confidence score.

Interface
---------
    identify(card_crop) -> (identity: str, role_class: str, confidence: float)

    - identity   : townee name (e.g. "Alchemist") or "unknown"
    - role_class : one of "villager" | "minion" | "outcast" | "demon" | "unknown"
    - confidence : float in [0.0, 1.0]

The stub always returns ("unknown", "unknown", 0.0) so the pipeline can run
before any recognizer is trained.

Research grounding
------------------
The real identifier will use a small frozen embedding backbone
(MobileNetV3-Small or similar) with nearest-neighbour lookup over a reference
gallery built from knowledge-base/card-art/.  On an art swap, only the gallery
references need to be re-embedded — no gradient steps.  This satisfies the
"cheap to retrain" constraint from CLAUDE.md.

An NCC/template fast-path handles clean axis-aligned crops; the embedding-NN
path handles ambiguous cards.  OCR of the name-label text can cross-check the
visual match.

See research/RESEARCH.md: "Card identification that is cheap to retrain when
art changes — 2026-06-21".
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Stub implementation
# ---------------------------------------------------------------------------


def identify(card_crop: np.ndarray) -> tuple[str, str, float]:
    """Stub identifier — returns "unknown" for everything.

    Parameters
    ----------
    card_crop:
        Cropped card region as a numpy array (HxWxC).  Channel order matches
        whatever the pipeline passed in (typically BGR from cv2).

    Returns
    -------
    (identity, role_class, confidence)
        ``identity``   — townee name or "unknown"
        ``role_class`` — role family or "unknown"
        ``confidence`` — 0.0 (stub always uncertain)

    Replacement path
    ----------------
    Swap this function (or pass a real callable as ``identifier`` in
    ``run_pipeline``) when the embedding-NN identifier is ready.  The
    function signature must remain identical so the pipeline needs no changes.
    """
    # The stub discards card_crop entirely.  A real identifier would:
    #   1. Preprocess the crop (resize to backbone input, normalize).
    #   2. Run a forward pass through a small frozen backbone.
    #   3. Compute cosine/L2 distance to every entry in the reference gallery.
    #   4. Return the closest match's identity/role_class and 1 - distance as
    #      the confidence score (clamped to [0, 1]).
    return ("unknown", "unknown", 0.0)
