"""
dbcv/gallery.py — Reference gallery builder for classical card identification.

Builds an in-memory gallery from the card-art directory tree at
``knowledge-base/card-art/<role_class>/<identity>/<file>.png``.

The gallery holds, for each reference image, three matchable representations
computed once at startup:
  1. A resized (normalised) thumbnail for fast pixel-level comparison.
  2. An HSV colour histogram (used as the primary matcher — colour is the
     most reliable discriminator between cartoon characters).
  3. ORB keypoints and descriptors for feature-point matching (secondary,
     used when colour histograms are too close to call).

Key design invariants
---------------------
- **Zero training.**  An art swap = re-run ``build_gallery()`` over the new
  directory.  No gradient steps, no stored model files, no retraining.  This
  is the "cheap to retrain" property mandated in CLAUDE.md.
- **All-in-memory.** No gallery artifact is persisted.  The ~67 small PNGs
  are read once at startup (~50–100 ms total); the resulting numpy arrays live
  in RAM for the lifetime of the process.
- **Minion / Twin_Minion tolerance.** Both map to identity "Minion" and
  role_class "minion" (documented game fact: they are functionally identical).
  Matching either is acceptable.
- **Puppet note.** Puppet (created by Puppeteer) has its own art and its own
  identity, but is still labelled role_class "minion" as it appears in that
  directory.

Research grounding
------------------
- HSV colour histograms for cartoon/game character identification: robust to
  moderate geometric variation (crop offsets, scale) because hue distribution
  is mostly invariant to translation and moderate rotation.  Source: OpenCV
  documentation on ``cv2.calcHist`` and compareHist.
- ORB as a lightweight fallback: rotation-invariant binary descriptors with no
  licence concerns.  Low runtime cost (~1-3 ms per crop on a modern CPU).
- Template NCC was evaluated and rejected as the primary method: the reference
  art is a clean illustration with white/transparent background; board crops
  include card borders, name labels, tinting, and often show only part of the
  art region — pixel-level alignment is not achievable without a known affine
  transform, making NCC unreliable as a primary signal.
  (See scrap_scripts/python/07_identify_probe.py for empirical evidence.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Thumbnail size for pixel-level representations (resized reference images).
# Small enough for fast comparison; large enough to preserve shape.
# Not a resolution assumption — this is the gallery's internal representation size.
_THUMB_SIZE: int = 64  # (64 x 64 px)

# Histogram parameters
_HIST_BINS_HUE: int = 32   # 32 hue buckets × 360°/179 ≈ coarse colour wheel
_HIST_BINS_SAT: int = 16   # 16 saturation buckets

# ORB configuration
_ORB_N_FEATURES: int = 200

# Art-region crop fractions: when building gallery entries, crop the reference
# art to focus on the character illustration (upper portion), avoiding the
# transparent/white margins that the art images have at the bottom.
# These are applied to the reference PNG, NOT to board crops.
_ART_CROP_TOP: float = 0.0
_ART_CROP_BOTTOM: float = 0.80   # keep top 80% of the reference image

# Alias for Twin_Minion → Minion (documented game fact)
_IDENTITY_ALIASES: dict[str, str] = {
    "Twin_Minion": "Minion",
}

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# ORB descriptor array may be None if too few keypoints were found
DescriptorArray = np.ndarray | None


class GalleryEntry(NamedTuple):
    """All precomputed representations for one reference image.

    One townee may have several entries (base art + skin variants).
    The identity and role_class are shared across all entries for that townee.
    """

    identity: str       # e.g. "Alchemist", "Baa", "Minion" (alias-resolved)
    role_class: str     # e.g. "villager", "demon", "minion"
    file_stem: str      # original filename stem, for debugging

    # Precomputed matchable representations
    thumb: np.ndarray            # BGR thumbnail (64×64)
    hsv_hist: np.ndarray         # 2-D HSV colour histogram, L1-normalised
    orb_kp: list                 # list of cv2.KeyPoint objects
    orb_desc: DescriptorArray    # (N, 32) uint8 ORB descriptors, or None


@dataclass
class Gallery:
    """Container for all loaded reference entries.

    Attributes
    ----------
    entries:
        All loaded reference images as GalleryEntry objects.
    townee_names:
        Sorted set of unique (alias-resolved) identity strings.
    role_classes:
        Sorted set of unique role_class strings (should be exactly 4).
    """

    entries: list[GalleryEntry] = field(default_factory=list)
    townee_names: list[str] = field(default_factory=list)
    role_classes: list[str] = field(default_factory=list)

    # Number of distinct townees (identities) — for tests and reporting.
    @property
    def n_townees(self) -> int:
        return len(self.townee_names)

    # Total number of reference images loaded (>= n_townees due to skins).
    @property
    def n_references(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ORB = None   # lazy singleton; avoid re-creating the detector per image


def _get_orb() -> cv2.ORB:
    """Return the module-level ORB detector (created once, reused)."""
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=_ORB_N_FEATURES)
    return _ORB


def _compute_hsv_hist(bgr: np.ndarray) -> np.ndarray:
    """Compute and L1-normalise a 2-D Hue×Saturation histogram.

    Why 2-D H×S rather than a 3-D H×S×V histogram?
    - Value (brightness) is sensitive to lighting/tinting.
    - Hue + saturation together capture colour identity well for cartoon art.
    - A 32×16 = 512-bin histogram is compact and fast to compare.

    Parameters
    ----------
    bgr:
        BGR image (any size; will be converted to HSV internally).

    Returns
    -------
    np.ndarray
        Float32 histogram of shape (32, 16), L1-normalised to sum to 1.0.
        An image with zero pixels returns a zero histogram (safe for comparison).
    """
    if bgr.size == 0:
        return np.zeros((_HIST_BINS_HUE, _HIST_BINS_SAT), dtype=np.float32)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Mask out near-black and near-white pixels (background / transparency fill).
    # Card art often has a white or black background that isn't part of the
    # character; excluding these makes the histogram more distinctive.
    # OpenCV HSV: H in [0,179], S in [0,255], V in [0,255].
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    mask = ((s_ch > 25) & (v_ch > 30) & (v_ch < 250)).astype(np.uint8) * 255

    hist = cv2.calcHist(
        [hsv],
        [0, 1],              # hue and saturation channels
        mask,
        [_HIST_BINS_HUE, _HIST_BINS_SAT],
        [0, 180, 0, 256],    # hue range [0,180), sat range [0,256)
    )

    norm = hist.sum()
    if norm > 0:
        hist = hist / norm   # L1 normalise → sum to 1.0
    return hist.astype(np.float32)


def _compute_orb(bgr: np.ndarray) -> tuple[list, DescriptorArray]:
    """Compute ORB keypoints and descriptors.

    Parameters
    ----------
    bgr:
        BGR image.

    Returns
    -------
    (keypoints, descriptors)
        ``descriptors`` is None if fewer than 2 keypoints were found.
    """
    orb = _get_orb()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    kps, descs = orb.detectAndCompute(gray, None)
    if descs is None or len(kps) < 2:
        return kps or [], None
    return list(kps), descs


def _load_reference_image(path: Path) -> np.ndarray | None:
    """Load a reference PNG and crop to the art region.

    Returns BGR numpy array, or None if the file cannot be decoded.
    Transparency (alpha channel in BGRA) is composited onto white,
    since the card reference art has transparent backgrounds.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Handle BGRA (transparent PNG) → composite onto white background
    if img.ndim == 3 and img.shape[2] == 4:
        bgra = img
        alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
        bgr_float = bgra[:, :, :3].astype(np.float32)
        white = np.ones_like(bgr_float) * 255.0
        bgr_float = bgr_float * alpha + white * (1.0 - alpha)
        img = np.clip(bgr_float, 0, 255).astype(np.uint8)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Crop to the art region (top 80% of the reference image)
    h = img.shape[0]
    crop_h = max(1, int(h * _ART_CROP_BOTTOM))
    img = img[:crop_h, :]

    return img


def _identity_from_dir(dir_name: str) -> str:
    """Convert a directory name to a canonical identity string.

    Applies alias resolution (Twin_Minion → Minion).
    Underscores are kept in the identity string (e.g. "Fortune_Teller").
    """
    return _IDENTITY_ALIASES.get(dir_name, dir_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_gallery(art_root: Path | str | None = None) -> Gallery:
    """Build an in-memory reference gallery from the card-art directory tree.

    Walks ``art_root/<role_class>/<identity>/<file>.png`` and, for each PNG,
    precomputes HSV histograms, ORB descriptors, and a thumbnail.  Returns
    a ``Gallery`` object ready for use by ``classify_crop``.

    **No files are written.**  Re-running this function on a new art directory
    (after an art swap) rebuilds the gallery with zero training.

    Parameters
    ----------
    art_root:
        Path to ``knowledge-base/card-art/``.  Defaults to the canonical
        location relative to the repo root (anchored via ``__file__``).

    Returns
    -------
    Gallery
        Populated in-memory gallery.  ``gallery.n_references`` == number of
        PNGs found; ``gallery.n_townees`` == number of distinct identities.

    Raises
    ------
    FileNotFoundError
        If ``art_root`` does not exist.
    ValueError
        If no PNG files are found under ``art_root``.
    """
    # Anchor default path to the repo root via __file__
    # __file__ = src/dbcv/gallery.py → parents[2] = repo root
    if art_root is None:
        _repo_root = Path(__file__).resolve().parents[2]
        art_root = _repo_root / "knowledge-base" / "card-art"
    else:
        art_root = Path(art_root).resolve()

    if not art_root.exists():
        raise FileNotFoundError(f"Card-art root not found: {art_root}")

    entries: list[GalleryEntry] = []
    seen_identities: set[str] = set()
    seen_role_classes: set[str] = set()

    # Walk role_class directories (villager, minion, outcast, demon)
    for role_class_dir in sorted(art_root.iterdir()):
        if not role_class_dir.is_dir():
            continue
        role_class = role_class_dir.name.lower()

        # Walk identity (townee) directories within each role class
        for identity_dir in sorted(role_class_dir.iterdir()):
            if not identity_dir.is_dir():
                continue

            raw_identity = identity_dir.name
            identity = _identity_from_dir(raw_identity)

            # Load every PNG in the identity directory (base + skins)
            for png_path in sorted(identity_dir.glob("*.png")):
                bgr = _load_reference_image(png_path)
                if bgr is None:
                    continue   # skip unreadable files

                # Resize to thumbnail
                thumb = cv2.resize(
                    bgr,
                    (_THUMB_SIZE, _THUMB_SIZE),
                    interpolation=cv2.INTER_AREA,
                )

                # Precompute matchable representations
                hsv_hist = _compute_hsv_hist(bgr)
                orb_kp, orb_desc = _compute_orb(bgr)

                entries.append(
                    GalleryEntry(
                        identity=identity,
                        role_class=role_class,
                        file_stem=png_path.stem,
                        thumb=thumb,
                        hsv_hist=hsv_hist,
                        orb_kp=orb_kp,
                        orb_desc=orb_desc,
                    )
                )

                seen_identities.add(identity)
                seen_role_classes.add(role_class)

    if not entries:
        raise ValueError(f"No PNG files found under {art_root}")

    gallery = Gallery(
        entries=entries,
        townee_names=sorted(seen_identities),
        role_classes=sorted(seen_role_classes),
    )
    return gallery
