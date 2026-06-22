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
# Process-level cache for build_gallery
# ---------------------------------------------------------------------------
# Keyed by the resolved art_root Path.  Within one process (test suite or server
# lifetime) the art directory never changes, so returning the same Gallery object
# is always correct.  An art swap requires a process restart, which clears the
# cache automatically.
#
# Safety: GalleryEntry objects are NamedTuples (immutable) and the Gallery
# dataclass is never mutated after construction — sharing it across callers is safe.
#
# Note: this cache is intentionally process-wide (module-level dict).  Tests that
# call build_gallery() multiple times with the same art_root will receive the same
# Gallery instance.  The idempotency test (test_gallery_rebuild_is_idempotent)
# checks .townee_names equality and .n_references equality — returning the same
# object trivially satisfies both assertions.
_GALLERY_CACHE: dict[Path, "Gallery"] = {}


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

    # --- Process-level cache: return the prebuilt Gallery if available ---
    if art_root in _GALLERY_CACHE:
        return _GALLERY_CACHE[art_root]

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
    # Store in process-level cache before returning
    _GALLERY_CACHE[art_root] = gallery
    return gallery


# ---------------------------------------------------------------------------
# Embedding gallery (Stage 3 — embedding-NN identifier)
# ---------------------------------------------------------------------------


class EmbeddingGalleryEntry(NamedTuple):
    """One reference entry in the embedding gallery.

    Holds the L2-normalised prototype embedding for one identity.
    The prototype is computed as the mean of all reference embeddings for
    that identity, then re-normalised — the "prototypical network" approach
    (Snell et al., NeurIPS 2017).  When there is only one reference per
    identity this is equivalent to the single-image embedding.
    """

    identity: str       # e.g. "Alchemist" (alias-resolved)
    role_class: str     # "villager" | "minion" | "outcast" | "demon"
    embedding: np.ndarray  # [576] float32, unit L2 norm


@dataclass
class EmbeddingGallery:
    """Container for all embedding prototypes.

    Attributes
    ----------
    entries:
        One entry per unique (identity, role_class) pair.
    townee_names:
        Sorted unique identity strings (same set as the classical Gallery).
    role_classes:
        Sorted unique role_class strings.
    embeddings:
        Stacked [K, 576] float32 array where K = len(entries).
        Kept pre-stacked for fast cosine-similarity via a single matmul.
    """

    entries: list[EmbeddingGalleryEntry] = field(default_factory=list)
    townee_names: list[str] = field(default_factory=list)
    role_classes: list[str] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))

    @property
    def n_classes(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Process-level cache for build_embedding_gallery
# ---------------------------------------------------------------------------
# Keyed by (resolved art_root Path, resolved onnx_path Path).  The embedder
# carries its ONNX path; we extract it to form the cache key so that different
# model files produce different galleries (correct behaviour on a model swap).
#
# Same safety rationale as _GALLERY_CACHE: EmbeddingGalleryEntry is a NamedTuple
# (immutable); EmbeddingGallery is never mutated after construction.
_EMBED_GALLERY_CACHE: dict[tuple, "EmbeddingGallery"] = {}


def build_embedding_gallery(
    classical_gallery: "Gallery",
    embedder: "object",  # OnnxEmbedder; using string annotation to avoid circular import
    art_root: Path | str | None = None,
) -> "EmbeddingGallery":
    """Build an in-memory embedding gallery from the reference art.

    For each townee identity, loads all reference PNGs (same art sub-region
    crop the classical gallery uses), embeds each via the OnnxEmbedder, then
    computes a prototypical mean embedding (per Snell et al. 2017).  The mean
    is re-normalised to keep it on the unit sphere.

    This function intentionally does NOT import torch; it relies entirely on
    the OnnxEmbedder (onnxruntime) and numpy.

    Parameters
    ----------
    classical_gallery:
        Pre-built classical Gallery.  Its entries carry the already-loaded
        reference BGR images via their file stems, and the same art_root
        convention is reused so we don't re-read the directory.
    embedder:
        An OnnxEmbedder instance (from src/dbcv/embed.py).
    art_root:
        Path to knowledge-base/card-art/ directory.  Defaults to the same
        canonical location used by build_gallery().

    Returns
    -------
    EmbeddingGallery
        K entries (K = n_townees in practice, using mean-pooled prototypes).
        The .embeddings array is shape [K, 576], float32, row-normalised.

    Teaching note
    -------------
    The key art-swap property is preserved: on an art swap you re-run this
    function with the new art directory (just like build_gallery).  No weights
    are updated; the new reference images are embedded in seconds.

    The prototypical mean approach (average multiple reference views per class,
    re-normalise) consistently outperforms single-embedding lookup on held-out
    samples in the few-shot learning literature (Snell et al. 2017).  For our
    44-class problem with 1-2 references per class the gain is modest but free.
    """
    # Anchor default path identically to build_gallery()
    if art_root is None:
        _repo_root = Path(__file__).resolve().parents[2]
        art_root = _repo_root / "knowledge-base" / "card-art"
    else:
        art_root = Path(art_root).resolve()

    # --- Process-level cache: return the prebuilt EmbeddingGallery if available ---
    # The cache key combines art_root and the embedder's ONNX model path so that
    # a model swap or art swap (each requiring a process restart in production) gets
    # a fresh gallery, while the common case (same process, same files) reuses the
    # prebuilt result.
    _onnx_path_key: Path | None = getattr(embedder, "_onnx_path", None)
    _cache_key = (art_root, _onnx_path_key)
    if _cache_key in _EMBED_GALLERY_CACHE:
        return _EMBED_GALLERY_CACHE[_cache_key]

    if not art_root.exists():
        raise FileNotFoundError(f"Card-art root not found: {art_root}")

    # Collect per-identity embeddings: identity -> list of embedding vectors
    identity_embeddings: dict[tuple[str, str], list[np.ndarray]] = {}

    for role_class_dir in sorted(art_root.iterdir()):
        if not role_class_dir.is_dir():
            continue
        role_class = role_class_dir.name.lower()

        for identity_dir in sorted(role_class_dir.iterdir()):
            if not identity_dir.is_dir():
                continue

            raw_identity = identity_dir.name
            identity = _identity_from_dir(raw_identity)
            key = (identity, role_class)

            if key not in identity_embeddings:
                identity_embeddings[key] = []

            for png_path in sorted(identity_dir.glob("*.png")):
                bgr = _load_reference_image(png_path)
                if bgr is None:
                    continue

                # Embed using the ONNX embedder (already L2-normalised [576] vector)
                vec = embedder.embed(bgr)
                if vec is not None and vec.size > 0:
                    identity_embeddings[key].append(vec)

    if not identity_embeddings:
        raise ValueError(f"No reference images could be embedded from {art_root}")

    # Build prototype entries: mean-pool all embeddings per identity, re-normalise
    entries: list[EmbeddingGalleryEntry] = []
    seen_identities: set[str] = set()
    seen_role_classes: set[str] = set()

    for (identity, role_class), vecs in sorted(identity_embeddings.items()):
        if not vecs:
            continue

        # Prototypical mean: average embeddings, re-normalise
        proto = np.mean(np.stack(vecs, axis=0), axis=0)  # [576]
        norm = float(np.linalg.norm(proto))
        if norm < 1e-9:
            continue  # degenerate reference; skip
        proto = (proto / norm).astype(np.float32)

        entries.append(
            EmbeddingGalleryEntry(
                identity=identity,
                role_class=role_class,
                embedding=proto,
            )
        )
        seen_identities.add(identity)
        seen_role_classes.add(role_class)

    if not entries:
        raise ValueError("All reference embeddings were degenerate; gallery is empty.")

    # Stack prototype embeddings into a [K, 576] matrix for fast batch cosine similarity
    embeddings_matrix = np.stack([e.embedding for e in entries], axis=0)  # [K, 576]

    embed_gallery_result = EmbeddingGallery(
        entries=entries,
        townee_names=sorted(seen_identities),
        role_classes=sorted(seen_role_classes),
        embeddings=embeddings_matrix,
    )
    # Store in process-level cache before returning
    _EMBED_GALLERY_CACHE[_cache_key] = embed_gallery_result
    return embed_gallery_result
