"""
tests/test_embed.py — Unit tests for the Stage 3 embedding identifier.

Tests that:
  - OnnxEmbedder loads successfully (ONNX file present).
  - embed() returns a unit-norm vector of the expected dimension (576).
  - embed() handles zero-size / blank crops gracefully (returns zeros, no crash).
  - preprocess() returns the expected shape / dtype.
  - EmbeddingGallery builds with the correct number of classes.
  - classify_crop_embedding() returns valid (str, str, float in [0,1]) schema.
  - The returned identity is either "unknown" or a gallery townee name.
  - classify_crop_embedding() returns "unknown" for a blank (black) crop.
  - make_embedding_identifier() wraps into the 1-arg pipeline interface.
  - End-to-end API test: POST /v1/snapshot returns 200 + valid snapshot
    with the embedding identifier wired via lifespan.

Performance note
----------------
OnnxEmbedder and EmbeddingGallery are session/module-scoped to avoid
rebuilding them for every test (~1–2 s build time combined).
The classical gallery fixture is also module-scoped, consistent with
the convention in test_identify.py.

Rule 1 compliance: no inline interpreter calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dbcv.api import app
from dbcv.embed import EMBEDDING_DIM, OnnxEmbedder, get_onnx_embedder
from dbcv.gallery import EmbeddingGallery, build_embedding_gallery, build_gallery
from dbcv.identify import classify_crop_embedding, make_embedding_identifier
from dbcv.schema import GameStateSnapshot

# ---------------------------------------------------------------------------
# Paths — anchored to repo root via __file__
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ART_ROOT = _REPO_ROOT / "knowledge-base" / "card-art"
_ONNX_PATH = _REPO_ROOT / "models" / "mobilenetv3_small_embed.onnx"
_FRAMES_DIR = _REPO_ROOT / "dataset" / "frames" / "Sample1"
_PREFERRED_STEM = "Sample1_003"

# ---------------------------------------------------------------------------
# Fixtures — module-scoped to share across tests (avoid per-test rebuild)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> OnnxEmbedder:
    """Load the ONNX embedder once for the module (or reuse the process-level cache)."""
    if not _ONNX_PATH.exists():
        pytest.skip(
            f"ONNX model not found: {_ONNX_PATH}\n"
            "Run: .venv\\Scripts\\python.exe utils\\python\\export_backbone.py"
        )
    return get_onnx_embedder(_ONNX_PATH)


@pytest.fixture(scope="module")
def classical_gallery():
    """Build the classical gallery once for the module."""
    if not _ART_ROOT.exists():
        pytest.skip(f"Card-art root not found: {_ART_ROOT}")
    return build_gallery(_ART_ROOT)


@pytest.fixture(scope="module")
def embed_gallery(classical_gallery, embedder) -> EmbeddingGallery:
    """Build the embedding gallery once for the module."""
    return build_embedding_gallery(classical_gallery, embedder, _ART_ROOT)


@pytest.fixture
def blank_crop() -> np.ndarray:
    """64x64 all-black image — structureless, no character art."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def natural_crop() -> np.ndarray:
    """A non-trivial 128x100 crop with random colour to exercise the embedder."""
    rng = np.random.default_rng(seed=7)
    return (rng.integers(0, 256, size=(100, 128, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# OnnxEmbedder tests
# ---------------------------------------------------------------------------


def test_embedder_loads(embedder: OnnxEmbedder) -> None:
    """OnnxEmbedder should load without raising FileNotFoundError."""
    # The fixture already loads it; if we get here without an exception, it passed.
    assert embedder is not None


def test_embed_returns_expected_shape(embedder: OnnxEmbedder, natural_crop: np.ndarray) -> None:
    """embed() must return a 1-D array of length EMBEDDING_DIM (576)."""
    vec = embedder.embed(natural_crop)
    assert vec.ndim == 1, f"Expected 1-D vector, got shape {vec.shape}"
    assert vec.shape[0] == EMBEDDING_DIM, (
        f"Expected {EMBEDDING_DIM}-dim embedding, got {vec.shape[0]}."
    )


def test_embed_returns_unit_norm(embedder: OnnxEmbedder, natural_crop: np.ndarray) -> None:
    """embed() must return a unit-norm (L2-normalised) vector.

    This is critical: the cosine similarity gallery lookup uses dot product
    on the assumption that all vectors are unit-norm.
    """
    vec = embedder.embed(natural_crop)
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5, (
        f"Embedding is not unit-norm: ||v|| = {norm:.6f} (expected ~1.0)."
    )


def test_embed_blank_crop_returns_no_crash(embedder: OnnxEmbedder, blank_crop: np.ndarray) -> None:
    """embed() on an all-black crop must not crash and must return a float32 array."""
    vec = embedder.embed(blank_crop)
    assert vec is not None
    assert vec.dtype == np.float32


def test_embed_zero_size_returns_zeros(embedder: OnnxEmbedder) -> None:
    """embed() with a zero-size array must return a zero vector of EMBEDDING_DIM."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    vec = embedder.embed(empty)
    assert vec.shape[0] == EMBEDDING_DIM
    assert np.all(vec == 0), "Expected zero vector for empty input."


def test_preprocess_output_shape(embedder: OnnxEmbedder, natural_crop: np.ndarray) -> None:
    """preprocess() must return shape [1, 3, 224, 224] float32."""
    x = embedder.preprocess(natural_crop)
    assert x.shape == (1, 3, 224, 224), (
        f"Expected (1, 3, 224, 224), got {x.shape}."
    )
    assert x.dtype == np.float32, (
        f"Expected float32, got {x.dtype}."
    )


def test_preprocess_imagenet_range(embedder: OnnxEmbedder, natural_crop: np.ndarray) -> None:
    """preprocess() output should span roughly the ImageNet normalised range.

    After ImageNet normalisation, pixel values roughly cover [-2.5, 2.5].
    A test image with varied colours should have min < 0 and max > 0.
    This guards against accidental omission of the normalisation step.
    """
    x = embedder.preprocess(natural_crop)
    assert x.min() < -0.1, "Normalised values should dip below 0 (ImageNet mean subtracted)."
    assert x.max() > 0.1, "Normalised values should rise above 0."


# ---------------------------------------------------------------------------
# EmbeddingGallery tests
# ---------------------------------------------------------------------------


def test_embedding_gallery_n_classes(embed_gallery: EmbeddingGallery) -> None:
    """EmbeddingGallery must have at least 43 classes (one per townee)."""
    assert embed_gallery.n_classes >= 43, (
        f"Expected >= 43 classes; got {embed_gallery.n_classes}."
    )


def test_embedding_gallery_matrix_shape(embed_gallery: EmbeddingGallery) -> None:
    """The stacked embeddings matrix must be [K, EMBEDDING_DIM]."""
    K = embed_gallery.n_classes
    assert embed_gallery.embeddings.shape == (K, EMBEDDING_DIM), (
        f"Expected ({K}, {EMBEDDING_DIM}), got {embed_gallery.embeddings.shape}."
    )


def test_embedding_gallery_prototypes_unit_norm(embed_gallery: EmbeddingGallery) -> None:
    """Every prototype embedding must be unit-norm (required for cosine=dot product)."""
    norms = np.linalg.norm(embed_gallery.embeddings, axis=1)
    max_dev = float(np.max(np.abs(norms - 1.0)))
    assert max_dev < 1e-5, (
        f"Gallery prototypes are not unit-norm: max |norm-1| = {max_dev:.2e}."
    )


# ---------------------------------------------------------------------------
# classify_crop_embedding tests
# ---------------------------------------------------------------------------

VALID_ROLE_CLASSES = {"villager", "minion", "outcast", "demon", "unknown"}


def test_classify_crop_embedding_returns_valid_schema(
    blank_crop: np.ndarray,
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
) -> None:
    """classify_crop_embedding must return (str, str, float) with confidence in [0,1]."""
    identity, role_class, confidence = classify_crop_embedding(blank_crop, embedder, embed_gallery)
    assert isinstance(identity, str)
    assert isinstance(role_class, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} outside [0,1]."


def test_classify_crop_embedding_identity_is_gallery_or_unknown(
    blank_crop: np.ndarray,
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
) -> None:
    """Identity must be a gallery townee name or 'unknown'."""
    identity, _, _ = classify_crop_embedding(blank_crop, embedder, embed_gallery)
    assert identity == "unknown" or identity in embed_gallery.townee_names, (
        f"Identity {identity!r} is neither 'unknown' nor a gallery townee name."
    )


def test_classify_crop_embedding_role_class_valid(
    blank_crop: np.ndarray,
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
) -> None:
    """role_class must be one of the five valid strings."""
    _, role_class, _ = classify_crop_embedding(blank_crop, embedder, embed_gallery)
    assert role_class in VALID_ROLE_CLASSES, (
        f"role_class {role_class!r} not in {VALID_ROLE_CLASSES!r}."
    )


def test_classify_crop_embedding_blank_returns_unknown(
    blank_crop: np.ndarray,
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
) -> None:
    """A completely black crop must return 'unknown'.

    The embed() method returns a zero vector for degenerate inputs, which
    produces zero cosine similarity against all gallery entries -> "unknown".
    """
    identity, role_class, confidence = classify_crop_embedding(
        blank_crop, embedder, embed_gallery
    )
    assert identity == "unknown", (
        f"Blank crop returned identity={identity!r}; expected 'unknown'."
    )
    assert role_class == "unknown", (
        f"Blank crop returned role_class={role_class!r}; expected 'unknown'."
    )


def test_classify_crop_embedding_zero_size_returns_unknown(
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
) -> None:
    """A zero-size array must return 'unknown', not crash."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    identity, role_class, confidence = classify_crop_embedding(empty, embedder, embed_gallery)
    assert identity == "unknown"
    assert role_class == "unknown"
    assert 0.0 <= confidence <= 1.0


def test_classify_crop_embedding_natural_crop_valid_schema(
    natural_crop: np.ndarray,
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
) -> None:
    """A non-trivial random crop must return a valid (str, str, float) tuple."""
    identity, role_class, confidence = classify_crop_embedding(
        natural_crop, embedder, embed_gallery
    )
    assert isinstance(identity, str)
    assert isinstance(role_class, str)
    assert 0.0 <= confidence <= 1.0
    assert role_class in VALID_ROLE_CLASSES
    assert identity == "unknown" or identity in embed_gallery.townee_names


# ---------------------------------------------------------------------------
# make_embedding_identifier tests
# ---------------------------------------------------------------------------


def test_make_embedding_identifier_returns_callable(
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
) -> None:
    """make_embedding_identifier must return a callable."""
    fn = make_embedding_identifier(embedder, embed_gallery)
    assert callable(fn)


def test_make_embedding_identifier_returns_valid_schema(
    embedder: OnnxEmbedder,
    embed_gallery: EmbeddingGallery,
    blank_crop: np.ndarray,
) -> None:
    """The returned callable must produce valid (str, str, float) tuples."""
    fn = make_embedding_identifier(embedder, embed_gallery)
    result = fn(blank_crop)
    assert len(result) == 3
    identity, role_class, confidence = result
    assert isinstance(identity, str)
    assert isinstance(role_class, str)
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# End-to-end API test with embedding identifier
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient that triggers the lifespan (builds gallery + embedder)."""
    with TestClient(app) as tc:
        yield tc


def _find_board_frame() -> Path | None:
    frames = sorted(_FRAMES_DIR.glob(f"{_PREFERRED_STEM}*.png"))
    return frames[0] if frames else None


def test_api_200_with_embedding_identifier(client: TestClient) -> None:
    """POST /v1/snapshot returns 200 when the embedding identifier is active."""
    frame_path = _find_board_frame()
    if frame_path is None:
        pytest.skip(f"{_PREFERRED_STEM}*.png not found -- run frame extraction.")

    with open(frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
            data={"video": "Sample1", "frame_index": 3, "timestamp_s": 460.0},
        )
    assert response.status_code == 200, (
        f"Expected 200; got {response.status_code}. Body: {response.text[:500]}"
    )


def test_api_snapshot_valid_with_embedding(client: TestClient) -> None:
    """Response with embedding identifier active must parse as valid GameStateSnapshot."""
    frame_path = _find_board_frame()
    if frame_path is None:
        pytest.skip(f"{_PREFERRED_STEM}*.png not found -- run frame extraction.")

    with open(frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    assert response.status_code == 200
    snap = GameStateSnapshot.model_validate(response.json())
    assert snap.schema_version == "0.2.0"
    assert snap.frame_state == "board"
    assert len(snap.cards) >= 4


def test_api_cards_confidence_in_unit_range_embedding(client: TestClient) -> None:
    """Every card confidence from the embedding identifier must be in [0,1]."""
    frame_path = _find_board_frame()
    if frame_path is None:
        pytest.skip(f"{_PREFERRED_STEM}*.png not found -- run frame extraction.")

    with open(frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    assert response.status_code == 200
    snap = GameStateSnapshot.model_validate(response.json())

    for i, card in enumerate(snap.cards):
        assert 0.0 <= card.confidence <= 1.0, (
            f"Card {i} has confidence={card.confidence} outside [0,1]."
        )
        assert card.identity == "unknown" or card.identity in embed_gallery_entries_from_api(), (
            f"Card {i} identity={card.identity!r} is not 'unknown' or a known townee."
        )
        assert card.role_class in VALID_ROLE_CLASSES


def embed_gallery_entries_from_api() -> list[str]:
    """Helper: return townee names from a freshly built gallery (for identity check)."""
    if not _ART_ROOT.exists():
        return []
    g = build_gallery(_ART_ROOT)
    return g.townee_names
