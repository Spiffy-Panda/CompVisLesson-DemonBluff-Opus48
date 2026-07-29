"""
tests/test_identify.py — Unit tests for the Stage 2 classical card identifier.

Tests that:
  - ``classify_crop`` returns a valid schema tuple (str, str, float in [0,1]).
  - The returned identity is either "unknown" or a gallery townee name.
  - The returned role_class is one of the five valid strings.
  - ``classify_crop`` with a blank (black) crop returns "unknown" with low confidence.
  - ``classify_crop`` with a reference image crop returns the matching townee.
  - ``make_gallery_identifier`` wraps the classifier into the 1-arg pipeline interface.
  - ``stub_identify`` and ``identify`` still return ("unknown", "unknown", 0.0).
  - The API end-to-end (POST /v1/snapshot) returns valid GameStateSnapshot when
    the gallery identifier is active via the lifespan.

Rule 1 compliance: no inline interpreter calls.  No ``import`` on the command
line.  All code runs through pytest.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dbcv.api import app
from dbcv.gallery import Gallery, build_gallery
from dbcv.identify import (
    classify_crop,
    combine_identifications,
    identify,
    make_ensemble_identifier,
    make_gallery_identifier,
    stub_identify,
)
from dbcv.schema import GameStateSnapshot

# ---------------------------------------------------------------------------
# Paths — anchored to repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ART_ROOT = _REPO_ROOT / "knowledge-base" / "card-art"
_FRAMES_DIR = _REPO_ROOT / "dataset" / "frames" / "Sample1"
_PREFERRED_STEM = "Sample1_003"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gallery() -> Gallery:
    if not _ART_ROOT.exists():
        pytest.skip(f"Card-art directory not found: {_ART_ROOT}")
    return build_gallery(_ART_ROOT)


@pytest.fixture
def blank_crop() -> np.ndarray:
    """A 64×64 all-black image — represents a structureless (e.g., face-down) crop."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def reference_crop(gallery: Gallery) -> tuple[np.ndarray, str, str]:
    """Load the actual reference PNG for one townee (Alchemist) as if it were a crop.

    This is the optimistic case: the reference image matched against itself
    should yield the highest possible histogram correlation.

    Returns (image, expected_identity, expected_role_class).
    """
    alchemist_path = _ART_ROOT / "villager" / "Alchemist" / "Alchemist.png"
    if not alchemist_path.exists():
        pytest.skip(f"Alchemist reference art not found: {alchemist_path}")
    img = cv2.imread(str(alchemist_path), cv2.IMREAD_COLOR)
    if img is None:
        pytest.skip("cv2.imread returned None for Alchemist.png")
    return img, "Alchemist", "villager"


# ---------------------------------------------------------------------------
# classify_crop: schema validity
# ---------------------------------------------------------------------------


VALID_ROLE_CLASSES = {"villager", "minion", "outcast", "demon", "unknown"}


def test_classify_crop_returns_valid_schema(gallery: Gallery, blank_crop: np.ndarray) -> None:
    """classify_crop must return (str, str, float) with confidence in [0, 1]."""
    identity, role_class, confidence = classify_crop(blank_crop, gallery)
    assert isinstance(identity, str)
    assert isinstance(role_class, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0, (
        f"Confidence {confidence} is outside [0, 1]."
    )


def test_classify_crop_identity_is_gallery_or_unknown(
    gallery: Gallery, blank_crop: np.ndarray
) -> None:
    """Identity must be a gallery townee name or 'unknown'."""
    identity, _, _ = classify_crop(blank_crop, gallery)
    assert identity == "unknown" or identity in gallery.townee_names, (
        f"Identity {identity!r} is neither 'unknown' nor a gallery townee name."
    )


def test_classify_crop_role_class_is_valid(
    gallery: Gallery, blank_crop: np.ndarray
) -> None:
    """role_class must be one of the five valid strings."""
    _, role_class, _ = classify_crop(blank_crop, gallery)
    assert role_class in VALID_ROLE_CLASSES, (
        f"role_class {role_class!r} is not in {VALID_ROLE_CLASSES!r}."
    )


# ---------------------------------------------------------------------------
# classify_crop: blank / face-down crop → "unknown"
# ---------------------------------------------------------------------------


def test_blank_crop_returns_unknown(gallery: Gallery, blank_crop: np.ndarray) -> None:
    """A black (blank) crop should return 'unknown' with low confidence.

    This simulates a face-down card: no character art, uniform colour.
    Low confidence + 'unknown' identity is the CORRECT result.
    """
    identity, role_class, confidence = classify_crop(blank_crop, gallery)
    assert identity == "unknown", (
        f"Blank crop returned identity={identity!r}; expected 'unknown'."
    )
    assert role_class == "unknown", (
        f"Blank crop returned role_class={role_class!r}; expected 'unknown'."
    )
    assert confidence < 0.40, (
        f"Blank crop confidence {confidence} >= 0.40 — "
        "the threshold should gate face-down cards as 'unknown'."
    )


def test_zero_size_crop_returns_unknown(gallery: Gallery) -> None:
    """A zero-size array (degenerate crop) must return 'unknown', not crash."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    identity, role_class, confidence = classify_crop(empty, gallery)
    assert identity == "unknown"
    assert role_class == "unknown"
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# classify_crop: reference image self-match
# ---------------------------------------------------------------------------


def test_reference_self_match_returns_correct_identity(
    gallery: Gallery,
    reference_crop: tuple[np.ndarray, str, str],
) -> None:
    """Matching a reference image against itself should yield the correct identity.

    This is the best-case scenario: the gallery histogram was built from this
    exact image, so the correlation should be near 1.0 and the identity should
    match.  If this test fails, the histogram comparison or gallery building
    has a bug.
    """
    img, expected_identity, expected_role_class = reference_crop
    identity, role_class, confidence = classify_crop(img, gallery)
    assert identity == expected_identity, (
        f"Self-match failed: expected identity={expected_identity!r}, "
        f"got {identity!r} with confidence={confidence:.3f}."
    )
    assert role_class == expected_role_class, (
        f"Self-match: expected role_class={expected_role_class!r}, got {role_class!r}."
    )
    assert confidence >= 0.40, (
        f"Self-match confidence {confidence:.3f} is below threshold 0.40 — "
        "the histogram comparison may be broken."
    )


# ---------------------------------------------------------------------------
# make_gallery_identifier: interface bridging
# ---------------------------------------------------------------------------


def test_make_gallery_identifier_returns_callable(gallery: Gallery) -> None:
    """make_gallery_identifier must return a callable."""
    fn = make_gallery_identifier(gallery)
    assert callable(fn), "make_gallery_identifier must return a callable."


def test_make_gallery_identifier_returns_valid_schema(gallery: Gallery) -> None:
    """The returned callable must produce valid (str, str, float) tuples."""
    fn = make_gallery_identifier(gallery)
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    result = fn(crop)
    assert len(result) == 3
    identity, role_class, confidence = result
    assert isinstance(identity, str)
    assert isinstance(role_class, str)
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# Legacy stubs
# ---------------------------------------------------------------------------


def test_stub_identify_returns_unknown() -> None:
    """stub_identify must return ('unknown', 'unknown', 0.0) always."""
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    result = stub_identify(crop)
    assert result == ("unknown", "unknown", 0.0), (
        f"stub_identify returned {result!r}; expected ('unknown', 'unknown', 0.0)."
    )


def test_legacy_identify_returns_unknown() -> None:
    """The legacy identify() function must still return the stub result."""
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    result = identify(crop)
    assert result == ("unknown", "unknown", 0.0), (
        f"identify() returned {result!r}; expected stub result."
    )


# ---------------------------------------------------------------------------
# Ensemble combiner (2026-07-29, plans/PLAN-live-capture.md Fix 3)
# ---------------------------------------------------------------------------
#
# combine_identifications is a pure function of two (identity, role_class,
# confidence) stub tuples -- no gallery/embedder needed, so these tests
# exercise every combination rule directly and unconditionally (no skips).


def test_combine_both_unknown_returns_unknown() -> None:
    classical = ("unknown", "unknown", 0.1)
    embedding = ("unknown", "unknown", 0.05)
    identity, role_class, confidence, source = combine_identifications(classical, embedding)
    assert (identity, role_class) == ("unknown", "unknown")
    assert confidence == 0.0
    assert source == "both_unknown"


def test_combine_agreement_boosts_confidence() -> None:
    """Same identity on both sides -> boosted confidence, source 'agree'."""
    classical = ("Wretch", "outcast", 0.60)
    embedding = ("Wretch", "outcast", 0.30)
    identity, role_class, confidence, source = combine_identifications(classical, embedding)
    assert identity == "Wretch"
    assert role_class == "outcast"
    assert source == "agree"
    # Boosted above the stronger (classical) input, capped at 1.0.
    assert confidence > 0.60
    assert confidence <= 1.0


def test_combine_agreement_confidence_is_capped_at_one() -> None:
    classical = ("Judge", "villager", 0.95)
    embedding = ("Judge", "villager", 0.90)
    _, _, confidence, source = combine_identifications(classical, embedding)
    assert source == "agree"
    assert confidence == 1.0


def test_combine_classical_abstains_adopts_embedding() -> None:
    """Classical unknown, embedding answers -> embedding's result, unchanged."""
    classical = ("unknown", "unknown", 0.10)
    embedding = ("Judge", "villager", 0.22)
    identity, role_class, confidence, source = combine_identifications(classical, embedding)
    assert (identity, role_class, confidence) == ("Judge", "villager", 0.22)
    assert source == "embedding_only"


def test_combine_embedding_abstains_adopts_classical() -> None:
    """Embedding unknown, classical answers -> classical's result, unchanged."""
    classical = ("Wretch", "outcast", 0.55)
    embedding = ("unknown", "unknown", 0.04)
    identity, role_class, confidence, source = combine_identifications(classical, embedding)
    assert (identity, role_class, confidence) == ("Wretch", "outcast", 0.55)
    assert source == "classical_only"


def test_combine_disagreement_abstains_not_higher_confidence() -> None:
    """Different identities on both sides -> abstain, even though classical's
    raw confidence is numerically higher.

    This is the exact Poisoner/Hunter shape found in eval_02
    (collect_02/018.png): classical said Poisoner@0.47, embedding said
    Hunter@0.20, and ground truth (visually confirmed) was Hunter. A
    "prefer higher confidence" rule would pick classical's wrong answer here
    -- the ensemble must NOT do that.
    """
    classical = ("Poisoner", "minion", 0.47)   # higher raw confidence, WRONG
    embedding = ("Hunter", "villager", 0.20)   # lower raw confidence, RIGHT
    identity, role_class, confidence, source = combine_identifications(classical, embedding)
    assert identity == "unknown", (
        "Disagreement must abstain, not silently pick the higher raw confidence "
        f"(which would wrongly select {classical[0]!r} here)."
    )
    assert role_class == "unknown"
    assert confidence == 0.0
    assert source == "disagree_abstain"


@pytest.mark.parametrize(
    "classical,embedding",
    [
        (("Empress", "villager", 0.9), ("Knight", "villager", 0.9)),
        (("Slayer", "villager", 0.41), ("Witch", "minion", 0.13)),
    ],
)
def test_combine_disagreement_always_abstains(
    classical: tuple[str, str, float], embedding: tuple[str, str, float]
) -> None:
    identity, role_class, confidence, source = combine_identifications(classical, embedding)
    assert identity == "unknown"
    assert role_class == "unknown"
    assert confidence == 0.0
    assert source == "disagree_abstain"


def test_make_ensemble_identifier_returns_callable() -> None:
    fn = make_ensemble_identifier(
        lambda crop: ("Wretch", "outcast", 0.6),
        lambda crop: ("unknown", "unknown", 0.05),
    )
    assert callable(fn)


def test_make_ensemble_identifier_calls_both_and_combines() -> None:
    """The pipeline-facing wrapper drops the source tag but keeps the combined result."""
    classical_stub = lambda crop: ("Wretch", "outcast", 0.6)  # noqa: E731
    embedding_stub = lambda crop: ("unknown", "unknown", 0.05)  # noqa: E731
    fn = make_ensemble_identifier(classical_stub, embedding_stub)

    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    result = fn(crop)

    assert len(result) == 3   # 3-tuple pipeline contract, no 4th 'source' element
    identity, role_class, confidence = result
    assert (identity, role_class, confidence) == ("Wretch", "outcast", 0.6)


def test_make_ensemble_identifier_disagreement_abstains() -> None:
    classical_stub = lambda crop: ("Poisoner", "minion", 0.47)  # noqa: E731
    embedding_stub = lambda crop: ("Hunter", "villager", 0.20)  # noqa: E731
    fn = make_ensemble_identifier(classical_stub, embedding_stub)

    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    identity, role_class, confidence = fn(crop)
    assert identity == "unknown"
    assert role_class == "unknown"
    assert confidence == 0.0


def test_ensemble_identifier_end_to_end_with_real_gallery(gallery: Gallery) -> None:
    """Real classify_crop wired through the ensemble on a blank crop stays honest."""
    real_classical = make_gallery_identifier(gallery)
    stub_embedding = lambda crop: ("unknown", "unknown", 0.0)  # noqa: E731
    fn = make_ensemble_identifier(real_classical, stub_embedding)

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    identity, role_class, confidence = fn(blank)
    # A blank/black crop should not be confidently identified by either arm.
    assert identity == "unknown"
    assert role_class == "unknown"
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# API end-to-end: gallery identifier wired via lifespan
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient that triggers the lifespan (gallery build)."""
    with TestClient(app) as tc:
        yield tc


def _find_board_frame() -> Path | None:
    """Find Sample1_003 (preferred validated board frame)."""
    frames = sorted(_FRAMES_DIR.glob(f"{_PREFERRED_STEM}*.png"))
    return frames[0] if frames else None


def test_api_returns_200_with_gallery(client: TestClient) -> None:
    """POST /v1/snapshot returns 200 when the gallery identifier is active."""
    frame_path = _find_board_frame()
    if frame_path is None:
        pytest.skip(f"{_PREFERRED_STEM}*.png not found — run frame extraction.")

    with open(frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
            data={"video": "Sample1", "frame_index": 3, "timestamp_s": 460.0},
        )
    assert response.status_code == 200, (
        f"Expected 200 with gallery active; got {response.status_code}. "
        f"Body: {response.text[:500]}"
    )


def test_api_snapshot_parses_valid_game_state(client: TestClient) -> None:
    """Response with gallery active must parse as a valid GameStateSnapshot."""
    frame_path = _find_board_frame()
    if frame_path is None:
        pytest.skip(f"{_PREFERRED_STEM}*.png not found — run frame extraction.")

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


def test_api_cards_have_valid_confidence(client: TestClient) -> None:
    """Every card in the gallery-backed snapshot must have confidence in [0, 1]."""
    frame_path = _find_board_frame()
    if frame_path is None:
        pytest.skip(f"{_PREFERRED_STEM}*.png not found — run frame extraction.")

    with open(frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    assert response.status_code == 200
    snap = GameStateSnapshot.model_validate(response.json())

    for i, card in enumerate(snap.cards):
        assert 0.0 <= card.confidence <= 1.0, (
            f"Card {i} has confidence={card.confidence} outside [0, 1]."
        )
        assert card.identity == "unknown" or card.identity in [
            e.identity for e in build_gallery().entries
        ], (
            f"Card {i} identity={card.identity!r} is not a gallery townee or 'unknown'."
        )
        assert card.role_class in VALID_ROLE_CLASSES, (
            f"Card {i} role_class={card.role_class!r} is not in {VALID_ROLE_CLASSES!r}."
        )
