"""
tests/test_api.py — Integration test for POST /v1/snapshot.

Tests that:
  - The endpoint returns HTTP 200 for a real sampled board frame.
  - The response validates as a GameStateSnapshot.
  - The ``resolution`` in the response matches the actual image dimensions
    (read independently from PIL — the server must have read them from the
    uploaded bytes, never from a hard-coded value).
  - At least 4 cards are present (the classical localizer finds ~8 on a full
    board; requiring >= 4 is robust to one or two edge-case missed detections).
  - Every bbox_rel component is in [0.0, 1.0].

Frame selection
---------------
The test targets ``Sample1_003_t00460s.png``, a known board frame validated in
the localizer spike (8/8 cards detected, 0 false positives).  Sample1_000 was
the previous first-glob result but is a modal frame where the classical
localizer correctly returns ~0 cards, which would break the ``>= 4`` assertion.

If the preferred frame is absent (e.g. on a fresh clone before frame extraction),
the test falls back to a glob of Sample1/ and skips if nothing is found.

Rule 1 compliance: no inline interpreter calls.  No ``import`` on the command
line.  All code runs through pytest.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dbcv.api import app
from dbcv.localize import classical_localize
from dbcv.schema import GameStateSnapshot, Resolution

# ---------------------------------------------------------------------------
# Locate a sample frame — anchored to repo root, no hard-coded filename
# ---------------------------------------------------------------------------

# conftest.py puts src/ on sys.path; __file__ is tests/test_api.py.
# parents[0] = tests/
# parents[1] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FRAMES_DIR = _REPO_ROOT / "dataset" / "frames" / "Sample1"

# Preferred frame: validated board frame (8/8 cards, 0 false positives in spike).
# Named with prefix match to be stable across any minor filename suffix changes.
_PREFERRED_STEM = "Sample1_003"


def _find_sample_frame() -> Path:
    """Return a deterministic board-state frame from dataset/frames/Sample1/.

    Preference order:
    1. Any frame whose stem starts with ``Sample1_003`` (the validated board frame).
    2. If absent, fall back to the first PNG by sorted name — but skip if empty.

    ``Sample1_000`` is intentionally NOT the first choice: it is a modal frame
    where the classical localizer correctly returns ~0 cards, which would cause
    the ``>= 4`` card assertion to fail.
    """
    # First: look for the preferred validated board frame
    preferred = sorted(_FRAMES_DIR.glob(f"{_PREFERRED_STEM}*.png"))
    if preferred:
        return preferred[0]

    # Fallback: grab whatever is first — warn the caller via pytest.skip if absent
    frames = sorted(_FRAMES_DIR.glob("*.png"))
    if not frames:
        pytest.skip(
            f"No PNG frames found under {_FRAMES_DIR} — run frame extraction first."
        )
    return frames[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient wrapping the FastAPI app (lifespan runs once per module)."""
    with TestClient(app) as tc:
        yield tc


@pytest.fixture(scope="module")
def sample_frame_path() -> Path:
    return _find_sample_frame()


@pytest.fixture(scope="module")
def actual_dimensions(sample_frame_path: Path) -> tuple[int, int]:
    """Read the PNG's (width, height) independently using PIL.

    This is the ground truth that the server's response must match.
    PIL and cv2 may produce subtly different dimension readings for exotic
    formats, but for standard PNGs they agree exactly.
    """
    with Image.open(sample_frame_path) as img:
        return img.width, img.height   # (w, h)


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


def test_snapshot_http_200(
    client: TestClient, sample_frame_path: Path
) -> None:
    """Endpoint returns HTTP 200 for a valid PNG frame."""
    with open(sample_frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
            data={"video": "Sample1", "frame_index": 3, "timestamp_s": 460.0},
        )
    assert response.status_code == 200, (
        f"Expected 200; got {response.status_code}. Body: {response.text[:500]}"
    )


def test_snapshot_parses_as_game_state(
    client: TestClient, sample_frame_path: Path
) -> None:
    """Response body validates as a GameStateSnapshot with schema version 0.2.0."""
    with open(sample_frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    snapshot = GameStateSnapshot.model_validate(data)
    # Schema bumped to 0.2.0 when frame_state field was added (Stage 0 gate).
    assert snapshot.schema_version == "0.2.0"


def test_resolution_matches_actual_image(
    client: TestClient,
    sample_frame_path: Path,
    actual_dimensions: tuple[int, int],
) -> None:
    """Server-reported resolution must match the actual PNG dimensions.

    This is the key test for the 'never bake in a resolution' constraint:
    if the server is reading dimensions from the decoded image (not from a
    constant), the numbers in the response will exactly match PIL's reading.
    """
    with open(sample_frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    actual_w, actual_h = actual_dimensions

    assert data["resolution"]["w"] == actual_w, (
        f"Server reported w={data['resolution']['w']}, "
        f"but PIL says the image is {actual_w} px wide."
    )
    assert data["resolution"]["h"] == actual_h, (
        f"Server reported h={data['resolution']['h']}, "
        f"but PIL says the image is {actual_h} px tall."
    )


def test_at_least_four_cards_returned(
    client: TestClient, sample_frame_path: Path
) -> None:
    """The classical localizer returns >= 4 cards on a full board frame.

    The validated spike result for Sample1_003 is 8 cards.  Requiring >= 4 is
    robust to any single-card misses at extreme frame positions while still
    catching a catastrophic failure (stub-style 3 boxes, or zero detections on
    a board frame).
    """
    with open(sample_frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    assert response.status_code == 200
    snapshot = GameStateSnapshot.model_validate(response.json())
    assert len(snapshot.cards) >= 4, (
        f"Expected >= 4 cards on a board frame; got {len(snapshot.cards)}. "
        "If using Sample1_003 this should be ~8 — check that classical_localize "
        "is the active default in pipeline.py."
    )


def test_bbox_rel_all_in_unit_range(
    client: TestClient, sample_frame_path: Path
) -> None:
    """Every bbox_rel component returned by the classical localizer must be in [0.0, 1.0]."""
    with open(sample_frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    assert response.status_code == 200
    snapshot = GameStateSnapshot.model_validate(response.json())

    for i, card in enumerate(snapshot.cards):
        for j, value in enumerate(card.bbox_rel):
            assert 0.0 <= value <= 1.0, (
                f"Card {i} bbox_rel[{j}] = {value} is outside [0, 1]."
            )


# ---------------------------------------------------------------------------
# Direct unit test: classical_localize on the validated board frame
# ---------------------------------------------------------------------------


def test_classical_localize_board_frame_count() -> None:
    """classical_localize returns 5–12 boxes on the validated board frame.

    Sample1_003_t00460s is the frame validated in the localizer spike (8/8 cards).
    Requiring between 5 and 12 allows for minor detection variance while
    definitively distinguishing a real-board detection (~8) from a stub result
    (3) or a zero/catastrophic failure.

    This is a *direct unit test* that calls classical_localize without the API
    or pipeline layers, so failures here isolate the localizer itself.
    """
    frame_path = sorted(_FRAMES_DIR.glob(f"{_PREFERRED_STEM}*.png"))
    if not frame_path:
        pytest.skip(
            f"Preferred board frame ({_PREFERRED_STEM}*.png) not found under "
            f"{_FRAMES_DIR} — run frame extraction first."
        )

    # Decode with cv2 exactly as the pipeline does (BGR, full colour)
    img = cv2.imread(str(frame_path[0]), cv2.IMREAD_COLOR)
    assert img is not None, f"cv2.imread returned None for {frame_path[0]}"

    h, w = img.shape[:2]
    resolution = Resolution(w=w, h=h)

    boxes = classical_localize(img, resolution)

    assert 5 <= len(boxes) <= 12, (
        f"Expected 5–12 boxes on a full board frame (spike result was 8); "
        f"got {len(boxes)}.  Frame: {frame_path[0].name}"
    )

    # All components must be in [0, 1] — belt-and-suspenders check
    for i, (bx, by, bw, bh) in enumerate(boxes):
        for label, val in (("x", bx), ("y", by), ("w", bw), ("h", bh)):
            assert 0.0 <= val <= 1.0, (
                f"Box {i} component {label}={val} is outside [0, 1]."
            )
