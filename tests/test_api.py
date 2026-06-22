"""
tests/test_api.py — Integration test for POST /v1/snapshot.

Tests that:
  - The endpoint returns HTTP 200 for a real sampled frame.
  - The response validates as a GameStateSnapshot.
  - The ``resolution`` in the response matches the actual image dimensions
    (read independently from PIL — the server must have read them from the
    uploaded bytes, never from a hard-coded value).
  - At least one card is present (the stub localizer guarantees >= 1 box).
  - Every bbox_rel component is in [0.0, 1.0].

The test frame is located by globbing ``dataset/frames/Sample1/`` rather than
hard-coding a filename, so it still works if frames are renamed or more are
added.  The path is anchored to the repo root via Path(__file__).

Rule 1 compliance: no inline interpreter calls.  No ``import`` on the command
line.  All code runs through pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dbcv.api import app
from dbcv.schema import GameStateSnapshot

# ---------------------------------------------------------------------------
# Locate a sample frame — anchored to repo root, no hard-coded filename
# ---------------------------------------------------------------------------

# conftest.py puts src/ on sys.path; __file__ is tests/test_api.py.
# parents[0] = tests/
# parents[1] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FRAMES_DIR = _REPO_ROOT / "dataset" / "frames" / "Sample1"


def _find_sample_frame() -> Path:
    """Return the first PNG in dataset/frames/Sample1/, sorted by name."""
    frames = sorted(_FRAMES_DIR.glob("*.png"))
    if not frames:
        pytest.skip(f"No PNG frames found under {_FRAMES_DIR} — run frame extraction first.")
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
# Tests
# ---------------------------------------------------------------------------


def test_snapshot_http_200(
    client: TestClient, sample_frame_path: Path
) -> None:
    """Endpoint returns HTTP 200 for a valid PNG frame."""
    with open(sample_frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
            data={"video": "Sample1", "frame_index": 0, "timestamp_s": 115.0},
        )
    assert response.status_code == 200, (
        f"Expected 200; got {response.status_code}. Body: {response.text[:500]}"
    )


def test_snapshot_parses_as_game_state(
    client: TestClient, sample_frame_path: Path
) -> None:
    """Response body validates as a GameStateSnapshot."""
    with open(sample_frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    assert response.status_code == 200
    data = response.json()
    snapshot = GameStateSnapshot.model_validate(data)
    assert snapshot.schema_version == "0.1.0"


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


def test_at_least_one_card_returned(
    client: TestClient, sample_frame_path: Path
) -> None:
    """The stub localizer guarantees at least one card box."""
    with open(sample_frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    assert response.status_code == 200
    snapshot = GameStateSnapshot.model_validate(response.json())
    assert len(snapshot.cards) >= 1, (
        "Expected at least one card; stub localizer should return 3 boxes."
    )


def test_bbox_rel_all_in_unit_range(
    client: TestClient, sample_frame_path: Path
) -> None:
    """Every bbox_rel component from the stub must be in [0.0, 1.0]."""
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
