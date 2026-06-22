"""
tests/test_frame_state.py — Unit and integration tests for the Stage 0 gate.

Tests that ``classify_frame_state`` correctly identifies:
  - Full board frames → "board"
  - Modal / overlay frames → "modal"
  - Partial-modal frame (Sample1_006) → documented handling

Also tests the pipeline integration:
  - Posting a MODAL frame returns frame_state="modal" and cards=[].
  - Posting a BOARD frame returns frame_state="board" and len(cards) >= 4.

Ground-truth labels (from task brief and visual inspection):
  BOARD:  Sample1_003, Sample1_018, Sample2_012, Sample2_023
  MODAL:  Sample1_000 (NEW CHARACTERS UNLOCKED)
          Sample2_000 (CURRENT DECK)
          Sample2_006 (CURRENT DECK with tooltip)
  PARTIAL MODAL: Sample1_006 (peripheral cards still visible) → treated as "board"

Rule 1 compliance: no inline interpreter calls.  No ``import`` on the command
line.  All code runs through pytest.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from dbcv.api import app
from dbcv.frame_state import classify_frame_state
from dbcv.schema import GameStateSnapshot

# ---------------------------------------------------------------------------
# Paths — anchored to repo root
# ---------------------------------------------------------------------------

# conftest.py puts src/ on sys.path; __file__ is tests/test_frame_state.py.
# parents[0] = tests/
# parents[1] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FRAMES_ROOT = _REPO_ROOT / "dataset" / "frames"


def _find(stem: str) -> Path | None:
    """Return the first PNG matching ``<stem>*.png`` in either Sample folder, or None."""
    for subfolder in ("Sample1", "Sample2"):
        matches = sorted((_FRAMES_ROOT / subfolder).glob(f"{stem}*.png"))
        if matches:
            return matches[0]
    return None


def _load(stem: str) -> tuple[Path, object]:
    """Return (path, bgr_image) for the given stem, or skip the test if absent."""
    path = _find(stem)
    if path is None:
        pytest.skip(f"Frame {stem}*.png not found — run frame extraction first.")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        pytest.skip(f"cv2.imread returned None for {path} — file may be corrupt.")
    return path, img


# ---------------------------------------------------------------------------
# Ground-truth label sets
# ---------------------------------------------------------------------------

BOARD_STEMS = ["Sample1_003", "Sample1_018", "Sample2_012", "Sample2_023"]
FULL_MODAL_STEMS = ["Sample1_000", "Sample2_000", "Sample2_006"]

# Sample1_006 is a PARTIAL modal: a dialog floats in the center but peripheral
# cards remain visible.  The center-vs-ring ratio lands at 0.94 (well below the
# threshold of 2.0), so the gate returns "board".  This is the CORRECT production
# decision: the localizer can still find the peripheral cards.
PARTIAL_MODAL_STEM = "Sample1_006"
PARTIAL_MODAL_EXPECTED = "board"   # documented choice: localizer runs on partial modal


# ---------------------------------------------------------------------------
# Unit tests: classify_frame_state on labeled frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", BOARD_STEMS)
def test_board_frame_classified_as_board(stem: str) -> None:
    """classify_frame_state returns 'board' for all labeled board frames."""
    _, img = _load(stem)
    state = classify_frame_state(img)
    assert state == "board", (
        f"Frame {stem}: expected 'board', got {state!r}. "
        "Check _MODAL_RATIO_THRESHOLD in frame_state.py."
    )


@pytest.mark.parametrize("stem", FULL_MODAL_STEMS)
def test_full_modal_frame_classified_as_modal(stem: str) -> None:
    """classify_frame_state returns 'modal' for all labeled full-modal frames."""
    _, img = _load(stem)
    state = classify_frame_state(img)
    assert state == "modal", (
        f"Frame {stem}: expected 'modal', got {state!r}. "
        "Check _MODAL_RATIO_THRESHOLD in frame_state.py — "
        "the center-vs-ring ratio for modals should be >> 2.0."
    )


def test_partial_modal_documented_handling() -> None:
    """classify_frame_state on Sample1_006 matches the documented decision.

    Sample1_006 shows a 'Pick 3 characters' dialog floating in the center,
    but the peripheral cards around the board ring remain partially visible.
    The center-vs-ring brightness ratio falls at ~0.94 (below the 2.0 threshold),
    so the gate correctly classifies it as 'board', meaning the localizer RUNS
    and can find the visible peripheral cards.

    This test documents the decision explicitly.  If the expected value is
    changed, update PARTIAL_MODAL_EXPECTED and this docstring.
    """
    _, img = _load(PARTIAL_MODAL_STEM)
    state = classify_frame_state(img)
    assert state == PARTIAL_MODAL_EXPECTED, (
        f"Frame {PARTIAL_MODAL_STEM}: expected {PARTIAL_MODAL_EXPECTED!r} "
        f"(documented partial-modal handling), got {state!r}."
    )


# ---------------------------------------------------------------------------
# Accuracy summary (informational — not a test; run separately if needed)
# ---------------------------------------------------------------------------
#
# Expected accuracy on labeled set:
#   Board frames (4 frames):          4/4 correct  (100 %)
#   Full modal frames (3 frames):     3/3 correct  (100 %)
#   Partial modal (1 frame):          1/1 documented-correct ("board")
#   Overall (7 firm + 1 documented):  7/7 classification, 8/8 documented


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    with TestClient(app) as tc:
        yield tc


def test_api_modal_frame_returns_modal_state_and_empty_cards(
    api_client: TestClient,
) -> None:
    """POSTing a modal frame yields frame_state='modal' and cards=[] from the API.

    Uses Sample1_000 (NEW CHARACTERS UNLOCKED), a full-modal frame.
    This is the key robustness test: the pipeline must NOT run the localizer
    on a modal frame and must return an empty card list.
    """
    path = _find("Sample1_000")
    if path is None:
        pytest.skip("Sample1_000 not found — run frame extraction first.")

    with open(path, "rb") as fh:
        response = api_client.post(
            "/v1/snapshot",
            files={"file": ("modal_frame.png", fh, "image/png")},
            data={"video": "Sample1", "frame_index": 0, "timestamp_s": 115.0},
        )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}. Body: {response.text[:500]}"
    )

    snap = GameStateSnapshot.model_validate(response.json())
    assert snap.frame_state == "modal", (
        f"Expected frame_state='modal' for a modal frame; got {snap.frame_state!r}."
    )
    assert snap.cards == [], (
        f"Expected empty cards list for a modal frame; got {len(snap.cards)} cards."
    )


def test_api_board_frame_returns_board_state_and_cards(
    api_client: TestClient,
) -> None:
    """POSTing a board frame yields frame_state='board' and len(cards) >= 4.

    Uses Sample1_003, the validated board frame from the localizer spike
    (8/8 cards detected, 0 false positives).
    """
    path = _find("Sample1_003")
    if path is None:
        pytest.skip("Sample1_003 not found — run frame extraction first.")

    with open(path, "rb") as fh:
        response = api_client.post(
            "/v1/snapshot",
            files={"file": ("board_frame.png", fh, "image/png")},
            data={"video": "Sample1", "frame_index": 3, "timestamp_s": 460.0},
        )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}. Body: {response.text[:500]}"
    )

    snap = GameStateSnapshot.model_validate(response.json())
    assert snap.frame_state == "board", (
        f"Expected frame_state='board' for a board frame; got {snap.frame_state!r}."
    )
    assert len(snap.cards) >= 4, (
        f"Expected >= 4 cards on a board frame (spike result was 8); "
        f"got {len(snap.cards)}."
    )
