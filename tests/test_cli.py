"""
tests/test_cli.py — Light unit tests for pure helpers in utils/python/run_pipeline.py.

Tests targeted helpers only — NO gallery build, NO full frame processing.
The full pipeline is already covered in test_frame_state.py and test_identify.py.
These tests run sub-second.

Rule 1 compliance: no inline interpreter calls.  All code runs through pytest.
Anchored to repo root via parents[1].
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure utils/python/ is importable (repo root → sys.path)
# __file__ = tests/test_cli.py → parents[1] = repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_UTILS_PYTHON = _REPO_ROOT / "utils" / "python"
if str(_UTILS_PYTHON) not in sys.path:
    sys.path.insert(0, str(_UTILS_PYTHON))

# Also ensure src/ is on path for the dbcv imports inside run_pipeline
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from run_pipeline import _rel_to_pixel, collect_frames, draw_overlay  # noqa: E402

# We import schema types for the draw_overlay test; no gallery is built.
from dbcv.schema import CardRead, GameStateSnapshot, Resolution, Source  # noqa: E402


# ---------------------------------------------------------------------------
# Tests: _rel_to_pixel
# ---------------------------------------------------------------------------


class TestRelToPixel:
    """Tests for the relative→pixel bbox converter."""

    def test_full_frame_box(self):
        """(0, 0, 1, 1) should map to (0, 0, W, H)."""
        x0, y0, x1, y1 = _rel_to_pixel((0.0, 0.0, 1.0, 1.0), w_img=640, h_img=360)
        assert x0 == 0
        assert y0 == 0
        assert x1 == 640
        assert y1 == 360

    def test_quarter_box(self):
        """(0.25, 0.25, 0.5, 0.5) should correctly map to pixel coords."""
        x0, y0, x1, y1 = _rel_to_pixel((0.25, 0.25, 0.5, 0.5), w_img=400, h_img=200)
        assert x0 == 100  # 0.25 * 400
        assert y0 == 50   # 0.25 * 200
        assert x1 == 300  # (0.25 + 0.50) * 400
        assert y1 == 150  # (0.25 + 0.50) * 200

    def test_returns_four_ints(self):
        """Output must always be four integers."""
        result = _rel_to_pixel((0.1, 0.2, 0.3, 0.4), w_img=1280, h_img=720)
        assert len(result) == 4
        for v in result:
            assert isinstance(v, int)

    def test_clamps_below_zero(self):
        """x0/y0 should never go below 0 even if rel coords imply it."""
        x0, y0, x1, y1 = _rel_to_pixel((-0.1, -0.1, 0.5, 0.5), w_img=100, h_img=100)
        assert x0 >= 0
        assert y0 >= 0

    def test_clamps_above_image_size(self):
        """x1/y1 should never exceed the image dimensions."""
        x0, y0, x1, y1 = _rel_to_pixel((0.8, 0.8, 0.5, 0.5), w_img=100, h_img=100)
        assert x1 <= 100
        assert y1 <= 100

    def test_roundtrip_precision(self):
        """Round-trip from a known pixel → relative → pixel should stay within 1px."""
        # Simulate what the localizer would produce for a card at pixel (128, 64)
        # of size 80x120 in a 640x360 frame.
        w, h = 640, 360
        px0, py0, pw, ph = 128, 64, 80, 120
        rel = (px0 / w, py0 / h, pw / w, ph / h)
        x0, y0, x1, y1 = _rel_to_pixel(rel, w_img=w, h_img=h)
        assert abs(x0 - px0) <= 1
        assert abs(y0 - py0) <= 1
        assert abs(x1 - (px0 + pw)) <= 1
        assert abs(y1 - (py0 + ph)) <= 1


# ---------------------------------------------------------------------------
# Tests: collect_frames
# ---------------------------------------------------------------------------


class TestCollectFrames:
    """Tests for the collect_frames directory/file resolver."""

    def test_directory_returns_sorted_pngs(self, tmp_path):
        """A directory of PNGs should return them sorted by name."""
        (tmp_path / "b.png").write_bytes(b"")
        (tmp_path / "a.png").write_bytes(b"")
        (tmp_path / "c.txt").write_bytes(b"")   # non-PNG, should be ignored
        result = collect_frames(tmp_path)
        assert [p.name for p in result] == ["a.png", "b.png"]

    def test_single_file(self, tmp_path):
        """Passing a single .png file should return a one-element list."""
        f = tmp_path / "frame.png"
        f.write_bytes(b"")
        result = collect_frames(f)
        assert result == [f]

    def test_empty_directory(self, tmp_path):
        """An empty directory should return an empty list."""
        result = collect_frames(tmp_path)
        assert result == []

    def test_nonexistent_path(self, tmp_path):
        """A path that does not exist should return an empty list."""
        result = collect_frames(tmp_path / "ghost")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: draw_overlay
# ---------------------------------------------------------------------------


def _make_snapshot(
    frame_state: str = "board",
    cards: list[CardRead] | None = None,
) -> GameStateSnapshot:
    """Build a minimal GameStateSnapshot for testing draw_overlay."""
    return GameStateSnapshot(
        source=Source(video="test", frame_index=0, timestamp_s=0.0),
        resolution=Resolution(w=100, h=100),
        frame_state=frame_state,  # type: ignore[arg-type]
        cards=cards or [],
    )


class TestDrawOverlay:
    """Tests for draw_overlay on tiny synthetic images."""

    def test_returns_copy(self):
        """draw_overlay must not modify the original image in place."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        snapshot = _make_snapshot(frame_state="board")
        result = draw_overlay(image, snapshot)
        assert result is not image
        # Original should still be all zeros
        assert image.sum() == 0

    def test_output_same_shape(self):
        """Output shape must match input shape exactly."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = _make_snapshot(frame_state="modal")
        result = draw_overlay(image, snapshot)
        assert result.shape == image.shape

    def test_modal_frame_has_banner_pixels(self):
        """A modal frame should have some non-zero pixels from the banner."""
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        snapshot = _make_snapshot(frame_state="modal")
        result = draw_overlay(image, snapshot)
        # The frame_state banner is drawn in the top-left corner; some pixels
        # should have been written (i.e. > 0).
        assert result.sum() > 0, "Expected banner pixels in the top-left"

    def test_board_frame_with_card_draws_box(self):
        """A board snapshot with one card should result in non-zero pixels."""
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        card = CardRead(
            bbox_rel=(0.1, 0.1, 0.3, 0.4),
            role_class="villager",
            identity="Scout",
            confidence=0.72,
        )
        snapshot = _make_snapshot(frame_state="board", cards=[card])
        result = draw_overlay(image, snapshot)
        assert result.sum() > 0, "Expected drawn bbox pixels"
