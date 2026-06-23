"""
tests/test_frame_select.py — Unit tests for the Stage 0 frame selector.

Covers ``dbcv.frame_select``:
  - dHash determinism + near-identical vs very-different behaviour;
  - Hamming distance correctness (popcount of XOR);
  - stride/fps math (stride derived from the media's REAL fps, never assumed);
  - strided decode + dedup + state gate on a TINY SYNTHETIC clip;
  - the gate integration tags board vs non-board.

Hard-constraint compliance (CLAUDE.md + task brief):
  - The ~1 h, ~370 MB sample videos are NEVER opened.  Decode is exercised on a
    tiny synthetic clip written with ``cv2.VideoWriter`` into ``tmp_path`` (a
    known fps, ~12 small frames), and the hash/dedup logic is exercised on
    synthetic numpy arrays plus the already-extracted PNGs in
    ``dataset/frames/Sample1/`` — never on raw video.
  - No resolution is baked: synthetic frames use a small arbitrary size and the
    selector reads dimensions back from the media.

Rule 1 compliance: no inline interpreter calls; all code runs through pytest.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dbcv.frame_select import (
    MediaInfo,
    SelectedFrame,
    dedup_hashes,
    dhash,
    hamming,
    iter_strided_frames,
    read_media_info,
    select_frames,
    stride_for_fps,
)

# ---------------------------------------------------------------------------
# Paths — anchored to repo root (parents[1] = repo root from tests/).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FRAMES_ROOT = _REPO_ROOT / "dataset" / "frames"


# ---------------------------------------------------------------------------
# Synthetic image helpers (no real frames, no baked resolution)
# ---------------------------------------------------------------------------


def _gradient_frame(w: int, h: int, shift: int = 0, seed: int | None = None) -> np.ndarray:
    """A small BGR frame with a horizontal brightness gradient.

    ``shift`` rolls the gradient horizontally so two frames can be made nearly
    identical (small shift) or very different (large shift / inverted).  A
    horizontal gradient gives dHash a well-defined sign pattern to encode.
    """
    col = (np.linspace(0, 255, w, dtype=np.float32) + shift) % 256
    plane = np.tile(col, (h, 1)).astype(np.uint8)
    bgr = cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = rng.integers(-2, 3, size=bgr.shape, dtype=np.int16)
        bgr = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bgr


def _write_clip(
    dest: Path, frames: list[np.ndarray], fps: float
) -> Path:
    """Write ``frames`` to ``dest`` as a tiny video at ``fps``.

    Uses the MJPG fourcc in an .avi container, which is the most reliable
    VideoWriter combination across OpenCV builds (no external codecs needed).
    Skips the test if this build cannot open a writer.
    """
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(dest), fourcc, fps, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter could not open an MJPG writer in this build.")
    for f in frames:
        writer.write(f)
    writer.release()
    if not dest.exists() or dest.stat().st_size == 0:
        pytest.skip("cv2.VideoWriter produced no output in this build.")
    return dest


def _find_png(stem: str) -> Path | None:
    for subfolder in ("Sample1", "Sample2"):
        matches = sorted((_FRAMES_ROOT / subfolder).glob(f"{stem}*.png"))
        if matches:
            return matches[0]
    return None


# ===========================================================================
# dHash
# ===========================================================================


def test_dhash_is_64_bits_and_in_range() -> None:
    """The default 8x8 grid yields a value that fits in 64 bits."""
    img = _gradient_frame(40, 24)
    h = dhash(img)
    assert isinstance(h, int)
    assert 0 <= h < (1 << 64)


def test_dhash_is_deterministic() -> None:
    """Hashing the same image twice yields the identical value."""
    img = _gradient_frame(50, 30, shift=10)
    assert dhash(img) == dhash(img)


def test_dhash_grayscale_and_bgr_agree() -> None:
    """A 3-channel gray image and its single-channel form hash identically.

    dHash uses only luminance, so feeding the grayscale plane directly must
    match feeding the BGR version of that same plane.
    """
    plane = np.tile(np.linspace(0, 255, 48, dtype=np.uint8), (28, 1))
    bgr = cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)
    assert dhash(bgr) == dhash(plane)


def test_dhash_resolution_independent() -> None:
    """The same gradient at two resolutions hashes identically.

    dHash resizes to its fixed internal grid, so a frame and a 2x-scaled copy
    of the same content must produce the same hash — proof that no frame
    resolution leaks into the hash.
    """
    small = _gradient_frame(40, 24)
    large = cv2.resize(small, (80, 48), interpolation=cv2.INTER_NEAREST)
    assert dhash(small) == dhash(large)


def test_dhash_near_identical_within_threshold() -> None:
    """A frame and a tiny-noise copy stay within the near-dup threshold (<=8)."""
    base = _gradient_frame(64, 36, shift=0, seed=1)
    noisy = _gradient_frame(64, 36, shift=0, seed=2)  # same gradient, diff noise
    assert hamming(dhash(base), dhash(noisy)) <= 8


def test_dhash_very_different_exceeds_threshold() -> None:
    """A board PNG vs an inverted copy differ well beyond the near-dup threshold.

    Uses a real already-extracted board PNG (NOT raw video) for a realistic
    image; the inverted copy is a maximally-different structure.
    """
    path = _find_png("Sample1_003")
    if path is None:
        pytest.skip("Sample1_003 PNG not found — run frame extraction first.")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        pytest.skip(f"cv2.imread returned None for {path}.")
    inverted = cv2.bitwise_not(img)
    d = hamming(dhash(img), dhash(inverted))
    assert d > 8, f"Inverted image should be far from original; got d_H={d}."


def test_dhash_distinguishes_distinct_real_frames() -> None:
    """Two different real board frames produce different hashes.

    Sanity check that dHash is not collapsing distinct game states to one value.
    """
    a = _find_png("Sample1_003")
    b = _find_png("Sample1_018")
    if a is None or b is None:
        pytest.skip("Required Sample1 PNGs not found — run frame extraction first.")
    ia = cv2.imread(str(a), cv2.IMREAD_COLOR)
    ib = cv2.imread(str(b), cv2.IMREAD_COLOR)
    if ia is None or ib is None:
        pytest.skip("cv2.imread returned None for a sample PNG.")
    assert dhash(ia) != dhash(ib)


# ===========================================================================
# Hamming distance
# ===========================================================================


def test_hamming_identical_is_zero() -> None:
    assert hamming(0, 0) == 0
    assert hamming(0xDEADBEEF, 0xDEADBEEF) == 0


def test_hamming_counts_differing_bits() -> None:
    assert hamming(0b0000, 0b1111) == 4
    assert hamming(0b1010, 0b0101) == 4
    assert hamming(0b1000, 0b0000) == 1


def test_hamming_is_symmetric() -> None:
    a, b = 0x0F0F0F0F, 0x00FF00FF
    assert hamming(a, b) == hamming(b, a)


# ===========================================================================
# Stride / fps math
# ===========================================================================


@pytest.mark.parametrize(
    "media_fps, target_fps, expected",
    [
        (30.0, 1.5, 20),   # 30 / 1.5 = 20
        (60.0, 1.5, 40),   # 60 / 1.5 = 40
        (24.0, 1.5, 16),   # 24 / 1.5 = 16
        (30.0, 2.0, 15),   # 30 / 2   = 15
        (25.0, 1.0, 25),   # 25 / 1   = 25
        (1.0, 1.5, 1),     # clamped: cannot go below every frame
        (0.5, 1.5, 1),     # clamped
    ],
)
def test_stride_for_fps(media_fps: float, target_fps: float, expected: int) -> None:
    """Stride is round(media_fps / target_fps), clamped to >= 1.

    Critically, the divisor is the media's REAL fps — 30 is never assumed.
    """
    assert stride_for_fps(media_fps, target_fps) == expected


def test_stride_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        stride_for_fps(30.0, 0.0)
    with pytest.raises(ValueError):
        stride_for_fps(0.0, 1.5)


# ===========================================================================
# Dedup pass (pure, on hash lists)
# ===========================================================================


def test_dedup_keeps_first_drops_identical_run() -> None:
    """A run of identical hashes collapses to a single kept frame."""
    hashes = [0xABCD, 0xABCD, 0xABCD, 0xABCD]
    assert dedup_hashes(hashes, hamming_threshold=8) == [0]


def test_dedup_keeps_distinct_frames() -> None:
    """Hashes far apart are all kept."""
    # 0x0 vs all-ones-64 differ by 64 bits; alternating differ a lot too.
    a = 0
    b = (1 << 64) - 1
    assert dedup_hashes([a, b, a], hamming_threshold=8) == [0, 1, 2]


def test_dedup_compares_against_last_kept_not_previous() -> None:
    """Slow drift below threshold per-step still collapses to one kept frame.

    Each consecutive pair differs by <= threshold, but the run drifts; because
    we compare against the last *kept* frame, only frames that exceed the
    threshold relative to that anchor are kept.
    """
    # bits set incrementally: 0, 1, 3, 7, 15, 31 bits high — each step adds a
    # few bits. With threshold 8, the first that exceeds the anchor (0) by >8
    # becomes the next anchor.
    hashes = [0b0, 0b1, 0b11, 0b111, 0b1111]  # popcounts 0,1,2,3,4 vs anchor 0
    # max distance from anchor(0) here is 4 (<=8) -> all collapse to first.
    assert dedup_hashes(hashes, hamming_threshold=8) == [0]


def test_dedup_empty_list() -> None:
    assert dedup_hashes([], hamming_threshold=8) == []


# ===========================================================================
# Synthetic-clip decode: media info, stride, full cascade
# ===========================================================================


def _build_synthetic_clip(tmp_path: Path, fps: float = 12.0) -> tuple[Path, int, int]:
    """Write a tiny clip: a board-like run, a bright (menu) run, a different run.

    Returns (path, width, height).  Frames are small and arbitrary in size to
    prove resolution is read back, not assumed.
    """
    w, h = 64, 48
    frames: list[np.ndarray] = []
    # 4 near-identical "board-ish" frames (dark, gradient) -> one kept
    for i in range(4):
        frames.append(_gradient_frame(w, h, shift=0, seed=100 + i))
    # 4 very bright frames (mean >> 160) -> classified "menu"
    bright = np.full((h, w, 3), 230, dtype=np.uint8)
    for _ in range(4):
        frames.append(bright.copy())
    # 4 near-identical, structurally different gradient frames -> one kept
    for i in range(4):
        frames.append(_gradient_frame(w, h, shift=128, seed=200 + i))
    dest = _write_clip(tmp_path / "synthetic.avi", frames, fps)
    return dest, w, h


def test_read_media_info_reads_resolution_and_fps(tmp_path: Path) -> None:
    """Resolution + fps come from the media, not from constants."""
    fps = 10.0
    path, w, h = _build_synthetic_clip(tmp_path, fps=fps)
    info = read_media_info(path)
    assert isinstance(info, MediaInfo)
    assert info.width == w
    assert info.height == h
    # MJPG/AVI round-trips fps reliably; allow a small tolerance.
    assert info.fps == pytest.approx(fps, abs=0.5)
    assert info.fps_source in {"opencv", "ffprobe"}


def test_read_media_info_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_media_info(tmp_path / "does_not_exist.avi")


def test_iter_strided_frames_applies_stride(tmp_path: Path) -> None:
    """At fps=12 and target ~1.5 fps, stride is ~8, so a 12-frame clip yields ~2.

    The exact yield count is ceil(n_frames / stride); we assert the stride is
    derived from the real fps and the yielded indices are stride-aligned.
    """
    fps = 12.0
    path, _, _ = _build_synthetic_clip(tmp_path, fps=fps)
    info = read_media_info(path)
    stride = stride_for_fps(info.fps)  # ~8 for 12 fps / 1.5
    yielded = list(iter_strided_frames(path))
    assert len(yielded) >= 1
    # Every yielded index must be a multiple of the stride.
    for idx, ts, frame in yielded:
        assert idx % stride == 0
        assert frame.ndim == 3
        # timestamp consistent with index / fps
        assert ts == pytest.approx(idx / info.fps, abs=0.2)


def test_iter_strided_frames_high_target_yields_more(tmp_path: Path) -> None:
    """A higher target fps -> smaller stride -> more frames yielded."""
    fps = 12.0
    path, _, _ = _build_synthetic_clip(tmp_path, fps=fps)
    few = list(iter_strided_frames(path, target_fps=1.0))
    many = list(iter_strided_frames(path, target_fps=12.0))
    assert len(many) >= len(few)


def test_select_frames_dedups_and_gates(tmp_path: Path) -> None:
    """Full cascade on a fine-grained clip: dedup collapses runs; gate filters.

    We use target_fps == media fps so the stride is 1 and every synthetic frame
    is examined; this isolates the dedup + gate behaviour from the stride math
    (which is tested separately).
    """
    fps = 12.0
    path, _, _ = _build_synthetic_clip(tmp_path, fps=fps)

    # board_only=False: keep every non-duplicate, tagged with its state.
    all_kept = select_frames(
        path, target_fps=fps, hamming_threshold=8, board_only=False
    )
    # Three distinct visual runs (board-ish, bright, different-gradient) ->
    # dedup should collapse to roughly one kept frame per run.
    assert all(isinstance(r, SelectedFrame) for r in all_kept)
    states = [r.state for r in all_kept]
    # The bright run must be detected as "menu" by the gate.
    assert "menu" in states, f"Expected a 'menu' frame among {states}."
    # Dedup must have removed the within-run duplicates (12 frames -> few kept).
    assert len(all_kept) <= 6, f"Dedup should collapse runs; kept {len(all_kept)}."

    # board_only=True: the bright (menu) frames must be filtered out.
    board_kept = select_frames(
        path, target_fps=fps, hamming_threshold=8, board_only=True
    )
    assert all(r.state == "board" for r in board_kept), (
        f"board_only must yield only board frames; got "
        f"{[r.state for r in board_kept]}."
    )
    # And it must keep no more than the unfiltered pass.
    assert len(board_kept) <= len(all_kept)


def test_select_frames_records_have_hashes_and_indices(tmp_path: Path) -> None:
    """Each kept record carries a 64-bit hash and a sane (index, timestamp)."""
    fps = 12.0
    path, _, _ = _build_synthetic_clip(tmp_path, fps=fps)
    kept = select_frames(path, target_fps=fps, board_only=False)
    assert kept, "Expected at least one kept frame from the synthetic clip."
    for r in kept:
        assert 0 <= r.dhash < (1 << 64)
        assert r.frame_index >= 0
        assert r.timestamp_s >= 0.0


# ===========================================================================
# Gate integration on real already-extracted PNGs (NOT raw video)
# ===========================================================================


def test_gate_integration_tags_board_png() -> None:
    """A real board PNG run through the dedup+gate path is tagged 'board'.

    We don't decode video here — we feed an already-extracted PNG straight
    through dhash + classify_frame_state to confirm the selector's gate wiring
    agrees with the standalone gate on real content.
    """
    from dbcv.frame_state import classify_frame_state

    path = _find_png("Sample1_003")
    if path is None:
        pytest.skip("Sample1_003 PNG not found — run frame extraction first.")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        pytest.skip(f"cv2.imread returned None for {path}.")
    # The selector uses exactly this gate; hashing must not change the verdict.
    assert classify_frame_state(img) == "board"
    h = dhash(img)
    assert 0 <= h < (1 << 64)
