"""
dbcv/frame_select.py  --  Stage 0: frame selection (dev / batch only).

A cheap, classical, CPU-only cascade that guarantees nothing downstream ever
processes raw full video.  Given a long gameplay capture (~1 h, ~370 MB in our
samples), it yields only the handful of frames worth analysing:

    1. low fixed-stride decode  -- decode at a target ~1-2 fps derived from the
       media's REAL fps (never an assumed frame rate, never a baked resolution);
    2. perceptual-hash dedup    -- drop frames within a small Hamming distance of
       the last kept frame (near-duplicates: idle board, no state change);
    3. board / menu / modal gate -- reuse ``classify_frame_state`` so only
       analysable board frames survive (optional via ``board_only``).

This is a DEV / BATCH selector, not a runtime path.  It decodes video with
OpenCV, which is fine for offline dataset building and the lesson's worked
example, but it must never be wired into the REST service: the runtime budget
(mid-grade gaming PC) forbids whole-video decoding per request.  The REST path
receives a single already-selected frame.

Research grounding
------------------
Follows the "Frame selection / keyframe extraction" entry in
``research/RESEARCH.md`` (2026-06-21).  Key guidance applied here:

- Decode at a **low fixed stride (1-2 fps)** read from the media's real fps;
  frame-skip-by-decode is the cheap classical choice for very long footage.
- Drop near-duplicates with a **perceptual hash** compared by **Hamming
  distance via popcount**, near-duplicate at d_H <= ~8.  Deep-embedding dedup
  only earns its cost under aggressive transforms that do not occur in a stable
  screen capture, so it is reserved for offline curation.
- Keep a **cheap board gate** (here: ``classify_frame_state``); scene-cut
  detection only flags *that* something changed, not *whether* a board is on
  screen.  PySceneDetect / deep methods are reserved for dev-only segmentation.

Why dHash (difference hash) rather than pHash (DCT)
---------------------------------------------------
Both are endorsed by the research entry.  dHash is chosen here because:

- It is trivial to implement with ``cv2`` + ``numpy`` alone -- no DCT, no new
  dependency (``imagehash`` is intentionally NOT a dependency).
- It is robust on **stable screen captures**: it encodes the sign of the
  horizontal brightness gradient, which is invariant to the global brightness /
  contrast shifts (fades, tooltip dimming) common between near-identical game
  frames, while still flipping bits when card art or panels actually change.
- It is trivially re-tunable on an art swap (the threshold is one constant).

All geometry / sizing here is internal to the hash (a fixed 9x8 hash grid is
the hash's own representation, NOT a frame resolution assumption).  Frame
resolution is read from the media at runtime and never hard-coded.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np

from dbcv.frame_state import FrameState, classify_frame_state

# ---------------------------------------------------------------------------
# Repo-anchored default paths (Rule 1: never assume the invocation directory).
# parents[0] = src/dbcv, parents[1] = src, parents[2] = repo root.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW_VIDEO_DIR = _REPO_ROOT / "dataset" / "raw-video"

# ---------------------------------------------------------------------------
# Tuning constants (all at module level; no pixel value is hard-coded).
# ---------------------------------------------------------------------------

# Target decode rate, in frames per second.  The research entry recommends a
# low fixed stride of 1-2 fps; 1.5 sits in the middle of that band.  The actual
# integer stride is derived from this and the media's real fps at runtime.
_DEFAULT_TARGET_FPS: float = 1.5

# Near-duplicate Hamming threshold.  The research entry puts the near-duplicate
# boundary at d_H <= ~8 over a 64-bit hash.  Frames within this distance of the
# last kept frame are treated as duplicates and dropped.
_DEFAULT_HAMMING_THRESHOLD: int = 8

# dHash grid: grayscale is resized to (HASH_W + 1) x HASH_H, then each pixel is
# compared to its right-hand neighbour, yielding HASH_W * HASH_H = 64 bits.
# This is the hash's internal representation size, NOT a frame resolution.
_HASH_H: int = 8
_HASH_W: int = 8

# Fallback fps if neither OpenCV nor ffprobe can report a usable frame rate.
# Used only as a last resort so the selector degrades gracefully rather than
# crashing; a warning-worthy condition, surfaced via MediaInfo.fps_source.
_FALLBACK_FPS: float = 30.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaInfo:
    """Resolution + frame rate read from the media at runtime.

    No field is ever assumed: ``width``/``height`` come from the first decoded
    frame (resolution is never baked), and ``fps`` is read from the container.
    ``fps_source`` records where the fps came from for debugging / lessons.
    """

    width: int
    height: int
    fps: float
    fps_source: str  # "opencv" | "ffprobe" | "fallback"


class SelectedFrame(NamedTuple):
    """Metadata for one kept frame (the selector's output record).

    The decoded pixels are intentionally NOT stored on this record so that a
    full selection pass does not hold every kept frame in memory.  Callers that
    need pixels can re-read by ``frame_index`` / ``timestamp_s`` or use the
    ``iter_selected_frames`` generator, which yields pixels alongside the record.
    """

    frame_index: int        # 0-based index into the decoded stream
    timestamp_s: float      # presentation timestamp in seconds
    state: FrameState       # "board" | "modal" | "menu"
    dhash: int              # 64-bit difference hash of the frame


# ---------------------------------------------------------------------------
# Media metadata: resolution + fps (read at runtime, never baked)
# ---------------------------------------------------------------------------


def _ffprobe_fps(video_path: Path) -> float | None:
    """Read fps from the container via ffprobe (fallback path).

    Mirrors the avg_frame_rate / r_frame_rate parsing in
    ``scrap_scripts/python/02_probe_video_meta.py``.  Returns ``None`` if
    ffprobe is unavailable or reports no usable rate.
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    video = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    afr = video.get("avg_frame_rate") or video.get("r_frame_rate")
    if afr and afr != "0/0":
        try:
            fps = float(Fraction(afr))
        except (ZeroDivisionError, ValueError):
            return None
        if fps > 0:
            return fps
    return None


def read_media_info(video_path: Path | str, cap: cv2.VideoCapture | None = None) -> MediaInfo:
    """Read resolution + fps from the media at runtime.

    Resolution comes from the media (``CAP_PROP_FRAME_WIDTH/HEIGHT``); fps is
    read from OpenCV (``CAP_PROP_FPS``) and, if that is missing or nonsensical,
    from ffprobe, and only then from a documented fallback.  No dimension or
    frame rate is ever hard-coded.

    Parameters
    ----------
    video_path:
        Path to the video file.
    cap:
        Optional already-open ``cv2.VideoCapture`` to avoid reopening.  If
        ``None``, one is opened and released internally.
    """
    video_path = Path(video_path)
    owned = False
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        owned = True
    if not cap.isOpened():
        if owned:
            cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cv_fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        if owned:
            cap.release()

    # cv2 reports 0.0 (or occasionally NaN / absurd values) when the container
    # has no frame-rate metadata it understands.  Treat anything outside a sane
    # band as missing and fall through to ffprobe.
    if 0.0 < cv_fps < 1000.0:
        return MediaInfo(width=width, height=height, fps=cv_fps, fps_source="opencv")

    ff_fps = _ffprobe_fps(video_path)
    if ff_fps is not None:
        return MediaInfo(width=width, height=height, fps=ff_fps, fps_source="ffprobe")

    return MediaInfo(
        width=width, height=height, fps=_FALLBACK_FPS, fps_source="fallback"
    )


def stride_for_fps(media_fps: float, target_fps: float = _DEFAULT_TARGET_FPS) -> int:
    """Derive the integer decode stride from the media's real fps.

    The stride is ``round(media_fps / target_fps)``, clamped to >= 1, so that
    decoding every ``stride``-th frame approximates ``target_fps``.  Never
    assumes 30 fps: the divisor is the media's measured rate.

    Examples (target_fps=1.5):
        media 30 fps  -> stride 20  (~1.5 fps)
        media 60 fps  -> stride 40  (~1.5 fps)
        media 24 fps  -> stride 16  (~1.5 fps)
        media 1  fps  -> stride 1   (clamped; cannot go below every frame)
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if media_fps <= 0:
        raise ValueError(f"media_fps must be positive, got {media_fps}")
    return max(1, int(round(media_fps / target_fps)))


# ---------------------------------------------------------------------------
# Perceptual hashing (dHash) + Hamming distance
# ---------------------------------------------------------------------------


def dhash(image: np.ndarray, hash_w: int = _HASH_W, hash_h: int = _HASH_H) -> int:
    """Compute a difference hash (dHash) of an image as a 64-bit integer.

    Algorithm (the classic dHash):
      1. convert to grayscale (channel-order independent);
      2. resize to ``(hash_w + 1) x hash_h`` (one extra column for the
         horizontal difference) using area interpolation;
      3. for each row, compare each pixel to its right-hand neighbour;
      4. pack the ``hash_w * hash_h`` boolean comparisons into one integer,
         most-significant bit first (row-major).

    With the default 8x8 grid this yields a 64-bit hash.  The hash encodes the
    *sign* of the horizontal brightness gradient, so it is stable under global
    brightness/contrast shifts but flips bits when structure changes.

    Resolution-agnostic: the input may be any size; only the fixed hash grid
    (an internal representation, not a frame dimension) is used.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        gray = image[:, :, 0]

    # Resize to (hash_h rows) x (hash_w + 1 cols).  cv2.resize takes (W, H).
    resized = cv2.resize(
        gray, (hash_w + 1, hash_h), interpolation=cv2.INTER_AREA
    ).astype(np.int16)

    # Compare each pixel to its right neighbour -> boolean grid of shape
    # (hash_h, hash_w).
    diff = resized[:, 1:] > resized[:, :-1]

    # Pack row-major, MSB first, into a single Python int (exact, no overflow).
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Hamming distance between two integer hashes (popcount of the XOR)."""
    return int(a ^ b).bit_count()


# ---------------------------------------------------------------------------
# Strided decode generator
# ---------------------------------------------------------------------------


def iter_strided_frames(
    video_path: Path | str,
    target_fps: float = _DEFAULT_TARGET_FPS,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, timestamp_s, bgr)`` at ~``target_fps``.

    Opens the video once, derives the integer stride from the media's real fps
    (``stride_for_fps``), and yields every ``stride``-th decoded frame.  Decode
    is sequential (``cap.read()`` then skip), which is the cheap, reliable
    pattern for long files; we do NOT random-seek per frame.

    The timestamp is computed from ``frame_index / media_fps`` rather than from
    ``CAP_PROP_POS_MSEC`` so it is stable even when the container reports no
    per-frame timestamps.  Resolution is whatever the media yields (read from
    the decoded frame); nothing is resized or cropped here.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        info = read_media_info(video_path, cap=cap)
        stride = stride_for_fps(info.fps, target_fps)

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                timestamp_s = idx / info.fps if info.fps > 0 else 0.0
                yield idx, timestamp_s, frame
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Dedup pass
# ---------------------------------------------------------------------------


def dedup_hashes(
    hashes: list[int],
    hamming_threshold: int = _DEFAULT_HAMMING_THRESHOLD,
) -> list[int]:
    """Return the indices of frames to KEEP after near-duplicate removal.

    Greedy "keep-against-last-kept": the first frame is always kept; each
    subsequent frame is kept only if its Hamming distance to the most recently
    kept frame's hash exceeds ``hamming_threshold``.  Comparing against the last
    *kept* frame (not the immediately preceding frame) collapses long runs of
    slowly-drifting near-duplicates (an idle board) into a single kept frame.

    Returns the kept positions as indices into ``hashes`` (ascending).
    """
    kept_indices: list[int] = []
    last_kept_hash: int | None = None
    for i, h in enumerate(hashes):
        if last_kept_hash is None or hamming(h, last_kept_hash) > hamming_threshold:
            kept_indices.append(i)
            last_kept_hash = h
    return kept_indices


# ---------------------------------------------------------------------------
# Full selection cascade
# ---------------------------------------------------------------------------


def iter_selected_frames(
    video_path: Path | str,
    target_fps: float = _DEFAULT_TARGET_FPS,
    hamming_threshold: int = _DEFAULT_HAMMING_THRESHOLD,
    board_only: bool = True,
) -> Iterator[tuple[SelectedFrame, np.ndarray]]:
    """Stream the kept frames as ``(SelectedFrame, bgr)`` pairs.

    Runs the full cascade in a single streaming pass (constant memory in the
    number of frames -- only the last kept hash is retained):

        strided decode  ->  dHash dedup vs last kept  ->  state gate

    Order matters: dedup runs *before* the gate so that a long run of identical
    board frames costs only one ``classify_frame_state`` call.  When
    ``board_only`` is ``True`` (the default for dataset building), only frames
    the gate calls ``"board"`` are emitted; non-board frames are still consumed
    for dedup bookkeeping so a modal->board transition is not masked.

    Yields pixels alongside the record so callers can save crops without a
    second decode pass.
    """
    last_kept_hash: int | None = None
    for idx, ts, frame in iter_strided_frames(video_path, target_fps=target_fps):
        h = dhash(frame)
        is_dup = (
            last_kept_hash is not None
            and hamming(h, last_kept_hash) <= hamming_threshold
        )
        if is_dup:
            continue
        # This frame is a non-duplicate: it becomes the new dedup anchor
        # regardless of its gate state, so we don't re-emit the next
        # near-identical frame after a state change.
        last_kept_hash = h

        state = classify_frame_state(frame)
        if board_only and state != "board":
            continue
        yield SelectedFrame(
            frame_index=idx, timestamp_s=ts, state=state, dhash=h
        ), frame


def select_frames(
    video_path: Path | str,
    target_fps: float = _DEFAULT_TARGET_FPS,
    hamming_threshold: int = _DEFAULT_HAMMING_THRESHOLD,
    board_only: bool = True,
) -> list[SelectedFrame]:
    """Run the full Stage-0 cascade and return the kept frames' metadata.

    Combines low fixed-stride decode (stride from the media's real fps) ->
    perceptual-hash dedup -> ``classify_frame_state`` gate, returning one
    ``SelectedFrame`` per kept frame (index, timestamp, state, dHash).

    This is the batch entry point.  It deliberately does NOT return pixels (use
    ``iter_selected_frames`` for that) so a whole-video pass stays cheap in
    memory.  Heavy decoding lives here, off any REST/runtime path.

    Parameters
    ----------
    video_path:
        Path to the (long) video.  Defaults elsewhere anchor to
        ``dataset/raw-video/`` via ``_RAW_VIDEO_DIR``; this function takes an
        explicit path so it never guesses.
    target_fps:
        Approximate decode rate; the integer stride is derived from this and
        the media's real fps.  Defaults to ~1.5 fps (research 1-2 fps band).
    hamming_threshold:
        Near-duplicate boundary over the 64-bit dHash (default 8).
    board_only:
        If ``True`` (default), keep only frames the gate classifies as
        ``"board"``.  If ``False``, keep every non-duplicate frame and tag it
        with its state (useful for auditing modal/menu coverage).
    """
    return [
        record
        for record, _frame in iter_selected_frames(
            video_path,
            target_fps=target_fps,
            hamming_threshold=hamming_threshold,
            board_only=board_only,
        )
    ]
