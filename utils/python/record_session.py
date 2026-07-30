"""
record_session.py — continuous Windows Graphics Capture session recorder for
the Demon Bluff game window, emitting CONSTANT-FRAME-RATE mp4 video.

Purpose: record long (30-60 minute) live gameplay sessions as mp4 video plus
frame-timestamp sidecars, so live footage can replace the two YouTube-sourced
sample videos as Stage 0 input (`src/dbcv/frame_select.py` stride-decodes the
mp4 exactly like it does the samples — no new consumption code needed). The
per-frame timestamps in `frames.jsonl` exist so future action logs can be
aligned to frames for weak temporal labels.

Usage (from repo root, with the project venv):
    .venv/Scripts/python.exe utils/python/record_session.py
        --game-version <build>            (REQUIRED, e.g. v0.762b)
        [--out-dir dataset/live_<UTC>/]   (default: new dataset/live_<UTC-timestamp>/)
        [--duration SECONDS]              (default: run until Ctrl+C or window close)
        [--fps N]                         (CFR target fps, default 30)
        [--max-fps N]                     (optional capture-side thinning cap)

`--game-version` is required with no default: the footage-version-drift rule
mandates recording the game build version for every newly captured recording.

Outputs (all inside --out-dir; `dataset/` is gitignored):
    session_seg1.mp4   (+ session_seg2.mp4, ... if the window is resized mid-session)
    frames.jsonl       one line per ENCODED (CFR) frame: encoded index, segment,
                       frame-in-segment index, ideal CFR timestamp, and the
                       monotonic + wall-clock timestamps of the SOURCE captured
                       frame it came from, plus a duplicate flag
    session.json       metadata: game version, start/end wall time, resolution(s),
                       target fps, measured capture fps, codec per segment,
                       duplicated / CFR-dropped / queue-dropped / thinned frame
                       counts, CFR epoch (see alignment below), segment list

!! FUTURE AGENTS: NEVER open the recorded .mp4 files directly into context —
!! they are large (GBs per hour). Inspect via session.json / frames.jsonl, or
!! via cv2.VideoCapture properties inside a script, or single extracted frames
!! only. Same standing rule as for the sample videos (CLAUDE.md).

Constant frame rate (CFR) — why and how:
    WGC delivers frames at a variable rate (~55 FPS on the capture box, but
    gapped/paused whenever the compositor has nothing new). Stage 0's stride
    decode derives its stride from the container's DECLARED fps, so a VFR-ish
    stream written into a fixed-fps cv2.VideoWriter silently breaks that
    arithmetic — wall-clock time drifts away from frame-index time over a
    30-60 minute session. The writer thread therefore RESAMPLES to true CFR:
    it emits exactly one frame per 1/fps tick of monotonic capture time,
    duplicating the most recent captured frame when capture is slow/gapped
    and dropping extras when capture is faster than the target. session.json
    records the target fps and the duplicated / dropped counts.
    ffprobe is not installed on this box, so CFR correctness is proven by
    construction plus the self-test
    (`scrap_scripts/python/17_record_session_selftest.py`), not by post-hoc
    container probing.

Frame/wall-clock alignment convention (for future action-log labeling):
    session.json records `cfr_epoch_wall` — the wall-clock time corresponding
    to encoded frame 0 (monotonic t=0 of the CFR clock). Any future action log
    carrying wall-clock timestamps maps an event to an encoded frame index via:

        frame_index = round((wall_ts - cfr_epoch_wall) * target_fps)

    A video plus a synchronized event log is self-labeling data; a video alone
    is just pixels (this is how the live_crops_v1 auto-labeling worked).

Design notes:
- The WGC `on_frame_arrived` callback must NEVER block on disk I/O: it only
  copies the BGRA buffer and pushes it onto a bounded queue. A separate writer
  thread does the CFR resampling, BGRA->BGR conversion and cv2.VideoWriter
  encoding. Frames that arrive while the queue is full are counted as
  queue-drops and reported at exit (and in session.json).
- Codec: tries `avc1` first, falls back to `mp4v`. opencv-python bundles its
  own FFmpeg DLLs; system ffmpeg is NOT required (and is not installed on the
  capture box).
- Resolution is read from the FIRST arriving frame — never hard-coded (hard
  project constraint) — and is constant within a segment. A mid-session
  frame-size change (window resize) closes the current segment and opens the
  next numbered segment file rather than crashing or corrupting the stream.
  Odd frame dimensions are trimmed by one pixel to even (codec alignment),
  derived per-frame, never baked in.
- Window finding is the hardened title-substring + owning-exe-verified
  approach from the WGC capture spikes (see `utils/python/grab_frame.py`):
  a File Explorer window browsing the install folder also matches the title,
  so only a window owned by `demon bluff.exe` is accepted.
- The writer/session logic (`SessionWriter`) is import-safe and WGC-free so it
  can be exercised by tests with synthetic frames.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Anchor to repo root regardless of invocation CWD (CLAUDE.md Rule 1).
REPO_ROOT = Path(__file__).resolve().parents[2]

WINDOW_TITLE_MATCH = "demon bluff"
EXPECTED_EXE_NAME = "demon bluff.exe"

DEFAULT_TARGET_FPS = 30.0
DEFAULT_QUEUE_SIZE = 60  # bounded frame queue (~1 s of 1080p BGRA ~= 0.5 GB worst case)

# Epsilon for the final tick-drain comparison only (catches a tick landing
# exactly on the last captured frame's timestamp despite float rounding).
_TICK_EPS = 1e-9


# ---------------------------------------------------------------------------
# Window finding / DPI awareness (hardened; copied from utils/python/grab_frame.py)
# ---------------------------------------------------------------------------

def get_process_exe_path(hwnd) -> str | None:
    """Resolve the full executable path of the process that owns hwnd.

    Uses QueryFullProcessImageNameW (PROCESS_QUERY_LIMITED_INFORMATION), which
    does not require debug/VM-read privileges. Returns None if it can't be
    resolved (e.g. elevated process, access denied).
    """
    import win32process

    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h_process = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not h_process:
            return None
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_uint(len(buf))
            ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(
                h_process, 0, buf, ctypes.byref(size)
            )
            return buf.value if ok else None
        finally:
            ctypes.windll.kernel32.CloseHandle(h_process)
    except Exception:
        return None


def set_dpi_awareness() -> None:
    """Per-monitor DPI awareness (PROCESS_PER_MONITOR_DPI_AWARE = 2).

    Must be called as early as possible, before any window-rect / client-rect
    queries, otherwise Windows will report virtualized (scaled) coordinates
    instead of true pixel coordinates.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception as exc:  # pragma: no cover - best effort on older Windows
        print(f"[warn] SetProcessDpiAwareness failed (non-fatal): {exc}")


def find_demon_bluff_window():
    """Enumerate top-level windows and find one whose title contains the match
    string, case-insensitively, verified against the owning process exe.
    Returns (hwnd, exact_title) or (None, None).
    """
    import win32gui

    matches = []

    def _enum_handler(hwnd, _extra):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and WINDOW_TITLE_MATCH in title.lower():
            matches.append((hwnd, title))

    win32gui.EnumWindows(_enum_handler, None)

    if not matches:
        return None, None

    verified = []
    for hwnd, title in matches:
        exe_path = get_process_exe_path(hwnd)
        exe_name = Path(exe_path).name.lower() if exe_path else None
        if exe_name == EXPECTED_EXE_NAME:
            verified.append((hwnd, title))
        else:
            print(f"[info] ignoring unverified title match: hwnd={hwnd} title={title!r} exe={exe_path!r}")

    if verified:
        if len(verified) > 1:
            print(f"[warn] {len(verified)} windows verified as {EXPECTED_EXE_NAME}, using first")
        return verified[0]

    return None, None


# ---------------------------------------------------------------------------
# Threaded CFR segment writer (WGC-free; importable and testable with
# synthetic frames)
# ---------------------------------------------------------------------------

class SessionWriter:
    """Resamples captured (variable-rate) frames to constant frame rate and
    encodes them to numbered mp4 segments on a dedicated writer thread, with
    frames.jsonl / session.json sidecars.

    Contract:
      - `submit(frame, t_mono, t_wall)` is cheap and never blocks: it applies
        the optional max-fps thinning cap, then `put_nowait`s onto a bounded
        queue. A full queue counts a queue-drop; nothing blocks the caller
        (which in live use is the WGC frame-arrived callback).
      - The writer thread runs the CFR resampler: encoded frame k has ideal
        monotonic time `cfr_epoch_mono + k / target_fps`; its pixel content is
        the most recent captured frame at-or-before that tick. Slow/gapped
        capture duplicates the previous frame (`duplicated_frames`); capture
        faster than the target drops never-emitted extras
        (`cfr_dropped_frames`).
      - Frame WxH is read from each frame; the first frame opens segment 1.
        Any subsequent size change closes the current segment and opens the
        next numbered one. Nothing about resolution is assumed.
      - `close()` flushes the queue, releases the encoder, closes the jsonl
        sidecar and writes session.json. Idempotent.
    """

    CODEC_CANDIDATES = ("avc1", "mp4v")

    def __init__(
        self,
        out_dir: Path | str,
        game_version: str,
        target_fps: float = DEFAULT_TARGET_FPS,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        max_fps: float | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.game_version = game_version
        self.target_fps = float(target_fps)
        self.max_fps = float(max_fps) if max_fps else None

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._queue_size = queue_size
        self._thread: threading.Thread | None = None
        self._writer = None  # current cv2.VideoWriter
        self._segments: list[dict] = []
        self._jsonl = None
        self._closed = False

        # Counters. submit() runs on the capture thread, the rest on the
        # writer thread; each counter has a single writing thread, so plain
        # ints are safe.
        self.submitted_frames = 0        # accepted past thinning, offered to queue
        self.encoded_frames = 0          # CFR frames actually written to mp4
        self.queue_dropped_frames = 0    # lost because the bounded queue was full
        self.cfr_dropped_frames = 0      # captured but never emitted (capture > target fps)
        self.duplicated_frames = 0       # emitted more than once (capture gap/slow)
        self.thinned_frames = 0          # skipped by the --max-fps capture cap

        self._last_accept_mono: float | None = None
        self._first_accept_mono: float | None = None
        self._last_submitted_mono: float | None = None
        self._start_wall: float | None = None
        self._end_wall: float | None = None

        # CFR resampler state (writer thread only).
        self.cfr_epoch_mono: float | None = None  # monotonic time of encoded frame 0
        self.cfr_epoch_wall: float | None = None  # wall-clock time of encoded frame 0
        self._tick_index = 0                      # next CFR tick to emit
        self._pending = None                      # (frame, t_mono, t_wall) latest captured
        self._pending_emits = 0                   # times _pending has been emitted

    @property
    def segments(self) -> list[dict]:
        """Per-segment metadata dicts (file, width, height, codec, frames)."""
        return self._segments

    # -- producer side ------------------------------------------------------

    def start(self) -> None:
        """Open the jsonl sidecar and start the writer thread."""
        self._start_wall = time.time()
        self._jsonl = open(self.out_dir / "frames.jsonl", "w", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="session-writer", daemon=True)
        self._thread.start()

    def submit(self, frame, t_mono: float, t_wall: float) -> bool:
        """Offer one captured frame (numpy HxWx4 BGRA or HxWx3 BGR array,
        already a private copy) for CFR encoding. Never blocks. Returns True
        if queued.
        """
        if self._closed:
            return False
        if self.max_fps is not None and self._last_accept_mono is not None:
            if (t_mono - self._last_accept_mono) < (1.0 / self.max_fps):
                self.thinned_frames += 1
                return False
        self._last_accept_mono = t_mono
        if self._first_accept_mono is None:
            self._first_accept_mono = t_mono
        self.submitted_frames += 1
        self._last_submitted_mono = t_mono
        try:
            self._queue.put_nowait((frame, t_mono, t_wall))
            return True
        except queue.Full:
            self.queue_dropped_frames += 1
            return False

    # -- writer thread: CFR resampler ---------------------------------------

    def _next_tick_mono(self) -> float:
        return self.cfr_epoch_mono + self._tick_index / self.target_fps

    def _drain_ticks(self, up_to_mono: float, inclusive: bool) -> None:
        """Emit every pending CFR tick at-or-before `up_to_mono` using the
        most recent captured frame (`self._pending`)."""
        frame, src_mono, src_wall = self._pending
        while True:
            tick = self._next_tick_mono()
            if inclusive:
                if tick > up_to_mono + _TICK_EPS:
                    break
            else:
                if tick >= up_to_mono:
                    break
            self._emit(frame, tick, src_mono, src_wall, dup=self._pending_emits > 0)
            self._pending_emits += 1
            self._tick_index += 1

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # sentinel from close()
                if self._pending is not None:
                    # Final drain: catch a tick landing exactly on the last
                    # captured frame's timestamp.
                    self._drain_ticks(self._pending[1], inclusive=True)
                    if self._pending_emits == 0:
                        self.cfr_dropped_frames += 1
                break
            if self._pending is None:
                # First captured frame defines the CFR epoch. Encoded frame 0
                # is this frame at monotonic tick 0.
                self.cfr_epoch_mono = item[1]
                self.cfr_epoch_wall = item[2]
                self._pending = item
                self._pending_emits = 0
                continue
            # Ticks strictly before the new frame's timestamp belong to the
            # previous (most recent at-or-before) captured frame.
            self._drain_ticks(item[1], inclusive=False)
            if self._pending_emits == 0:
                self.cfr_dropped_frames += 1  # superseded before any tick used it
            self._pending = item
            self._pending_emits = 0

    def _open_segment(self, width: int, height: int) -> None:
        import cv2

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        seg_index = len(self._segments) + 1
        name = f"session_seg{seg_index}.mp4"
        path = self.out_dir / name
        for codec in self.CODEC_CANDIDATES:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, self.target_fps, (width, height))
            if writer.isOpened():
                self._writer = writer
                self._segments.append(
                    {"file": name, "width": width, "height": height,
                     "codec": codec, "frames": 0}
                )
                print(f"[ok] opened segment {name} ({width}x{height}, codec={codec})")
                return
            writer.release()
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"[warn] codec {codec!r} failed to open, trying next")
        raise RuntimeError(f"No usable codec among {self.CODEC_CANDIDATES} for {width}x{height}")

    def _emit(self, frame, t_ideal_mono: float, src_mono: float, src_wall: float,
              dup: bool) -> None:
        """Encode one CFR frame and append its frames.jsonl record."""
        import cv2

        # Even-align dimensions (codec requirement); derived per-frame, never
        # assumed.
        h, w = frame.shape[:2]
        w -= w % 2
        h -= h % 2
        frame = frame[:h, :w]

        cur = self._segments[-1] if self._segments else None
        if self._writer is None or cur is None or (cur["width"], cur["height"]) != (w, h):
            self._open_segment(w, h)
            cur = self._segments[-1]

        if frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr = frame
        self._writer.write(bgr)

        record = {
            "frame": self.encoded_frames,       # encoded (CFR) index, global
            "segment": len(self._segments),
            "segment_frame": cur["frames"],
            "t_ideal_mono": t_ideal_mono,       # cfr_epoch_mono + frame / target_fps
            "t_source_mono": src_mono,          # captured source frame, monotonic
            "t_source_wall": src_wall,          # captured source frame, wall clock
            "dup": dup,
        }
        self._jsonl.write(json.dumps(record) + "\n")

        cur["frames"] += 1
        self.encoded_frames += 1
        if dup:
            self.duplicated_frames += 1

    # -- shutdown -----------------------------------------------------------

    def capture_fps_measured(self) -> float | None:
        """Measured fps of the accepted capture stream (pre-CFR)."""
        if (
            self._first_accept_mono is None
            or self._last_submitted_mono is None
            or self._last_submitted_mono <= self._first_accept_mono
            or self.submitted_frames < 2
        ):
            return None
        span = self._last_submitted_mono - self._first_accept_mono
        return (self.submitted_frames - 1) / span

    def close(self) -> None:
        """Flush the queue, release the encoder, and write session.json."""
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            self._queue.put(None)  # sentinel; blocking put is fine off the capture thread
            self._thread.join()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
        self._end_wall = time.time()
        self._write_session_json()

    def _write_session_json(self) -> None:
        def _iso(epoch: float | None) -> str | None:
            if epoch is None:
                return None
            return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

        meta = {
            "recorder": "utils/python/record_session.py",
            "game_version": self.game_version,
            "start_wall_epoch": self._start_wall,
            "start_wall_iso": _iso(self._start_wall),
            "end_wall_epoch": self._end_wall,
            "end_wall_iso": _iso(self._end_wall),
            "target_fps": self.target_fps,
            "cfr_epoch_mono": self.cfr_epoch_mono,
            "cfr_epoch_wall": self.cfr_epoch_wall,
            "cfr_epoch_wall_iso": _iso(self.cfr_epoch_wall),
            "alignment_formula": "frame_index = round((wall_ts - cfr_epoch_wall) * target_fps)",
            "capture_fps_measured": self.capture_fps_measured(),
            "max_fps_cap": self.max_fps,
            "queue_size": self._queue_size,
            "submitted_frames": self.submitted_frames,
            "encoded_frames": self.encoded_frames,
            "duplicated_frames": self.duplicated_frames,
            "cfr_dropped_frames": self.cfr_dropped_frames,
            "queue_dropped_frames": self.queue_dropped_frames,
            "thinned_frames": self.thinned_frames,
            "resolutions": sorted({(s["width"], s["height"]) for s in self._segments}),
            "codecs": sorted({s["codec"] for s in self._segments}),
            "segments": self._segments,
        }
        with open(self.out_dir / "session.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Live WGC capture loop
# ---------------------------------------------------------------------------

def record_live(hwnd, writer: SessionWriter, duration: float | None) -> int:
    """Run a WGC capture session against hwnd, feeding frames to `writer`.

    Blocks until --duration elapses, Ctrl+C, or the window closes. Returns an
    exit code. The caller owns writer.close().
    """
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl

    stop_event = threading.Event()
    closed_event = threading.Event()

    capture = WindowsCapture(window_hwnd=hwnd)

    @capture.event
    def on_frame_arrived(frame: "Frame", capture_control: "InternalCaptureControl"):
        # MUST stay cheap: copy + non-blocking queue offer only. No disk I/O.
        if stop_event.is_set():
            capture_control.stop()
            return
        t_mono = time.perf_counter()
        t_wall = time.time()
        writer.submit(frame.frame_buffer.copy(), t_mono, t_wall)

    @capture.event
    def on_closed():
        print("[info] game window closed — stopping capture.")
        # Finalize outputs BEFORE the native session tears down: stopping the
        # windows-capture session can terminate the whole process (observed
        # live 2026-07-30: no traceback, atexit/finally never ran), so the
        # mp4 moov atom and session.json must already be on disk by then.
        writer.close()
        closed_event.set()
        stop_event.set()

    # Run the blocking capture.start() on its own thread so the main thread
    # stays free to catch Ctrl+C (a signal handler cannot preempt a blocking
    # native call) and to enforce --duration.
    cap_thread = threading.Thread(target=capture.start, name="wgc-capture", daemon=True)
    writer.start()
    cap_thread.start()

    t0 = time.perf_counter()
    last_report = t0
    try:
        while cap_thread.is_alive() and not closed_event.is_set():
            now = time.perf_counter()
            if duration is not None and (now - t0) >= duration:
                print(f"[info] --duration {duration:g}s reached — stopping capture.")
                # Close (finalize mp4 + session.json) BEFORE signalling the
                # capture callback to stop the native session — see on_closed.
                writer.close()
                stop_event.set()
            if now - last_report >= 10.0:
                last_report = now
                print(f"[info] t={now - t0:7.1f}s encoded={writer.encoded_frames} "
                      f"dup={writer.duplicated_frames} qdrop={writer.queue_dropped_frames}")
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("[info] Ctrl+C — stopping capture.")
        writer.close()  # finalize before the native session stops — see on_closed
        stop_event.set()

    # The callback stops the session on the next arriving frame; if no frames
    # are arriving (e.g. minimized window), don't hang forever.
    cap_thread.join(timeout=5.0)
    if cap_thread.is_alive():
        print("[warn] capture thread did not stop within 5 s (no frames arriving?); "
              "finalizing outputs anyway.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continuously record the Demon Bluff game window (WGC) to "
                    "constant-frame-rate mp4 with frame-timestamp sidecars."
    )
    parser.add_argument(
        "--game-version", required=True,
        help="Game build version of the footage being captured (REQUIRED — "
             "footage-version-drift rule; e.g. v0.762b).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output folder (default: dataset/live_<UTC-timestamp>/ under the repo root).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Stop after this many seconds (default: run until Ctrl+C or window close).",
    )
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_TARGET_FPS,
        help=f"Constant-frame-rate target fps of the output mp4 (default "
             f"{DEFAULT_TARGET_FPS:g}). Encoded frame count == duration * fps; "
             "capture gaps become duplicated frames, never time skew.",
    )
    parser.add_argument(
        "--max-fps", type=float, default=None,
        help="Optional frame-thinning cap at capture time, before CFR "
             "resampling (default: no cap; WGC delivers ~55 FPS on the "
             "capture box).",
    )
    parser.add_argument(
        "--queue-size", type=int, default=DEFAULT_QUEUE_SIZE,
        help=f"Bounded frame-queue size before drops (default {DEFAULT_QUEUE_SIZE}).",
    )
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO_ROOT / "dataset" / f"live_{stamp}"

    set_dpi_awareness()

    hwnd, title = find_demon_bluff_window()
    if hwnd is None:
        print(f"[FAIL] No verified window found with title containing '{WINDOW_TITLE_MATCH}' "
              f"owned by {EXPECTED_EXE_NAME}. Is the game running?")
        return 1
    print(f"[ok] Found window hwnd={hwnd} title={title!r}")
    print(f"[ok] Recording to {out_dir}  (game version {args.game_version}, "
          f"CFR {args.fps:g} fps)")

    writer = SessionWriter(
        out_dir,
        game_version=args.game_version,
        target_fps=args.fps,
        queue_size=args.queue_size,
        max_fps=args.max_fps,
    )
    try:
        rc = record_live(hwnd, writer, args.duration)
    finally:
        writer.close()

    cap_fps = writer.capture_fps_measured()
    cap_txt = f", capture measured {cap_fps:.1f} fps" if cap_fps else ""
    print(f"[ok] session closed: {writer.encoded_frames} CFR frames encoded at "
          f"{writer.target_fps:g} fps across {len(writer.segments)} segment(s){cap_txt}")
    print(f"[ok] duplicated (capture gaps): {writer.duplicated_frames}, "
          f"CFR-dropped (capture>target): {writer.cfr_dropped_frames}, "
          f"queue-dropped (writer behind): {writer.queue_dropped_frames}, "
          f"thinned (--max-fps): {writer.thinned_frames}")
    print(f"[ok] metadata: {out_dir / 'session.json'}")
    if writer.encoded_frames == 0:
        print("[FAIL] No frames were encoded.")
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
