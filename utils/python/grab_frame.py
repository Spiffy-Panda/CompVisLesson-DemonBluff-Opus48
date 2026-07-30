"""
grab_frame.py — grab exactly ONE frame of the "Demon Bluff" game window via
Windows Graphics Capture (windows-capture) and save it to a caller-supplied
PNG path.

The single-frame counterpart of `utils/python/record_session.py` (continuous
session recorder): use this for one-shot state checks, dataset spot-grabs, and
capture-stack sanity tests without opening a recording session. Promoted
2026-07-29 (was `scrap_scripts/python/04_grab_frame.py`); its hardened
window-finding/DPI code is also the shared basis of `record_session.py`.

What it does:
  1. Sets per-monitor DPI awareness (must happen before any window/geometry
     query).
  2. Finds the game window the hardened way validated in the WGC capture
     spikes: title-substring match, verified against the owning process exe
     (Demon Bluff.exe), so we never accidentally target a File Explorer
     window browsing the install folder.
  3. Starts a WindowsCapture session, grabs the FIRST frame that arrives,
     stops the session immediately (keeps it fast — no fixed sleep/duration).
  4. Saves the frame (BGRA -> BGR via OpenCV) to --out, creating parent dirs
     as needed.
  5. Prints the saved path and the frame's WxH (read from the frame itself,
     never hard-coded).

Run with: .venv/Scripts/python.exe utils/python/grab_frame.py --out <path>
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

# Anchor to repo root regardless of invocation CWD (CLAUDE.md Rule 1).
REPO_ROOT = Path(__file__).resolve().parents[2]

WINDOW_TITLE_MATCH = "demon bluff"
EXPECTED_EXE_NAME = "demon bluff.exe"
FRAME_TIMEOUT_SECONDS = 10.0


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


def _save_png(bgra_buf, path: Path) -> None:
    import cv2

    bgr = cv2.cvtColor(bgra_buf, cv2.COLOR_BGRA2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def grab_one_frame(hwnd, out_path: Path) -> tuple[int, int] | None:
    """Start a capture session, save the first frame that arrives, stop.

    Returns (width, height) on success, None on failure.
    """
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl

    state = {"width": None, "height": None, "saved": False, "start": time.time()}

    capture = WindowsCapture(window_hwnd=hwnd)

    @capture.event
    def on_frame_arrived(frame: "Frame", capture_control: "InternalCaptureControl"):
        if state["saved"]:
            return
        state["width"] = frame.width
        state["height"] = frame.height
        buf = frame.frame_buffer.copy()
        _save_png(buf, out_path)
        state["saved"] = True
        capture_control.stop()

    @capture.event
    def on_closed():
        pass

    try:
        capture.start()
    except Exception as exc:
        print(f"[FAIL] capture.start() raised: {exc}")
        return None

    if not state["saved"]:
        return None
    return state["width"], state["height"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Grab one frame of the Demon Bluff game window.")
    parser.add_argument("--out", required=True, help="Output PNG path (parent dirs created as needed).")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    set_dpi_awareness()

    hwnd, title = find_demon_bluff_window()
    if hwnd is None:
        print(f"[FAIL] No verified window found with title containing '{WINDOW_TITLE_MATCH}' "
              f"owned by {EXPECTED_EXE_NAME}. Is the game running?")
        return 1

    print(f"[ok] Found window hwnd={hwnd} title={title!r}")

    dims = grab_one_frame(hwnd, out_path)
    if dims is None:
        print("[FAIL] No frame captured.")
        return 1

    width, height = dims
    print(f"[ok] Saved frame -> {out_path} ({width}x{height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
