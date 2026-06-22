#!/usr/bin/env python3
"""
run_pipeline.py — Offline CLI runner for the Demon Bluff CV pipeline.

Exercises the full frame → GameStateSnapshot pipeline (gate → localize →
identify) on sampled frames from dataset/frames/, without requiring the HTTP
server.  Writes per-frame snapshot JSON and, optionally, annotated overlay
PNGs to dataset/pipeline-out/.

Designed as a hands-on, reproducible teaching artifact: the lesson plan points
students here to see the end-to-end pipeline in action on real frames.

The gallery is built ONCE at startup and reused for every frame — mirroring
the "load-once" pattern used by the REST API's lifespan context manager.

Anchored to the repo root via Path(__file__).resolve().parents[2].
The src/ directory is prepended to sys.path so `import dbcv` works without
installing the package.

Rule 1 compliance: this file exists in utils/python/ (a promoted, durable
location).  No inline interpreter calls.  No `python -c` with imports.

Usage examples
--------------
# Run all Sample1 frames with overlays:
    .venv/Scripts/python.exe utils/python/run_pipeline.py --overlay

# Limit to 8 frames, custom output dir:
    .venv/Scripts/python.exe utils/python/run_pipeline.py --limit 8 --overlay --out dataset/pipeline-out

# Run a different sample set:
    .venv/Scripts/python.exe utils/python/run_pipeline.py --frames dataset/frames/Sample2

# Skip identification (gate + localize only — faster, no gallery build):
    .venv/Scripts/python.exe utils/python/run_pipeline.py --no-gallery --overlay

# Run a single frame file:
    .venv/Scripts/python.exe utils/python/run_pipeline.py --frames dataset/frames/Sample1/Sample1_000_t00115s.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root — anchor: this file lives at repo/utils/python/run_pipeline.py
#   parents[0] = utils/python/
#   parents[1] = utils/
#   parents[2] = repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"

# Prepend src/ so `import dbcv` resolves without a package install.
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2  # noqa: E402  (after sys.path manipulation)
import numpy as np  # noqa: E402

from dbcv.gallery import build_gallery  # noqa: E402
from dbcv.identify import make_gallery_identifier  # noqa: E402
from dbcv.pipeline import run_pipeline  # noqa: E402
from dbcv.schema import GameStateSnapshot, Source  # noqa: E402


# ---------------------------------------------------------------------------
# Overlay drawing helpers
# ---------------------------------------------------------------------------


def _rel_to_pixel(bbox_rel: tuple[float, float, float, float],
                  w_img: int, h_img: int) -> tuple[int, int, int, int]:
    """Convert a relative (x, y, w, h) bounding box to absolute pixel coords.

    Returns (x0, y0, x1, y1) — top-left and bottom-right corners.

    This is the ONLY place in the runner where relative → pixel conversion
    happens, mirroring the single-place rule enforced in pipeline.py
    (crop_relative).  Reading dimensions from the image, not from a constant.

    Parameters
    ----------
    bbox_rel:
        (x, y, w, h) fractions in [0, 1].
    w_img, h_img:
        Frame dimensions in pixels, read from image.shape — never a constant.

    Returns
    -------
    (x0, y0, x1, y1) in pixel coordinates, clamped to valid range.
    """
    x_rel, y_rel, w_rel, h_rel = bbox_rel
    x0 = int(round(x_rel * w_img))
    y0 = int(round(y_rel * h_img))
    x1 = int(round((x_rel + w_rel) * w_img))
    y1 = int(round((y_rel + h_rel) * h_img))
    # Clamp to valid pixel range
    x0 = max(0, min(x0, w_img - 1))
    y0 = max(0, min(y0, h_img - 1))
    x1 = max(0, min(x1, w_img))
    y1 = max(0, min(y1, h_img))
    return x0, y0, x1, y1


# Role-class colour map for overlay boxes (BGR).
# Distinct colours make board-frame overlays immediately readable.
_ROLE_COLOURS: dict[str, tuple[int, int, int]] = {
    "villager": (80, 200, 80),    # green
    "demon":    (30, 30, 220),    # red
    "minion":   (200, 120, 30),   # blue-ish orange
    "outcast":  (200, 60, 200),   # purple
    "unknown":  (160, 160, 160),  # grey
}
_DEFAULT_COLOUR = (160, 160, 160)


def draw_overlay(
    image: np.ndarray,
    snapshot: GameStateSnapshot,
) -> np.ndarray:
    """Draw card bboxes + identity labels + frame_state onto a copy of image.

    Uses only cv2 drawing primitives — no GUI, no display windows.

    Parameters
    ----------
    image:
        Original BGR frame (not modified in place — a copy is returned).
    snapshot:
        The GameStateSnapshot produced by run_pipeline for this frame.

    Returns
    -------
    np.ndarray
        BGR frame copy with overlays drawn.
    """
    canvas = image.copy()
    h_img, w_img = canvas.shape[:2]

    # Draw per-card boxes and labels (only present when frame_state == "board")
    for card in snapshot.cards:
        x0, y0, x1, y1 = _rel_to_pixel(card.bbox_rel, w_img, h_img)
        colour = _ROLE_COLOURS.get(card.role_class, _DEFAULT_COLOUR)

        # Bounding box rectangle
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thickness=2)

        # Label: "identity conf" (e.g. "Alchemist 0.72")
        label = f"{card.identity} {card.confidence:.2f}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )

        # Small background rectangle so the label is readable over any frame
        label_y0 = max(y0 - lh - baseline - 2, 0)
        label_y1 = label_y0 + lh + baseline + 2
        label_x1 = min(x0 + lw + 2, w_img)
        cv2.rectangle(canvas, (x0, label_y0), (label_x1, label_y1), colour, -1)
        cv2.putText(
            canvas, label, (x0 + 1, label_y0 + lh),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Frame-state banner in the top-left corner
    state_label = f"frame_state={snapshot.frame_state}"
    (sw, sh), sbl = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    banner_colour = (0, 200, 255) if snapshot.frame_state == "board" else (50, 50, 220)
    cv2.rectangle(canvas, (4, 4), (sw + 12, sh + sbl + 8), banner_colour, -1)
    cv2.putText(
        canvas, state_label, (8, sh + 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return canvas


# ---------------------------------------------------------------------------
# Frame collection helpers
# ---------------------------------------------------------------------------


def collect_frames(frames_path: Path) -> list[Path]:
    """Return a sorted list of PNG files from a file path or a directory.

    Parameters
    ----------
    frames_path:
        Either a single .png file, or a directory to glob for *.png files.

    Returns
    -------
    list[Path]
        Sorted list of PNG paths (may be empty).
    """
    if frames_path.is_file():
        return [frames_path] if frames_path.suffix.lower() == ".png" else []
    if frames_path.is_dir():
        return sorted(frames_path.glob("*.png"))
    return []


# ---------------------------------------------------------------------------
# Per-frame stdout summary line
# ---------------------------------------------------------------------------


def _format_summary_line(stem: str, snapshot: GameStateSnapshot) -> str:
    """Build the single-line per-frame console summary.

    Format:
        <stem>  state=<frame_state>  cards=<N>  [identity@conf ...]

    Examples:
        Sample1_000  state=modal  cards=0
        Sample1_003  state=board  cards=8  Wretch@0.65 Alchemist@0.71 ...
    """
    parts = [
        f"{stem:<35}",
        f"state={snapshot.frame_state:<7}",
        f"cards={len(snapshot.cards)}",
    ]
    if snapshot.cards:
        id_parts = [f"{c.identity}@{c.confidence:.2f}" for c in snapshot.cards]
        parts.append("  " + "  ".join(id_parts))
    return "  ".join(parts[:3]) + ("  " + "  ".join(id_parts) if snapshot.cards else "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argparse parser."""
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description=(
            "Offline CLI runner for the Demon Bluff CV pipeline.\n"
            "Runs gate -> localize -> identify on sampled frames and writes "
            "snapshot JSON (and optionally annotated PNG overlays) to --out.\n\n"
            "The gallery is built once and reused for all frames, mirroring "
            "the load-once pattern of the REST API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=_REPO_ROOT / "dataset" / "frames" / "Sample1",
        metavar="PATH",
        help=(
            "A single PNG file or a directory of PNGs to process. "
            "Default: dataset/frames/Sample1"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "dataset" / "pipeline-out",
        metavar="DIR",
        help=(
            "Output directory for snapshot JSON files (and overlays if --overlay). "
            "Created if it does not exist. Default: dataset/pipeline-out"
        ),
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help=(
            "Also save an annotated PNG for each frame: bboxes coloured by "
            "role class, identity+confidence labels, frame_state banner. "
            "Saved as <stem>_overlay.png in --out."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of frames processed (useful for quick checks).",
    )
    parser.add_argument(
        "--no-gallery",
        dest="no_gallery",
        action="store_true",
        help=(
            "Skip gallery build and use the stub identifier (returns 'unknown'). "
            "Useful for fast gate+localize-only runs; gallery build takes ~0.5 s."
        ),
    )
    return parser


def main() -> int:
    """Entry point.  Returns an exit code (0 = success, 1 = error)."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- Resolve and validate --frames ---
    frames_path: Path = args.frames.resolve()
    if not frames_path.exists():
        print(f"ERROR: --frames path does not exist: {frames_path}", file=sys.stderr)
        return 1

    frame_files = collect_frames(frames_path)
    if not frame_files:
        print(
            f"ERROR: No PNG files found at: {frames_path}\n"
            "Provide a .png file or a directory containing .png files.",
            file=sys.stderr,
        )
        return 1

    # Apply --limit
    if args.limit is not None and args.limit > 0:
        frame_files = frame_files[: args.limit]

    # --- Set up output directory ---
    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build gallery ONCE (unless --no-gallery) ---
    identifier_fn = None
    if args.no_gallery:
        print("Gallery skipped (--no-gallery).  Identifier will return 'unknown'.")
    else:
        print("Building reference gallery (loads card-art PNGs, computes HSV + ORB)...")
        gallery = build_gallery()
        identifier_fn = make_gallery_identifier(gallery)
        print(
            f"Gallery ready: {gallery.n_references} reference images, "
            f"{gallery.n_townees} distinct townees."
        )

    print(f"\nProcessing {len(frame_files)} frame(s) from: {frames_path}")
    print(f"Output dir:  {out_dir}")
    print()

    # Header line for stdout summary
    print(f"{'Frame':<35}  {'State':<12}  {'Cards'}  Identities")
    print("-" * 78)

    n_ok = 0
    n_err = 0

    for frame_path in frame_files:
        stem = frame_path.stem

        # Decode frame — never open raw videos; operate on the sampled PNGs
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [SKIP] Could not decode: {frame_path.name}", file=sys.stderr)
            n_err += 1
            continue

        # Derive source metadata from the filename (video stem + frame index from name)
        # Sample filenames look like:  Sample1_003_t00460s.png
        # We treat the first two segments as video and frame-in-sequence.
        name_parts = stem.split("_")
        video_id = name_parts[0] if name_parts else stem
        # Frame index: second segment if it's a zero-padded digit
        try:
            frame_idx = int(name_parts[1]) if len(name_parts) > 1 else 0
        except ValueError:
            frame_idx = 0
        # Timestamp: third segment like "t00460s" → 460 s
        try:
            ts_str = name_parts[2] if len(name_parts) > 2 else "t0s"
            timestamp_s = float(ts_str.lstrip("t").rstrip("s"))
        except (ValueError, IndexError):
            timestamp_s = 0.0

        source = Source(video=video_id, frame_index=frame_idx, timestamp_s=timestamp_s)

        # Run the pipeline — pass identifier only if gallery was built
        pipeline_kwargs: dict = {}
        if identifier_fn is not None:
            pipeline_kwargs["identifier"] = identifier_fn

        snapshot: GameStateSnapshot = run_pipeline(
            image=image,
            source=source,
            **pipeline_kwargs,
        )

        # Print per-frame summary
        print(_format_summary_line(stem, snapshot))

        # Write snapshot JSON
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

        # Optionally write overlay PNG
        if args.overlay:
            overlay_img = draw_overlay(image, snapshot)
            overlay_path = out_dir / f"{stem}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay_img)

        n_ok += 1

    print()
    print(f"Done.  {n_ok} frames written to {out_dir}  ({n_err} skipped).")
    if args.overlay:
        print(f"Overlays saved as *_overlay.png in {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
