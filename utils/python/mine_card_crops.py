#!/usr/bin/env python3
"""mine_card_crops.py — Round-2 dataset crop miner (dev/batch).

Promoted from scrap_scripts/python/12_mine_card_crops.py (Rule 1: it produces
a dataset artifact — dataset/crops/ + manifest.jsonl — so it graduates to
utils/ with a stable name and a row in utils/README.md).

Streams both sample videos through the Stage 0 selector
(dbcv.frame_select.iter_selected_frames — constant memory, board frames only),
localizes cards with classical_localize, crops each slot via
pipeline.crop_relative, saves crop PNGs to dataset/crops/<Sample>/, and records
BOTH identifier proposals (classical gallery matcher + fine-tuned embedding-NN)
per crop in dataset/crops/manifest.jsonl, with a `label: null` field for the
wave-2 labeling pass.

Dev/batch tooling only — decodes video, so it must NEVER be wired into the
REST path (runtime budget forbids whole-video decode per request).

Rule 1 compliance: this file lives in utils/python/ (a promoted, durable
location).  No inline interpreter calls.  Anchored to the repo root; no CWD
assumptions.

Usage
-----
# Smoke run (first 3 kept board frames per video):
    .venv/Scripts/python.exe utils/python/mine_card_crops.py --limit 3

# Full pass over both videos:
    .venv/Scripts/python.exe utils/python/mine_card_crops.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root — anchor: this file lives at repo/utils/python/mine_card_crops.py
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

from dbcv.embed import get_onnx_embedder  # noqa: E402
from dbcv.frame_select import (  # noqa: E402
    iter_selected_frames,
    read_media_info,
    stride_for_fps,
)
from dbcv.gallery import build_embedding_gallery, build_gallery  # noqa: E402
from dbcv.identify import (  # noqa: E402
    make_embedding_identifier,
    make_gallery_identifier,
)
from dbcv.localize import classical_localize  # noqa: E402
from dbcv.pipeline import crop_relative  # noqa: E402
from dbcv.schema import Resolution  # noqa: E402

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


def collect_videos(videos_dir: Path) -> list[Path]:
    """Return a sorted list of video files under videos_dir."""
    return sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    )


def probe_decode_estimate(video_path: Path, target_fps: float | None) -> tuple[int, int]:
    """Return (total_frames, stride) for reporting — fps read from the media.

    total_frames comes from CAP_PROP_FRAME_COUNT (container metadata); the
    stride is derived from the media's measured fps, never an assumed rate.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        info = read_media_info(video_path, cap=cap)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    kwargs = {} if target_fps is None else {"target_fps": target_fps}
    stride = stride_for_fps(info.fps, **kwargs)
    return total, stride


def mine_video(
    video_path: Path,
    out_dir: Path,
    manifest_fh,
    classical_identifier,
    embedding_identifier,
    target_fps: float | None,
    limit: int | None,
) -> dict:
    """Mine one video: select board frames, crop slots, record proposals.

    Returns a stats dict for the run report.
    """
    stem = video_path.stem
    crops_dir = out_dir / stem
    crops_dir.mkdir(parents=True, exist_ok=True)

    total_frames, stride = probe_decode_estimate(video_path, target_fps)
    est_decoded = total_frames // stride if stride else total_frames
    print(f"\n[{stem}] container frames={total_frames}  stride={stride}  "
          f"~strided decodes={est_decoded}")

    stats = {
        "video": stem,
        "container_frames": total_frames,
        "stride": stride,
        "est_strided_decodes": est_decoded,
        "kept_board_frames": 0,
        "crops": 0,
        "agree": 0,
        "embed_abstain": 0,
        "classical_unknown": 0,
    }

    iter_kwargs = {} if target_fps is None else {"target_fps": target_fps}
    t0 = time.perf_counter()

    for selected, bgr in iter_selected_frames(video_path, board_only=True, **iter_kwargs):
        if limit is not None and stats["kept_board_frames"] >= limit:
            break
        stats["kept_board_frames"] += 1

        h_img, w_img = bgr.shape[:2]  # resolution read from the frame, never baked
        resolution = Resolution(w=w_img, h=h_img)
        boxes = classical_localize(bgr, resolution)

        for slot, bbox_rel in enumerate(boxes):
            crop = crop_relative(bgr, bbox_rel)
            if crop.size == 0:  # degenerate box — skip, matching pipeline guard
                continue

            crop_name = (
                f"{stem}_{selected.frame_index:06d}"
                f"_t{selected.timestamp_s:07.1f}s_s{slot:02d}.png"
            )
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            c_identity, c_role, c_conf = classical_identifier(crop)
            e_identity, e_role, e_margin = embedding_identifier(crop)
            e_abstained = e_identity == "unknown"
            agreement = c_identity == e_identity

            record = {
                "video": stem,
                "frame_index": selected.frame_index,
                "timestamp_s": round(selected.timestamp_s, 3),
                "slot": slot,
                "bbox_rel": [round(v, 6) for v in bbox_rel],
                "crop_path": crop_path.relative_to(_REPO_ROOT).as_posix(),
                "classical": {
                    "identity": c_identity,
                    "role_class": c_role,
                    "confidence": round(float(c_conf), 4),
                },
                "embedding": {
                    "identity": e_identity,
                    "role_class": e_role,
                    "margin": round(float(e_margin), 4),
                    "abstained": e_abstained,
                },
                "agreement": agreement,
                "label": None,
            }
            manifest_fh.write(json.dumps(record) + "\n")

            stats["crops"] += 1
            stats["agree"] += int(agreement)
            stats["embed_abstain"] += int(e_abstained)
            stats["classical_unknown"] += int(c_identity == "unknown")

        if stats["kept_board_frames"] % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[{stem}]  kept={stats['kept_board_frames']}  "
                  f"crops={stats['crops']}  t={selected.timestamp_s:7.1f}s  "
                  f"elapsed={elapsed:6.1f}s", flush=True)

    stats["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argparse parser."""
    parser = argparse.ArgumentParser(
        prog="mine_card_crops",
        description=(
            "Round-2 dataset miner: Stage 0 frame selection over the raw sample "
            "videos -> classical_localize -> per-slot crop PNGs + a JSONL "
            "manifest recording both identifier proposals (classical + "
            "embedding) with a null label field for the wave-2 labeling pass."
        ),
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=_REPO_ROOT / "dataset" / "raw-video",
        metavar="DIR",
        help="Directory containing the raw sample videos. Default: dataset/raw-video",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "dataset" / "crops",
        metavar="DIR",
        help="Output root for crop PNGs + manifest.jsonl. Default: dataset/crops",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        metavar="FPS",
        help=(
            "Stage 0 strided-decode target rate. Omit to use frame_select's "
            "own default (currently 1.5)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap kept board frames per video (smoke runs).",
    )
    return parser


def main() -> int:
    """Entry point.  Returns an exit code (0 = success, 1 = error)."""
    parser = build_arg_parser()
    args = parser.parse_args()

    videos_dir: Path = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        print(f"ERROR: --videos-dir does not exist: {videos_dir}", file=sys.stderr)
        return 1
    videos = collect_videos(videos_dir)
    if not videos:
        print(f"ERROR: no video files found under: {videos_dir}", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build both identifiers ONCE (load-once pattern, as in the API lifespan) ---
    print("Building classical gallery (HSV + ORB)...")
    gallery = build_gallery()
    classical_identifier = make_gallery_identifier(gallery)
    print(f"Classical gallery ready: {gallery.n_references} refs, "
          f"{gallery.n_townees} townees.")

    print("Loading ONNX embedder + building embedding gallery...")
    embedder = get_onnx_embedder()
    embed_gallery = build_embedding_gallery(gallery, embedder)
    embedding_identifier = make_embedding_identifier(embedder, embed_gallery)
    print(f"Embedding gallery ready: {embed_gallery.n_classes} prototypes.")

    manifest_path = out_dir / "manifest.jsonl"
    all_stats: list[dict] = []
    t_start = time.perf_counter()

    with manifest_path.open("w", encoding="utf-8") as manifest_fh:
        for video_path in videos:
            stats = mine_video(
                video_path=video_path,
                out_dir=out_dir,
                manifest_fh=manifest_fh,
                classical_identifier=classical_identifier,
                embedding_identifier=embedding_identifier,
                target_fps=args.target_fps,
                limit=args.limit,
            )
            all_stats.append(stats)
            print(f"[{stats['video']}] done: kept={stats['kept_board_frames']}  "
                  f"crops={stats['crops']}  elapsed={stats['elapsed_s']}s")

    total_elapsed = time.perf_counter() - t_start
    total_crops = sum(s["crops"] for s in all_stats)
    total_agree = sum(s["agree"] for s in all_stats)
    total_abstain = sum(s["embed_abstain"] for s in all_stats)
    total_c_unknown = sum(s["classical_unknown"] for s in all_stats)

    print("\n=== Mining summary ===")
    for s in all_stats:
        print(f"  {s['video']}: container_frames={s['container_frames']}  "
              f"stride={s['stride']}  ~decoded={s['est_strided_decodes']}  "
              f"kept_board={s['kept_board_frames']}  crops={s['crops']}  "
              f"elapsed={s['elapsed_s']}s")
    if total_crops:
        print(f"  total crops:              {total_crops}")
        print(f"  identity agreement:       {total_agree}/{total_crops} "
              f"({100.0 * total_agree / total_crops:.1f}%)")
        print(f"  embedding abstentions:    {total_abstain}/{total_crops} "
              f"({100.0 * total_abstain / total_crops:.1f}%)")
        print(f"  classical 'unknown':      {total_c_unknown}/{total_crops} "
              f"({100.0 * total_c_unknown / total_crops:.1f}%)")
    print(f"  total runtime:            {total_elapsed:.1f}s")
    print(f"  manifest:                 {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
