"""
utils/python/embed_eval.py — Head-to-head eval: classical vs embedding-NN.

Runs both identifiers on the same board frames from Sample1 (frames 001-018,
skipping non-board frames) and prints per-card results side-by-side.

Rule 1 compliance: no inline interpreter calls. Run as a file:
    .venv\\Scripts\\python.exe utils\\python\\embed_eval.py

All paths are anchored to the repo root via Path(__file__).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Anchor: utils/python/embed_eval.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2
import numpy as np

from dbcv.embed import OnnxEmbedder
from dbcv.frame_state import classify_frame_state
from dbcv.gallery import build_embedding_gallery, build_gallery
from dbcv.identify import (
    classify_crop,
    classify_crop_embedding,
)
from dbcv.localize import classical_localize
from dbcv.pipeline import crop_relative
from dbcv.schema import Resolution

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_FRAMES_DIR = _REPO_ROOT / "dataset" / "frames" / "Sample1"
_ART_ROOT = _REPO_ROOT / "knowledge-base" / "card-art"
_DEFAULT_ONNX_PATH = _REPO_ROOT / "models" / "mobilenetv3_small_embed.onnx"

# Frames to evaluate: 001 through 018 (skip 000 which is known modal)
_FRAME_INDICES = list(range(1, 19))


def load_frame(stem_prefix: str) -> np.ndarray | None:
    matches = sorted(_FRAMES_DIR.glob(f"{stem_prefix}*.png"))
    if not matches:
        return None
    img = cv2.imread(str(matches[0]), cv2.IMREAD_COLOR)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Head-to-head: classical vs embedding-NN on real frames.")
    ap.add_argument("--onnx", type=str, default=str(_DEFAULT_ONNX_PATH),
                    help="Embedding ONNX model to evaluate (default: frozen ImageNet backbone). "
                         "Pass a fine-tuned export to A/B it against the frozen baseline.")
    args = ap.parse_args()
    onnx_path = Path(args.onnx)

    # --- Build classical gallery ---
    print("Building classical gallery ...")
    gallery = build_gallery(_ART_ROOT)
    print(f"  {gallery.n_references} references, {gallery.n_townees} townees")

    # --- Build embedding gallery ---
    if not onnx_path.exists():
        print(
            f"ERROR: ONNX model not found at {onnx_path}\n"
            "Run: .venv\\Scripts\\python.exe utils\\python\\export_backbone.py"
        )
        sys.exit(1)

    print(f"Loading OnnxEmbedder ({onnx_path.name}) ...")
    embedder = OnnxEmbedder(onnx_path)
    print("Building embedding gallery ...")
    embed_gallery = build_embedding_gallery(gallery, embedder, _ART_ROOT)
    print(f"  {embed_gallery.n_classes} classes (prototypes)")

    # --- Eval ---
    print()
    print("=" * 100)
    print("HEAD-TO-HEAD EVAL: Classical HSV vs Embedding-NN")
    print("=" * 100)

    total_cards = 0
    classical_identified = 0   # confident (non-unknown) classical predictions
    embed_identified = 0       # confident (non-unknown) embedding predictions
    classical_unknown = 0
    embed_unknown = 0

    # Per-frame stats: did they agree?
    agree_count = 0
    disagree_count = 0

    for frame_idx in _FRAME_INDICES:
        stem = f"Sample1_{frame_idx:03d}"
        img = load_frame(stem)
        if img is None:
            print(f"\n[{stem}] -- NOT FOUND, skipping")
            continue

        # Check frame state
        state = classify_frame_state(img)
        if state != "board":
            print(f"\n[{stem}] frame_state={state} -- skipping (not a board frame)")
            continue

        h, w = img.shape[:2]
        resolution = Resolution(w=w, h=h)
        boxes = classical_localize(img, resolution)

        if not boxes:
            print(f"\n[{stem}] -- no cards localised, skipping")
            continue

        print(f"\n[{stem}]  resolution={w}x{h}  cards={len(boxes)}")
        print(f"  {'Card':>4}  {'Classical identity':22}  {'Cl.conf':7}  {'Embedding identity':22}  {'Em.conf':7}  Match?")
        print(f"  {'----':>4}  {'-'*22}  {'-'*7}  {'-'*22}  {'-'*7}  ------")

        for card_i, bbox_rel in enumerate(boxes):
            crop = crop_relative(img, bbox_rel)
            if crop.size == 0:
                continue

            # Classical
            cl_id, cl_role, cl_conf = classify_crop(crop, gallery)

            # Embedding
            em_id, em_role, em_conf = classify_crop_embedding(crop, embedder, embed_gallery)

            total_cards += 1
            if cl_id != "unknown":
                classical_identified += 1
            else:
                classical_unknown += 1

            if em_id != "unknown":
                embed_identified += 1
            else:
                embed_unknown += 1

            match = (cl_id == em_id) or (cl_id == "unknown" and em_id == "unknown")
            if match:
                agree_count += 1
            else:
                disagree_count += 1

            match_str = "agree" if match else "DIFFER"
            print(
                f"  {card_i:>4}  {cl_id:22}  {cl_conf:7.3f}  {em_id:22}  {em_conf:7.3f}  {match_str}"
            )

    # --- Summary ---
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Total card slots evaluated : {total_cards}")
    print()
    print(f"  Classical HSV:")
    print(f"    Identified (non-unknown)  : {classical_identified} / {total_cards}"
          + (f"  ({100*classical_identified/total_cards:.1f}%)" if total_cards else ""))
    print(f"    Returned 'unknown'        : {classical_unknown}")
    print()
    print(f"  Embedding-NN:")
    print(f"    Identified (non-unknown)  : {embed_identified} / {total_cards}"
          + (f"  ({100*embed_identified/total_cards:.1f}%)" if total_cards else ""))
    print(f"    Returned 'unknown'        : {embed_unknown}")
    print()
    print(f"  Agreement (both same, or both unknown) : {agree_count}")
    print(f"  Disagreements                          : {disagree_count}")
    print()
    print("IMPORTANT: 'identified' means the identifier returned a non-unknown prediction,")
    print("NOT that it was correct.  Manual inspection of the frame required to judge accuracy.")
    print("Face-down cards SHOULD return 'unknown' -- those are correct even as 'unknown'.")
    print()
    print("Decision rule: embedding-NN is the default UNLESS it is dramatically worse")
    print("than classical (a sign of a preprocessing/export bug).")


if __name__ == "__main__":
    main()
