"""
utils/python/finetune_embedding.py — Domain fine-tune the card-embedding backbone
on the local GPU (Titan Xp), then re-export to ONNX for CPU serving.

This is the generator of the SERVED model artifact
(models/mobilenetv3_small_embed.onnx).  export_backbone.py produces the frozen
ImageNet baseline (models/mobilenetv3_small_embed_frozen.onnx) used only for the
head-to-head comparison.

WHY THIS EXISTS
---------------
The shipped embedder is a *frozen* ImageNet MobileNetV3-Small. On our 43
stylised cartoon card-characters its features collapse: inter-prototype cosine
0.65-0.94 (everything looks alike), so embedding-NN over-identifies and does not
beat the conservative classical matcher.  research/RESEARCH.md (2026-06-22)
diagnoses this as **domain shift of frozen ImageNet features to a stylised /
fine-grained domain** (Chen et al. ICLR'19) and prescribes:
  * fine-tune the *same* backbone (keep it tiny / ONNX-CPU-servable),
  * a metric-learning loss with an inter-class margin (**Proxy-Anchor**),
  * **LP-FT**: warm the proxies on frozen features, then unfreeze the top
    blocks at a small LR (Kumar et al. ICLR'22 — full FT distorts features on
    <1e3 images),
  * strong augmentation synthesised from the clean reference art (no real
    board-crop labels exist yet — that is the explicit Round-2 lever).

DESIGN CHOICES (so the ONNX contract never moves)
-------------------------------------------------
  * No projection head. We fine-tune the backbone so the *actual 576-d pooled
    embedding* — the exact vector the runtime serves — separates the classes.
    Downstream (gallery, identify.py, tests) is untouched; only the .onnx swaps.
  * Proxy-Anchor is implemented inline (≈30 lines) — no new runtime/dev dep, and
    it doubles as the worked example for the lesson module.
  * Proxies are warm-started from the frozen prototypes (a clean LP init).
  * The honest, leak-proof verdict is the **inter-prototype cosine on CLEAN
    references** before vs after (uses zero augmentation), reported alongside a
    synthetic top-1 retrieval sanity check.  Real-frame generalisation is judged
    separately by 09_embed_eval.py --onnx <this file's output>.

DEV-ONLY — imports torch/torchvision.  The runtime (src/dbcv/) stays torch-free;
this only regenerates the gitignored models/*.onnx artifact.

Rule 1 compliance: no inline interpreter calls.  Run as a file:
    .venv\\Scripts\\python.exe utils\\python\\finetune_embedding.py
All paths are anchored to the repo root via Path(__file__).resolve().parents[2].
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# --- Path anchoring: scrap_scripts/python/10_*.py -> parents[2] = repo root ----
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_ART_ROOT = _REPO_ROOT / "knowledge-base" / "card-art"
_MODELS_DIR = _REPO_ROOT / "models"

# Reuse the EXACT production reference loader (alpha->white composite, crop top
# 80%) and the EXACT export wrapper architecture so train==serve.
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "utils" / "python"))
from dbcv.gallery import _identity_from_dir, _load_reference_image  # noqa: E402
import export_backbone as eb  # noqa: E402  (build_embedder, validate_parity)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

# ImageNet normalisation — MUST match src/dbcv/embed.py OnnxEmbedder.preprocess.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_INPUT = 224


# ===========================================================================
# Data — one list of clean source images per class, loaded like production
# ===========================================================================


def load_class_sources() -> tuple[list[str], list[list[np.ndarray]]]:
    """Walk card-art/<role>/<identity>/*.png -> per-class lists of BGR images.

    Identity keying matches build_embedding_gallery exactly (Twin_Minion ->
    Minion alias via _identity_from_dir).  Returns (class_names, per_class_bgr).
    """
    if not _ART_ROOT.exists():
        raise FileNotFoundError(f"Card-art root not found: {_ART_ROOT}")

    by_key: dict[tuple[str, str], list[np.ndarray]] = {}
    for role_dir in sorted(_ART_ROOT.iterdir()):
        if not role_dir.is_dir():
            continue
        role = role_dir.name.lower()
        for ident_dir in sorted(role_dir.iterdir()):
            if not ident_dir.is_dir():
                continue
            identity = _identity_from_dir(ident_dir.name)
            key = (identity, role)
            for png in sorted(ident_dir.glob("*.png")):
                bgr = _load_reference_image(png)
                if bgr is not None and bgr.size:
                    by_key.setdefault(key, []).append(bgr)

    items = sorted((k for k in by_key if by_key[k]), key=lambda k: (k[0], k[1]))
    class_names = [f"{ident}" for (ident, _role) in items]
    per_class = [by_key[k] for k in items]
    return class_names, per_class


def _bgr_to_uint8_rgb_chw(bgr: np.ndarray) -> torch.Tensor:
    """BGR HWC uint8 -> RGB CHW uint8 tensor (input to the augmentation pipe)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()


def _clean_preprocess(bgr: np.ndarray) -> torch.Tensor:
    """Deterministic preprocess identical to OnnxEmbedder.preprocess (no aug).

    Used to build prototypes and to embed the val-base images, so the reported
    cosines reflect exactly what the served ONNX model will see.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_INPUT, _INPUT), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(_IMAGENET_MEAN, np.float32)) / np.array(_IMAGENET_STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).contiguous()


def make_train_transform() -> T.Transform:
    """Synthetic 'board-like' augmentation from clean reference art.

    No horizontal flip: in-game cards are never mirrored, so flipping would
    teach a false invariance.  Geometric (scale/rotate/perspective) + RandAugment
    (photometric) + blur (video softness) + erasing (badge/clue occlusion).
    """
    return T.Compose(
        [
            T.RandomResizedCrop(_INPUT, scale=(0.55, 1.0), ratio=(0.8, 1.25), antialias=True),
            T.RandomPerspective(distortion_scale=0.2, p=0.5, fill=255),
            T.RandomRotation(degrees=6, fill=255),
            T.RandAugment(num_ops=2, magnitude=7),
            T.ToDtype(torch.float32, scale=True),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.6)),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            T.RandomErasing(p=0.3, scale=(0.02, 0.12), value=0.0),
        ]
    )


# ===========================================================================
# Proxy-Anchor loss (Kim et al., CVPR 2020) — inline, cosine distance
# ===========================================================================


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes: int, dim: int, margin: float, alpha: float,
                 init_proxies: torch.Tensor | None = None) -> None:
        super().__init__()
        if init_proxies is not None:
            self.proxies = nn.Parameter(init_proxies.detach().clone().float())
        else:
            self.proxies = nn.Parameter(torch.randn(num_classes, dim) * 0.1)
        self.nc = num_classes
        self.margin = margin
        self.alpha = alpha

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = F.normalize(embeddings, dim=1)            # [B, D]
        p = F.normalize(self.proxies, dim=1)          # [C, D]
        cos = x @ p.t()                               # [B, C]
        oh = F.one_hot(labels, self.nc).float()       # [B, C]

        pos = torch.exp(-self.alpha * (cos - self.margin))
        neg = torch.exp(self.alpha * (cos + self.margin))
        p_sum = (pos * oh).sum(dim=0)                 # [C]
        n_sum = (neg * (1.0 - oh)).sum(dim=0)         # [C]

        present = oh.sum(dim=0) > 0                    # [C]
        n_present = present.sum().clamp(min=1)
        pos_term = torch.log1p(p_sum)[present].sum() / n_present
        neg_term = torch.log1p(n_sum).sum() / self.nc
        return pos_term + neg_term


# ===========================================================================
# Train / freeze helpers
# ===========================================================================


def feature_blocks(model: nn.Module) -> list[nn.Module]:
    """Top-level blocks of MobileNetV3-Small.features (a Sequential)."""
    return list(model.features.children())


def set_top_blocks_trainable(model: nn.Module, k: int) -> list[nn.Parameter]:
    """Unfreeze the last k feature blocks; keep the rest frozen.

    Critical BN handling: frozen blocks are put in eval() so their running stats
    do NOT drift; only the unfrozen top blocks train (BN adapts to the cartoon
    domain).  Returns the list of now-trainable backbone parameters.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()  # all BN frozen by default...

    blocks = feature_blocks(model)
    trainable: list[nn.Parameter] = []
    for blk in blocks[-k:] if k > 0 else []:
        blk.train()                       # ...then re-enable BN updates on top-k
        for p in blk.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def make_batch(per_class_src: list[list[torch.Tensor]], views: int,
               transform: T.Transform, device: torch.device
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """One batch = every class present, `views` augmented views each."""
    imgs: list[torch.Tensor] = []
    labels: list[int] = []
    for ci, srcs in enumerate(per_class_src):
        src = srcs[random.randrange(len(srcs))]
        for _ in range(views):
            imgs.append(transform(src))
        labels.extend([ci] * views)
    batch = torch.stack(imgs, dim=0).to(device, non_blocking=True)
    return batch, torch.tensor(labels, dtype=torch.long, device=device)


# ===========================================================================
# Embedding / eval helpers (all on CLEAN references unless stated)
# ===========================================================================


@torch.no_grad()
def embed_clean(model: nn.Module, bgr_list: list[np.ndarray], device: torch.device,
                batch: int = 64) -> np.ndarray:
    """Embed clean BGR images (production preprocess) -> L2-normed [N, 576]."""
    was_training = model.training
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(bgr_list), batch):
        chunk = bgr_list[i:i + batch]
        x = torch.stack([_clean_preprocess(b) for b in chunk]).to(device)
        v = model(x).cpu().numpy()
        out.append(v)
    if was_training:
        model.train()
    embs = np.concatenate(out, axis=0)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    return embs.astype(np.float32)


def build_prototypes(model: nn.Module, per_class_bgr: list[list[np.ndarray]],
                     device: torch.device) -> np.ndarray:
    """Mean-pooled, renormalised prototype per class -> [C, 576] (like prod)."""
    protos = []
    for srcs in per_class_bgr:
        e = embed_clean(model, srcs, device)
        m = e.mean(axis=0)
        m /= (np.linalg.norm(m) + 1e-9)
        protos.append(m)
    return np.stack(protos, axis=0).astype(np.float32)


def interproto_stats(protos: np.ndarray) -> dict[str, float]:
    """Off-diagonal cosine stats of the prototype matrix (the collapse metric)."""
    m = protos @ protos.T
    off = m[~np.eye(m.shape[0], dtype=bool)]
    return {
        "mean": float(off.mean()), "median": float(np.median(off)),
        "p95": float(np.percentile(off, 95)), "max": float(off.max()),
        "min": float(off.min()),
    }


@torch.no_grad()
def synthetic_top1(model: nn.Module, per_class_bgr: list[list[np.ndarray]],
                   protos: np.ndarray, transform: T.Transform, device: torch.device,
                   views: int = 8, seed: int = 1234) -> tuple[float, float]:
    """Augment each class `views` times, match vs clean protos -> (top1, mean-margin).

    Synthetic (optimistic) — measures augmentation robustness, not new-pose
    generalisation.  The real check is 09_embed_eval.py on actual frames.
    """
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    random.seed(seed)
    P = torch.from_numpy(protos).to(device)            # [C, D]
    correct = 0
    total = 0
    margins: list[float] = []
    for ci, srcs in enumerate(per_class_bgr):
        src = _bgr_to_uint8_rgb_chw(srcs[0])
        xs = torch.stack([transform(src) for _ in range(views)]).to(device)
        v = F.normalize(model(xs), dim=1)              # [V, D]
        cos = v @ P.t()                                # [V, C]
        top2 = torch.topk(cos, k=2, dim=1)
        pred = top2.indices[:, 0]
        correct += int((pred == ci).sum().item())
        total += views
        margins.extend((top2.values[:, 0] - top2.values[:, 1]).cpu().tolist())
    if was_training:
        model.train()
    return correct / max(total, 1), float(np.mean(margins))


def fmt(s: dict[str, float]) -> str:
    return (f"mean={s['mean']:.3f} median={s['median']:.3f} "
            f"p95={s['p95']:.3f} max={s['max']:.3f} min={s['min']:.3f}")


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the card-embedding backbone (Proxy-Anchor, LP-FT).")
    ap.add_argument("--steps-a", type=int, default=250, help="Phase A steps (proxy warmup, backbone frozen).")
    ap.add_argument("--steps-b", type=int, default=600, help="Phase B steps (top-block fine-tune).")
    ap.add_argument("--unfreeze", type=int, default=4, help="How many top feature blocks to unfreeze in Phase B.")
    ap.add_argument("--views", type=int, default=4, help="Augmented views per class per batch.")
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-proxy", type=float, default=1e-2)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default="", help="Optional .pt state_dict to start from (Round 2).")
    ap.add_argument("--out-onnx", type=str, default=str(_MODELS_DIR / "mobilenetv3_small_embed.onnx"),
                    help="Served-model output (default: the canonical served path).")
    ap.add_argument("--out-pt", type=str, default=str(_MODELS_DIR / "mobilenetv3_small_embed.pt"),
                    help="Torch weights output (for Round-2 --resume / re-export).")
    ap.add_argument("--tag", type=str, default="round1")
    args = ap.parse_args()

    # Windows consoles default to cp1252 and choke on stray unicode; force UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"FINE-TUNE EMBED [{args.tag}]  device={device}  "
          f"{'(' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else '(CPU — slow!)'}")
    print("=" * 78)

    # --- Data -------------------------------------------------------------
    class_names, per_class_bgr = load_class_sources()
    n_classes = len(class_names)
    n_refs = sum(len(s) for s in per_class_bgr)
    print(f"Loaded {n_classes} classes, {n_refs} reference images "
          f"(min/max per class: {min(len(s) for s in per_class_bgr)}/{max(len(s) for s in per_class_bgr)}).")
    per_class_src = [[_bgr_to_uint8_rgb_chw(b) for b in srcs] for srcs in per_class_bgr]
    transform = make_train_transform()

    # --- Frozen baseline (untouched ImageNet) for honest before/after -----
    print("\nBuilding FROZEN baseline (ImageNet) prototypes ...")
    frozen = eb.build_embedder().to(device).eval()
    frozen_protos = build_prototypes(frozen, per_class_bgr, device)
    frozen_stats = interproto_stats(frozen_protos)
    frozen_top1, frozen_margin = synthetic_top1(frozen, per_class_bgr, frozen_protos, transform, device)
    print(f"  FROZEN inter-prototype cosine: {fmt(frozen_stats)}")
    print(f"  FROZEN synthetic top-1: {frozen_top1*100:.1f}%   mean top1-top2 margin: {frozen_margin:.3f}")

    # --- Model under training --------------------------------------------
    model = eb.build_embedder().to(device)
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd)
        print(f"\nResumed weights from {args.resume}")

    # Warm-start proxies from current prototypes (clean LP init).
    init_protos = torch.from_numpy(build_prototypes(model, per_class_bgr, device))
    criterion = ProxyAnchorLoss(n_classes, 576, args.margin, args.alpha, init_protos).to(device)

    # ---- Phase A: proxies only, backbone fully frozen (BN in eval) --------
    print(f"\n[Phase A] proxy warmup - backbone frozen - {args.steps_a} steps")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    optA = torch.optim.AdamW(criterion.parameters(), lr=args.lr_proxy, weight_decay=0.0)
    for step in range(1, args.steps_a + 1):
        xb, yb = make_batch(per_class_src, args.views, transform, device)
        with torch.no_grad():
            emb = model(xb)
        loss = criterion(emb, yb)
        optA.zero_grad(set_to_none=True)
        loss.backward()
        optA.step()
        if step % 50 == 0 or step == 1:
            print(f"  A step {step:4d}/{args.steps_a}  loss={loss.item():.4f}")

    # ---- Phase B: unfreeze top blocks, joint fine-tune at low LR ----------
    print(f"\n[Phase B] fine-tune top {args.unfreeze} blocks - {args.steps_b} steps")
    bb_params = set_top_blocks_trainable(model, args.unfreeze)
    n_bb = sum(p.numel() for p in bb_params)
    print(f"  trainable backbone params: {n_bb:,} "
          f"({n_bb / sum(p.numel() for p in model.parameters()) * 100:.1f}% of backbone)")
    optB = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": args.lr_backbone, "weight_decay": 1e-4},
            {"params": criterion.parameters(), "lr": args.lr_proxy * 0.1, "weight_decay": 0.0},
        ]
    )
    for step in range(1, args.steps_b + 1):
        xb, yb = make_batch(per_class_src, args.views, transform, device)
        emb = model(xb)
        loss = criterion(emb, yb)
        optB.zero_grad(set_to_none=True)
        loss.backward()
        optB.step()
        if step % 50 == 0 or step == 1:
            print(f"  B step {step:4d}/{args.steps_b}  loss={loss.item():.4f}")

    # --- After: honest before/after on CLEAN references -------------------
    model.eval()
    ft_protos = build_prototypes(model, per_class_bgr, device)
    ft_stats = interproto_stats(ft_protos)
    ft_top1, ft_margin = synthetic_top1(model, per_class_bgr, ft_protos, transform, device)

    print("\n" + "=" * 78)
    print("VERDICT - inter-prototype cosine on CLEAN references (lower off-diag = better)")
    print("=" * 78)
    print(f"  FROZEN : {fmt(frozen_stats)}")
    print(f"  FT     : {fmt(ft_stats)}")
    print(f"  delta mean off-diagonal: {ft_stats['mean'] - frozen_stats['mean']:+.3f}  "
          f"(want strongly negative)")
    print(f"\n  Synthetic top-1 retrieval (augmented->clean proto):")
    print(f"  FROZEN : {frozen_top1*100:5.1f}%   margin {frozen_margin:.3f}")
    print(f"  FT     : {ft_top1*100:5.1f}%   margin {ft_margin:.3f}")
    print("\n  (Synthetic = optimistic; real-frame check: 09_embed_eval.py --onnx <out>)")

    # --- Export to ONNX (CPU) + parity, save weights + results -----------
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_onnx = Path(args.out_onnx)
    out_pt = Path(args.out_pt)
    model_cpu = model.to("cpu").eval()
    torch.save(model_cpu.state_dict(), out_pt)
    print(f"\nSaved weights: {out_pt}")

    print(f"Exporting ONNX -> {out_onnx}")
    dummy = torch.randn(1, 3, _INPUT, _INPUT, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model_cpu, dummy, str(out_onnx), opset_version=13,
            input_names=["image"], output_names=["embedding"],
            dynamic_axes={"image": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            do_constant_folding=True,
        )
    max_diff = eb.validate_parity(model_cpu, out_onnx)

    results = out_onnx.with_name(f"finetune_{args.tag}_results.txt")
    with results.open("w", encoding="utf-8") as fh:
        fh.write(f"Fine-tune {args.tag}\n")
        fh.write(f"classes={n_classes} refs={n_refs} views={args.views} "
                 f"steps_a={args.steps_a} steps_b={args.steps_b} unfreeze={args.unfreeze}\n")
        fh.write(f"lr_backbone={args.lr_backbone} lr_proxy={args.lr_proxy} "
                 f"alpha={args.alpha} margin={args.margin} seed={args.seed}\n\n")
        fh.write(f"FROZEN inter-proto: {fmt(frozen_stats)}\n")
        fh.write(f"FT     inter-proto: {fmt(ft_stats)}\n")
        fh.write(f"FROZEN synth top1={frozen_top1*100:.1f}% margin={frozen_margin:.3f}\n")
        fh.write(f"FT     synth top1={ft_top1*100:.1f}% margin={ft_margin:.3f}\n")
        fh.write(f"onnx={out_onnx.name} torch<->onnx max_abs_diff={max_diff:.2e}\n")
    print(f"Wrote results: {results}")
    print("\nDone. Next: .venv\\Scripts\\python.exe scrap_scripts\\python\\09_embed_eval.py "
          f"--onnx {out_onnx}")


if __name__ == "__main__":
    main()
