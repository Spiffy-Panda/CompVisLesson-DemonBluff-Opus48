# PLAN — round2-dataset

**Goal:** mine and label real board-frame card crops from the two sample videos, producing the labeled dataset that gates margin calibration, face-down measurement, fine-tune round 2, and Stage 3 OCR validation.

**Status:** wave-1 mining complete 2026-07-29 (tool promoted to `utils/python/mine_card_crops.py`; full pass run over both videos). Wave 2 (labeling, calibration, face-down inventory) not started.

## Tasks

- [x] Mining tool: frame_select → classical_localize → per-card crops + manifest (proposals from both identifiers with margins) — `utils/python/mine_card_crops.py` (promoted from `scrap_scripts/python/12_mine_card_crops.py`; cataloged in `utils/README.md`; output contract in `CodeDocs/io/outputs.md`; `dataset/crops/` gitignored)
- [x] Run over both samples; record counts + runtime — see "Wave-1 run record" below
- [ ] Labeling pass (wave 2) — fill the `label: null` field in `dataset/crops/manifest.jsonl`
- [ ] Margin-threshold calibration from labels (wave 2)
- [ ] Face-down crop inventory (wave 2)
- [ ] *(new, wave 2 — from the 2026-07-29 roster check)* **Art-set version is a per-crop attribute.** The footage is **≤ v0.389 (Oct 2025)**; `card-art/` is v0.762a. Record the art-set version on every crop and every reference image, and never pool two sets silently. Concrete checks this enables: (a) any crop labeled `Rambler` or `Investigator` is a **known-bad read** — those roles postdate the footage, so they are free negatives for identifier evaluation; (b) `Chancellor` appears in-footage as **`COUNSELLOR`**; (c) confirm the two sample videos are from the *same* build before pooling their crops. Rationale in `knowledge-base/wiki/townees/ROSTER.md` § Footage-version drift.
- [ ] *(new, wave 2)* Re-examine the **44% both-confident agreement** result against art drift — the wiki gallery is a *current*-art prior applied to old-art cards, so some disagreement may be version skew rather than identifier weakness. Separate the two before drawing lesson-plan conclusions.

## Wave-1 run record (2026-07-29, full pass, defaults: target_fps=1.5, hamming≤8, board-only)

| | Sample1 | Sample2 | total |
|---|---|---|---|
| container frames | 172,768 | 223,638 | 396,406 |
| stride (from measured fps) | 40 | 40 | — |
| ~strided decodes | 4,319 | 5,590 | 9,909 |
| kept board frames (selector) | 460 | 324 | 784 |
| frames yielding ≥1 crop | 450 | 317 | 767 |
| crops written | 3,596 | 2,606 | **6,202** (~8.0–8.2/frame) |
| classical identified (non-unknown) | 26.3% | 27.4% | 26.8% |
| embedding identified (margin ≥ 0.12) | 17.5% | 16.6% | 17.1% (abstention 82.9%) |
| identity agreement (string-equal incl. unknowns) | 62.6% | 65.5% | 63.8% |
| both-confident, and agreeing | 172 → 33.1% | 158 → 56.3% | 330 → 44.2% |
| mining runtime | 1,078 s | 1,503 s | **2,581 s (~43 min)** |

Artifacts: `dataset/crops/<Sample>/<Sample>_<frame>_t<sec>s_s<slot>.png` + `dataset/crops/manifest.jsonl` (6,202 lines, one per crop, `label: null` awaiting wave 2). All gitignored/regenerable.

## Spot-check observations (6 crops, wave-1)

- Crops are genuine card faces at usable resolution (~150–190 px wide); name banner usually included, sometimes cropped off.
- `Sample1_068280_s05` / `Sample2_158360_s08` — clean face-up **Wretch**; embedding correct (margins 0.128 / 0.339), classical says **Architect at 0.70–0.71 confidence both times** → classical has a systematic high-confidence Wretch→Architect confusion.
- `Sample2_109800_s04` — clean face-up **Knitter**, in-gallery, yet BOTH identifiers miss it (classical 0.24, embedding margin 0.034) → real face-up misses exist even on clean crops; direct motivation for fine-tune round 2.
- `Sample1_089520_s01` — revealed red demon-frame card partially occluded by a dialog panel, with a "#4" token overlay; embedding claims Hunter at margin 0.355 (dubious) → high margin is not proof of correctness under occlusion; labeling must not blindly trust high-margin proposals.
- `Sample1_103120_s10` — **localizer false positive**: a "Good" UI banner fragment, not a card (slot 10 of an ~11-box frame). Expect a tail of non-card crops; wave-2 labeling needs a `not_a_card` label value.
- `Sample2_149720_s07` — face-up card with a token overlay at top; classical Doppelganger@0.63 vs embedding Slayer@0.23 disagree → overlays (tokens, arrows, dialogs) are a distinct nuisance category worth tracking in labels.

## Surprises / open questions for wave 2

1. **Both-confident disagreement is high** (only 44% of the 330 both-confident crops agree) — auto-labeling by "identifiers agree" would be unsafe; agreement can only *seed* labels, a human/visual pass must confirm.
2. **Overall crop volume is ~50× round 1** (6,202 vs 125), so even the ~17% embedding-confident subset (~1,060 crops) is a meaningful calibration set once labeled.
3. **Localizer emits occasional non-card boxes** on busy frames (UI banners); round-1's "0 false positives" was measured on a handful of frames — at 767-frame scale a small FP tail appears. Quantify it during labeling before trusting crops-per-frame as card count.
4. **Duplicate identity across time:** the same physical card appears in many kept frames (e.g. Wretch above at t=1138 s and t=2639 s in *different* videos showing the same art). Labeling can exploit temporal runs (label propagation within a frame run) to cut labeling cost.
5. ~10 kept board frames per pass yielded zero localizer boxes (460→450, 324→317) — probably near-empty boards or transition frames; harmless, but they leave no manifest trace. If wave 2 wants frame-level accounting, extend the manifest with per-frame records.
