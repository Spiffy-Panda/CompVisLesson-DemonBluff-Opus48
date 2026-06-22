# 00_PROJECT — code overview

Project-level map of `src/`. Read before opening any source file.

## What the code does

Turns *Demon Bluff* **video → structured game state → REST response**, entirely from pixels. No audio, no game files.

## Intended component flow

```
raw video (dataset/raw-video/*.mp4, gitignored)
   │
   ▼  [frame selection]   — pick the few frames worth analyzing; never load whole video downstream
selected frames
   │
   ▼  [Stage 0: frame-state gate]  — classify as board / modal / menu
   │                                 Non-board frames → skip to assembly with cards=[]
   │  (board frames only)
   ▼  [card localization] — find card regions in a frame (resolution-agnostic)
card crops
   │
   ▼  [card identification] — name each card (villager/minion/outcast/demon + identity); cheap to retrain when art swaps
   ▼  [on-card reading]     — text / numbers / state markers (OCR or narrow recognizer)
   │
   ▼  [state assembly]    — merge per-card reads (and temporal cues across frames) into one snapshot
game-state snapshot  ──►  [REST service]  ──►  JSON over HTTP
```

Each arrow is a design fork the lesson plan will teach, with the chosen technique justified from `research/RESEARCH.md`.

## Runtime vs. dev compute split

- **Runtime (must fit a mid-grade gaming PC):** frame selection, localization, identification, on-card reading, assembly, REST. Any learned model here must train on a single Titan XP or free Colab.
- **Dev / offline only (heavy models allowed):** dataset creation, auto-labeling, ground-truth generation, debugging visualizations, evaluation.

## Conventions specific to the code

- **Resolution is never hard-coded.** Read frame dimensions from the media; express geometry in relative terms.
- **Card recognition is retrain-cheap by construction** — the art set is assumed to change.
- Anchor every script to the repo root (`Path(__file__).resolve().parents[N]`); no CWD assumptions.

## Status

**Stage 2 (classical identification baseline) landed 2026-06-22.**
`src/dbcv/` now contains the full package through Stage 2. 46 tests pass.

- Stage 0 (frame-state gate): `frame_state.py` — `classify_frame_state`
  correctly classifies all labeled board and modal frames (7/7, 100 %).
- Stage 1 (card localizer): `localize.py` — `classical_localize` validated
  (8/8 and 9/9 exact on board frames, 0 false positives).
- Stage 2 (classical identification baseline):
  - `gallery.py` — `build_gallery()` loads 67 reference PNGs into a
    Gallery of 43 townees with precomputed HSV histograms + ORB descriptors.
    Zero training. Art swap = re-run `build_gallery()`.
  - `identify.py` — `classify_crop(crop, gallery)` uses HSV histogram
    correlation (primary) + ORB tiebreaker. Returns `("unknown","unknown",low)`
    for face-down cards. `make_gallery_identifier(gallery)` bridges into the
    1-arg pipeline interface.
  - `api.py` lifespan now builds the gallery and stores a real identifier on
    `app.state.identifier`.
- Schema: `schema_version` = `"0.2.0"`; `frame_state` field present.

See `CODE-DESIGN.md` for the per-file overview index.
