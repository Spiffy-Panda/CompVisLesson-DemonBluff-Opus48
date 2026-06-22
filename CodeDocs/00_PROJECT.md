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

No source files yet. When the first lands, add `CodeDocs/sources/<Project>/<File>.md` and keep `io/inputs.md` / `io/outputs.md` in sync (Rule 3).
