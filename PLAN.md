# PLAN — index

Index of per-slug plan files in `plans/PLAN-<slug>.md`. One line per entry. This file is an **index only** — detail lives in the slug files.

## Active

- [PLAN-pipeline](plans/PLAN-pipeline.md) — video → game state → REST, in 6 stages; Stages 0–2 + REST serving shipped (110 tests green), remaining: Stage 3 OCR + Stage 4 assembly — see the slug file.
- [PLAN-round2-dataset](plans/PLAN-round2-dataset.md) — mine + label real board crops from the sample videos; feeds margin calibration, face-down measurement, fine-tune round 2, and OCR validation.
- [PLAN-stage3-ocr](plans/PLAN-stage3-ocr.md) — closed-vocab on-card/HUD text reading: region catalog, vocabulary, rendered training set, tiny recognizer + mobile-OCR fallback.
- [PLAN-temporal-assembly](plans/PLAN-temporal-assembly.md) — Stage 4 design: temporal depth + REST contract options (research → decision memo → implementation).
- [PLAN-live-capture](plans/PLAN-live-capture.md) — live-frame eval findings → pipeline fixes: HUD-zone widen (shipped), classical+embedding ensemble (shipped), gate-detection prototype (scrap only), embedding margin recalibration (deferred).

## Completed

_None yet._

## How to use this index

- Create `plans/PLAN-<slug>.md` for any unit of forward work; add one line here under **Active**.
- When every checkbox in a slug is done, move its line to **Completed** (keep the link).
- Keep lines to a single sentence — the slug file holds the chat dump, task list, and open questions.
