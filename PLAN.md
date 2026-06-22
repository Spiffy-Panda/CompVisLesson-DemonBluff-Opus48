# PLAN — index

Index of per-slug plan files in `plans/PLAN-<slug>.md`. One line per entry. This file is an **index only** — detail lives in the slug files.

## Active

- [PLAN-pipeline](plans/PLAN-pipeline.md) — video → game state → REST, in 6 stages. **Shipped 2026-06-22 (classical/CPU, in `src/dbcv/`):** Stage 0 frame-state gate, Stage 1 localization, Stage 2 identification baseline, the REST slice (`POST /v1/snapshot`), and a `utils/` CLI runner — 60 tests green. **Remaining:** Stage 3 on-card OCR, Stage 4 temporal assembly, the Stage 2 embedding-NN upgrade, and Stage 5 ONNX serving — all gated on the first heavier dependencies (onnxruntime/torch), deferred per the conservative-path directive.

## Completed

_None yet._

## How to use this index

- Create `plans/PLAN-<slug>.md` for any unit of forward work; add one line here under **Active**.
- When every checkbox in a slug is done, move its line to **Completed** (keep the link).
- Keep lines to a single sentence — the slug file holds the chat dump, task list, and open questions.
