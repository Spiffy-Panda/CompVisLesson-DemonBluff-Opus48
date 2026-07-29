# CODE-DESIGN — index into CodeDocs

The map of `src/`. Read this and `CodeDocs/00_PROJECT.md` **before** opening any source file — an overview may answer your question without a read, and keeps the docs honest (Rule 3).

## Layout

- [`CodeDocs/00_PROJECT.md`](CodeDocs/00_PROJECT.md) — project-level code overview: components, data flow, and the runtime/dev compute split.
- [`CodeDocs/io/inputs.md`](CodeDocs/io/inputs.md) — formats consumed (video, frames, cached wiki data, card-art templates, configs).
- [`CodeDocs/io/outputs.md`](CodeDocs/io/outputs.md) — formats produced (selected frames, the game-state schema, the REST response shape).
- `CodeDocs/sources/<Project>/<File>.md` — one overview per source file (signatures, line numbers, status, who-uses-it). **Empty until code exists.**

## Status

**Stages 0–2 (classical + fine-tuned embedding-NN) + REST + CLI landed 2026-06-22.** (Stage 3 = OCR, not built yet.)
`src/dbcv/` now contains the full package through Stage 2 identification:
- Stage 0 (frame selection — dev/batch): `frame_select.py` (stride-decode + dHash dedup → reuses the gate)
- Stage 0 (frame-state gate): `frame_state.py`
- Stage 1 (classical localizer): `localize.py`
- Stage 2 (identification): classical baseline — `gallery.py` + `identify.py` (`classify_crop`);
  **adopted default** — embedding-NN over a fine-tuned backbone: `embed.py` + `gallery.py`
  (`build_embedding_gallery`) + `identify.py` (`classify_crop_embedding`)
- REST: `api.py` — the fine-tuned embedding-NN is the default identifier; classical retained as fallback
- CLI: `utils/python/run_pipeline.py` (wires the classical identifier)
- Dev model tools (require torch): `utils/python/finetune_embedding.py` (generates the served
  fine-tuned model) and `utils/python/export_backbone.py` (generates the frozen baseline)

**123 passed / 25 skipped** (was 110 tests / 81 prior + 29 Stage 0 frame-selection). Suite runtime **~15-20 s**
(the embedding suite was made fast 2026-06-22 via process-level memoization of `build_gallery`,
`OnnxEmbedder`, and `build_embedding_gallery`; see DEV-LOG entry 2026-06-22 for rationale).
**2026-07-29:** +38 tests for the live-eval fixes (`plans/PLAN-live-capture.md`) — HUD-zone
masking/recall (`tests/test_localize.py`), Kill-Mode red-tint detection (additions to
`tests/test_frame_state.py`), the ensemble combiner (additions to `tests/test_identify.py`),
and the tint-discount pipeline wiring (`tests/test_pipeline.py`).

Source overviews live under `CodeDocs/sources/dbcv/`:
- [`schema.md`](sources/dbcv/schema.md) — Pydantic v2 models (`GameStateSnapshot`, etc.)
- [`frame_select.md`](sources/dbcv/frame_select.md) — Stage 0 frame selector (dev/batch: stride-decode + dHash dedup → gate; `select_frames`/`iter_selected_frames`)
- [`frame_state.md`](sources/dbcv/frame_state.md) — Stage 0 frame-state gate (`classify_frame_state`)
- [`localize.md`](sources/dbcv/localize.md) — localizer interface + classical implementation
- [`gallery.md`](sources/dbcv/gallery.md) — classical gallery + embedding gallery (Stage 2)
- [`embed.md`](sources/dbcv/embed.md) — `OnnxEmbedder` (Stage 2 embedding runtime, fine-tuned model, no torch)
- [`identify.md`](sources/dbcv/identify.md) — classical HSV matcher + fine-tuned embedding-NN identifier (Stage 2)
- [`assemble.md`](sources/dbcv/assemble.md) — snapshot assembly (pure function)
- [`pipeline.md`](sources/dbcv/pipeline.md) — end-to-end orchestration + `crop_relative`
- [`api.md`](sources/dbcv/api.md) — FastAPI app, lifespan (builds both galleries), `POST /v1/snapshot`
- [`config.md`](sources/dbcv/config.md) — pydantic-settings `Settings`

I/O contracts updated in `CodeDocs/io/`:
- `inputs.md` — card-art gallery documented as active Stage 2 input; API upload format
- `outputs.md` — `GameStateSnapshot` schema at version **0.2.0** (added `frame_state` field)
