# CODE-DESIGN — index into CodeDocs

The map of `src/`. Read this and `CodeDocs/00_PROJECT.md` **before** opening any source file — an overview may answer your question without a read, and keeps the docs honest (Rule 3).

## Layout

- [`CodeDocs/00_PROJECT.md`](CodeDocs/00_PROJECT.md) — project-level code overview: components, data flow, and the runtime/dev compute split.
- [`CodeDocs/io/inputs.md`](CodeDocs/io/inputs.md) — formats consumed (video, frames, cached wiki data, card-art templates, configs).
- [`CodeDocs/io/outputs.md`](CodeDocs/io/outputs.md) — formats produced (selected frames, the game-state schema, the REST response shape).
- `CodeDocs/sources/<Project>/<File>.md` — one overview per source file (signatures, line numbers, status, who-uses-it). **Empty until code exists.**

## Status

**Stages 0–2 (classical/CPU) + the REST slice landed 2026-06-22.**
`src/dbcv/` contains the frame-state gate, classical localizer, gallery +
classical identifier, assembly, pipeline, and the FastAPI app. A CLI runner
lives at `utils/python/run_pipeline.py` (cataloged in `utils/README.md`).
**60 tests pass** (the `src/dbcv` suite + the CLI helper tests). Note: the suite
runs ~70s because the card-art gallery rebuilds across test files — a tracked
cleanup task exists to share one session-scoped gallery.

Source overviews live under `CodeDocs/sources/dbcv/`:
- [`schema.md`](sources/dbcv/schema.md) — Pydantic v2 models (`GameStateSnapshot`, etc.)
- [`frame_state.md`](sources/dbcv/frame_state.md) — Stage 0 frame-state gate (`classify_frame_state`)
- [`localize.md`](sources/dbcv/localize.md) — localizer interface + classical implementation
- [`gallery.md`](sources/dbcv/gallery.md) — **NEW** gallery builder (`build_gallery`, `Gallery`)
- [`identify.md`](sources/dbcv/identify.md) — classical HSV matcher + stub (Stage 2)
- [`assemble.md`](sources/dbcv/assemble.md) — snapshot assembly (pure function)
- [`pipeline.md`](sources/dbcv/pipeline.md) — end-to-end orchestration + `crop_relative`
- [`api.md`](sources/dbcv/api.md) — FastAPI app, lifespan (builds gallery), `POST /v1/snapshot`
- [`config.md`](sources/dbcv/config.md) — pydantic-settings `Settings`

I/O contracts updated in `CodeDocs/io/`:
- `inputs.md` — card-art gallery documented as active Stage 2 input; API upload format
- `outputs.md` — `GameStateSnapshot` schema at version **0.2.0** (added `frame_state` field)
