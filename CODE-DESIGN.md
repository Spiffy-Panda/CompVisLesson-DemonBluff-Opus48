# CODE-DESIGN — index into CodeDocs

The map of `src/`. Read this and `CodeDocs/00_PROJECT.md` **before** opening any source file — an overview may answer your question without a read, and keeps the docs honest (Rule 3).

## Layout

- [`CodeDocs/00_PROJECT.md`](CodeDocs/00_PROJECT.md) — project-level code overview: components, data flow, and the runtime/dev compute split.
- [`CodeDocs/io/inputs.md`](CodeDocs/io/inputs.md) — formats consumed (video, frames, cached wiki data, card-art templates, configs).
- [`CodeDocs/io/outputs.md`](CodeDocs/io/outputs.md) — formats produced (selected frames, the game-state schema, the REST response shape).
- `CodeDocs/sources/<Project>/<File>.md` — one overview per source file (signatures, line numbers, status, who-uses-it). **Empty until code exists.**

## Status

**Vertical-slice skeleton landed 2026-06-22.** `src/dbcv/` now exists.

Source overviews live under `CodeDocs/sources/dbcv/`:
- [`schema.md`](sources/dbcv/schema.md) — Pydantic v2 models (`GameStateSnapshot`, etc.)
- [`frame_state.md`](sources/dbcv/frame_state.md) — Stage 0 frame-state gate (`classify_frame_state`)
- [`localize.md`](sources/dbcv/localize.md) — localizer interface + stub
- [`identify.md`](sources/dbcv/identify.md) — identifier interface + stub
- [`assemble.md`](sources/dbcv/assemble.md) — snapshot assembly (pure function)
- [`pipeline.md`](sources/dbcv/pipeline.md) — end-to-end orchestration + `crop_relative`
- [`api.md`](sources/dbcv/api.md) — FastAPI app, lifespan, `POST /v1/snapshot`
- [`config.md`](sources/dbcv/config.md) — pydantic-settings `Settings`

I/O contracts updated in `CodeDocs/io/`:
- `inputs.md` — documents new API frame upload format
- `outputs.md` — `GameStateSnapshot` schema at version **0.2.0** (added `frame_state` field)
