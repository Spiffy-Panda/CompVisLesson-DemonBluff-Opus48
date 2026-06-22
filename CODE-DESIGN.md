# CODE-DESIGN — index into CodeDocs

The map of `src/`. Read this and `CodeDocs/00_PROJECT.md` **before** opening any source file — an overview may answer your question without a read, and keeps the docs honest (Rule 3).

## Layout

- [`CodeDocs/00_PROJECT.md`](CodeDocs/00_PROJECT.md) — project-level code overview: components, data flow, and the runtime/dev compute split.
- [`CodeDocs/io/inputs.md`](CodeDocs/io/inputs.md) — formats consumed (video, frames, cached wiki data, card-art templates, configs).
- [`CodeDocs/io/outputs.md`](CodeDocs/io/outputs.md) — formats produced (selected frames, the game-state schema, the REST response shape).
- `CodeDocs/sources/<Project>/<File>.md` — one overview per source file (signatures, line numbers, status, who-uses-it). **Empty until code exists.**

## Status

No `src/` code yet. This index and the `CodeDocs/io/` stubs describe the *intended* contracts so the first code has a target to hit. Keep them in sync the moment real files land (Rule 3): every new source file gets a `CodeDocs/sources/...` overview, and any change to a consumed/produced format updates `io/inputs.md` or `io/outputs.md`.
