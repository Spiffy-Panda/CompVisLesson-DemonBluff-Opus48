# io/inputs — formats the code consumes

Update this whenever a consumed format changes (Rule 3). Stub until code exists; describes the *intended* input contracts.

| Input | Location | Format | Notes |
|-------|----------|--------|-------|
| Sample footage | `dataset/raw-video/*.mp4` (gitignored) | video container | ~370 MB, ~1 h each. **Never opened directly** — only via the frame-selection stage. Resolution read from the media, never assumed. |
| Selected frames | `dataset/frames/` (gitignored) | image files | Output of frame selection; the first thing downstream stages actually see. |
| Card-art templates | `knowledge-base/card-art/` (gitignored) | image files | Reference art per townee for identification; assumed swappable, so kept easy to re-fit. |
| Cached game facts | `knowledge-base/wiki/*.md` | markdown | Transcribed/transformed wiki content (roles, mechanics). The raw fetch cache (`wiki/_raw_cache/`) is gitignored. |
| Run config | `src/.../config.*` (TBD) | TBD (likely YAML/JSON) | Paths, thresholds, model selection. No resolution baked in. |

## Open contract questions

- Frame-selection output naming and metadata sidecar (timestamp, source video, frame index).
- Card-art template manifest format (how identities map to files, versioned for art swaps).
