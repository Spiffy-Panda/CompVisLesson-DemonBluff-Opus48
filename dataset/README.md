# dataset/ — footage and derived data (large files, GITIGNORED contents)

Holds the sample footage and everything derived from it. **The heavy contents are gitignored** — only the tracked `README.md` files survive in git. This keeps ~740 MB of video (and the frames/crops derived from it) out of history while keeping the layout self-documented.

## Layout

| Path | What | Tracked? |
|------|------|----------|
| `raw-video/Sample1.mp4`, `Sample2.mp4` | The two source recordings (~370 MB each, ~1 h) | **no** — gitignored |
| `frames/` | Frames chosen by the frame-selection stage (+ provenance sidecars) | **no** — gitignored, regenerable |
| `crops/` | Localized card-region crops | **no** — gitignored, regenerable |
| `state/` | Debug game-state snapshots (JSON) | **no** — gitignored, regenerable |

`frames/`, `crops/`, and `state/` are created by the pipeline when it runs; they don't exist yet.

## Hard rules (from CLAUDE.md)

- **Never open the raw videos directly into context.** They are huge. Use the frame-selection script (`scrap_scripts/` → promoted to `utils/`) and inspect only selected frames.
- **Never bake in the resolution.** Both samples share one resolution, but read it from the media every time — no hard-coded width/height/crop pixels.
- Everything under here except the `README.md`s is **regenerable** from `raw-video/` + the pipeline, by design.
