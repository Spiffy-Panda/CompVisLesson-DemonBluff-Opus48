# io/outputs — formats the code produces

Update this whenever a produced format changes (Rule 3).

| Output | Location | Format | Notes |
|--------|----------|--------|-------|
| Stage 0 selected-frame records | in-memory (list / stream) | `SelectedFrame` NamedTuples | **Dev/batch selector output, NOT a REST response.** Produced by `frame_select.select_frames()` (returns `list[SelectedFrame]`) and `frame_select.iter_selected_frames()` (streams `(SelectedFrame, bgr)`). Kept-frame metadata only. See schema below and `CodeDocs/sources/dbcv/frame_select.md`. |
| Selected frames | `dataset/frames/` (gitignored) | PNG image files | Chosen frames with provenance encoded in the filename (`<Set>_<NNN>_t<SSSSS>s.png`). |
| Card crops | in-memory only (slice) | numpy array (HxWxC, BGR) | Localized card regions in pixel coordinates, derived from relative boxes. |
| Game-state snapshot | in-memory / HTTP response | JSON (`GameStateSnapshot`) | The structured board read per frame.  See schema below.  Version: **0.2.0**. |
| REST response | `POST /v1/snapshot` | JSON | Snapshot returned over HTTP.  Shape is identical to the game-state schema. |
| Batch pipeline output | `dataset/pipeline-out/` (gitignored, regenerable) | JSON + PNG pairs | Produced by `utils/python/run_pipeline.py` over already-selected frames: per frame a `<Sample>_<NNN>_t<SSSSS>s.json` (`GameStateSnapshot`, same schema as the REST response) plus, with `--overlay`, a `<same stem>_overlay.png` annotated visual overlay. Dev/debug artifacts only — never served. |

---

## GameStateSnapshot schema — version 0.2.0 (current)

Implemented in `src/dbcv/schema.py`.

```jsonc
{
  "source": {
    "video": "<stem of source video, no path>",    // str
    "frame_index": 0,                               // int, zero-based
    "timestamp_s": 0.0                              // float, seconds
  },
  "resolution": {
    "w": 1920,    // int — read from decoded image, NEVER hard-coded
    "h": 1080     // int — read from decoded image, NEVER hard-coded
  },
  "frame_state": "board",   // NEW in 0.2.0 — Stage 0 gate result
                             // "board" | "modal" | "menu" | "unknown"
                             // "unknown" only on hand-constructed test fixtures
  "cards": [
    {
      "bbox_rel": [0.08, 0.30, 0.18, 0.30],   // [x, y, w, h] fractions in [0,1]
                                                // origin top-left
      "role_class": "villager",                 // "villager"|"minion"|"outcast"|"demon"|"unknown"
      "identity": "Alchemist",                  // townee name or "unknown"
      "readings": {
        "text": null,                            // str | null — on-card text (OCR, Stage 3 — not built yet)
        "number": null,                          // int | null — numeric reading
        "state": null                            // str | null — discrete state marker
      },
      "confidence": 0.91                        // float in [0.0, 1.0] — see note below
    }
  ],
  "schema_version": "0.2.0"  // bump on any breaking change
}
```

**`confidence` semantics depend on the identifier (no schema change):**

- **Embedding identifier (the default, Stage 2).** `confidence` is the **top1−top2 cosine
  margin** — how decisively the nearest prototype beat the runner-up — clamped into [0, 1].
  After the 2026-06-22 fine-tune this replaced the old `(cos+1)/2` absolute-cosine remap,
  because fine-tuning compressed the absolute cosine scale and only the margin still
  discriminates. The abstention gate is `_EMBED_MARGIN_THRESHOLD = 0.12` (provisional). See
  `CodeDocs/sources/dbcv/identify.md`.
- **Classical identifier (fallback).** `confidence` is the HSV-histogram Pearson correlation of
  the best match, clamped to [0, 1] (gate `_CONFIDENCE_THRESHOLD = 0.40`).

Both are floats in [0.0, 1.0] and validated by the schema; the wire shape is unchanged. The
number is **not** comparable across the two identifiers (different quantities).

**Invariant introduced in 0.2.0:** when `frame_state` is `"modal"` or
`"menu"`, `cards` is always `[]`.  The localizer is intentionally skipped
for non-board frames to prevent false detections.

---

## SelectedFrame — Stage 0 selector record (dev/batch only)

Implemented in `src/dbcv/frame_select.py` as a `NamedTuple`. This is the
**output of the offline frame-selection cascade**, not a wire format and not a
REST response — it never travels over HTTP. `select_frames()` returns a
`list[SelectedFrame]` (metadata only, no pixels); `iter_selected_frames()`
streams `(SelectedFrame, bgr)` pairs when the kept image is also needed.

| Field | Type | Notes |
|-------|------|-------|
| `frame_index` | `int` | 0-based index into the decoded stream |
| `timestamp_s` | `float` | Presentation timestamp = `frame_index / media_fps` (fps read from the media, never assumed) |
| `state` | `"board" \| "modal" \| "menu"` | Result of reusing `classify_frame_state` as the gate |
| `dhash` | `int` | 64-bit difference hash of the kept frame (used for near-duplicate dedup) |

Pixels are intentionally **not** stored on the record so a whole-video pass
stays cheap in memory. See `CodeDocs/sources/dbcv/frame_select.md` for the
cascade and tuning constants.

---

## Schema changelog

### 0.2.0 (2026-06-22) — Stage 0 frame-state gate

| Field | 0.1.0 | 0.2.0 | Notes |
|-------|-------|-------|-------|
| `frame_state` | absent | `"board"\|"modal"\|"menu"\|"unknown"` | Stage 0 gate result |
| `schema_version` default | `"0.1.0"` | `"0.2.0"` | Bumped for field addition |
| `cards` semantics | populated when localizer runs | guaranteed `[]` when `frame_state != "board"` | Pipeline enforces this |

### 0.1.0 (2026-06-22) — Initial slice

| Field | Draft (0.0.0) | Implemented (0.1.0) | Notes |
|-------|--------------|---------------------|-------|
| `schema_version` | `"0.0.0"` | `"0.1.0"` | Frozen for slice |
| `bbox_rel` type | array (JSON) | `tuple[float,float,float,float]` | Serialises as array — no wire change |
| `role_class` | string | `Literal[...]` with Pydantic validation | Now validated at parse time |
| `confidence` | present | `ge=0.0, le=1.0` validators added | Out-of-range values raise ValidationError |
| `readings.*` | all present | all present, all nullable | No change in shape |

---

### Coordinate convention

`bbox_rel` is **(x, y, w, h)** where:
- `x`, `y` — top-left corner as a fraction of the frame **width** and **height** respectively
- `w`, `h` — box width and height as fractions of the frame **width** and **height** respectively
- Origin is the **top-left** corner of the frame
- All four values must be in **[0.0, 1.0]**

Convert to pixels: `pixel_x = round(x * frame_width)`.  This conversion
happens exactly once, in `pipeline.crop_relative()`.
