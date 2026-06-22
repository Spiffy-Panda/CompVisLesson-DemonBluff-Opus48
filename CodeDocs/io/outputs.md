# io/outputs — formats the code produces

Update this whenever a produced format changes (Rule 3).

| Output | Location | Format | Notes |
|--------|----------|--------|-------|
| Selected frames | `dataset/frames/` (gitignored) | PNG image files | Chosen frames with provenance encoded in the filename (`<Set>_<NNN>_t<SSSSS>s.png`). |
| Card crops | in-memory only (slice) | numpy array (HxWxC, BGR) | Localized card regions in pixel coordinates, derived from relative boxes. |
| Game-state snapshot | in-memory / HTTP response | JSON (`GameStateSnapshot`) | The structured board read per frame.  See schema below.  Version: **0.1.0**. |
| REST response | `POST /v1/snapshot` | JSON | Snapshot returned over HTTP.  Shape is identical to the game-state schema. |

---

## GameStateSnapshot schema — version 0.1.0 (implemented, frozen for slice)

Implemented in `src/dbcv/schema.py`.  This section is authoritative; the
`io/outputs.md` draft was version 0.0.0 / "not frozen."

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
  "cards": [
    {
      "bbox_rel": [0.08, 0.30, 0.18, 0.30],   // [x, y, w, h] fractions in [0,1]
                                                // origin top-left
      "role_class": "villager",                 // "villager"|"minion"|"outcast"|"demon"|"unknown"
      "identity": "Alchemist",                  // townee name or "unknown"
      "readings": {
        "text": null,                            // str | null — on-card text (OCR, Stage 3)
        "number": null,                          // int | null — numeric reading
        "state": null                            // str | null — discrete state marker
      },
      "confidence": 0.91                        // float in [0.0, 1.0]
    }
  ],
  "schema_version": "0.1.0"  // bump on any breaking change
}
```

### Changes from the 0.0.0 draft (in `io/outputs.md` prior to 2026-06-22)

| Field | Draft (0.0.0) | Implemented (0.1.0) | Notes |
|-------|--------------|---------------------|-------|
| `schema_version` | `"0.0.0"` | `"0.1.0"` | Frozen for slice |
| `bbox_rel` type | array (JSON) | `tuple[float,float,float,float]` | Serialises as array — no wire change |
| `role_class` | string | `Literal[...]` with Pydantic validation | Now validated at parse time |
| `confidence` | present | `ge=0.0, le=1.0` validators added | Out-of-range values raise ValidationError |
| `readings.*` | all present | all present, all nullable | No change in shape |

### Coordinate convention

`bbox_rel` is **(x, y, w, h)** where:
- `x`, `y` — top-left corner as a fraction of the frame **width** and **height** respectively
- `w`, `h` — box width and height as fractions of the frame **width** and **height** respectively
- Origin is the **top-left** corner of the frame
- All four values must be in **[0.0, 1.0]**

Convert to pixels: `pixel_x = round(x * frame_width)`.  This conversion
happens exactly once, in `pipeline.crop_relative()`.
