# io/outputs — formats the code produces

Update this whenever a produced format changes (Rule 3). Stub until code exists; describes the *intended* output contracts.

| Output | Location | Format | Notes |
|--------|----------|--------|-------|
| Selected frames | `dataset/frames/` (gitignored) | image files (+ sidecar) | Chosen frames with provenance (source video, timestamp/index). |
| Card crops | in-memory / `dataset/crops/` (gitignored) | image regions | Localized card regions, resolution-agnostic coordinates. |
| Game-state snapshot | in-memory / `dataset/state/` (debug) | JSON | The structured board read: per-card identity + role class + on-card readings + confidences. Schema TBD. |
| REST response | HTTP endpoint | JSON | The snapshot served over the API. Shape mirrors the game-state schema. |

## Intended game-state schema (draft — not yet implemented)

```jsonc
{
  "source": { "video": "<id>", "frame_index": 0, "timestamp_s": 0.0 },
  "resolution": { "w": 0, "h": 0 },          // read from media, never assumed
  "cards": [
    {
      "bbox_rel": [0.0, 0.0, 0.0, 0.0],      // relative coords, resolution-agnostic
      "role_class": "villager|minion|outcast|demon|unknown",
      "identity": "<townee name | unknown>",
      "readings": { "text": null, "number": null, "state": null },
      "confidence": 0.0
    }
  ],
  "schema_version": "0.0.0"
}
```

Treat this as a target for the first code, not a frozen contract — refine it in `src/` and re-sync here.
