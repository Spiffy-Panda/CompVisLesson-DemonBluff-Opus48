# CodeDocs/sources/dbcv/assemble.py

**Status:** slice/active — stateless assembly, no temporal logic yet.

**Purpose:** Packages per-card localizer and identifier outputs into a
`GameStateSnapshot`.  Pure data transformation; no I/O.

**Who uses it:**
- `dbcv/pipeline.py` — calls `assemble(source, resolution, boxes, identities)`

---

## Key signatures (with line numbers)

### `assemble(source, resolution, boxes, identities) -> GameStateSnapshot` — line 33
```python
def assemble(
    source: Source,
    resolution: Resolution,
    boxes: list[tuple[float, float, float, float]],
    identities: list[tuple[str, str, float]],
) -> GameStateSnapshot:
```
**Parameters:**
- `source` — provenance metadata (video, frame_index, timestamp_s)
- `resolution` — frame dimensions from `image.shape` (never assumed)
- `boxes` — relative bounding boxes from the localizer (parallel to `identities`)
- `identities` — `(identity, role_class, confidence)` triples from identifier

**Returns:** `GameStateSnapshot` with `schema_version="0.1.0"`.

**Raises:** `ValueError` if `len(boxes) != len(identities)` — indicates a
pipeline logic error, should not be caught.

---

## Design notes

- `role_class` values outside the known set are normalised to `"unknown"` at
  line 68 rather than raising — defensive against unexpected identifier output
  during development.
- `Readings` is constructed empty (`Readings()`) for every card in the slice;
  the OCR stage (Stage 3) will populate it.
- No temporal state is held; this is a pure function.  Temporal smoothing
  across frames will be added as a later stage.
