# CodeDocs/sources/dbcv/schema.py

**Status:** slice/active — Pydantic v2 models are frozen for the 0.1.0 schema.

**Purpose:** Defines the data contract for the entire pipeline and REST API.
All other modules import from here; `schema.py` imports from nothing in the
package (no circular risk).

**Who uses it:**
- `dbcv/assemble.py` — constructs `CardRead`, `GameStateSnapshot`
- `dbcv/pipeline.py` — constructs `Resolution`, `Source`
- `dbcv/api.py` — declares `response_model=GameStateSnapshot`; constructs `Source`
- `dbcv/localize.py` — imports `Resolution` for the localizer interface
- `tests/test_schema.py` — tests round-trip serialisation of all models
- `tests/test_api.py` — validates the REST response as `GameStateSnapshot`

---

## Key signatures (with line numbers)

### `class Source(BaseModel)` — line 40
```python
class Source(BaseModel):
    video: str          # identifier for source video (stem only, no path)
    frame_index: int    # zero-based frame index
    timestamp_s: float  # seconds from start of video
```

### `class Resolution(BaseModel)` — line 51
```python
class Resolution(BaseModel):
    w: int  # frame width in pixels — read from decoded image, NEVER assumed
    h: int  # frame height in pixels — read from decoded image, NEVER assumed
```

### `class Readings(BaseModel)` — line 62
```python
class Readings(BaseModel):
    text: str | None = None      # role-name text found on the card
    number: int | None = None    # numeric reading (ability count, HUD digit)
    state: str | None = None     # discrete state marker (e.g. 'poisoned')
```

### `class CardRead(BaseModel)` — line 77
```python
class CardRead(BaseModel):
    bbox_rel: tuple[float, float, float, float]
    #   (x, y, w, h) as fractions of frame width/height, origin top-left
    #   all values in [0.0, 1.0]
    role_class: Literal["villager", "minion", "outcast", "demon", "unknown"]
    identity: str        # townee name or "unknown"
    readings: Readings   # default_factory=Readings (all None)
    confidence: float    # [0.0, 1.0], ge=0.0 le=1.0
```

### `class GameStateSnapshot(BaseModel)` — line 110
```python
class GameStateSnapshot(BaseModel):
    source: Source
    resolution: Resolution
    cards: list[CardRead]       # default_factory=list
    schema_version: str = "0.1.0"
```

---

## Design notes

- `bbox_rel` is a tuple of 4 floats, not a list, to encourage treating it as
  an immutable (x, y, w, h) record rather than an arbitrary sequence.
- `confidence` has `ge=0.0, le=1.0` Pydantic validators — values outside
  [0, 1] raise `ValidationError` at construction time.
- `schema_version` is a plain string (not semver validated) to keep the model
  lightweight; validate it in client code if needed.
- No resolution is hard-coded here.  `Resolution.w` and `Resolution.h` are
  always populated from the decoded image's `image.shape`.
