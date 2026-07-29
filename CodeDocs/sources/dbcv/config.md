# CodeDocs/sources/dbcv/config.py

**Status:** slice/active — minimal settings; placeholder `confidence_threshold`.
**2026-07-29:** added `identifier` (embedding/classical/ensemble selector,
plans/PLAN-live-capture.md Fix 3).

**Purpose:** pydantic-settings `Settings` class for project-wide runtime
configuration.  Values come from environment variables (prefixed `DBCV_`) or
a `.env` file.  No resolution is baked in.

**Who uses it:**
- `dbcv/api.py` — calls `get_settings()` in `lifespan` to store on `app.state`
- Any future module needing paths or thresholds

---

## Key signatures (with line numbers)

### `_REPO_ROOT` — line 41
```python
_REPO_ROOT = Path(__file__).resolve().parents[2]
# parents[0] = src/dbcv, parents[1] = src, parents[2] = repo root
```
Anchored to the repo root per Rule 1 — never assumes CWD.

### `class Settings(BaseSettings)` — line 50
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DBCV_", ...)
    frames_dir: Path = Field(default=_REPO_ROOT / "dataset" / "frames")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    identifier: Literal["embedding", "classical", "ensemble"] = Field(default="embedding")
```
- `DBCV_FRAMES_DIR` overrides `frames_dir`
- `DBCV_CONFIDENCE_THRESHOLD` overrides `confidence_threshold`
- `DBCV_IDENTIFIER` overrides `identifier` — (added 2026-07-29, line 102)
  selects which Stage 2 identifier `api.py`'s lifespan builds and serves.
  `"embedding"` (default) reproduces pre-2026-07-29 behaviour exactly;
  `"classical"` and `"ensemble"` (plans/PLAN-live-capture.md Fix 3) are
  opt-in.

### `get_settings() -> Settings` — line 119
```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
```
Singleton: environment variables are read exactly once.  Call in `lifespan`,
not inside every request handler.

---

## Adding new settings

Add a field to `Settings` with a type and default.  The env var name is
`DBCV_<FIELD_NAME_UPPERCASED>`.  Never add a hard-coded resolution.
