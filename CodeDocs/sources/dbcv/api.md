# CodeDocs/sources/dbcv/api.py

**Status:** active — single endpoint; the fine-tuned embedding-NN identifier (Stage 2) is built in lifespan as the default (classical retained as fallback).
**2026-07-29:** lifespan now also builds the opt-in ensemble identifier when
`Settings.identifier == "ensemble"` (plans/PLAN-live-capture.md Fix 3).

**Purpose:** FastAPI application object and the `POST /v1/snapshot` endpoint.
Follows the lifespan + plain-def patterns from research/RESEARCH.md entry 5.

**Who uses it:**
- `tests/test_api.py` — imports `app` for TestClient
- Production: run via `uvicorn dbcv.api:app`

---

## Key signatures (with line numbers)

### `lifespan(application: FastAPI)` async context manager — line 55
```python
@asynccontextmanager
async def lifespan(application: FastAPI):
```
Runs at startup (before first request) and shutdown.  Sets on `application.state`:
- `settings` — `Settings` instance from `get_settings()`
- `gallery` — `Gallery` object from `build_gallery()` (67 references, 43 townees)
- `classical_identifier` — callable from `make_gallery_identifier(gallery)` (classical baseline, retained as fallback; **always built**, regardless of `Settings.identifier`)
- `embedder` — `OnnxEmbedder` instance via `get_onnx_embedder()` (process-cached; loads `models/mobilenetv3_small_embed.onnx`, now the **fine-tuned** served model)
- `embed_gallery` — `EmbeddingGallery` from `build_embedding_gallery()` (43 prototypes, [43,576] matrix)
- `identifier` — selected by `Settings.identifier` (line 117, `DBCV_IDENTIFIER`
  env var; added 2026-07-29):
  - `"embedding"` (default) → `make_embedding_identifier(embedder, embed_gallery)`, unchanged pre-2026-07-29 behaviour
  - `"classical"` → `application.state.classical_identifier`
  - `"ensemble"` → `make_ensemble_identifier(classical_identifier, embedding_identifier)` (plans/PLAN-live-capture.md Fix 3)

If the ONNX file is absent, the lifespan emits a `warnings.warn` and falls back
to `classical_identifier` regardless of the requested selection (embedding
and ensemble both need the ONNX model); `embedder` and `embed_gallery` = None.

**Teaching note:** `@app.on_event("startup")` is deprecated — `lifespan` is
the current recommended pattern (FastAPI docs, research/RESEARCH.md entry 5 src 1).

### `app = FastAPI(...)` — line 164
FastAPI application object.  Title: "Demon Bluff CV — snapshot API".
Version: "0.1.0".  Passed `lifespan=lifespan`.

### `snapshot(file, video, frame_index, timestamp_s) -> GameStateSnapshot` — line 186
```python
@app.post("/v1/snapshot", response_model=GameStateSnapshot)
def snapshot(
    file: UploadFile = File(...),
    video: str = Form(default="unknown"),
    frame_index: int = Form(default=0),
    timestamp_s: float = Form(default=0.0),
) -> GameStateSnapshot:
```
**Plain `def`** (not `async def`) so FastAPI runs it in the threadpool.
CPU-bound CV inference in `async def` would block the event loop
(research/RESEARCH.md entry 5, source 2 and 5).

**Image decoding (line 225):**
```python
image_array = cv2.imdecode(
    np.frombuffer(raw_bytes, dtype=np.uint8),
    cv2.IMREAD_COLOR,
)
```
Returns `None` on failure → raises `HTTPException(422)`.

**Resolution:** read from `image_array.shape` inside `run_pipeline` — not from
the request, not from a constant.

---

## Running locally

```
# src/ must be on PYTHONPATH (no editable install in the slice)
$env:PYTHONPATH = "src"
.venv\Scripts\uvicorn.exe dbcv.api:app --reload
```

Or equivalently with inline env:
```
$env:PYTHONPATH = "src"; .venv\Scripts\uvicorn.exe dbcv.api:app --reload
```
