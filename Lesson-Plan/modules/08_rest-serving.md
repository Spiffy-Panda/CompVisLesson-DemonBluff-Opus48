# Module 08 — Serving it: the REST contract and game-state schema

**The problem (in the pipeline):** The localizer finds cards and the identifier names them. Now what? A CV pipeline that lives inside a single script is useful for development; one that exposes a stable HTTP interface is useful to everything else — a web overlay, a game-state logger, a statistics tool, another process on the same machine. The challenge is not just writing an endpoint. It is three coupled problems: (1) how to load expensive resources without paying that cost on every request; (2) how to safely run CPU-bound inference inside an async web framework; and (3) how to shape the output so that a client can depend on it and detect when it changes.

**What you'll be able to do:**

1. Explain why model loading belongs in a `lifespan` context manager rather than at import time or inside the request handler, and describe what goes wrong in each of the wrong places.
2. Articulate the counter-intuitive FastAPI concurrency rule: why CPU-bound inference must be a plain `def` endpoint, not `async def`, to avoid blocking the event loop.
3. Read `src/dbcv/api.py`'s `lifespan` function and `snapshot` endpoint and explain every decision, including why `app.state` is used and why the gallery is built there rather than globally.
4. Describe the `GameStateSnapshot` schema's versioning discipline — what changed between 0.1.0 and 0.2.0, and what a client should do when it sees an unexpected `schema_version`.
5. Run the API locally and POST a frame to it; interpret the JSON response structure and verify the `resolution` field using the test suite.

---

## The options

The research entry that grounds this module is `research/RESEARCH.md`, "Serving CV inference over a REST API (Python) — 2026-06-21."

### Model loading: three candidate locations

**At import time.** Build the gallery and load weights as module-level code — `gallery = build_gallery()` at the top of `api.py`. This works until you import the module in a test, a script, or a subprocessor: the gallery builds immediately, taking ~100 ms and emitting filesystem I/O, even when the caller has no intention of serving requests. It also prevents the loading code from appearing in profiling tools under a meaningful name. Import-time loading is appropriate for pure-Python constants; it is wrong for disk-backed resources.

**Inside the request handler.** Build the gallery each time a request arrives. The first request pays the ~100 ms build cost; so does every subsequent request. At even moderate request rates, the per-request load becomes the dominant latency cost, dwarfing the actual CV work. The gallery's reference images do not change between requests — rebuilding it on each request is pure waste.

**In a `lifespan` context manager (chosen).** A `lifespan` function is an async context manager registered with the FastAPI app. It runs once at startup, before the first request is accepted, and once at shutdown, after the last request completes. Resources built in the startup phase are stored on `app.state` and are shared across every request handler without rebuilding. This is the pattern documented in the FastAPI lifespan docs (`research/RESEARCH.md` entry 5, source 1) as the recommended replacement for the deprecated `@app.on_event("startup")` decorator.

The research entry also flags a tension in the official FastAPI documentation: the *lifespan example* shows an `async def predict(...)` endpoint calling a model directly, while the *concurrency page* explicitly states that CPU-bound work in `async def` blocks the event loop. The shipped code follows the concurrency page. The lesson here is that even official documentation can contain internally inconsistent examples — always check the specific page that addresses your question.

### Sync vs. async endpoints: the non-obvious choice

FastAPI is built on Starlette's async event loop. When a request arrives, the event loop dispatches it to a handler. What happens next depends on whether the handler is `async def` or plain `def`:

**`async def` endpoint:** The handler runs directly on the event loop. While it is executing, no other request can be dispatched. For I/O-bound work (database queries, HTTP calls), this is fine — the handler `await`s the I/O and yields the loop to other requests while waiting. For CPU-bound work (NumPy operations, HSV segmentation, contour finding), there is no `await` — the handler holds the loop for the entire duration of the inference, blocking every other request.

**Plain `def` endpoint (chosen):** Starlette detects that the handler is not a coroutine and runs it in an external thread pool (`run_in_threadpool`). The event loop remains free; other requests can be dispatched while inference runs in a worker thread. NumPy and OpenCV release the GIL during native compute, so multiple frames can process concurrently (up to the thread pool size, typically the number of physical cores).

This is documented explicitly in the FastAPI concurrency docs (`research/RESEARCH.md` entry 5, source 2), but the phrasing is easy to misread. The page says: "If you have a path operation function that is declared as a normal `def` function, it is run in an external thread pool." Many readers see `async def` as "the modern, correct way" and `def` as "the old way." For I/O-bound endpoints, that reading is defensible. For CPU-bound inference, it produces a service that serializes all requests and degrades under any concurrent load.

The research entry also notes that pure-Python pre- and post-processing *serializes under load* (the GIL is held), so the parallelism benefit applies mainly to the native compute inside OpenCV and NumPy, not to Python-level loops over card boxes. For a low-concurrency teaching service, this is an acceptable limitation.

### Schema shape and versioning

The output of the pipeline must be a typed contract, not an unstructured dict. Three options:

**Untyped dict / plain JSON.** Easy to produce; impossible to depend on. A client parsing `response.json()["cards"][0]["bbox"]` has no way to know when the key changes to `bbox_rel`, or when a new required field appears.

**Typed Pydantic model without versioning.** The schema is stable at development time, but there is no signal in the payload when it changes. A client on version 0.1.0 that reads a 0.2.0 response may silently misparse the new fields.

**Typed Pydantic model with `schema_version` (chosen).** A `schema_version: str` field is included in the top-level response. Clients check this field before parsing. When the schema changes in a breaking way, the version string changes. Old clients can detect the mismatch and emit a meaningful error rather than silently consuming a mismatched payload.

---

## What we chose and why

**FastAPI + Pydantic + lifespan + plain `def` endpoint.** The decisions are recorded in `src/dbcv/api.py`'s module docstring and grounded in `research/RESEARCH.md` entry 5.

The four specific choices:

1. **`lifespan` for loading:** resources are built once, stored on `app.state`, and shared across all requests without rebuilding.
2. **Plain `def` endpoint:** CPU-bound CV inference runs in Starlette's thread pool, keeping the event loop free.
3. **Pydantic `response_model`:** the endpoint's return type annotation (`-> GameStateSnapshot`) is passed to `@app.post(response_model=GameStateSnapshot)`, which both validates the output and drives the OpenAPI schema. The client gets a typed contract.
4. **`schema_version` in the payload:** the field is set to `"0.2.0"` in the current schema; clients can detect a version mismatch without inspecting the response body structure.

---

## The shipped implementation: reading the code

### `lifespan` in `api.py`

```python
@asynccontextmanager
async def lifespan(application: FastAPI):
    # --- startup ---
    settings = get_settings()
    application.state.settings = settings

    gallery = build_gallery()
    application.state.gallery = gallery
    application.state.identifier = make_gallery_identifier(gallery)

    yield  # <-- application is live here

    # --- shutdown ---
    # Gallery is in-memory only; nothing to close.
```

`build_gallery()` loads ~67 small PNGs from `knowledge-base/card-art/`, computes HSV histograms and ORB descriptors for each, and returns a reference gallery. This takes roughly 100 ms and touches the filesystem. It runs exactly once per server process, not once per request.

`make_gallery_identifier(gallery)` wraps the gallery in a callable that accepts a card crop and returns `(identity, role_class, confidence)`. The callable is stored on `app.state.identifier` so the endpoint can retrieve it without a module-level reference to the gallery (which would cause the import-time problem described above).

The `yield` is the dividing line: everything before it is startup; everything after it (inside a `finally` block in practice) is shutdown. The app is alive and serving requests only while suspended at `yield`.

**Art-swap note:** on an art swap, rebuild the gallery by restarting the server. Because `build_gallery()` reads from `knowledge-base/card-art/`, swapping the PNG files and restarting is all that is required. No model weights change. No training runs. This is the "no gradient steps on an art swap" principle from Module 05 in action, now enforced at the serving layer.

The comment in the `lifespan` docstring shows where future ONNX model loading would slot in:

```python
# Example (future):
#   application.state.ort_session = ort.InferenceSession(
#       model_path,
#       providers=["CPUExecutionProvider"],
#   )
```

When real ONNX models are wired in, they will load here, once, and live on `app.state` for the server's lifetime. The research entry recommends setting `intra_op_num_threads` to the number of physical cores and `ORT_ENABLE_ALL` for the optimization level.

### `POST /v1/snapshot` in `api.py`

```python
@app.post(
    "/v1/snapshot",
    response_model=GameStateSnapshot,
    summary="Analyse a game frame and return a structured board snapshot.",
)
def snapshot(
    file: UploadFile = File(...),
    video: str = Form(default="unknown"),
    frame_index: int = Form(default=0),
    timestamp_s: float = Form(default=0.0),
) -> GameStateSnapshot:
```

Four things to notice:

1. **Plain `def`, not `async def`.** This is the sync-endpoint choice. Starlette routes this into the thread pool.
2. **`UploadFile`** rather than `bytes`: FastAPI streams the file into a spooled temporary file and exposes it through the `UploadFile` interface. `file.file.read()` returns the raw bytes.
3. **`response_model=GameStateSnapshot`**: FastAPI uses this to validate the return value and generate the OpenAPI schema. A return that does not conform to `GameStateSnapshot` raises an internal validation error before the response is sent.
4. **Resolution from the decoded image:** The endpoint decodes the uploaded bytes with `cv2.imdecode`, hands the array to `run_pipeline`, and never queries a resolution from the request metadata. `run_pipeline` measures `image.shape[:2]` on its own. The `resolution` field in the response reflects the actual dimensions of the uploaded file.

The error handling follows the HTTP convention: a 400 for an empty upload, a 422 for a file that cannot be decoded as an image. Both are documented in the OpenAPI schema automatically.

### `GameStateSnapshot` in `schema.py`

```python
class GameStateSnapshot(BaseModel):
    source: Source
    resolution: Resolution
    frame_state: Literal["board", "modal", "menu", "unknown"]
    cards: list[CardRead]
    schema_version: str = Field(default="0.2.0")
```

The `schema_version` field has a default value of `"0.2.0"` — the current version. Any code that constructs a `GameStateSnapshot` gets this version automatically. When the schema changes in a breaking way, the default is bumped.

**The 0.1.0 → 0.2.0 story:** The `frame_state` field was added in version 0.2.0. In 0.1.0, a response with zero cards was ambiguous: was the localizer called and found nothing, or was the frame a modal that was intentionally skipped? With `frame_state`, the distinction is explicit: `"board"` means the localizer ran; `"modal"` or `"menu"` means it was skipped intentionally; `"unknown"` means the gate did not execute (a test fixture or a snapshot constructed without running the full pipeline). A client on 0.1.0 receiving a 0.2.0 response will see an unexpected `frame_state` field (Pydantic ignores extra fields by default) and a `schema_version` of `"0.2.0"`. A well-written client checks `schema_version` first.

The version string is semver: a minor bump for backward-compatible additions, a major bump for removals or renamed fields. This is the versioning discipline the research entry recommends.

---

## The contract check: the test suite

`tests/test_api.py` serves as the live contract check. Its five tests form a hierarchy:

1. `test_snapshot_http_200` — the endpoint accepts a real frame and returns 200. This catches server startup failures, import errors, and gallery-build failures.
2. `test_snapshot_parses_as_game_state` — the response body validates as `GameStateSnapshot` with `schema_version == "0.2.0"`. This catches any response that does not match the schema (wrong field name, missing required field, wrong type).
3. `test_resolution_matches_actual_image` — the server's `resolution` field matches PIL's independent measurement of the same file. This is the resolution-agnostic correctness check: if the server ever uses a hard-coded resolution, this test catches it on the next run.
4. `test_at_least_four_cards_returned` — the classical localizer finds at least 4 cards on the validated board frame (spike result: 8). This catches a regression from classical to stub localizer.
5. `test_bbox_rel_all_in_unit_range` — every bbox_rel component is in [0.0, 1.0]. This catches a localizer that returns pixel coordinates instead of relative fractions.

The tests run against the FastAPI `TestClient`, which invokes the full lifespan (gallery builds, identifier is created) and then routes requests through the actual endpoint. They are integration tests, not unit tests, and they are the right level of coverage for a serving contract: they test the whole stack, not a mocked subset.

To run them:

```
# Windows:
PYTHONPATH=src .venv\Scripts\python.exe -m pytest tests/test_api.py -v

# macOS / Linux:
PYTHONPATH=src .venv/bin/python -m pytest tests/test_api.py -v
```

---

## Running the server

```
# Windows:
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m uvicorn dbcv.api:app --reload

# macOS / Linux:
PYTHONPATH=src .venv/bin/python -m uvicorn dbcv.api:app --reload
```

Navigate to `http://127.0.0.1:8000/docs` for the interactive OpenAPI interface. The `POST /v1/snapshot` endpoint accepts a multipart form upload. You can POST any board frame from `dataset/frames/` and inspect the full JSON response.

To POST from the command line:

```
curl -X POST http://127.0.0.1:8000/v1/snapshot \
     -F "file=@dataset/frames/Sample1/Sample1_003_t00460s.png" \
     -F "video=Sample1" \
     -F "frame_index=3" \
     -F "timestamp_s=460.0"
```

The response includes `resolution`, `frame_state`, `schema_version`, and the list of `cards` with their `bbox_rel` tuples and `identity` fields.

---

## Honest notes and forward pointers

**No batching.** This API processes one frame per request. The research entry notes that micro-batching helps only under sustained high request rates — for a low-concurrency teaching service where frames arrive interactively (a student POSTing one at a time), per-request inference is the right default. A production service serving a live game monitor at steady state would introduce batching and a queue; that is out of scope for this course.

**No ONNX session yet.** The current pipeline uses a classical gallery identifier (HSV histograms + ORB descriptors, no neural weights). The `lifespan` docstring shows where an `ort.InferenceSession` would load; when the card identification module produces a trained ONNX model, it will slot in there. The serving layer does not need to change. This is the art-swap principle at the serving layer: retrain, re-export to ONNX, restart the server — the endpoint contract is unchanged.

**`schema_version` as a forcing function.** Adding `schema_version` to a schema feels like overhead when the schema is stable. Its value becomes obvious the moment a collaborator (or future self) changes a field name and the downstream consumer fails silently for a week before anyone notices. The discipline of bumping the version string on every breaking change and checking it on every parse is the difference between a contract and a guess.

---

## Failure modes

**`async def` inference blocks all concurrent requests.** If a future contributor refactors the `snapshot` endpoint from `def` to `async def` without adding an explicit `await asyncio.get_event_loop().run_in_executor(...)` call, the first slow frame will block every other request for its duration. Under a load test, the service degrades immediately. The symptom is correct single-request latency and near-zero multi-request throughput. The fix is reverting to plain `def` or adding an explicit executor call.

**Gallery not built before first request.** If the gallery is built inside the endpoint (not in `lifespan`), the first request pays the 100 ms build cost and may race with a second concurrent request that starts the build simultaneously. Depending on whether `build_gallery` is idempotent, this can corrupt the in-memory gallery or simply waste work. The `lifespan` pattern makes this impossible: the app does not accept requests until `yield` returns, and the gallery is built before that.

**`schema_version` not checked by a client.** A client that parses `response["cards"]` without checking `schema_version` will silently misparse a 0.2.0 response if it was written expecting 0.1.0. The `frame_state` field is new in 0.2.0; a 0.1.0 client will ignore it (Pydantic's default), which may be acceptable for `frame_state` but becomes dangerous for a field that changes the meaning of an existing key. The correct client pattern is: read `schema_version` first, reject if not the expected version, then parse the body.

**`UploadFile` not closed after reading.** The shipped code calls `file.file.read()` without an explicit close. FastAPI's `UploadFile` wraps a `SpooledTemporaryFile`; the file handle is closed and cleaned up when the request scope exits. This is safe in practice but worth noting: if the bytes were consumed inside an `async with` context that re-read `file.file` after `read()`, it would appear empty (the file pointer is at EOF). Always read once and keep the bytes.

---

## Further reading

Sources are from `research/RESEARCH.md`, "Serving CV inference over a REST API (Python) — 2026-06-21" (authority: A for official docs, B for practitioners):

- *Lifespan Events* — FastAPI official docs (A) — https://fastapi.tiangolo.com/advanced/events/ — the authoritative source for the `lifespan` pattern and the deprecation of `@app.on_event("startup")`.
- *Concurrency and async / await* — FastAPI official docs (A) — https://fastapi.tiangolo.com/async/ — the page that establishes the `def` vs. `async def` rule for CPU-bound endpoints.
- *FastAPI ML Deployment course* — apxml.com (B) — https://apxml.com/courses/fastapi-ml-deployment/ — chapter 2 (Pydantic response models) and chapter 3 (lifespan model loading) for practitioner-level worked examples.
- *Building Low-Latency Inference APIs Using FastAPI and ONNX* — mljourney.com (B) — https://mljourney.com/building-low-latency-inference-apis-using-fastapi-and-onnx/ — the ONNX Runtime startup/session configuration recommendations.
- *Make FastAPI CPU-bound Endpoints 2X Faster* — amirkarimi.dev (B) — https://amirkarimi.dev/blog/2023/07/23/make-fastapi-cpu-bound-endpoints-2x-faster/ — the worker-thread analysis, including the GIL behaviour during native NumPy/ONNX compute.
