# Module 03 — Resolution-agnostic geometry: never hard-code a pixel

**The problem (in the pipeline):** Before the pipeline can return a single bounding box, it must answer a deceptively simple question: *how big is the frame I am looking at?* Not 1280 × 720. Not 1920 × 1080. Whatever the decoded image actually is right now, measured from its own bytes. The alternative — writing a constant into the code — seems harmless until the footage resolution changes, the streamer switches capture devices, or a learner runs the pipeline on their own clips. At that point every hard-coded pixel offset, every threshold, every crop silently lands in the wrong place. The code does not crash. It produces wrong answers with no complaint.

This module explains the resolution-agnostic design enforced throughout this pipeline and shows, with a self-correcting test, that the discipline is actually honoured.

**What you'll be able to do:**

1. State the single constraint that motivates the whole design: resolution is always read from the decoded image, never assumed.
2. Identify the four places in the pipeline where the principle is enforced — the `Resolution` type, `bbox_rel`, the relative thresholds in `classical_localize`, and `crop_relative` — and explain what would break if each were removed.
3. Read `tests/test_api.py`'s `test_resolution_matches_actual_image` and explain why it is a self-correcting test: it passes for any input size and fails the moment a constant appears.
4. Tell the 1280 × 720 vs. 1920 × 1080 story from this pipeline's own sample footage, and explain why the code did not care.
5. Describe the honest caveat: relative coordinates protect against resolution changes, but HSV thresholds and morphological scales are still tuned to content, not just geometry.

---

## Why this matters: a real discovery from the sample footage

When this pipeline first ran against its two sample clips, the frames that came out of `dataset/frames/Sample1/` measured 1280 × 720. The original capture was almost certainly 1920 × 1080 — Demon Bluff's streamed footage is typically recorded at 1080p — but the sample clips had been downscaled. The pipeline ran without modification and returned correct bounding boxes on both resolutions. It did not care, because it never stored a resolution assumption to invalidate.

This is not a subtle achievement. Most introductory CV tutorials write pixel coordinates directly: `x = 640`, `y = 200`, `crop = frame[100:300, 50:250]`. That code works exactly once, for exactly the frame size it was tuned on. Switching resolutions requires a re-audit of every hard-coded number — a task that is error-prone and leaves the new constants as fragile as the old ones.

The constraint is stated explicitly in `CLAUDE.md`: *"Never bake in a resolution."* This module shows how that constraint propagates into code.

---

## The options

When your pipeline needs to work with image coordinates, you have two broad choices. Evaluating them is straightforward once you accept that the frame resolution is an input, not a constant.

### Option A: Pixel coordinates throughout

Write all offsets, thresholds, and crop bounds as integer pixel values derived from — or assumed to match — one specific resolution. For example: crop the top HUD strip as `frame[0:65, :]`, set the minimum contour area to 500 pixels squared, scale morphological kernels to 13 pixels.

**Accuracy:** Works perfectly on the exact resolution it was tuned for.

**Fragility:** Breaks silently on every other resolution. A 65-pixel strip represents 9% of a 720-pixel-tall frame; on a 1080-pixel frame it represents 6% and misses the bottom of the HUD band. The morphological kernel that was right for 720p may be too small to bridge card gaps at 1080p, producing fragmented detections. There is no error or warning — just wrong output.

**Retrain cost on a resolution change:** Full code audit. Every hard-coded number must be reviewed against the new frame dimensions.

### Option B: Relative coordinates throughout, pixels only at the edge

Store all geometry as fractions of the frame's measured width and height. Convert to pixels exactly once, at the moment you need to index into the array. Express all thresholds relative to `min(w, h)` or as a fraction of frame area. The frame's actual size is measured from `image.shape[:2]` at the start of every function that needs it.

**Accuracy:** Identical to Option A on the training resolution; identical on every other resolution.

**Fragility:** Essentially none for resolution changes. The only code that touches pixels is the final crop conversion — one multiplication, one clamp — and it derives its multipliers from the image it just measured.

**Cost on a resolution change:** Zero. Run the pipeline. It adapts automatically.

**Honest overhead:** Relative coordinates add a tiny conversion burden (`x_px = round(x_rel * w)`) and require discipline: every function that works in pixels must not be called with relative values, and vice versa. A type alias (`BboxRel = tuple[float, float, float, float]`) makes this explicit but does not enforce it at runtime. The pipeline enforces it with an assertion in `classical_localize` that fires if the `Resolution` arg does not match `image.shape`.

---

## What we chose and why

**Option B: relative coordinates throughout, pixels only at the edge.** The constraint is recorded in `CLAUDE.md` ("Never bake in a resolution") and shapes every coordinate-bearing module in the pipeline.

The deciding factor is not elegance — it is correctness across an unpredictable input space. Sample footage may come from different capture sessions, different streamers, different recording software. A learner working through this course will run the pipeline on their own clips. The pipeline must be right on all of them without manual adjustment.

---

## How the principle is enforced: four interlocking decisions

Reading the code is part of the hands-on experience. The resolution-agnostic discipline is not a single function — it is enforced at four distinct points in the codebase.

### 1. `Resolution` — the schema type that carries measured size

`src/dbcv/schema.py` defines:

```python
class Resolution(BaseModel):
    w: int = Field(description="Frame width in pixels, read from the decoded image.")
    h: int = Field(description="Frame height in pixels, read from the decoded image.")
```

The docstring is a contract: `w` and `h` are populated by reading `image.shape[:2]` at runtime. There is no default value, no fallback constant. Every pipeline function that creates a `Resolution` does it from a live image, not from a stored value.

In `pipeline.py`, the measurement happens at the start of `run_pipeline`:

```python
h_img, w_img = image.shape[:2]
resolution = Resolution(w=w_img, h=h_img)
```

This is the one moment the pipeline "sees" the frame dimensions. Everything downstream receives this measured `Resolution` or derives its own dimensions from `image.shape` directly — no function may store the dimensions and re-use them on a different image.

### 2. `bbox_rel` — bounding boxes as fractions, not pixels

`src/dbcv/schema.py` documents `CardRead.bbox_rel`:

```python
bbox_rel: tuple[float, float, float, float] = Field(
    description=(
        "Bounding box as (x, y, w, h) relative to the frame "
        "(fractions in [0, 1], origin top-left)."
    )
)
```

Every localizer in the pipeline must return boxes in this form. The contract is defined by `LocalizerCallable` (a Protocol in `localize.py`) so any future localizer — a learned detector, a prototype, a stub — must satisfy the same interface. A box stored as fractions is meaningless until multiplied by a frame's measured dimensions, which forces the conversion to happen where the frame dimensions are known.

### 3. Thresholds relative to `min(w, h)` — no pixel constants in the localizer

`src/dbcv/localize.py` contains no integer pixel constants. Every threshold is derived from the image's shape:

```python
h, w = image.shape[:2]

# HUD strips as fractions of measured height / width:
work[: int(h * 0.09), :] = 0           # top strip: 9% of height
work[int(h * 0.86) :, :] = 0           # bottom strip
work[:, : int(w * 0.13)] = 0           # left panel: 13% of width
work[:, int(w * 0.92) :] = 0           # right edge: 8% of width

# Morphological kernels proportional to the shorter dimension:
k_close = max(9, int(min(w, h) * 0.018))
k_open  = max(3, int(min(w, h) * 0.006))

# Area thresholds as fractions of frame area:
min_area = w * h * 0.0015
max_area = w * h * 0.09
```

The HUD zone table stored inside `classical_localize` (`HUD_ZONES`) is also expressed as relative fractions: `(0.00, 0.00, 0.13, 1.00)` for the left score panel, not `(0, 0, 166, 720)`. On a different resolution, the fraction maps correctly; the pixel constant would not.

The function includes an explicit sanity-check assertion:

```python
assert resolution.w == w and resolution.h == h, (
    f"resolution arg ({resolution.w}×{resolution.h}) does not match "
    f"image.shape ({w}×{h}).  Always pass the Resolution measured from "
    f"this exact image."
)
```

This fires only if a caller passes a `Resolution` measured from a *different* image — a programming error that would otherwise produce subtle, hard-to-trace misfires.

### 4. `crop_relative` — the single pixel-conversion site

`src/dbcv/pipeline.py` defines `crop_relative`, the only place in the entire pipeline where relative coordinates become pixels:

```python
def crop_relative(image: np.ndarray, bbox_rel: BboxRel) -> np.ndarray:
    h_img, w_img = image.shape[:2]   # read from the array — never a constant
    x_rel, y_rel, w_rel, h_rel = bbox_rel
    x0 = round(x_rel * w_img)
    y0 = round(y_rel * h_img)
    x1 = round((x_rel + w_rel) * w_img)
    y1 = round((y_rel + h_rel) * h_img)
    ...
    return image[y0:y1, x0:x1]
```

The teaching note in the docstring is explicit: "`pixel = round(fraction * dimension)` is the only place in the whole pipeline where a relative coordinate becomes a pixel coordinate." By making this conversion a named function with a single responsibility, the code makes the boundary visible and testable. Every other pipeline stage operates in relative-coordinate space or in `image.shape` space — never in a stored constant space.

---

## The self-correcting test

`tests/test_api.py` includes `test_resolution_matches_actual_image`. Here is how it works:

1. It opens the sample PNG independently using PIL and reads the true dimensions: `img.width, img.height`.
2. It POSTs the same file to the API.
3. It asserts that the `resolution` field in the JSON response matches the PIL-measured dimensions exactly.

```python
def test_resolution_matches_actual_image(
    client, sample_frame_path, actual_dimensions
):
    with open(sample_frame_path, "rb") as fh:
        response = client.post(
            "/v1/snapshot",
            files={"file": ("frame.png", fh, "image/png")},
        )
    actual_w, actual_h = actual_dimensions
    assert data["resolution"]["w"] == actual_w
    assert data["resolution"]["h"] == actual_h
```

Why is this described as "self-correcting"? Because the test does not assert specific pixel values. It asserts that the server's answer matches the file's own truth. Run it against a 1280 × 720 PNG: it passes if the server reports 1280 × 720. Run it against a 1920 × 1080 PNG: it passes if the server reports 1920 × 1080. The only way the test fails is if the server has a hard-coded constant somewhere that overrides the measured value — which is exactly the bug the test is designed to catch.

The frame targeted by this test — `Sample1_003_t00460s.png` — is the validated board frame from the localizer spike (8/8 cards, zero false positives). The test file's own comment notes that `Sample1_000` was intentionally *not* the target: it is a modal frame where the localizer returns near-zero cards, which would break the card-count assertion in a companion test. Choosing the right frame for a test is itself a design decision; the test file documents the reasoning.

---

## Hands-on: verifying the principle

The pipeline's full run path is exposed through the API and through `utils/python/run_pipeline.py`. To see the resolution field in action:

```
# Run the test suite (Windows):
.venv\Scripts\python.exe -m pytest tests/test_api.py::test_resolution_matches_actual_image -v

# Run the test suite (macOS / Linux):
.venv/bin/python -m pytest tests/test_api.py::test_resolution_matches_actual_image -v
```

To observe the `resolution` field in a live response, start the server and POST any frame:

```
PYTHONPATH=src .venv\Scripts\python.exe -m uvicorn dbcv.api:app --reload   # Windows
PYTHONPATH=src .venv/bin/python -m uvicorn dbcv.api:app --reload           # macOS / Linux
```

Then POST a frame with curl or the interactive docs at `http://127.0.0.1:8000/docs`. The `resolution` field in the JSON response will reflect the dimensions of whatever file you uploaded — not a constant.

---

## The honest caveat

Relative coordinates insulate the pipeline against resolution changes. They do not insulate it against content changes.

The HSV thresholds in Stage 2 of `classical_localize` were tuned to the current card art palette (the specific hue ranges of Demon Bluff's purple role rings, orange card backs, and demon-red accents). Those thresholds are stored as numeric constants — not pixel constants, but colour constants. An art swap changes which colours appear on the cards, which changes which HSV ranges are needed. The 15–30 minute re-tune described in Module 04 is not avoidable by the resolution-agnostic design.

Similarly, the morphological kernel size (`k_close = max(9, int(min(w, h) * 0.018))`) is expressed relative to frame dimensions, but the `0.018` coefficient was tuned to the scale at which card features appear in the current footage. A scene filmed with a very different camera distance — card sprites at 5% of the frame height instead of 15% — would need the coefficient revisited. The formula adapts to resolution; it does not adapt to arbitrary content scale.

This is the honest boundary of the principle: *same content, different resolution* is handled automatically; *different content* requires re-tuning the content-driven parameters. Understanding this boundary is what separates a practitioner who knows why the design works from one who only knows that it does.

---

## Failure modes

**Mixing pixel and relative coordinates silently.** The type alias `BboxRel = tuple[float, float, float, float]` communicates intent but does not enforce it at the Python runtime. If a caller passes a pixel-coordinate tuple to a function expecting `BboxRel`, the values will be treated as fractions and produce tiny, near-origin crops with no error. The `resolution.w == w` assertion in `classical_localize` catches one version of this mistake (passing a wrong-sized Resolution), but it does not catch a pixel tuple smuggled in as `bbox_rel`. The defence is naming discipline and test coverage; both are in place.

**Assuming a fixed aspect ratio.** The pipeline reads `w` and `h` independently and uses them independently. A function that computes `h = w * 9/16` — assuming 16:9 footage — would appear resolution-agnostic but would fail on 4:3 or 21:9 sources. No such assumption exists in the current code, but it is a class of mistake worth naming because it is common and produces off-axis crops rather than visible errors.

**HUD zone fractions tuned to one layout version.** The HUD zone table in `classical_localize` was measured from the current build of Demon Bluff. A future game update that repositions the name strip or the timer cluster would require updating the fractions. The fractions are stored in one place and are easy to find, but they are not automatically correct for every game build.

---

## Further reading

- `CLAUDE.md` — the project constraint that gates this whole design: "Never bake in a resolution." The constraint is not derived from a cited paper; it is a project engineering decision made explicit here as its source.
- `research/RESEARCH.md`, "Card/region localization robust to art swaps under a tight compute budget — 2026-06-21" — the localization research entry; the note on "reading the frame resolution" and "keeping boxes relative" is the direct research grounding for the bbox_rel design.
- `src/dbcv/schema.py` — `Resolution`, `CardRead.bbox_rel`, docstrings.
- `src/dbcv/localize.py` — `classical_localize`, the assertion, `HUD_ZONES` as fractions.
- `src/dbcv/pipeline.py` — `crop_relative`, the teaching note on the single conversion site.
- `tests/test_api.py` — `test_resolution_matches_actual_image`, `actual_dimensions` fixture.
