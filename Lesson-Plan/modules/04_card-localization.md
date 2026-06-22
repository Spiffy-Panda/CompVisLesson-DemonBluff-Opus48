# Module 04 — Finding the cards: classical layout-based localization vs. a trained detector

**The problem (in the pipeline):** Before the pipeline can name a card, read its text, or assemble a game-state snapshot, it must answer a more primitive question: *where are the cards in this frame?* A frame of *Demon Bluff* footage contains between 8 and 9 cards arranged in a radial ring, surrounded by HUD elements, streamer overlays, and occasionally modal dialogs that occlude part of the board. The pipeline needs to return a bounding box for each card — reliably, quickly, and in a form that does not break the moment the card art set is swapped for the alternate set.

**What you'll be able to do:**

1. Articulate the design decision space for localization in a fixed-layout UI: classical geometry-based methods vs. trained object detectors vs. large foundation models, on the axes of accuracy, speed, training-data need, retrain cost, and compute budget.
2. Explain why layout-based classical localization is the appropriate choice for this pipeline, given the project's specific constraints.
3. Read and explain each step of `src/dbcv/localize.py`'s `classical_localize` function — HSV segmentation, morphological cleanup, contour filtering, HUD-zone exclusion, and IoU-NMS.
4. Run the localizer against the sample frames and interpret the output boxes.
5. Describe honestly what the classical approach does *not* handle well, and what would need to change if those cases became common.

---

## The options

When you need to find regions of interest in an image, the field offers three broad families of approaches. Evaluating them requires being specific about the axes that matter for this project: accuracy (can it find all cards without false positives?), speed (can it run on a mid-grade gaming PC?), training-data need (how many labeled examples are required?), retrain cost after an art swap (the central constraint of this pipeline), and compute budget for training.

### Option A: Classical, geometry-based localization

The observation that motivates this approach: the *layout* of a Demon Bluff board is stable even when the card *art* changes. Cards sit in a radial ring at known relative positions. The UI chrome — card borders, position badges, background, HUD bands — is drawn by the game engine independently of the card art. This means "find the cards" is largely a geometry problem: segment regions with the colour profile of cards and UI chrome, filter by shape and size, suppress duplicates.

Tools: OpenCV's HSV colour segmentation, morphological operators (close/open), contour detection (`findContours`), geometric filtering by area and aspect ratio, and IoU-based non-maximum suppression.

**Accuracy:** On clean board frames, high. On this pipeline's sample footage, the classical approach hit **8/8 and 9/9 cards exact with zero false positives** (spike validated against `dataset/frames/Sample1/` and `dataset/frames/Sample2/`, 2026-06-22). The board/modal state gate is still weak — see "Failure modes" below.

**Speed:** Roughly 10 ms per frame on a CPU, implemented in OpenCV's native C++ layer. No GPU required.

**Training data:** Zero labeled examples. The classifier learns nothing; it applies deterministic rules derived from the UI geometry.

**Retrain cost after an art swap:** Near-zero. The HSV colour thresholds are tuned to the current art palette and would need re-tuning (a 15–30 minute manual pass with an HSV visualiser), but the morphology, aspect-ratio filters, and NMS logic are geometry-derived and art-independent. No re-annotation, no training run, no GPU time. The retune is done once per art set, not per frame.

**Compute budget:** Negligible. Runs on any CPU; no training required.

### Option B: Trained object detector (e.g. YOLO11n / YOLOv8n)

A small trained detector — the nano-class models, around 2–3 million parameters — can learn to detect card regions from labeled examples. On standard benchmarks these models run at roughly 56 ms/frame on a CPU (~18 FPS) and around 1.5 ms on a T4 GPU; for comparison, the peer-reviewed game-UI benchmark cited in `research/RESEARCH.md` found classical OpenCV at ~12.1 ms vs. YOLO at ~19.4 ms for UI-element recognition in a similar context. Nano-class detectors train in a few hours on a free Colab T4 given ~150–400 labeled images.

The critical problem for *this* pipeline is the art-swap constraint. If the detector is trained on card *appearance* (which is the only signal that distinguishes a card from background when the layout is not fixed), it must be **re-annotated and retrained on every art swap**. Re-annotation means drawing bounding boxes on hundreds of frames from the new art set; retraining means a GPU training run. Both are feasible (the nano models train cheaply), but both are *required* on every art change — whereas the classical approach requires only a colour re-tune.

There is a version of the detector approach that avoids art-coupling: train the detector on art-*independent* features (card border geometry, the radial ring pattern) rather than card appearance. But at that point you have essentially built the classical approach in a learned form, with all of the cost and none of the additional accuracy.

**Accuracy:** High for the training distribution; can be lower for frames outside that distribution (unusual lighting, occlusion patterns not in training data).

**Training data:** 150–400 labeled frames minimum; more improves robustness.

**Retrain cost after an art swap:** Re-annotation plus retraining — proportional to art-change frequency. For a pipeline that explicitly states "card art can change," this is a recurring cost the classical approach avoids.

**Compute budget:** Training fits on a single Titan XP or Colab T4. Inference cost is slightly higher than classical at ~56 ms CPU vs. ~10 ms for the classical approach.

### Option C: Foundation model (SAM, Grounding-DINO) — dev-only

Large foundation models (Segment Anything Model: SAM ViT-H at 632 million parameters; Grounding-DINO: documented as "too slow for real-time even on an A100") can localize objects from natural-language prompts or point-click inputs without training data. They are genuinely impressive, and they are the right tool for dataset creation and offline labeling: you can use Grounding-DINO with the prompt "card" to generate bounding-box annotations for the training set you would feed to a nano-detector.

At runtime, however, they violate the compute budget (`research/RESEARCH.md`, entry "Runtime compute budget — 2026-06-21"). A mid-grade gaming PC cannot run SAM ViT-H in real time. These models are **dev/labeling-only** for this project — explicitly off the table as runtime components.

---

## What we chose and why

**Classical layout-based localization.** The decision is recorded as confirmed in `PROJECT-PITCH.md` (2026-06-22 decisions table) and grounded in the localization research entry: `research/RESEARCH.md`, "Card/region localization robust to art swaps under a tight compute budget — 2026-06-21."

The deciding factors:

1. **Art-swap robustness by construction.** The classical approach does not look at card art; it looks at colour families and geometry. An art swap changes the HSV ranges (re-tune, minutes, no GPU) but leaves the morphology and contour logic untouched.
2. **Speed.** ~10 ms CPU vs. ~19 ms for YOLO nano on similar game-UI benchmarks. For a real-time pipeline on a mid-grade PC, this difference matters and the classical approach wins.
3. **Zero labels.** No annotation effort is required before the first working localizer.
4. **Validation.** The approach was spiked against real sample footage before being committed. It is not a guess.

The trained detector (Option B) is the approach we *rejected for production*. It is not wrong — for a system whose layout is unpredictable, or whose art never changes, a nano-detector would be entirely reasonable. But on the axes that matter for this project (retrain cost, compute at inference, zero labels), the classical approach wins cleanly.

---

## Hands-on: running the localizer

The localizer lives in `src/dbcv/localize.py`. The pipeline default is `classical_localize`; the alias `localize` at the bottom of the module points to it. The function signature:

```python
from dbcv.localize import classical_localize
from dbcv.schema import Resolution
import cv2

image = cv2.imread("dataset/frames/Sample1/Sample1_003.jpg")
h, w = image.shape[:2]
boxes = classical_localize(image, Resolution(w=w, h=h))
# Returns a list of (x_rel, y_rel, w_rel, h_rel) tuples, all in [0.0, 1.0]
```

All bounding box coordinates are returned as *relative fractions* of the frame dimensions — not pixel coordinates. This is the resolution-agnostic contract described in CLAUDE.md and enforced by the schema: no pipeline stage may assume a particular resolution, because `resolution` is always read from the media at runtime.

The module also keeps `stub_localize` — the teaching "before" baseline — which returns three hard-coded approximate boxes without looking at the image at all. The diff between `stub_localize` and `classical_localize` is the entire delta introduced by adding vision: it shows what each step of the algorithm adds.

The test suite (`tests/test_localize.py`) runs the localizer against a known board frame and asserts the expected card count. To run:

```
.venv/Scripts/python.exe -m pytest tests/test_localize.py -v   # Windows
.venv/bin/python        -m pytest tests/test_localize.py -v   # macOS/Linux
```

---

## Inside `classical_localize`: the five steps

Reading the code is part of the hands-on experience. Here is what each step does and *why* it was designed that way. All pixel math is derived from `image.shape[:2]` — nothing is hard-coded to a specific resolution.

### Step 1 — HUD-strip exclusion

Before segmenting for card colours, the function zeroes out the parts of the frame it knows are not cards: the objective bar at the top (~9% of height), the name-label strip at the bottom (~14%), the score panel on the left (~13% of width), and the icon cluster on the right (~8%). This is done in pixel space by setting those regions to black before the HSV conversion.

The alternative — trying to distinguish HUD elements from card elements purely by colour — does not work reliably because the HUD contains colourful elements (red timers, orange tokens, bright text) that overlap the same colour ranges as card borders. Zeroing the HUD zones first makes the colour segmentation downstream dramatically cleaner.

### Step 2 — HSV colour segmentation

The function converts the HUD-masked image to HSV and thresholds for the colour families that appear on Demon Bluff card borders: purple (role-colour rings), orange (card backs / villain styling), red (demon accent, which wraps around 0 in OpenCV's 0–179 hue scale and needs two ranges), and a broad "bright and saturated" catch-all for vivid card art not captured by the narrower ranges.

HSV rather than BGR thresholds: hue is the perceptually meaningful dimension for colour. A BGR range conflates brightness and colour and becomes fragile when streaming capture settings shift exposure or gamma slightly. An HSV hue range is stable across moderate brightness variation.

**The honest caveat here:** these HSV ranges were tuned to the *current* art set. They encode knowledge about which colours appear on the current cards. An art swap means re-tuning these specific ranges — still label-free, but not zero work. The 15–30 minute estimate comes from the module docstring, which reflects the development team's own experience.

### Step 3 — Morphological cleanup

A closing pass (dilate then erode) with a large proportional kernel joins the colour-segmented blobs that belong to a single card. Card art typically has interior dark gaps — between the role-colour ring and the art panel, between the art panel and text areas — that would otherwise split one card into several fragments. Closing bridges those gaps. An opening pass (erode then dilate) afterward removes isolated speckle noise that survived the colour threshold.

Kernel sizes are proportional to `min(w, h)`, so the same relative amount of morphological work is done regardless of the input resolution.

### Step 4 — Contour filtering

`findContours` with `RETR_EXTERNAL` extracts the outermost contours in the cleaned mask. Each contour's bounding rectangle is computed and tested against four criteria:

- Too small (< 0.15% of frame area): noise or a partial card edge, discarded.
- Too large (> 9% of frame area): a HUD panel or full-board overlay, discarded.
- Extreme aspect ratio (< 0.38 or > 1.40): bar-shaped UI elements (health bars, progress bands), discarded.
- Overlaps a known HUD zone by > 40% of its area: a colourful HUD element that survived both Step 1 zeroing and the size/aspect filters, discarded.

The specific threshold values (0.0015, 0.09, 0.38, 1.40, 0.40) were tuned on the sample frames. They are documented in the code alongside the geometry reasoning behind each.

### Step 5 — IoU non-maximum suppression

A single physical card often produces 2–4 overlapping contour blobs — one for the art panel, one for the border ring, one for the role-colour accent — that each pass the geometry filters. NMS collapses them to one box per card.

The algorithm is greedy: sort surviving boxes by area descending (larger boxes capture more of the full card), then for each box compute IoU against every already-kept box. If IoU > 0.30, suppress the candidate. The IoU threshold of 0.30 was chosen empirically: 0.50 would allow adjacent sub-region boxes from the same card to survive; 0.10 would suppress boxes for physically adjacent cards that happen to nearly touch.

---

## Failure modes

**Badge blob-detection does not work as a primary anchor.** The numbered position badges (`#1`, `#2`, ...) on each card are art-independent UI chrome and were the first candidate for a reliable geometric anchor. In the spike, direct blob detection on the badge regions over-fired at 30–60 detections per frame because the card's clue/ability text panels produce blobs that are visually indistinguishable from the badge blobs at the segmentation stage. Badges remain useful for *ordering* already-detected boxes (by running targeted OCR on the `#` glyph), but they are not a viable primary localization anchor. This finding is noted in `DEV-LOG.md` (2026-06-22) and in `knowledge-base/lessons/observed-board-layout.md`, which now carries the blob-detection caveat alongside the "geometrically ideal landmarks" note.

**Board-vs-modal state gating (now handled upstream).** During the spike, a naive center-brightness heuristic (bright center = board) misread dark-background modal dialogs as board frames, because *Demon Bluff*'s modals put bright art and text on the same dark starfield as the board — the heuristic fired in exactly the wrong direction, and the localizer would return a few stray boxes (0–2) on a modal. This is now fixed *upstream* of localization by the Stage 0 frame-state gate (`src/dbcv/frame_state.py`), which uses a center-vs-ring brightness *ratio* and runs before the localizer, so localization only ever sees board frames (see Module 02). The residual caveat for localization itself: on a *partial* modal that the gate still (correctly) classifies as "board," expect a reduced ring count — callers should not read "fewer boxes than usual" as "the board is empty."

**HSV ranges are art-tuned.** As noted in Step 2 above: an art swap requires re-tuning the HSV colour ranges. This is a one-time cost per art set, not a per-frame cost, and the estimate is 15–30 minutes. But it is a real cost, and learners should expect to do it. The geometry logic (morphology, contour filters, NMS) is art-independent and would not need adjustment.

**Streamer overlays and occlusion.** The sample footage includes a streamer handle ("Benji") rendered over the board. The current implementation is tolerant of this specific overlay because it occupies a region that the HUD-exclusion strips already handle, but a different overlay in a different position — particularly one in the ring area — could produce false positive detections. The pipeline does not have a general overlay-detection mechanism.

---

## Further reading

Sources are from `research/RESEARCH.md`, "Card/region localization robust to art swaps under a tight compute budget — 2026-06-21" (authority: A/B mixed):

- *Automated game testing using computer vision methods* (peer-reviewed, A) — https://www.researchgate.net/publication/357080752_Automated_game_testing_using_computer_vision_methods — the source for the ~12.1 ms classical vs. ~19.4 ms YOLO comparison on a game-UI localization task.
- *Small Object Detection with YOLO: A Performance Analysis Across Model Versions and Hardware* (2025, arXiv, A/B) — https://arxiv.org/html/2504.09900v1 — per-hardware throughput numbers for nano-class YOLO models.
- Ultralytics YOLO11 official docs (A) — https://docs.ultralytics.com/models/yolo11/ — model parameters, FLOPs, and inference times.
- OpenCV `findContours` documentation (A) — https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html — the retrieval mode (`RETR_EXTERNAL`) and approximation method (`CHAIN_APPROX_SIMPLE`) choices documented in the code.
- *On Efficient Variants of SAM* (arXiv 2410.04960, A) — https://arxiv.org/html/2410.04960v1 — the source for the SAM ViT-H parameter count and the "too slow for real-time" characterisation of Grounding-DINO.
