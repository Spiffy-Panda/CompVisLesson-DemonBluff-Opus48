# Module 05 — Naming the cards: a classical baseline that motivates the embedding upgrade

**The problem (in the pipeline):** The localizer hands us a cropped image of a card. We need to answer: *which of the ~44 Demon Bluff townees is this?* And crucially: if the card art set is swapped for the alternate set, we need to re-fit the answer cheaply — without re-annotating hundreds of frames or running a multi-hour training job.

**What you'll be able to do:**

1. Describe the four families of card identification methods on the axes that matter for this pipeline: accuracy, speed, data need, retrain cost after an art swap, and compute budget.
2. Explain why a trained classifier — the simplest and most accurate-sounding option — is the one approach explicitly rejected for production.
3. Read `src/dbcv/gallery.py` and `src/dbcv/identify.py` and explain each design decision: why 2-D HSV histograms, why value is excluded, why ORB is a tiebreaker and not a primary, what the confidence threshold guards against.
4. Interpret the pipeline's honest measured accuracy (~40–60% on face-up cards, 100% correct "unknown" on face-down) as a lower bound that motivates a specific deferred upgrade.
5. Run the identifier against sample frames using `utils/python/run_pipeline.py` and observe where it succeeds and where it fails.

---

## The options

The research entry that grounds this module is `research/RESEARCH.md`, "Card identification that is cheap to retrain when art changes — 2026-06-21."

The four families that came out of the literature:

### Option A: Template matching / NCC

Compare the card crop to a reference image of each townee using normalised cross-correlation (NCC). If the reference matches the crop, the top NCC score identifies the card. Zero training; an art swap means swapping the reference images and the NCC scores are immediately valid.

**Accuracy:** High on clean, axis-aligned, full-card crops under consistent illumination. Degrades substantially with scale variation, partial occlusion, crop offset, or state-tinting applied by the game (a "poisoned" or "cursed" card has colour overlaid on its art). The cited survey (`research/RESEARCH.md` entry 3, source 4) documents NCC's brittleness to these transforms. Board crops are partial, bordered, and tinted — NCC is unreliable as a primary signal. This was evaluated and rejected as the primary method (see `src/dbcv/gallery.py` module comment and `scrap_scripts/python/07_identify_probe.py`).

**Retrain cost after art swap:** near-zero — replace reference images, no training.

### Option B: Embedding similarity (small frozen backbone + nearest-neighbour gallery)

Run each card crop through a small, frozen image-embedding backbone (MobileNetV3-Small, ResNet18, or a frozen CLIP encoder), then find the nearest neighbour in a gallery of pre-embedded reference images using cosine distance. The backbone produces a feature vector that is trained to be invariant to lighting, partial occlusion, and moderate geometric variation. On an art swap: re-embed the new reference images with the same frozen backbone. No weight updates, no gradient descent. The cited NeurIPS and ECCV papers (entry 3, sources 1 and 2) demonstrate that nearest-neighbour retrieval over learned embeddings can match or exceed a trained softmax classifier on identification tasks with small class counts.

**Retrain cost after art swap:** re-embed ~44 reference images (seconds, no GPU required for the reference set). No re-annotation, no training run. This is the property that makes it the research-recommended approach.

**Compute:** one forward pass through a MobileNetV3-Small or similar model per crop (~13 ms CPU; ~8 MB ONNX weights) plus a 44-way distance comparison (negligible). This requires `onnxruntime` as a dependency — the first genuinely heavier dep in the pipeline.

**Status: deferred.** This is the approach the research recommends and the pipeline will eventually use. It is deferred because it requires `onnxruntime`, a model export/download step, and a one-time backbone selection decision. Per the project's conservative path, it is held for a dedicated later round. The classical baseline below exists precisely to measure how far we can get without it.

### Option C: Prototypical networks / few-shot learning

Prototypical networks (Snell et al., NeurIPS 2017, `research/RESEARCH.md` entry 3, source 1) extend the embedding approach: each class is represented as the *mean* of several embedded support images. New classes are added at inference by providing a few examples — no parameter updates. This is the formal version of Option B for the case where you have multiple reference images per card (which we do: 24 townees have skin variants in the gallery).

In practice for this pipeline, collapsing each townee's gallery entries to their mean embedding gives a prototypical representation at essentially no extra cost over Option B. This is how Option B would be implemented once the backbone is in place.

### Option D: Trained classifier (small CNN)

Train a small CNN (or fine-tune a MobileNet head) to classify crops into 44 card classes. This is the most familiar framing: it is a standard image classification problem with a fixed, known label set.

**Why we reject this for production:** a trained classifier is the only approach in this list that must be **re-annotated and retrained on every art swap**. Re-annotation means drawing bounding boxes and labelling hundreds of frames from the new art set. Retraining means running a training job (hours on a Colab T4, or longer on the Titan XP in FP32 — Pascal's INT8 quantisation path is unusable, per `research/RESEARCH.md`, "Runtime compute budget — 2026-06-21"). Both steps are feasible, but both are *required* every time the art changes. The project constraint is explicit: recognition must be cheap to retrain. Option D fails this test structurally, not incidentally.

---

## What we chose and why

**The shipped classical baseline (Options A/B hybrid, grounded in entry 3 of `research/RESEARCH.md`):** 2-D HSV colour histogram correlation as the primary signal, with ORB feature matching as a tiebreaker. This is a classical approximation to the embedding-similarity approach: histograms capture the colour distribution of the character art (which is the most stable visual signal across crop offsets and moderate state-tinting) without requiring a neural backbone.

The deciding factor is the conservative path: the classical baseline requires only `opencv-python-headless` and `numpy`, both already in `requirements.txt`. It gives us a measurable lower bound on identification accuracy, quantifies the gap that motivates the embedding-NN upgrade, and is itself zero-training (art swap = re-run `build_gallery()` over the new reference images, no gradient steps).

The trained classifier (Option D) is the explicitly rejected option. The embedding-NN (Option B) is the explicitly deferred upgrade.

---

## The shipped implementation: `src/dbcv/gallery.py` and `src/dbcv/identify.py`

Reading these files alongside this section is part of the hands-on experience.

### Building the gallery (`src/dbcv/gallery.py`)

`build_gallery()` walks `knowledge-base/card-art/<role_class>/<identity>/<file>.png` and builds an in-memory gallery. For this pipeline's current art set: **43 distinct townees, 67 reference images** (24 townees have skin variants, all loaded). The directory name is the label: `<role_class>/<identity>` gives both the role family (villager / minion / outcast / demon) and the townee name. One alias is applied: `Twin_Minion` maps to `Minion` (a documented game fact — the two are functionally identical for CV purposes).

For each reference PNG the function precomputes three representations:
- A 64×64 thumbnail (for any future pixel-level debug display)
- A 2-D HSV colour histogram (the primary matcher — computed once, reused for every crop)
- ORB keypoints and descriptors (the tiebreaker — computed once, reused)

**No file is written.** The gallery lives in RAM. Re-running `build_gallery()` on a new art directory rebuilds it with zero training — this is the art-swap property the design is built around.

The gallery is built once per process, in the API's `lifespan` context manager, stored on `app.state`, and reused for every request. `utils/python/run_pipeline.py` mirrors this pattern: gallery built once at startup, passed to every frame call.

### Why 2-D HSV histograms, and why value is excluded

The histogram is computed over hue and saturation only (32 hue bins × 16 saturation bins = 512 values). Value (brightness) is excluded deliberately.

The reason: the game applies state-tinting to cards. A "poisoned" card has a green tint overlaid on its art; a "cursed" card has a purple one; night-phase cards may be darkened. These tints shift the value (brightness) dimension substantially while leaving the hue distribution recognisable. A 2-D Hue×Saturation histogram survives moderate tinting; a 3-D H×S×V histogram would misclassify a tinted card as a different character.

Additionally, near-black and near-white pixels are masked out before computing the histogram (saturation > 25, value between 30 and 250). These are background pixels (the card-art PNGs have white or transparent backgrounds that are composited onto white when loaded); including them would dilute the character's colour signal.

The histogram is L1-normalised so that differently sized crops and references produce comparable distributions.

### The ORB tiebreaker

When the top two histogram scores are within 0.05 of each other, the code invokes ORB feature matching to break the tie. ORB (Oriented FAST and Rotated BRIEF, Rublee et al. 2011) detects keypoints and computes binary descriptors. Brute-force Hamming matching with Lowe's ratio test (0.75) produces a count of "good" matches. If the runner-up gets more good ORB matches than the top histogram candidate (and the count meets a minimum threshold of 4), the runner-up wins.

ORB is not the primary signal because: board crops are partial and tinted, reference images are clean full-card illustrations, and the spatial misalignment between the two makes feature-point matching unreliable as a *primary* method. But when two characters are similarly coloured (both predominantly brown-beige, for example), histogram correlation cannot separate them — ORB adds a structural signal that can.

### Confidence and the "unknown" return

Confidence is defined as the top histogram correlation score, clamped to [0, 1]. Below a threshold of 0.40, the function returns `("unknown", "unknown", low_score)` rather than the best match.

**Face-down cards are the key test case.** A face-down card shows the uniform brown/orange card-back pattern. Its HSV histogram is dominated by the card-back's colour distribution, which matches no character's art histogram. Face-down cards consistently score below the threshold and return `"unknown"` — which is the correct answer. A system that confidently identified a face-down card as a specific townee would be wrong; "unknown" is honest.

This is also the right behaviour for any crop that is too occluded, too small, or too blurred to identify — the threshold acts as a minimum evidence gate.

---

## Honest results: what the classical baseline actually achieves

The measured performance on the current pipeline (from `DEV-LOG.md`, 2026-06-22, Stage 2 identification entry):

- **~40–60% on face-up cards** — a meaningful fraction of face-up townees are correctly identified, with real identities surfacing: `Wretch@0.80`, `Baa@0.69`, `Confessor`, `Hunter`, `Druid`, `Scout`, `Fortune_Teller`, `Doppelganger` among the successes.
- **100% correct "unknown" on face-down cards** — the uniform card back matches no character, which is the right answer and the gate behaves exactly as designed.
- **One false match observed:** `Scout` was predicted twice in a single frame. In Demon Bluff, the same townee cannot appear twice in the same round; this is an impossible in-game configuration and a real false match.

**Verdict from the DEV-LOG:** "classical histograms are a useful, honest lower bound but insufficient for production." The ~40–60% face-up accuracy is the gap that motivates the deferred embedding-NN upgrade. An embedding backbone would learn to bridge the gap between clean reference art and partial, tinted, bordered board crops — the classical approach approximates this with a colour distribution, which is invariant but coarse.

The embedding-NN path (Option B above) is the explicitly deferred next step: a small frozen backbone such as MobileNetV3-Small, exported to ONNX, used to embed both reference images and board crops, with nearest-neighbour lookup over the gallery. This adds `onnxruntime` as the first genuinely heavier dependency and requires a one-time model export step. The classic approach runs today on the already-installed `opencv-python-headless`.

---

## A debugging lesson: the degenerate-input bug

During development, a subtle bug was found that is worth teaching explicitly: **`cv2.compareHist(zeros, anything)` returns 1.0**.

OpenCV's `HISTCMP_CORREL` computes the Pearson correlation between two histograms. When both histograms sum to zero (which happens when the crop is an all-black rectangle, a pitch-dark frame, or a fully masked-out region with no colour signal), the Pearson formula divides zero by zero. OpenCV returns 1.0 as the result.

A degenerate crop (all-black card frame, fully dark region, heavily masked background) was therefore "perfectly matching" every character in the gallery, returning a confidence of 1.0 and claiming a definitive match. The correct answer for a structureless crop is "unknown".

The fix in `src/dbcv/identify.py` is a zero-sum guard before the gallery comparison loop:

```python
if crop_hist.sum() < 1e-6:
    return ("unknown", "unknown", 0.0)
```

**The lesson:** always test degenerate inputs. A function that receives an image should be tested with an all-black image, an all-white image, and a single-pixel image before it is considered correct. The statistical formula that looks correct in the non-degenerate case may return a confident but wrong answer when the input has no signal. This applies to any histogram or correlation computation, not just OpenCV's specific implementation.

---

## Hands-on: running the identifier

`utils/python/run_pipeline.py` runs the full cascade including identification. The gallery is built once at startup and the identity + confidence for each card appears in the output.

```
# Windows — run on all Sample1 frames, build gallery, print identities
.venv\Scripts\python.exe utils/python/run_pipeline.py

# With overlay PNGs showing identity labels and role-class colours
.venv\Scripts\python.exe utils/python/run_pipeline.py --overlay

# macOS / Linux
.venv/bin/python utils/python/run_pipeline.py --overlay
```

Expected output (board frame):
```
Sample1_003_t00460s                 state=board    cards=8  Wretch@0.65  Baa@0.70  unknown@0.22 ...
```

Cards where identification succeeds show a townee name and a confidence above 0.40. Cards where it fails (face-down, occluded, or misidentified) show `unknown` and a low confidence. The role-class colour in the overlay PNG reflects the predicted role (green = villager, red = demon, grey = unknown).

To observe the gallery's size and content before running frames:

```python
# From a Python session with the venv activated:
from dbcv.gallery import build_gallery
gallery = build_gallery()
print(gallery.n_townees, "townees,", gallery.n_references, "references")
print(gallery.role_classes)
```

This should print `43 townees, 67 references` and `['demon', 'minion', 'outcast', 'villager']`.

---

## Failure modes

**The reference-vs-crop gap is fundamental.** Reference images are clean, full-character illustrations. Board crops are partial (the card frame and badge clip the art), tinted (state overlays shift colours), and may include the name label and ability text. The 2-D HSV histogram approximates around this gap; an embedding backbone would learn through it. The ~40–60% face-up accuracy directly reflects how much gap remains after the approximation.

**Tonally similar characters confuse the histogram.** Characters whose colour palettes are dominated by the same hue families (two brown-robed figures, two blue-armoured characters) will have similar histograms. The ORB tiebreaker helps when the histogram delta is small, but does not fully resolve it when both characters are visually similar enough that their keypoint distributions also overlap.

**The art-crop heuristic is approximate.** `identify.py` uses a fixed art-band extraction (rows 12–62% of crop height, columns 8–92% of width) to focus on the character illustration and avoid the name label and ability text. This band was set by inspection on the current sample frames. A significantly different card layout — a different screen resolution, a different camera crop, a heavily occluded card — would shift the band to the wrong region. The embedding approach would learn the art-region implicitly.

**A false match cannot be detected from the score alone.** If the pipeline returns `Scout@0.65` it cannot know whether this is a correct match at 65% confidence or a plausible-looking wrong match. The threshold of 0.40 filters out low-confidence guesses, but confident wrong answers pass through. Cross-checking against the on-card name label (a future OCR stage) is the designed validation layer.

---

## Further reading

From `research/RESEARCH.md`, "Card identification that is cheap to retrain when art changes — 2026-06-21":

- Snell et al., *Prototypical Networks for Few-shot Learning* (NeurIPS 2017) — https://arxiv.org/abs/1703.05175 — the canonical few-shot learning paper; shows that class prototypes (mean embeddings) are an effective representation for nearest-neighbour identification, and that adding new classes at inference requires no parameter updates.
- Wu, Efros, Yu, *Improving Generalization via Scalable Neighborhood Component Analysis* (ECCV 2018) — https://arxiv.org/pdf/1808.04699 — demonstrates that nearest-neighbour retrieval over learned embeddings can match or exceed a trained softmax classifier on image recognition tasks, and is more interpretable.
- Howard et al., *Searching for MobileNetV3* (ICCV 2019) — https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf — the backbone family recommended for deployment on a mid-grade CPU or mobile device; the small variant fits in under 8 MB and runs at approximately 13 ms per crop on CPU.
- *Limitations of Template Matching* — https://apxml.com/courses/introduction-to-computer-vision/chapter-5-introduction-object-recognition/limitations-template-matching — a concise practitioner account of the conditions under which NCC fails; the cases it documents (scale change, partial occlusion, illumination shift, deformation) are exactly the conditions present in board crops.
