# Module 09 — Staying alive: keeping a recognizer working when the art changes

**The problem (in the pipeline):** A computer-vision system that works on the day it ships is not the same as a system that still works six months later. Inputs drift — the game patches, the capture setup changes, the streamer switches to a widescreen monitor. For this pipeline, the most concrete and predictable threat is the one the project constraint names explicitly: *Demon Bluff* has an alternate card-art set, and it may be deployed. A recognizer that needs a full relabeling and retraining pass every time the art changes is a liability. This module traces what the pipeline was designed to do in that case, is honest about what it does not currently do, and explains why the re-fit story was a first-class design constraint — not an afterthought — from the first day the pipeline was planned.

**What you'll be able to do:**

1. Describe the three pipeline stages that change when the card art is swapped, and explain concretely what "re-fitting" means for each one — no gradient steps for localization (HSV re-tune) and for the *classical* identification fallback (gallery rebuild), a short **re-fine-tune** for the *adopted* embedding identifier, and a font re-render for the OCR stage. Explain the accuracy ↔ retrain-cost tradeoff that the adopted identifier now makes.
2. Explain why the trained classifier was rejected for production identification, using both the art-swap argument and the training-cost caveat for Pascal hardware.
3. Identify what a production deployment would add that this pipeline does not currently build: drift/health monitoring, confidence tracking over time, and an automated re-fit trigger.
4. Describe the failure modes that arise when these missing production concerns are absent — silent drift, a re-tune that overfits one art set, confidence thresholds that go stale.

---

## The art-swap scenario

The alternate art set for *Demon Bluff* uses different character illustrations on the cards. The game rules, the card names, the role assignments, the number of cards — all of these remain the same. What changes is the visual appearance of each card face.

From the pipeline's perspective, this event touches three things:

1. **Localization** relies on HSV colour ranges tuned to the current art palette (the purple role rings, the orange card backs, the red demon accents). Different art may use different dominant hues.
2. **Identification** relies on a reference gallery built from images of the current card art. The same character in different art is a visually distinct image. The *classical* fallback rebuilds its gallery with zero training; the *adopted* fine-tuned embedding identifier is fit to the current characters, so it is best **re-fine-tuned** on a swap (still cheap, still label-free) — see below.
3. **On-card OCR** (the deferred Stage 3) relies on a custom recognizer trained on rendered crops of the current card font. A different art set may use a different font rendering.

How expensive each re-fit is — that question was answered at design time, before the pipeline was written.

---

## The shipped re-fit story

### Localization: re-tune HSV ranges (~15–30 minutes, no training)

The classical localizer in `src/dbcv/localize.py` finds cards using HSV colour segmentation — it looks for regions whose hue, saturation, and value fall within ranges associated with card borders and card-face colours. These ranges were tuned manually against the current art set on a 15–30 minute pass with an HSV visualizer. They are documented in the source with the explicit caveat that they encode knowledge about the current art palette.

On an art swap, the procedure is: extract a handful of representative frames from the new footage, open the HSV visualizer, adjust the hue/saturation/value bounds until the colour mask cleanly captures card regions, validate against a broader frame set. The morphological operators (kernel sizes proportional to `min(w, h)`), the contour-area filters (expressed as fractions of frame area), the aspect-ratio bounds, and the NMS IoU threshold are all geometry-derived — they do not encode art knowledge and would not change.

Estimated re-tune cost: 15–30 minutes, no GPU, no labeled data, no code change to anything except the threshold constants in the module docstring and colour-range definitions.

**Why is this possible?** Because localization was built to be art-independent by construction. The cards' *positions* and *sizes* are stable properties of the game layout; only their *colours* depend on the art set. The research and design rationale is in `research/RESEARCH.md`, "Card/region localization robust to art swaps under a tight compute budget — 2026-06-21," and is recorded as a confirmed decision in `PROJECT-PITCH.md` (2026-06-22 decisions table).

### Identification: a tradeoff that changed (classical = zero training; adopted model = re-fine-tune)

This is the one re-fit story the project **revised in practice**, and the revision is itself the lesson. The original design treated identification as strictly zero-training on a swap. That is still true of the *classical fallback*, but the *adopted default* changed it — and the change was a deliberate accuracy ↔ retrain-cost trade, not a regression.

**The classical fallback — still zero training, ~seconds.** `src/dbcv/gallery.py` and `src/dbcv/identify.py` store the card art as an in-memory reference gallery: for each character, one or more reference PNGs are loaded at startup, and their HSV histograms and ORB descriptors are precomputed. Matching a board crop is a nearest-neighbor lookup over these precomputed representations — no weights trained, no gradient computed. On an art swap: drop the new PNGs into `knowledge-base/card-art/<role_class>/<identity>/` and re-run `build_gallery()` (~50–100 ms over ~67 images, no GPU). The gallery-builder's own module comment states it directly:

> "**Zero training.** An art swap = re-run `build_gallery()` over the new directory. No gradient steps, no stored model files, no retraining."

That classical baseline reaches ~40–60% on face-up cards in the current art — a measured lower bound, not a production target.

**The adopted default — a fine-tuned embedding-NN, re-fine-tuned on a swap.** Module 05 tells the full story: a *frozen* ImageNet embedding backbone was supposed to keep the zero-training property (re-embed the new references, done) — but it collapsed the stylised characters and over-identified, *losing to the classical baseline*. The fix was to **domain-fine-tune** the backbone (Proxy-Anchor LP-FT), which separated the characters (inter-prototype cosine 0.85→0.41) and was adopted as the default. The consequence for *this* module: the served backbone is now fit to the **current** 43 characters, so a genuinely new art set is best handled by **re-fine-tuning** it — `utils/python/finetune_embedding.py`, ~minutes on the Titan Xp — and then rebuilding the gallery. A quick re-embed against the existing fine-tuned backbone still *works*, but won't separate unfamiliar art as cleanly.

**The honest tradeoff.** We gave up the pure zero-gradient re-fit for identification's *default* in exchange for a large accuracy gain on the current art. The new cost is still cheap by any production standard — minutes, **no human labeling** (the fine-tune augments the clean reference art synthetically), no leaving the dev box — and the truly zero-gradient classical path remains available as a fallback. This is the kind of trade a real system makes consciously: a recognizer that is *too* cheap to adapt may simply not be good enough, and the right answer is often a cheap-but-nonzero re-fit rather than a free-but-weak one.

**Why was the trained classifier rejected?** A small CNN trained to classify card crops into ~44 character classes would likely achieve higher accuracy than the classical baseline. This is explicitly discussed in Module 05 as Option D. It is rejected because it is the only approach in the identification space that *must be relabeled and retrained on every art swap*. Relabeling means drawing bounding boxes on hundreds of new-art frames and assigning character labels; retraining means running a training job. On a free Colab T4 this takes a few hours for a nano-class model. On the Titan XP (Pascal architecture, no Tensor Cores), INT8 quantization and mixed-precision training provide no speedup, and training is done in FP32 — slower still. The compute budget research entry (`research/RESEARCH.md`, "Runtime compute budget — 2026-06-21") documents the Pascal limitation explicitly: "keep dev work FP32" and "slower than Ampere, no mixed precision."

The structural argument: at every art swap, a trained-classifier approach demands annotation cost plus training cost. That cost scales with art-swap frequency. If the art changes once, it is a day's work. If it changes repeatedly — or if a learner wants to adapt the pipeline to a different game altogether — the classifier becomes an anchor that prevents cheap adaptation. The gallery-based approaches do not.

### OCR re-fit: re-render the font (retrain in minutes, no new labels)

The OCR stage (Module 06, not yet shipped) is planned as a tiny custom recognizer over the closed glyph set of role names and ability counts. The design recorded in `PROJECT-PITCH.md` is: train on synthetically rendered crops of the current card font; re-fit by re-rendering crops of the new font and running the training job again.

This approach satisfies the art-swap constraint because the font is known (the course ships with reference renders), the glyph set is closed and small (role names + 1–2-digit counts), and re-rendering the training set is automated — no human labeling is required. The training job for a tiny CNN/CRNN on a small closed glyph set is on the order of minutes on a Colab T4, not hours. This is documented in `research/RESEARCH.md`, "Lightweight OCR for short on-card text and numbers in game UI — 2026-06-21" (fit-for-constraints section).

The cross-check between the visual identification result and the OCR-read name label is a validation layer: if identification says "Scout" and the name label reads "Baa," one of the two stages has made an error, and the discrepancy can be flagged even before a monitoring system is in place.

---

## What is NOT built: the honest production gap

The re-fit story above describes *how to re-fit when you know the art has changed*. That leaves a gap that a real production deployment would close.

### Drift and health monitoring

The pipeline does not track confidence distributions over time. In a healthy pipeline, the median identification confidence on face-up cards should be roughly stable. If the art changes — or if the capture setup shifts (resolution, compression, exposure) — the confidence distribution would degrade: median confidence falls, "unknown" returns increase. A monitoring layer that records these statistics per video session and alerts when they drift outside a reference window would detect this automatically.

This is not built. The pipeline returns per-card confidences in the snapshot JSON, and `utils/python/run_pipeline.py` prints them to stdout — the raw signal is available — but no aggregation, no alert threshold, and no drift history exist.

### Confidence thresholds going stale

The identification confidence threshold (currently 0.40 — below which a card is returned as "unknown") was set by inspection on the current art set. If the art changes and the gallery is rebuilt, the new art's reference images may produce systematically different histogram correlation scores. The 0.40 threshold was not derived from a calibration procedure; it was tuned on one art set. A different art set could make 0.40 too strict (causing too many "unknown" returns on recognizable cards) or too lenient (causing low-confidence wrong matches to pass through).

A production system would re-calibrate this threshold against a labeled validation set from the new art before deploying. This pipeline does not have an automated calibration procedure.

### Automated re-fit trigger

The system has no mechanism to detect that the art has changed and initiate a re-fit. The current workflow is: a human observes degraded performance, diagnoses the cause as an art swap, manually collects new reference PNGs, runs `build_gallery()`, and optionally retunes the HSV localizer. Automating this — detecting the drift, classifying its cause (art change vs. capture degradation vs. new game layout), and triggering the appropriate re-fit procedure — would require at minimum a reference distribution for the current art set and a statistical test for deviation.

These are standard production concerns in deployed CV systems. They are absent here because the pipeline is a teaching artifact rather than a deployed service, and because building them correctly requires design decisions (what distribution to track, what significance threshold to use, what the re-fit trigger actually does) that are out of scope for the current development stage.

### Silent drift

The most dangerous failure mode in a deployed recognizer is the one that produces no visible error: confidence scores that are slightly wrong, identifications that are slightly off, and a game-state snapshot that looks plausible but is subtly incorrect. A human watching the overlay might not notice that one character is consistently misidentified, or that face-down cards are occasionally being labeled with a low-but-passing confidence rather than "unknown."

The absence of monitoring means silent drift goes undetected. The only current safeguard is the confidence threshold, which catches very-low-confidence matches but does nothing about moderately-wrong-but-confident ones. The OCR cross-check (when the OCR stage is built) will add a second validation signal — but even two signals can both drift in the same direction if the art change affects both the visual appearance and the text rendering.

---

## Why "cheap to re-fit" was first-class, not an afterthought

The constraint appears explicitly in `CLAUDE.md`: "Card art can change to a limited alternate set. Anything that recognizes a card must be cheap to retrain — assume the art will be swapped and the recognizer re-fit." It was recorded in the first decisions table entry for identification in `PROJECT-PITCH.md` (2026-06-21): "Art swap re-fits with new reference images and zero training; classifier would need relabel+retrain."

This means the re-fit story was not added after the pipeline was built to justify a design choice that happened to work out. It was the design requirement that ruled out the trained classifier before the first identification prototype was written. The gallery-based approach exists *because* zero-training re-fit was required, not as a happy side effect of choosing it for other reasons.

The same logic applies to localization: classical layout-based detection was chosen partly for its art-swap property (re-tune HSV ranges, no annotation), not only for its speed advantage over a nano-detector.

This is the design posture the course is trying to teach: anticipate the failure modes and maintenance costs of a system *before* building it, and let those anticipated costs shape the technique choices from the start.

---

## Failure modes

**A re-tune that overfits one art set.** When re-tuning the HSV localizer for new art, it is possible to narrow the colour ranges so tightly that they work perfectly on the reference frames used for tuning but fail on frames from later in the session (different lighting, different frame-state transitions, cards in shadow or under a modal's partial occlusion). The symptom is high accuracy on the re-tune sample and degraded accuracy on held-out frames. Mitigation: tune on a representative sample that includes non-ideal board states, and validate against a held-out set before declaring the re-tune complete.

**Silent drift.** As described above: confidence scores degrade slowly, identifications become slightly wrong, no error is raised. The only current protection is the threshold and the (future) OCR cross-check. The correct fix is monitoring; the short-term mitigation is periodic spot-checking of the overlay output.

**Confidence thresholds calibrated for the wrong art.** After an art swap and gallery rebuild, the 0.40 threshold may be wrong for the new art's histogram distributions. A character whose new-art illustration has a histogram that happens to correlate weakly with the gallery (perhaps the new art is more muted, or uses colour palettes closer to other characters) may systematically return low-but-nonzero confidence, either passing the threshold with wrong matches or failing it with "unknown." Calibrating the threshold after every gallery rebuild is the correct procedure; currently it is a manual step with no automated guidance.

**Gallery rebuild without validation.** `build_gallery()` will succeed as long as the reference PNGs are present and readable. It will not raise an error if the gallery covers only 30 of 44 characters, or if two characters share a directory by mistake. A production build step would include a validation pass: assert that all expected identities are present, that no two entries share an identity unexpectedly, and that the gallery's held-out accuracy on a labelled validation set meets a defined threshold before the gallery is promoted to the running service.

---

## Further reading

- `research/RESEARCH.md`, "Card identification that is cheap to retrain when art changes — 2026-06-21" — the original identification research entry; sources 1 (Snell et al., prototypical networks), 2 (Wu et al., NN retrieval vs. classifier), and 3 (Howard et al., MobileNetV3) ground the embedding-NN approach and the zero-training-on-swap *aspiration*.
- `research/RESEARCH.md`, "Why a frozen-ImageNet embedding + NN gallery collapses on stylized cards, and how to fix it — 2026-06-22" — the follow-up entry that records *why* the zero-training frozen approach lost to classical (domain shift) and prescribes the adopted fine-tune (Proxy-Anchor LP-FT). This is the basis for the revised re-fit story above: the adopted identifier re-fine-tunes on a swap rather than re-embedding for free.
- `research/RESEARCH.md`, "Runtime compute budget: what fits a mid-grade gaming PC and trains on a Titan XP / Colab T4 — 2026-06-21" — the compute-budget entry; the Pascal/Titan XP FP32 caveat and the "~2.5 h Colab T4 fine-tune" figure are documented there and bear on the cost of the rejected trained-classifier path.
- `research/RESEARCH.md`, "Lightweight OCR for short on-card text and numbers in game UI — 2026-06-21" — the OCR research entry; the "fit for our constraints" section describes the synthetic-render re-training path for the closed-vocabulary recognizer.
- `src/dbcv/gallery.py` — the gallery builder; the module docstring states the zero-training re-fit guarantee explicitly.
- `src/dbcv/localize.py` — the classical localizer; the "art-swap caveat" in the module docstring describes what changes (HSV ranges, 15–30 min re-tune) and what does not (morphology, contour filters, NMS).
