# Module 05 — Naming the cards: a classical baseline, a deep model that *loses*, and the fine-tune that fixes it

**The problem (in the pipeline):** The localizer hands us a cropped image of a card. We need to answer: *which of the ~44 Demon Bluff townees is this?* And crucially: if the card art set is swapped for the alternate set, we need to re-fit the answer cheaply — without re-annotating hundreds of frames or running a multi-hour training job.

This module follows a real arc that played out while building the pipeline, and it is the most teachable result in the project: a conservative **classical** matcher first **beat** a naively-applied **deep embedding** model; we then diagnosed *why* (correctly — it is **domain shift**, not "neural collapse"), applied the **principled fix** (domain fine-tuning with a metric-learning loss under tiny-data discipline), changed the **abstention rule** to match the new model, and **adopted** the fine-tuned embedding-NN as the default identifier.

**What you'll be able to do:**

1. Describe the four families of card identification methods on the axes that matter for this pipeline: accuracy, speed, data need, retrain cost after an art swap, and compute budget.
2. Explain why a trained-from-data classifier is the one family explicitly rejected for production, and why an *off-the-shelf frozen* embedding backbone is not an automatic win either.
3. Read `src/dbcv/gallery.py` and `src/dbcv/identify.py` and explain each design decision: why 2-D HSV histograms, why value is excluded, why ORB is a tiebreaker and not a primary, what the confidence thresholds guard against (both the classical correlation gate and the embedding *margin* gate).
4. Diagnose the frozen-embedding failure correctly as **domain shift of ImageNet features to a stylised, fine-grained domain** (and explain why "neural collapse" is the *wrong* name), citing the literature.
5. Explain the adopted fix — **Proxy-Anchor loss + LP-FT + strong synthetic augmentation** — and why each piece is chosen for a tiny, imbalanced dataset; explain why abstention switched from an absolute cosine to a **top1−top2 margin**.
6. Run the identifier against sample frames using `utils/python/run_pipeline.py` and observe where it succeeds and where it abstains.

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

**Status: built, and now the adopted default — but with an important twist.** This is the approach the research recommended, and the pipeline now uses it. The twist is the heart of this module: applying it *off the shelf* (a **frozen** ImageNet backbone) did **not** beat the classical baseline — it over-identified. The fix was to **fine-tune** the backbone to our domain. The classical baseline below exists precisely to measure how far we get without any deep model, and it set the bar that the *frozen* deep model then failed to clear. See "When the deep model lost" and "The fix" below.

### Option C: Prototypical networks / few-shot learning

Prototypical networks (Snell et al., NeurIPS 2017, `research/RESEARCH.md` entry 3, source 1) extend the embedding approach: each class is represented as the *mean* of several embedded support images. New classes are added at inference by providing a few examples — no parameter updates. This is the formal version of Option B for the case where you have multiple reference images per card (which we do: 24 townees have skin variants in the gallery).

In practice for this pipeline, collapsing each townee's gallery entries to their mean embedding gives a prototypical representation at essentially no extra cost over Option B. This is how Option B would be implemented once the backbone is in place.

### Option D: Trained classifier (small CNN)

Train a small CNN (or fine-tune a MobileNet head) to classify crops into 44 card classes. This is the most familiar framing: it is a standard image classification problem with a fixed, known label set.

**Why we reject this for production:** a trained classifier is the only approach in this list that must be **re-annotated and retrained on every art swap**. Re-annotation means drawing bounding boxes and labelling hundreds of frames from the new art set. Retraining means running a training job (hours on a Colab T4, or longer on the Titan XP in FP32 — Pascal's INT8 quantisation path is unusable, per `research/RESEARCH.md`, "Runtime compute budget — 2026-06-21"). Both steps are feasible, but both are *required* every time the art changes. The project constraint is explicit: recognition must be cheap to retrain. Option D fails this test structurally, not incidentally.

---

## What we chose and why

**The shipped classical baseline (Options A/B hybrid, grounded in entry 3 of `research/RESEARCH.md`):** 2-D HSV colour histogram correlation as the primary signal, with ORB feature matching as a tiebreaker. This is a classical approximation to the embedding-similarity approach: histograms capture the colour distribution of the character art (which is the most stable visual signal across crop offsets and moderate state-tinting) without requiring a neural backbone.

The deciding factor for shipping the classical baseline *first* was the conservative path: it requires only `opencv-python-headless` and `numpy`, both already in `requirements.txt`. It gives us a measurable lower bound on identification accuracy, quantifies the gap that motivates the embedding-NN upgrade, and is itself zero-training (art swap = re-run `build_gallery()` over the new reference images, no gradient steps).

The trained-from-scratch classifier (Option D) is the explicitly rejected option. The embedding-NN (Option B) is what we then built — and the rest of this module is the honest story of what happened when we did: the *frozen* version lost to the classical baseline, we diagnosed why, fine-tuned it, and adopted it. The classical matcher is retained as a selectable fallback.

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

**Verdict from the DEV-LOG:** "classical histograms are a useful, honest lower bound but insufficient for production." The ~40–60% face-up accuracy is the gap that motivates the embedding-NN upgrade. An embedding backbone would learn to bridge the gap between clean reference art and partial, tinted, bordered board crops — the classical approach approximates this with a colour distribution, which is invariant but coarse. So we built it. What happened next is the lesson.

---

## When the deep model lost: a frozen backbone that *under*-performs

We implemented Option B exactly as the research recommended: a small **frozen** MobileNetV3-Small (ImageNet-pretrained, classifier head stripped), exported to ONNX, used to embed both the reference art and the board crops, with cosine nearest-neighbour over per-identity prototype embeddings. This added `onnxruntime` as the first genuinely heavier dependency.

And it **lost to the classical baseline.** Not by crashing — by being *confidently wrong*. The failure has a precise, measurable signature:

- **The 43 character prototypes collapsed into one tight cluster.** The cosine similarity *between different characters'* prototype embeddings ran **0.65–0.94** (mean ≈ 0.85). In a healthy embedding space, two different classes should be nearly orthogonal (cosine near 0); here every character looked almost identical to every other.
- **Consequence: it over-identified.** Because every prototype was close to every crop, *some* prototype always scored highly, so the model confidently named nearly every slot — including face-down cards it should have rejected. The conservative classical matcher beat it simply by honestly saying "unknown" more often.

This is the moment the module is built around: **the fancy model is not automatically better.** A naively-applied deep model can be strictly worse than a simple one, and worse in a way that looks like success (high confidence) unless you measure the right thing.

### Diagnosing it correctly: domain shift, *not* "neural collapse"

It is tempting — and wrong — to call this "neural collapse." Naming the failure correctly is the whole point, because the name determines the fix.

- **What it actually is: domain shift (domain gap) of frozen ImageNet features to a stylised, fine-grained domain.** ImageNet features are trained on natural photographs. Our inputs are stylised cartoon illustrations of 43 visually-similar fantasy characters. Two literature results pin this down: rendering ImageNet images as cartoons/drawings makes pretrained accuracy *drop significantly* (ImageNet-Cartoon/-Drawing; the PACS Photo→Art/Cartoon/Sketch benchmark shows the same), and Kornblith et al. show the ImageNet-accuracy↔transfer correlation **breaks down on fine-grained tasks**, where the discriminative cues are subclass-specific rather than generic. Off-the-shelf features simply do not separate our characters.
- **Why our prototype/NN structure was *not* the problem.** Chen et al.'s keystone result ("A Closer Look at Few-Shot Classification"): a cosine classifier on a frozen backbone is competitive with prototypical/meta-learning methods *only when the base and novel classes share a domain*. Under a domain shift, the advantage of any fancy distance metric or few-shot head disappears, and **the limiting factor becomes the backbone's features, not the classifier head.** Our gallery-of-prototypes *is* that inference structure — it was never wrong; it was being fed un-adapted features.
- **Why "neural collapse" is the wrong label.** Neural collapse is a *training-dynamics* phenomenon where a network's *own trained* class means converge to a symmetric (equiangular tight frame) arrangement on its *in-distribution training* data. That is a different thing entirely from generic frozen features failing to separate an unseen, out-of-distribution class set. Calling our failure "neural collapse" would point you at the classifier head; calling it "domain shift" points you at the features — which is where the fix lives.

The full diagnosis with citations is `research/RESEARCH.md`, "Why a frozen-ImageNet embedding + NN gallery collapses on stylized cards, and how to fix it — 2026-06-22."

---

## The fix: domain fine-tuning under tiny-data discipline

The diagnosis says: **adapt the features.** But we have very little data — 1–67 reference images per character across 43 classes, and (so far) *no labelled real board crops at all*. So the fix has to adapt the backbone without overfitting or destroying what pretraining gave us. Three decisions, each grounded in the literature:

### 1. The loss: Proxy-Anchor (a metric-learning loss with an inter-class margin)

The measured failure was *insufficient margin between classes*, so we use a loss that **directly enforces an angular/cosine margin**. We chose **Proxy-Anchor loss** (Kim et al., CVPR 2020), implemented inline (~30 lines — no new dependency, and it doubles as a worked example). Why Proxy-Anchor specifically:

- It compares each class to a single **learnable proxy**, so it works with **few examples per class** — the natural training-time analogue of our inference-time prototype gallery.
- It **converges fast** and is **robust to noisy/outlier samples**, and it avoids the O(n²) pair/triplet mining that plagues contrastive/triplet losses.
- Honest caveat: angular-margin *softmax* losses (ArcFace/CosFace) are reported to degrade when classes have ~1 example or near-collapsed intra-class statistics — i.e. they are *not* an automatic win at our extreme low-shot tail. Proxy-based losses degrade more gracefully there, which is why we reached for Proxy-Anchor first. (SupCon / supervised-contrastive is the documented Round-2 lever if needed.)

We deliberately use **no projection head** — we fine-tune the backbone so the *actual 576-d pooled vector the runtime serves* is the one that separates the classes. That keeps the ONNX contract, the gallery, `identify.py`, and the tests all untouched; only the `.onnx` file swaps.

### 2. The unfreeze schedule: LP-FT (linear-probe, then fine-tune)

How *much* of the backbone to unfreeze is itself a tiny-data decision. Kumar et al. (ICLR 2022, oral) show that **full fine-tuning can distort good pretrained features and underperform out-of-distribution** on small data, while head-only training is stable but can't close a domain gap. Their **LP-FT** recipe — warm up the head/proxies first on frozen features, *then* unfreeze and fine-tune at a small learning rate — beats both, because warming the head first means the fine-tuning step perturbs the trunk less. We follow it directly:

- **Phase A:** 250 steps of Proxy-Anchor warm-up with the **backbone frozen** (proxies warm-started from the frozen prototypes — a clean linear-probe initialisation).
- **Phase B:** 600 steps with the **top-4 feature blocks unfrozen** (~736k trainable params) at a small LR — a partial unfreeze, the documented middle path between head-only and full-FT.

### 3. The data: strong augmentation synthesised from the clean reference art

With only a handful of clean references and zero real-crop labels, **augmentation is the highest-payoff regulariser** (it is the canonical small-data move). We synthesise training crops from the reference art with `torchvision.transforms.v2`: `RandomResizedCrop` + perspective warp + ≤6° rotation + **RandAugment** + Gaussian blur + `RandomErasing`. These mimic the real capture variation (the card frame clips the art, the board tints and borders it). **No horizontal flip** — card art is not left-right symmetric, so a flip would teach a false invariance.

All of this trains in **minutes on the Titan Xp** (FP32) and changes nothing about the ONNX-on-CPU serving path.

### What the fine-tune actually did (leak-proof numbers)

The honest, leak-proof verdict is the inter-prototype cosine measured on the **clean references** (zero augmentation), before vs after, plus a synthetic held-out retrieval sanity check:

| Metric | Frozen baseline | Fine-tuned (adopted) |
|--------|-----------------|----------------------|
| Inter-prototype cosine, mean | 0.850 | **0.409** |
| Inter-prototype cosine, max | 0.939 | **0.536** |
| Synthetic top-1 retrieval | 79.9% | **100%** |
| top1−top2 margin (synthetic) | 0.031 | **0.405** |
| torch↔ONNX parity (max abs diff) | — | **5.5e-6** |

The cluster *un-collapsed*: characters that were ≈0.85 similar are now ≈0.41 similar — the model learned to tell them apart.

**Stay honest about what this does and does not show.** The synthetic eval scores *augmented reference art*, not real board crops — it is optimistic by construction. Real-frame generalisation beyond the few confidently-identified cards is the explicit **Round-2 lever** (collect/label real crops; SupCon and/or a deeper unfreeze; class-balancing for the 1-vs-67 imbalance). We report the synthetic numbers *as* synthetic, next to the real-frame behaviour below, rather than letting the optimistic number stand alone.

---

## A second debugging lesson: when the fix breaks your threshold

Fine-tuning fixed the model and *broke the abstention rule* — a subtle, teachable second-order effect.

The frozen model abstained on an **absolute cosine**: `confidence = (cosine + 1) / 2`, reject below 0.60. That worked because frozen cosines were spread across a wide range. But fine-tuning **compressed the absolute cosine scale**: a *correct* match now sits around cosine 0.6 and an *unrelated* prototype around 0.4. The old 0.60 cutoff, applied to the new model, **over-identified 125/125 real cards** — it could no longer tell a match from a non-match, because *every* score now lived in a narrow band.

The fix is to abstain on the **top1−top2 cosine margin** instead of the absolute value — *how decisively did the nearest prototype beat the runner-up?* A genuine match pulls clearly ahead; a face-down or ambiguous crop sits roughly equidistant from several prototypes (small margin) → "unknown". In `src/dbcv/identify.py` the gate is now `_EMBED_MARGIN_THRESHOLD = 0.12` (provisional), and **the reported `confidence` field is now that margin** — a semantics change worth knowing when reading a snapshot. (The research entry prescribes exactly this: port a min-margin abstention onto the learned head so it can still say "unknown" instead of over-identifying.)

**The lesson:** a metric that calibrated one model can be meaningless for a retrained one. When you change the model, re-examine every threshold and confidence definition downstream of it — the numbers moved even though the code around them didn't.

### Adopted real-frame behaviour

With the fine-tuned backbone and the margin gate, on the real sample frames:

- **Embedding identifier:** **30 / 125 (24%)** cards identified confidently, abstains ("unknown") on the other 95.
- **Classical baseline:** 44 / 125 (35%) identified.
- **Agreement** between the two rose from **27 → 90** cards after the fine-tune — the fine-tuned embedder now mostly agrees with the classical matcher where the classical one is confident, and abstains elsewhere.

The fine-tuned embedding-NN is the **adopted default**; the classical matcher is retained as a selectable fallback. Note the embedding model is *more conservative* here (24% vs 35%) — that is the margin gate doing its job after the over-identification scare, and pushing that confident fraction up on real frames is precisely the Round-2 work.

### What this changed about the art-swap promise

Module framing so far (and Module 09, staying-alive) leaned on identification being **zero-gradient** on an art swap: re-embed the new references, done. That was true of the *frozen* backbone. The adopted fine-tuned backbone is fit to the *current* 43 characters, so a genuinely new art set is now best handled by **re-fine-tuning** (`utils/python/finetune_embedding.py`, ~minutes on the Titan Xp) and then rebuilding the gallery. A quick re-embed against the existing fine-tuned backbone still *works*, but won't separate unfamiliar art as cleanly. This is an honest **accuracy ↔ retrain-cost tradeoff**: we bought a large accuracy gain on the current art at the cost of a (still cheap, still label-free, minutes-long) training step on an art swap. The classical fallback remains truly zero-gradient on a swap.

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

> **Note — two entry points, two identifiers.** This CLI runner (`run_pipeline.py`) wires the **classical** identifier (`make_gallery_identifier`), so the confidences above are HSV-correlation scores gated at 0.40 — handy for seeing the classical baseline directly. The **REST API** (`src/dbcv/api.py`) wires the **adopted fine-tuned embedding-NN** as its default instead, where `confidence` is the top1−top2 *margin* gated at 0.12 (semantics described above). Same gallery, different identifier and different confidence meaning — don't compare the two numbers directly.

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

For the frozen-fails / fine-tune-fixes arc, from `research/RESEARCH.md`, "Why a frozen-ImageNet embedding + NN gallery collapses on stylized cards, and how to fix it — 2026-06-22":

- Chen, Liu, Kira, Wang, Huang, *A Closer Look at Few-Shot Classification* (ICLR 2019) — https://openreview.net/pdf?id=HkxLXnAcFQ — the keystone result: a cosine classifier on a frozen backbone is competitive only when base and novel classes share a domain; under domain shift the limiting factor is the *features*, not the head. This is why our prototype gallery was fine and the backbone was not.
- Kumar, Raghunathan, Jones, Ma, Liang, *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (ICLR 2022, oral) — https://arxiv.org/abs/2202.10054 — full fine-tuning distorts good features on small data; **LP-FT** (linear-probe then fine-tune) beats both head-only and full-FT. The basis for our two-phase schedule.
- Kim, Kim, Cho, Kwak, *Proxy Anchor Loss for Deep Metric Learning* (CVPR 2020) — https://arxiv.org/pdf/2003.13911 — the metric-learning loss we adopted: enforces an inter-class margin, converges fast, tolerates few examples per class. (Plus the *PyTorch Metric Learning* loss catalog — https://kevinmusgrave.github.io/pytorch-metric-learning/losses/ — for ArcFace/CosFace/SupCon alternatives.)
- Salvador & Oberman, *ImageNet-Cartoon and ImageNet-Drawing* (ICML 2022 "Shift Happens" workshop) — https://tiagosalvador.github.io/projects/imagenet-shift/ — direct evidence that ImageNet-pretrained accuracy drops significantly on cartoon/drawing renderings: the domain-shift diagnosis in concrete form.
- Cubuk et al., *RandAugment* (NeurIPS 2020) — https://arxiv.org/pdf/1909.13719 — the augmentation policy used to synthesise training crops from the clean reference art (the highest-payoff small-data regulariser).
