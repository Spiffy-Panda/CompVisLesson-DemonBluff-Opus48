# RESEARCH.md — outside-knowledge log

Every **non-Demon-Bluff** subject researched before a decision. Newest entries on top. One entry per subject. This is the evidence base the lesson plan cites and the gate the pipeline passes through (research-before-deciding).

## How to use

- **Append, don't rewrite.** If findings change, add a new dated entry that references the old one.
- **Cite the real source.** Prefer primary, recent, and authoritative material.
- **Rate trust honestly** using the rubric below.
- **The abstract reports what was found**, including null/negative results — not what we wished were true.

## Authority / trust rubric

| Tier | Meaning |
|------|---------|
| **A** | Peer-reviewed paper, official library/standard documentation, or a recognized authority (e.g. OpenCV docs, a maintainer, a benchmark paper). |
| **B** | Credible practitioner source with shown evidence (reproducible blog post with code/numbers, conference talk, well-known engineering blog). |
| **C** | Community/anecdotal (forum answer, single blog assertion, undated tutorial) — usable as a lead, not as proof. |
| **D** | Unverified / AI-generated / marketing — record only with a flag, verify before relying. |

## Entry skeleton

```
## <subject> — <YYYY-MM-DD>

- **Source:** <title + URL / citation>  (+ additional sources as a list)
- **Authority / trust:** <A|B|C|D> — <one line on why this rating>
- **Reason for inclusion / what we were looking for:** <the decision this informs>
- **Abstract of findings:** <2–6 sentences of what the source actually says, including caveats and anything that argues *against* using it>
```

---

<!-- entries appended below, newest first. The 2026-06-21 batch is ordered by
     pipeline stage (frame-selection → localization → identification → OCR →
     serving → compute budget) rather than by clock time, since they share a date. -->

## Python environment & dependency management (venv/virtualenv/pip/conda/uv) — 2026-06-22

- **Source:**
  - *`venv` — Creation of virtual environments* — Python 3 standard library docs — https://docs.python.org/3/library/venv.html
  - *pip User Guide* — https://pip.pypa.io/en/stable/user_guide/
  - *conda documentation* — Anaconda / conda-forge — https://docs.conda.io/projects/conda/en/stable/
  - *uv documentation* — Astral — https://docs.astral.sh/uv/
- **Authority / trust:** **A** — all four are official documentation for their respective projects (Python stdlib, pip maintainers, Anaconda, Astral).
- **Reason for inclusion / what we were looking for:** Choosing a reproducible environment strategy for a course whose learners run on diverse machines, and teaching the tradeoffs between the main alternatives — specifically to support the user-requested environment-management lesson module.
- **Abstract of findings:** `venv` (stdlib since Python 3.3) creates an isolated directory with a private `site-packages` and a `python`/`pip` pointing into it; it does not install packages or manage interpreter versions. `pip` (bundled since Python 3.4) installs packages from PyPI into a target environment; a pinned `requirements.txt` (`package==x.y.z`) is the minimal reproducibility mechanism. `virtualenv` is the third-party predecessor to `venv`; it adds faster env creation and broader configuration but is rarely needed directly on modern Python 3. `conda` (Anaconda/Miniconda/miniforge) manages environments *and* installs packages including non-Python native binaries (CUDA, BLAS, MKL); it is not Python-specific, operates on its own channel ecosystem (conda-forge), and provides pre-built binaries for a wider platform matrix than PyPI wheels — but mixing conda and pip installs into the same environment is a documented source of dependency corruption. `uv` (Astral, 2024+) reimplements pip + venv in Rust; per its documentation it installs packages significantly faster than pip via aggressive caching and parallel downloads; it also produces `uv.lock` lockfiles pinning all transitive dependencies, and newer versions manage Python interpreter versions. `uv` is pip-compatible (reads `requirements.txt`, `pyproject.toml`) but is a separate installation step not bundled with Python. Caveats: `conda` and `pip` should not be mixed in the same environment without care; `uv` does not manage non-Python system libraries the way conda does; no vendor speed claims are cited as numbers here because figures vary substantially across machine configurations and cache states.
- **Fit for our constraints:** The project chose `venv` + pinned `requirements.txt` as the zero-install, universally reproducible baseline — every Python ships `venv`, every learner can follow along without first installing a tool. `uv` (speed) and `conda` (native binary stacks) are taught as alternatives in Module 00 of the lesson plan, with honest tradeoffs. `opencv-python-headless` (not `opencv-python`) is used because the pipeline is server/batch — no GUI, no GTK/Qt dependency. Packages not yet justified by research are consciously deferred from `requirements.txt` (conservative path: onnxruntime, torch, imagehash held back until a pipeline stage warrants them).

## Frame selection / keyframe extraction from long gameplay video — 2026-06-21

- **Source:**
  - *Scene Detection Policies and Keyframe Extraction Strategies for Large-Scale Video Analysis* (2025) — https://arxiv.org/html/2506.00667v1
  - *PySceneDetect 0.7 — Detectors API* (official docs) — https://www.scenedetect.com/docs/latest/api/detectors.html ; CLI/performance — https://www.scenedetect.com/docs/latest/cli.html
  - *Comparative Evaluation of Perceptual Hashing and Deep Embedding Methods for Robust and Efficient Image Deduplication*, MDPI Electronics 15(7):1493 (2025) — https://www.mdpi.com/2079-9292/15/7/1493 (full text 403; used abstract only)
  - `akamhy/videohash` — https://akamhy.github.io/videohash/ ; `knjcode/imgdupes` — https://github.com/knjcode/imgdupes
  - OpenCV *Template Matching* tutorial — https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html
- **Authority / trust:** **A** for PySceneDetect & OpenCV docs; **A/B** for arXiv 2506.00667 (recent preprint, but production-scale evidence: 600k+ hours, >95% extraction on commodity 8-core CPU); **A/B** for the MDPI benchmark (only the abstract was readable). `videohash`/`imgdupes` are **B**. 2024–2025 VLM-keyframe papers were treated as context only — they optimize for a VLM's QA budget, a different objective than ours.
- **Reason for inclusion / what we were looking for:** Choosing the frame-selection stage that guarantees nothing downstream ever processes raw full video (~1 h, ~370 MB samples).
- **Abstract of findings:** The best-matching source recommends a **duration-aware policy**: content-based scene detection for medium clips, **fixed-interval splitting** for very long footage, decoding at **~2 fps resized to 256×144** then scoring ~5 equidistant frames/segment — real-time on a standard 8-core CPU, no GPU. PySceneDetect confirms the cheap classical toolbox: `ContentDetector` (HSV frame-diff, default threshold 27), `ThresholdDetector` (fades to black — i.e. menu transitions), plus `HashDetector`/`HistogramDetector`; **downscaling speeds processing ~4× per integer increment**, while frame-skip is discouraged. For near-duplicate removal, perceptual hashing (pHash/dHash → 64-bit, compared by **Hamming distance via popcount**, near-dup at d_H ≤ ~8) is the standard cheap method; deep embeddings only earn their cost under aggressive transforms that don't occur in a stable screen capture. **Caveats:** (1) `ContentDetector` is tuned for *filmed* content — a screen capture has no camera motion but does have sudden UI repaints, card-flip animations, and tooltip/particle effects, so thresholds must be tuned on our footage. (2) The VLM-selection literature's "adaptive beats uniform" conclusion is about a tiny QA frame budget, not our "don't miss any board state, don't re-process identical frames" goal. (3) Scene-cut detection only flags *that* something changed, not *whether a board is on screen* — a separate cheap board-gate is still needed.
- **Fit for our constraints:** A **cheap, classical, CPU-only cascade, no runtime model**: (1) decode at a low fixed stride (1–2 fps) read from the media's real fps — never a baked resolution; (2) drop near-duplicates with a perceptual hash + Hamming threshold (essentially free, trivially re-tunable on art swap); (3) gate "board vs. menu/transition" with `matchTemplate`/HSV-histogram checks on stable UI anchors, optionally backed by `ThresholdDetector` for fade-to-menu. Reserve `ContentDetector`/`HashDetector` for **dev-only** segmentation of the long sample (use `-d 2`/`-d 3` for faster-than-real-time passes); reserve CLIP/BLIP scoring and deep-embedding dedup strictly for **offline dataset curation**.

---

## Card/region localization robust to art swaps under a tight compute budget — 2026-06-21

- **Source:**
  1. *Small Object Detection with YOLO: A Performance Analysis Across Model Versions and Hardware* (2025) — https://arxiv.org/html/2504.09900v1
  2. Ultralytics — train-on-custom-dataset guidance — https://github.com/orgs/ultralytics/discussions/6249 ; https://learnopencv.com/train-yolov8-on-custom-dataset/
  3. *Automated game testing using computer vision methods* (classical OpenCV ~12.1 ms vs YOLO ~19.37 ms/frame for UI recognition) — https://www.researchgate.net/publication/357080752_Automated_game_testing_using_computer_vision_methods
  4. LearnCodeByGaming — *How To Build a Bot with OpenCV* (template matching on fixed game UI) — https://learncodebygaming.com/blog/how-to-build-a-bot-with-opencv
  5. *SIFT vs. ORB* practitioner comparison (ORB ~25× faster; SIFT more robust to scale/rotation) — https://medium.com/@beauc_37732/comparing-sift-and-orb-for-feature-matching-a-visual-and-practical-exploration-6c194c72e4d6
- **Authority / trust:** **A/B.** (1) 2025 arXiv benchmark with per-hardware numbers (A for data, not yet peer-reviewed). (2) vendor docs (A for "how to train," B for rosier accuracy claims). (3) peer-reviewed conference paper (A). (4) credible practitioner walkthrough (B). (5) community but reproduces a well-established result (B/C).
- **Reason for inclusion / what we were looking for:** Choosing how to find card regions in a frame, robustly and cheaply.
- **Abstract of findings:** The decisive fact is that **localization and recognition are separable**, and our UI *layout* is stable even when card *art* swaps — so "find the card regions" is largely a geometry problem, not a learned-detection problem. Classical anchoring (detect a few stable, art-independent UI landmarks — panel borders, slot rectangles, the trial-row band — via contour/edge/`HoughLines` + template-match on UI chrome, then parse the fixed grid relative to those landmarks, scaled by the measured resolution) is cheap, deterministic, needs **zero training data**, and is *immune to art swaps by construction*. The peer-reviewed game-CV benchmark found classical OpenCV UI recognition ran ~12.1 ms/frame vs ~19.37 ms for YOLO — the learned detector is both slower *and* needs labels here. Small detectors (YOLOv8n/YOLO11n, ≈2.6–3.2 M params) train cheaply (~150–400 labeled images, minutes on a Titan XP / Colab T4) and run real-time, **but** (a) a detector keyed to card *appearance* must be **re-annotated and retrained on every art swap** — violating "cheap to retrain"; (b) small-object mAP degrades on tiny targets and CPU throughput drops sharply beyond 320×320; (c) template/grid parsing is brittle to scale/rotation/perspective — fine for a flat 2D UI captured head-on, dangerous if the capture is skewed.
- **Fit for our constraints:** **Classical, layout-driven localization as the primary path; do not train a detector for the "where are the cards" step.** Read the frame resolution, locate 2–4 art-independent UI landmarks with edge+contour+line detection plus small template matches on *UI chrome* (not card art), then slice the known card grid relative to them — ~10 ms CPU, deterministic, no labels, survives art swaps untouched. **Reserve a learned detector (YOLO11n) only if footage proves the layout isn't reliably parseable** (resizable/overlapping panels, animated reflow, skewed captures); it trains in minutes on Titan XP/Colab but costs re-annotation per art set. Keep identity recognition a separate, cheap-to-retrain module so an art swap only re-fits that. *(Confirmed against sample frames: cards sit in a stable radial ring with numbered position badges — geometry localization is clearly viable; board card-count varies between clips, so derive structure from layout, not a fixed count.)*

---

## Card identification that is cheap to retrain when art changes — 2026-06-21

- **Source:**
  1. Snell et al., *Prototypical Networks for Few-shot Learning* (NeurIPS 2017) — https://arxiv.org/abs/1703.05175
  2. Wu, Efros, Yu, *Improving Generalization via Scalable Neighborhood Component Analysis* (ECCV 2018) — https://arxiv.org/pdf/1808.04699 (softmax classifier vs. NN retrieval over learned embeddings)
  3. Howard et al., *Searching for MobileNetV3* (ICCV 2019) — https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf
  4. *Limitations of Template Matching* — https://apxml.com/courses/introduction-to-computer-vision/chapter-5-introduction-object-recognition/limitations-template-matching ; survey arXiv:1610.07231 — https://arxiv.org/pdf/1610.07231
  5. LanceDB, *Zero-Shot Image Classification with Vector Search* (2024) — https://lancedb.com/blog/zero-shot-image-classification-with-vector-search/
- **Authority / trust:** **A** for 1–3 (canonical NeurIPS/ECCV/ICCV papers). **B** for 4 (survey [A] + teaching page [B]) and 5 (vendor engineering blog with working code; promotional framing).
- **Reason for inclusion / what we were looking for:** Naming each localized card (~44 roles + role class) in a way that is cheap to re-fit when the art set changes — the *retrain cost* on an art swap is the central question.
- **Abstract of findings:** The four families split cleanly on retrain cost. **(a) Template matching / NCC** needs zero training (swap the reference template) but degrades with even small rotation, scale, illumination, occlusion, or deformation; wants one template per *appearance*. **(b) Embedding similarity / metric learning** (small backbone — MobileNetV3-Small / small ResNet / frozen CLIP — + nearest-neighbor over a reference gallery) is the sweet spot: on an art swap you re-embed the new references and change *no weights*; Wu et al. show NN retrieval can *beat* a softmax classifier and is more interpretable. **(c) Few-shot / prototypical networks** formalize this — each class is the mean of a few embedded supports, new classes added at inference with *no parameter updates*. **(d) A small trained CNN classifier** is cheapest at inference and most accurate *for its training art*, but is the only family that must be **relabeled and retrained** on an art swap. Caveats: (b)/(c) push cost to inference (trivial at ~44 classes); a *frozen generic* backbone may underperform on stylized art without light adaptation. Honest counter-argument: at ~44 well-localized frontal crops, even template matching or a tiny classifier may hit high accuracy — the embedding/few-shot machinery earns its keep specifically because of the *art-swap constraint*.
- **Fit for our constraints:** Recommend **(b) a small frozen embedding backbone + nearest-neighbor over a per-art reference gallery**, collapsing to **(c) prototypical** by averaging a few references per card. On an art swap, (a)/(b)/(c) need **only the new reference images and zero gradient steps** — re-embed ~44 cards in seconds — whereas **(d) requires re-cropping, relabeling, retraining**. The backbone fits easily on Titan XP/Colab (or off-the-shelf); inference is one forward pass + a 44-way distance compare (negligible on a mid-grade PC). Pragmatic hybrid for the course: keep an **NCC/template fast-path** for clean axis-aligned crops (teaches the classical baseline, free to re-fit), fall back to **embedding-NN** for ambiguous cards, and frame the trained classifier as the approach we *reject* for production because of its retrain cost. *(Sample frames confirm a second identity signal: a name-label text under each card — OCR can corroborate the visual match; see the OCR entry.)*

---

## Lightweight OCR for short on-card text and numbers in game UI — 2026-06-21

- **Source:**
  - *PP-OCRv5: A Specialized 5M-Parameter Model Rivaling Billion-Parameter VLMs on OCR* (arXiv, 2025/2026) — https://arxiv.org/html/2603.24373v1
  - *PP-OCRv4_mobile_rec* model card — https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_rec ; Paddle2ONNX export — https://paddlepaddle.github.io/PaddleOCR/main/en/version2.x/legacy/paddle2onnx.html
  - *PaddleOCR vs Tesseract* benchmark (2026) — https://www.codesota.com/ocr/paddleocr-vs-tesseract
  - *Tesseract PSMs Explained* — https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/ ; preprocessing — https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
  - *On the Accuracy of CRNNs for Line-Based OCR* (arXiv:2008.02777) — https://arxiv.org/pdf/2008.02777 ; `gasparian/CRNN-OCR-lite` — https://github.com/gasparian/CRNN-OCR-lite
- **Authority / trust:** **A/B mixed.** PP-OCRv5 paper, PaddleOCR docs/model cards, the CRNN arXiv study, PyImageSearch are A. CodeSOTA and freeCodeCamp are B (single-test-image benchmarks — treat exact figures as directional). One "Tesseract is Dead" Medium piece was **D** and not relied on.
- **Reason for inclusion / what we were looking for:** Reading short text / numbers / state markers on cards (role name, ability counts, HUD digits) within the compute budget.
- **Abstract of findings:** For *general* OCR the field favors PaddleOCR's mobile line: **PP-OCRv4_mobile_rec ~11 MB, ~83% avg accuracy**; **PP-OCRv5 mobile ~5 M params, 370+ chars/sec on one CPU core**, both ONNX-exportable. Caveat: the *full* pipeline (det+cls+rec) is heavy — one 2026 benchmark measured **~4.85 s/image, ~500 MB RAM on CPU** vs **Tesseract 5.5 at ~0.77 s, ~10 MB** (where Tesseract made 3 char errors, PaddleOCR 0) — so "small" applies to the recognizer weights, not the default install. Tesseract (LSTM) and EasyOCR degrade sharply on stylized/decorative fonts; Tesseract works better on grayscale than naive binarization, and even with a digit whitelist can mis-read tight synthetic glyphs (last resort `--psm 13`). Counter-argument to all general OCR: a card has a **known, fixed, small glyph set** (closed vocabulary of role names + 1–2-digit counts), and a **tiny custom CNN/CRNN reaches >95% on a fixed lexicon** at a fraction of the size/latency.
- **Fit for our constraints:** Split the problem. (1) **Role-name text and state markers are a closed vocabulary** → prefer a **tiny custom classifier over the known glyph set** (per-field CNN for one-of-N labels, or small depthwise-separable CRNN+CTC for variable strings), trained on synthetically-rendered crops of the current card font; >95%, sub-ms, few MB, ONNX-exportable, and **retrains in minutes on Titan XP/Colab** by re-rendering the new font — directly satisfying the art-swap rule. (2) Keep **PP-OCRv4/v5 mobile (ONNX)** as the **narrow fallback** for genuinely free-form text, sizing in just the recognizer (~5–11 MB), fed **2–4× upscaled, grayscale (not binarized) crops**. Treat **Tesseract/EasyOCR as dev/debug baselines, not the runtime path**. Preprocessing = locate-region → upscale → grayscale → contrast-normalize, never global binarization.

---

## Serving CV inference over a REST API (Python) — 2026-06-21

- **Source:**
  1. *Lifespan Events* — FastAPI official docs — https://fastapi.tiangolo.com/advanced/events/
  2. *Concurrency and async / await* — FastAPI official docs — https://fastapi.tiangolo.com/async/
  3. FastAPI ML Deployment course — response models — https://apxml.com/courses/fastapi-ml-deployment/chapter-2-data-validation-pydantic/response-model-definition ; lifespan model-loading — https://apxml.com/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/loading-models-fastapi
  4. *Building Low-Latency Inference APIs Using FastAPI and ONNX* — https://mljourney.com/building-low-latency-inference-apis-using-fastapi-and-onnx/
  5. *Make FastAPI CPU-bound Endpoints 2X Faster* — https://amirkarimi.dev/blog/2023/07/23/make-fastapi-cpu-bound-endpoints-2x-faster/
- **Authority / trust:** **A** for 1–2 (official FastAPI/Starlette docs). **B** for 3–5 (credible practitioners with runnable code; ONNX latency figures illustrative, not benchmarked on our hardware).
- **Reason for inclusion / what we were looking for:** Designing the REST service exposing the game-state snapshot — loading models out of the request path, schema shape/versioning, sync vs async for CPU-bound inference on one PC.
- **Abstract of findings:** Load expensive models **once** via a `lifespan` async context manager (`@app.on_event("startup")` is deprecated — source 4 still uses it); store on `app.state`. On concurrency the docs are explicit and counter-intuitive: a path op declared **`def` runs in an external threadpool** (won't block the loop), whereas CPU-bound work in **`async def` blocks the loop and stalls every request** — so synchronous CV inference belongs in plain `def` (or an explicit `run_in_threadpool`/`run_in_executor`). Flagged tension: the official *lifespan* example shows `async def predict(...)` calling the model directly, contradicting the concurrency page — treat inference as blocking. NumPy/ONNX release the GIL during native compute (so a small threadpool genuinely parallelizes), but pure-Python pre/post-processing serializes under load (escalate to `ProcessPoolExecutor`). Pydantic `response_model` gives a typed, auto-filtered schema + OpenAPI; version by pinning a `schema_version` field plus a path/header version and evolving via new models. ONNX Runtime: build `InferenceSession` once at startup, cache IO names, set `intra_op_num_threads`≈physical cores, `ORT_ENABLE_ALL`, `CPUExecutionProvider`; micro-batching helps only under sustained high request rates.
- **Fit for our constraints:** Load each CV model once in `lifespan` onto `app.state`; expose `POST /v1/snapshot` taking a frame and returning a versioned Pydantic `GameStateSnapshot` (per-card identity, role class, on-card readings, per-field confidences, and `resolution` read from the media — never hard-coded) with an explicit `schema_version`. Declare the inference endpoint **plain `def`** so it runs in the threadpool and one slow frame can't freeze the API; push heavy Python pre/post behind `run_in_executor`. **Default to no batching** (interactive, low-concurrency teaching use). Serve small recognizers via **ONNX Runtime on CPU**; re-exporting a re-fit model to ONNX leaves the serving layer untouched (consistent with the art-swap rule).

---

## Runtime compute budget: what fits a mid-grade gaming PC and trains on a Titan XP / Colab T4 — 2026-06-21

- **Source:**
  1. Ultralytics YOLO11 official model docs (params, FLOPs, CPU-ONNX & T4-TensorRT ms) — https://docs.ultralytics.com/models/yolo11/
  2. ONNX Runtime — *Quantize ONNX models* (INT8 needs Tensor Cores: T4/A100; older HW won't benefit) — https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
  3. LearnOpenCV — *Performance Comparison of YOLO Models* (nano/tiny FPS on CPU and RTX 4090) — https://learnopencv.com/performance-comparison-of-yolo-models/
  4. *On Efficient Variants of SAM: A Survey* (arXiv 2410.04960) + Grounding-DINO 1.5 (arXiv 2405.10300) — dev-only upper bounds — https://arxiv.org/html/2410.04960v1
  5. EmergentMind/MobileNetV3 — small-classifier params/size/latency — https://www.emergentmind.com/topics/mobilenetv3
- **Authority / trust:** **A** for 1, 2, 4 (official Ultralytics & ONNX Runtime docs; peer-reviewed/arXiv survey + primary papers); **B** for 3, 5 (credible practitioner benchmarks, single-setup). Titan-Xp Pascal FP16 ≈1/64 FP32 corroborated across hardware reviews (B).
- **Reason for inclusion / what we were looking for:** The model-size budget that gates every model choice.
- **Abstract of findings:** A small detector is effectively free at runtime: **YOLO11n = 2.6 M params / 6.5 BFLOPs, ~1.5 ms on T4 (TensorRT)**, YOLOv8n **~1.76 ms on an RTX 3060** (500+ FPS headroom); on CPU YOLO11n ~56 ms (~18 FPS), nano/tiny 20–30 FPS on a 6-core CPU. Small classifiers are smaller: **MobileNetV3-Small 2.5 M params, <8 MB, ~13 ms CPU**; **ResNet18 11.2 M params, ~43 MB, ~14 ms CPU**. Mobile OCR (~8 MB) is the heaviest per-call item (~420 ms CPU / ~140 ms T4 per crop) → run only on selected regions. The hard ceiling is **dev/labeling-only** foundation models: **SAM ViT-H 632 M params**; **Grounding-DINO "too slow for real-time even on A100."** Caveats: **INT8 helps mainly on CPU** (x86 VNNI); on GPU it **needs Tensor Cores (T4/A100)** — the **Titan Xp/Pascal has none and runs FP16 at ~1/64 FP32**, so keep dev work FP32. A custom YOLOv8 fine-tune is **~2.5 h on a free Colab T4**; small models fit the Titan Xp's 12 GB in FP32 (slower than Ampere, no mixed precision).
- **Fit for our constraints:** **Runtime (ship): YOLOv8n/YOLO11n nano-detector class (≤~3 M params, ≤~10 BFLOPs) and MobileNetV3-Small/ResNet18-class classifiers (≤~12 M params, ≤~45 MB)** — real-time on a 3060 with margin, 10–30 FPS on CPU fallback; mobile-OCR (~8 MB) fine **if gated to detected text regions**. Rule of thumb: **≤~30 M params / ≤~100 MB ≈ comfortably real-time on a mid-grade GPU**; reserve large ViT / SAM (600 M+) / Grounding-DINO strictly for **dev-only** dataset labeling. Train custom runtime models **FP32, ≤12 GB**: the nano-detector + small-classifier recipe trains in a few hours on a free Colab T4 (feasible if slower on Titan Xp), satisfying "cheap to retrain on art swap" — but don't rely on mixed-precision/INT8 speedups on Pascal.
