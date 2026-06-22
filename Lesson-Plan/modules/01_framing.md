# Module 01 — Framing: reading game state from pixels under a compute budget

**The problem (in the pipeline):** Before writing a single line of vision code, this course has to answer a prior question: *why does this system exist as a teaching vehicle?* And before a learner runs any code, they need to understand the constraints that will drive every technique choice from here on. This module establishes both: the shape of the problem, the constraints that make it interesting, and the arc of decisions the rest of the course will trace.

**What you'll be able to do:**

1. Describe what "reading game state from video frames" means concretely for *Demon Bluff* — what the input is, what structured output is expected, and why the problem is not trivial.
2. Sketch the pipeline's seven-stage arc (frame selection → state gate → localization → identification → on-card reading → assembly → REST serving) and explain why each stage exists.
3. Name the six constraints that shape every technique choice in this course and explain how each one closes off options that would otherwise be available.
4. Identify which pipeline stages are classical (no learned model at runtime) and which are deferred to a learned model — and understand why those are deliberate choices, not oversights.

---

## Why this course exists

Most computer-vision teaching works from toy datasets: clean images, fixed resolutions, labels already provided, a benchmark accuracy to chase. Those courses teach the mechanics of CV methods reasonably well. What they hide is the part that actually requires judgment: which technique to reach for, when a cheap deterministic method beats a large neural one, how to design a system so it stays alive after inputs shift, and how to explain those choices to someone who has to maintain the code later.

This course refuses that artificiality. The mandate is stated directly in `PROJECT-PITCH.md`:

> "This project refuses that artificiality. It teaches modern CV by building **one genuinely constrained system** and showing the reasoning at every fork."

The system is a *Demon Bluff* game-state reader. The course is built around it because it is simultaneously small enough to finish and real enough to surface all the decisions that matter. The lesson plan is the primary deliverable; the pipeline is the worked example the course is built around.

---

## The problem: extracting board state from a stream of pixels

*Demon Bluff* is a social-deduction game — think *One Night Ultimate Werewolf* or Mafia — whose board is a layout of role **cards** arranged in a radial ring. Each card shows a townee character and encodes structured information: which role (villager / minion / outcast / demon), which specific character identity, the current state of that slot (face-up, face-down, under a modal), and any ability counts or text on the card face.

The input to the pipeline is **video footage of a game session** captured from a screen. The input is *only* pixels: no audio channel is read, no game save files are accessed, no API into the game process is available. Everything must be inferred from frames.

The output is a structured `GameStateSnapshot` — a JSON object (versioned as Pydantic schema v0.2.0 in `src/dbcv/schema.py`) containing:

- The frame state (board frame, modal, non-board)
- A list of detected cards, each with a bounding box (in relative coordinates — never pixel-absolute), a predicted identity, a role class, and per-field confidences
- Source metadata (video id, frame index, timestamp)
- The measured resolution of the source frame

That snapshot is served over a REST endpoint: `POST /v1/snapshot` accepts a frame upload and returns the structured JSON.

---

## The pipeline arc: seven stages as a preview

The course teaches CV by building these stages in order. Here is the arc:

**Stage 0 — Frame-state gate.** Not every frame of a game session shows the board. Modals (role-reveal popups), menu screens, and loading transitions interrupt the board view. A cheap center-vs-ring brightness-ratio gate (`src/dbcv/frame_state.py`) classifies each frame as `board` or not-board before any downstream processing runs. Localization only ever sees board frames; everything else is discarded early.

**Stage 1 — Frame selection.** An hour of game video at 30 fps is ~108 000 frames. Downstream stages cannot process every one in real time; more importantly, most adjacent frames are near-identical and redundant. A perceptual-hash deduplication step, running on a low fixed-stride decode (~1–2 fps), collapses the stream to a manageable set of distinct frames. Module 02 teaches this stage.

**Stage 2 — Resolution-agnostic geometry.** Every geometry computation in the pipeline is derived from the measured frame dimensions (`image.shape[:2]`), never from a hard-coded constant. This is enforced as a project constraint (see `CLAUDE.md`). Module 03 teaches why this matters and how the pipeline enforces it.

**Stage 3 — Card localization.** Given a board frame, find the bounding box of each card. The pipeline uses a classical, layout-based approach: HSV colour segmentation → morphological cleanup → contour filtering → HUD-zone exclusion → IoU NMS. This is `src/dbcv/localize.py`. Module 04 teaches this stage and explains the choice over a trained detector.

**Stage 4 — Card identification.** Given a cropped card image, name the townee. The shipped classical baseline uses a 2-D HSV colour histogram matched against an in-memory reference gallery (`src/dbcv/gallery.py`, `src/dbcv/identify.py`). The deferred upgrade uses a small frozen embedding backbone with nearest-neighbour retrieval. Module 05 teaches both.

**Stage 5 — On-card reading.** Given a card crop, read the role-name text and ability counts. A closed-vocabulary recognizer (tiny CNN/CRNN over the known glyph set) or a lightweight OCR fallback reads these fields. Module 06 teaches this stage.

**Stage 6 — Assembly and temporal logic.** Assemble per-card results into a coherent board snapshot. Handle transient modals that occlude cards by carrying forward state across frames. Module 07 covers this.

**Stage 7 — REST serving.** Expose the snapshot over a FastAPI endpoint with ONNX Runtime inference on CPU, versioned Pydantic schemas, and a lifespan-managed model load. Module 08 teaches this stage (`src/dbcv/api.py`).

The course closes with Module 09: what happens when the card art changes, and how the system was designed so that "cheap to re-fit" is a structural property, not a post-hoc retrofit.

---

## The constraints: the characters in every later decision

The constraints below are not background context. Each one closes off options that would otherwise seem attractive and motivates a technique that the course will teach. Understanding them before diving in means the reasoning at each module's fork will feel earned rather than arbitrary.

These constraints are stated in `PROJECT-PITCH.md` (constraints table) and enforced in `CLAUDE.md`.

### Constraint 1 — Runtime: mid-grade gaming PC

The pipeline must run at usable frame rates on consumer hardware — specifically on the class of machine a hobbyist or educator is likely to own (e.g. a system with an NVIDIA 3060). This means inference-time models must be small and fast.

The research entry that grounds the compute budget (`research/RESEARCH.md`, "Runtime compute budget: what fits a mid-grade gaming PC and trains on a Titan XP / Colab T4 — 2026-06-21") establishes concrete limits. The rule of thumb: **≤~30 M parameters / ≤~100 MB** for any runtime model. Nano-class detectors (YOLOv8n, YOLO11n: ~2–3 M params) and small classifiers (MobileNetV3-Small: ~2.5 M params, ~8 MB) satisfy this. Foundation models do not: SAM ViT-H at 632 M parameters, and Grounding-DINO documented as "too slow for real-time even on an A100," are explicitly off the table as runtime components.

A companion constraint is the training budget: **custom models must be trainable on a single Titan XP (12 GB, Pascal architecture, FP32 only — no Tensor Cores) or a free Google Colab T4**. Pascal's lack of Tensor Cores means mixed-precision and INT8 quantization provide no speedup on that hardware; training happens in FP32. The compute budget research entry documents this explicitly as a caveat.

The consequence: every stage where a learned model appears, the course shows why the chosen model fits the budget and why a larger alternative was excluded.

### Constraint 2 — Video only

No signal beyond pixels is available: no audio, no save files, no game process API. Everything is inferred from frames. This means all the structured information in a snapshot — role class, character identity, card state — must be recoverable from a single image of the board, or from temporal patterns across nearby images.

This constraint rules out shortcuts that would dramatically simplify identification: reading the game's own state file, listening for audio cues, or intercepting game network packets. The pipeline must work from a screen recording without any cooperation from the game itself.

### Constraint 3 — Huge input (frame selection is mandatory)

The sample footage runs ~1 hour at ~370 MB per clip. Processing every frame is not possible at any practical frame rate. A frame selection stage is therefore mandatory — not a performance optimization, but a prerequisite.

The frame-selection research entry (`research/RESEARCH.md`, "Frame selection / keyframe extraction from long gameplay video — 2026-06-21") gives the approach: fixed-stride decode at ~1–2 fps (read from the media's actual fps, never a hard-coded stride), perceptual-hash deduplication using pHash/dHash with Hamming-distance comparison, and a board-vs-not-board gate before localization. Near-duplicate removal with a 64-bit perceptual hash and Hamming distance threshold ≤~8 is the standard cheap method; deep-embedding dedup is reserved for offline dataset curation where the computational cost is acceptable.

### Constraint 4 — No baked-in resolution

Both sample clips happen to share the same resolution. The pipeline does not care. Every geometric computation is derived from `image.shape[:2]` — the dimensions measured from the decoded image at runtime — not from any constant. `CLAUDE.md` states this constraint verbatim: "Never bake in a resolution."

The practical reason: a learner running the pipeline on footage from a different capture setup, a different monitor, or a future game version should not need to change any code. Resolution-agnosticism is what makes the pipeline actually portable. Module 03 teaches the mechanisms that enforce this.

### Constraint 5 — Card art may be swapped (recognizers must be cheap to re-fit)

*Demon Bluff* has an alternate card-art set. Any recognizer that requires a full relabeling and retraining pass every time the art changes is a liability — particularly given that training on a Titan XP or Colab T4 is feasible but takes hours. The project constraint requires that identification be re-fittable cheaply.

This constraint directly determines the technique choices in Modules 04 and 05, and is the organizing theme of Module 09. It is why the trained classifier is the approach *rejected* for production identification: it is the one approach that structurally requires re-annotation and retraining on every art swap.

### Constraint 6 — Teaching-first

Explainability and honest trade-off discussion outrank cleverness. A technique is included in the course because the pipeline needed it, not because it rounds out a curriculum. A technique is excluded when the evidence does not justify it.

This constraint shapes how the course is written as much as how the pipeline is built: every module explains *why* the chosen technique was chosen over its alternatives, names the approach that was rejected and why, and is honest about what the chosen approach does not handle well.

---

## What to expect by the end

By the end of this course, a learner will be able to:

- Take an hour of game footage and run it through a pipeline that returns structured JSON game state, with a working REST endpoint.
- Explain every technique choice — from the perceptual-hash deduplication to the classical localizer to the gallery-based identifier — citing the research and the constraint it responds to.
- Understand which stages ship a classical baseline first (localization, identification) and why the baseline exists as a deliberate lower-bound measurement rather than a placeholder.
- Know what "cheap to re-fit" means structurally, and which design decisions guarantee it.
- Have an honest account of where the pipeline's current accuracy falls short, and what the planned upgrade path is.

Some stages in the course ship a classical baseline and explicitly defer a heavier model. This is not an apology. The classical baseline is measurable, explainable, and already satisfies the art-swap constraint. The deferred embedding-NN upgrade for identification is held back not because it is wrong but because it adds a new dependency (`onnxruntime`, a model export step, a backbone selection decision) and the gap between baseline and upgrade is more instructive to measure than to paper over.

---

## Further reading

- `research/RESEARCH.md`, "Runtime compute budget: what fits a mid-grade gaming PC and trains on a Titan XP / Colab T4 — 2026-06-21" — the compute-budget entry that grounds every model-size decision in the course. The Titan XP / Pascal FP32 caveat and the INT8/mixed-precision restriction are documented there.
- `PROJECT-PITCH.md` — the long-arc design narrative, the constraints table, and the decisions table that records every confirmed technique choice with its date and rationale.
- `src/dbcv/schema.py` — the versioned Pydantic schema for `GameStateSnapshot`; seeing the output structure before the pipeline stages are taught gives the stages a concrete target to build toward.
