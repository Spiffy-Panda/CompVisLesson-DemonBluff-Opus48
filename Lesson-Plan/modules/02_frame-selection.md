# Module 02 — Taming an hour of video: frame selection and the board/modal/menu gate

**The problem (in the pipeline):** One hour of 60 fps gameplay is around 216,000 frames. Running the full localizer + identifier on every one of them would take the better part of a day on a CPU. Worse, most of those frames are useless: duplicates of the frame before them, loading screens, menus, and modal dialogs that occlude the board entirely. The pipeline must answer two questions before a single frame reaches the localizer: *Is this frame different enough from the last to be worth processing?* and *Does this frame even show the board?*

**What you'll be able to do:**

1. Explain why raw-stride decode, scene detection, and perceptual-hash deduplication occupy different positions in a frame-selection pipeline, and when each earns its cost.
2. Describe the three-class frame-state gate (`board` / `modal` / `menu`) and why it runs before the localizer.
3. Walk through the center-vs-ring brightness ratio, explain why the earlier absolute-brightness approach failed on Demon Bluff's dark-background modals, and identify the invariant that the ratio exploits.
4. Read `src/dbcv/frame_state.py` and predict which class a novel frame will receive.
5. Run the gate against sample frames using `utils/python/run_pipeline.py` and interpret the `frame_state` field in the output.

---

## The options

The research entry that grounds this module is `research/RESEARCH.md`, "Frame selection / keyframe extraction from long gameplay video — 2026-06-21."

When you need to reduce a long video to a manageable, useful set of frames, the field offers three tools that are often combined:

### Option A: Fixed-stride decode

Decode one frame every N seconds — for example, one frame every 0.5 seconds from a one-hour video yields 7,200 frames, a roughly 30× reduction from 60 fps capture. The stride is duration-aware: the cited research recommends decoding at approximately 2 fps for long game footage, then scoring a few equidistant frames per segment rather than every decoded frame. This is the cheapest option on every axis — no model, no comparison, just a counter — and it is the right starting point.

**Trade-off:** stride gives you *coverage* (you will not miss an event that lasts longer than one stride period), but it gives you no *deduplication*. A 30-second round where nothing changes on screen still produces 60 decoded frames at 2 fps. That is probably fine for a teaching pipeline, but a production system would add the next step.

### Option B: Perceptual-hash deduplication

After stride-decoding, compute a 64-bit perceptual hash (pHash or dHash — a coarse discrete-cosine or difference hash of a downscaled, grayscale version of the frame) for each frame and drop any frame whose Hamming distance from the last kept frame is below a threshold, typically 8 bits or fewer (out of 64). The comparison cost is essentially free: Hamming distance on 64-bit integers is a single CPU instruction (`popcount`). Per the cited benchmark (MDPI Electronics 15(7):1493, 2025), perceptual hashing is the standard cheap method for near-duplicate removal and earns its keep over the stride-decode output; deep embeddings only add value under aggressive geometric transforms that do not occur in a stable screen capture.

**Trade-off:** the threshold must be tuned. A threshold that is too tight will pass almost every frame (no deduplication); one that is too loose will drop frames that show a new card arrangement after a fast transition. Tuning requires labelled examples from the real footage. This step is **not yet built** in the pipeline as of this module's writing — stride decode + perceptual-hash deduplication is a known owed item, forward-referenced from the current state gate. The module teaches the design now so that implementation slot is clear.

### Option C: Scene detection (content-based)

Tools like PySceneDetect's `ContentDetector` compute per-frame HSV difference scores and emit a cut event when the score exceeds a threshold. This is the right tool for segmenting a long video into scenes (finding where game rounds begin and end, or where a modal appears). It runs in real time on a standard 8-core CPU at roughly 2–3 fps video speed per the PySceneDetect documentation.

**Trade-off:** scene detection is tuned for *filmed* content — camera cuts, scene transitions. A screen capture has no camera motion, but it does have sudden UI repaints, card-flip animations, and tooltip particle effects that can all fire the detector spuriously. Per the research entry, `ContentDetector` thresholds must be re-tuned on screen-capture footage, and even then it tells you only *that* something changed, not *whether a board is on screen*. For the Demon Bluff pipeline, scene detection is useful for **offline dataset curation** (finding interesting round transitions to sample from) but is not the right tool for the per-frame production gate.

### Option D: Foundation models — dev/labeling only

CLIP, BLIP, and similar vision-language models can score frames for relevance ("does this show a game board?") with no task-specific training. They are genuinely useful for building the labeled dataset that a tiny per-frame classifier would need. At runtime, however, they are dev-only: as documented in `research/RESEARCH.md`, "Runtime compute budget — 2026-06-21," SAM ViT-H runs at 632 million parameters and is not real-time on any mid-grade consumer GPU, and Grounding-DINO is described as "too slow for real-time even on an A100."

---

## What we chose and why

**A CPU-only, two-part cascade:**

1. **Fixed-stride decode** — the first gatekeeper; cheap, no model, gives coverage.
2. **Frame-state gate** — a classical brightness-ratio classifier that runs on each decoded frame and decides whether to forward it to the localizer.

Perceptual-hash deduplication (Option B) is designed and owed; it goes between stride decode and the gate. Scene detection (Option C) is reserved for offline dataset work. Foundation models (Option D) are dev-only.

The gate is grounded in the same research entry (entry 1 of `research/RESEARCH.md`), specifically the finding that "a separate cheap board-gate is still needed" because scene-cut detection does not answer the board-vs-modal question.

**Compute:** the gate is a handful of NumPy region means and a single division — it runs in under 1 ms per frame. The full cascade (stride + gate + localizer) fits comfortably on a mid-grade gaming PC with no GPU.

---

## The frame-state gate: from a failed idea to the shipped discriminator

This section teaches the worked example in `src/dbcv/frame_state.py`. Reading that file alongside this section is part of the hands-on experience.

### The problem: you cannot threshold absolute brightness

The obvious first guess is: a board frame has a dark center (the pentagram), so "mean brightness of the center region" will be low on board frames and high on modal frames. This was the first spike, and it scored **0 out of 3 on modal frames**.

Why? *Demon Bluff*'s modal dialogs — the deck viewer, the role-reveal screen, the "pick three characters" panel — are **dark-background panels**. The game's art direction uses the same deep dark starfield for the board background and for the background area *around* a modal panel. The modal panel itself is bright, but the entire ring around the modal is dark. Measure the mean brightness of the whole center region and you get a moderate value, not much higher than a normal board frame where the pentagram is also dark. The naive heuristic fires in the wrong direction.

The empirical proof (from `src/dbcv/frame_state.py`, documented in the module comments):

| Frame type | Measured center mean brightness |
|---|---|
| Board frames | moderate — cards distributed radially |
| Modal frames | also moderate — bright panel surrounded by dark ring |

Absolute threshold fails because both classes occupy the same absolute-brightness range in the center region.

### The fix: find the invariant

The insight is to stop measuring an absolute quantity and instead measure a *ratio*. On a board frame, the ring of cards around the center is itself bright — the cards' colourful art fills the ring. On a modal frame, the ring is dark (it is the game's background), but the center panel is bright. The ratio of center brightness to ring brightness is high on modal frames and close to 1.0 on board frames.

This is the discriminating invariant the gate exploits:

```
ratio = mean_brightness(center box) / mean_brightness(surrounding ring)
```

The measured ratios on the labeled set (7 frames, all from `dataset/frames/Sample1/` and `dataset/frames/Sample2/`):

| Frame | Type | Center/ring ratio |
|---|---|---|
| Sample1_003 | board | 1.063 |
| Sample1_007 | board | 1.021 |
| Sample1_009 | board | 1.109 |
| Sample1_010 | board | 1.097 |
| Sample1_000 | modal | 3.099 |
| Sample2_000 | modal | 5.878 |
| (a third modal) | modal | 4.121 |
| Sample1_006 | partial modal | 0.943 |

Board frames cluster between 0.94 and 1.11. Full modal frames cluster between 3.10 and 5.88. There is a **3× gap** between the highest board score (1.11) and the lowest modal score (3.10). The threshold is set at **2.0**, midpoint of the gap — not at the data boundary, but in the clean empty space between the two clusters, providing margin in both directions.

The partial-modal case (`Sample1_006`, the "Pick 3 characters" dialog with peripheral cards still visible around the edges) scores 0.943 — well below 2.0 — and is correctly classified as `board`. This is the **right production decision**: the localizer can still find the peripheral cards and the game state is partially readable. Calling it `modal` and skipping localization would discard real information.

### The algorithm in `src/dbcv/frame_state.py`

The function is `classify_frame_state(image: np.ndarray) -> FrameState` where `FrameState = Literal["board", "modal", "menu"]`. Its four steps:

1. **Convert to single-channel brightness** — the frame is decoded as BGR by OpenCV; `cv2.cvtColor` to grayscale. Channel order does not affect the result because only brightness (luminance) is used.

2. **Menu check** (Gate 1) — compute the whole-frame mean brightness. If it exceeds 160 out of 255, return `"menu"`. A loading screen or main menu is typically light-coloured; no in-game frame in these samples comes close to this threshold. This gate runs first so the ratio test — which would be meaningless on a uniformly bright image — is never reached for menu screens.

3. **Ratio test** (Gate 2) — compute the mean of the center box (30–70% × 30–70% of the frame, expressed as fractions) and the mean of the surrounding ring (10–90% × 10–90% minus the center). If `center_mean / ring_mean >= 2.0`, return `"modal"`. The ring mean is clamped to a minimum of 1.0 to guard against an all-black image causing division by zero.

4. **Default** — return `"board"`.

All geometry is expressed as fractions of `image.shape[:2]`. No pixel values are hard-coded. The same function works at any resolution.

### What the pipeline does with the gate result

`run_pipeline` runs the gate first. On a `"modal"` or `"menu"` frame, it returns immediately with `cards=[]` and does not invoke the localizer at all. On a `"board"` frame it proceeds. The `frame_state` field in `GameStateSnapshot` carries the gate result, so the API consumer always knows which class the frame received.

---

## Hands-on: running the gate

`utils/python/run_pipeline.py` runs the full cascade (gate → localize → identify) on sampled PNGs. The `frame_state` column in its output is the gate result.

```
# Windows — run on all Sample1 frames with overlay images
.venv\Scripts\python.exe utils/python/run_pipeline.py --overlay

# Run a single frame to see the modal gate fire
.venv\Scripts\python.exe utils/python/run_pipeline.py --frames dataset/frames/Sample1/Sample1_000_t00115s.png

# Gate + localizer only (skip gallery build — faster)
.venv\Scripts\python.exe utils/python/run_pipeline.py --no-gallery --overlay
```

Expected output for a modal frame:
```
Sample1_000_t00115s                 state=modal    cards=0
```

Expected output for a board frame:
```
Sample1_003_t00460s                 state=board    cards=8  Wretch@0.65 ...
```

The overlay PNGs (written to `dataset/pipeline-out/` when `--overlay` is passed) show a coloured `frame_state=board` or `frame_state=modal` banner in the top-left corner. This is the same `draw_overlay` function in `utils/python/run_pipeline.py`.

---

## What is not yet built

**Stride decode and perceptual-hash deduplication** (`select_frames`) are designed but not yet implemented in `src/`. The pipeline currently operates on a pre-sampled set of PNGs in `dataset/frames/`. The owed work is:

- Read the video's real frame rate from `ffprobe` metadata (never from a constant).
- Decode at a configurable stride (default: 1 frame / 2 fps equivalent).
- Compute a pHash for each decoded frame and compare against the last kept frame's hash by Hamming distance; discard near-duplicates (Hamming distance ≤ 8 out of 64).
- Pass survivors to the gate.

When that step is built, it will live in `utils/python/` per Rule 1's promotion rule, and it will be pointed at by the hands-on section of this module.

---

## Failure modes

**The gap could close.** The 3× margin between board (max 1.11) and modal (min 3.10) is reassuring on the labeled set, but it rests on 7 frames from two videos. A modal with a very small bright panel — or a board frame with an unusually bright center artifact — could approach the 2.0 threshold. `Sample1_000` already scores the lowest of the three full-modal frames at 3.099, suggesting the gap's floor is not infinitely far from 2.0. Adding more labeled frames from diverse round types will either confirm the gap or reveal it needs adjustment.

**Partial modals are a policy decision.** Calling `Sample1_006` "board" because peripheral cards are still visible is the right production choice, but it means the localizer runs on a frame that is partially occluded. It will find some cards and miss others. Consumers of `frame_state` should not interpret `"board"` as "all cards are visible" — only as "the localizer should run."

**The menu gate is barely validated.** The 160/255 full-frame brightness threshold has not been tested against menu or loading screen frames from these videos; the labeled set contained none. This gate catches obviously light screens but could miss a dark-themed loading screen or be tripped by an unusually bright board frame at game start.

**Perceptual-hash dedup is owed.** Until stride decode + deduplication is built, the pipeline processes every frame in `dataset/frames/` regardless of whether it is a duplicate of the previous one. This does not affect correctness (the gate still fires correctly on duplicates) but it does mean the identification stage processes redundant frames and the overall throughput is lower than it needs to be.

---

## Further reading

From `research/RESEARCH.md`, "Frame selection / keyframe extraction from long gameplay video — 2026-06-21":

- *Scene Detection Policies and Keyframe Extraction Strategies for Large-Scale Video Analysis* (2025) — https://arxiv.org/html/2506.00667v1 — the duration-aware policy (fixed-interval for long footage, ~2 fps decode, ~5 equidistant frames/segment); production-scale evidence at 600k+ hours on commodity 8-core CPUs.
- *PySceneDetect 0.7 — Detectors API* (official docs) — https://www.scenedetect.com/docs/latest/api/detectors.html — `ContentDetector` (HSV frame-diff), `ThresholdDetector` (fades to black), `HashDetector`; the note that downscaling speeds processing approximately 4× per integer increment while frame-skip is actively discouraged.
- *Comparative Evaluation of Perceptual Hashing and Deep Embedding Methods for Robust and Efficient Image Deduplication*, MDPI Electronics 15(7):1493 (2025) — https://www.mdpi.com/2079-9292/15/7/1493 — the standard result that pHash / dHash at Hamming threshold ≤ 8 is sufficient for stable screen-capture deduplication; deep embeddings only justify their cost under aggressive transforms that do not occur here.
