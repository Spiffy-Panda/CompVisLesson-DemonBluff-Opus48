# CodeDocs/sources/dbcv/frame_select.md

**Status:** active — Stage 0 frame selector (stride-decode + pHash dedup),
implemented and tested. Completes the part of Stage 0 the plan marked "owed"
(the gate was already done in `frame_state.py`).

**Purpose:** A cheap, classical, CPU-only **dev/batch** cascade that picks the
few frames worth analysing from a long capture, so nothing downstream ever
processes raw full video:

```
low fixed-stride decode (stride from media's REAL fps)
   → perceptual-hash (dHash) dedup vs last kept (Hamming ≤ threshold)
   → board / menu / modal gate (reuses classify_frame_state)
```

**Runtime boundary:** this is **dev/batch only** — it decodes video with
OpenCV, which is fine offline but must never be wired into the REST service
(the runtime budget forbids whole-video decode per request). The REST path
receives a single already-selected frame.

**Who uses it:**
- `tests/test_frame_select.py` — unit-tests the hash/dedup/stride logic on
  synthetic arrays + already-extracted PNGs, and the full cascade on a tiny
  synthetic `cv2.VideoWriter` clip (the real ~370 MB samples are never opened).
- *(Intended)* a `utils/python/` batch runner for dataset building — **not yet
  promoted** (see FLAGS in the handoff). `scrap_scripts/python/03_sample_frames.py`
  is the uniform-sampling predecessor this module supersedes for selection.

**Reuses:** `dbcv.frame_state.classify_frame_state` (the Stage 0 gate) and its
`FrameState` type — the selector does not reinvent the gate.

---

## Key signatures (with line numbers)

### `MediaInfo` dataclass — line ~104
```python
@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    fps: float
    fps_source: str  # "opencv" | "ffprobe" | "fallback"
```
Resolution + frame rate **read from the media at runtime** (never baked).
`fps_source` records provenance for debugging/lessons.

### `SelectedFrame` NamedTuple — line ~120
```python
class SelectedFrame(NamedTuple):
    frame_index: int     # 0-based index into the decoded stream
    timestamp_s: float   # presentation timestamp (index / media_fps)
    state: FrameState    # "board" | "modal" | "menu"
    dhash: int           # 64-bit difference hash
```
The selector's output record. Pixels are intentionally **not** stored on the
record so a whole-video pass stays cheap in memory.

### `read_media_info(video_path, cap=None) -> MediaInfo` — line ~170
Reads width/height from `CAP_PROP_FRAME_WIDTH/HEIGHT`; fps from OpenCV
`CAP_PROP_FPS`, falling back to **ffprobe** (`avg_frame_rate`/`r_frame_rate`,
the `02_probe_video_meta.py` pattern), then a documented fallback. Accepts an
already-open `VideoCapture` to avoid reopening.

### `stride_for_fps(media_fps, target_fps=1.5) -> int` — line ~215
`max(1, round(media_fps / target_fps))`. **The divisor is the media's measured
fps — 30 is never assumed.** Raises `ValueError` on non-positive inputs.

### `dhash(image, hash_w=8, hash_h=8) -> int` — line ~243
Difference hash: grayscale → resize to `(9×8)` → compare adjacent columns →
pack 64 bits (MSB-first, row-major). Encodes the *sign* of the horizontal
brightness gradient (stable under global brightness/contrast shifts).
Resolution-agnostic: only the fixed internal hash grid is used.

### `hamming(a, b) -> int` — line ~285
Popcount of the XOR (`(a ^ b).bit_count()`).

### `iter_strided_frames(video_path, target_fps=1.5) -> Iterator[(idx, ts, bgr)]` — line ~295
Opens the video once, derives the stride from the media's real fps, yields
every `stride`-th decoded frame. Sequential decode (read-then-skip), no
per-frame random seek. Timestamp = `idx / media_fps`.

### `dedup_hashes(hashes, hamming_threshold=8) -> list[int]` — line ~335
Pure helper: greedy "keep-against-last-**kept**" → indices to keep. Comparing
against the last *kept* (not the previous) frame collapses long runs of
slowly-drifting near-duplicates into one kept frame.

### `iter_selected_frames(video_path, target_fps=1.5, hamming_threshold=8, board_only=True) -> Iterator[(SelectedFrame, bgr)]` — line ~360
Full cascade in one streaming pass (constant memory: only the last kept hash is
held). **Order: dedup before gate**, so a run of identical board frames costs
one `classify_frame_state` call. Non-board frames still update the dedup anchor
so a modal→board transition is not masked. Yields pixels alongside the record.

### `select_frames(video_path, target_fps=1.5, hamming_threshold=8, board_only=True) -> list[SelectedFrame]` — line ~400
Batch entry point. Returns one `SelectedFrame` per kept frame (no pixels — use
`iter_selected_frames` for those). Heavy decoding lives here, off any REST path.

---

## Tuning constants (module level, lines ~80–95; no pixel value hard-coded)

| Constant | Value | Purpose |
|----------|-------|---------|
| `_DEFAULT_TARGET_FPS` | 1.5 | Target decode rate (mid of research 1–2 fps band) |
| `_DEFAULT_HAMMING_THRESHOLD` | 8 | Near-dup boundary over the 64-bit dHash (research d_H ≤ ~8) |
| `_HASH_H` / `_HASH_W` | 8 / 8 | dHash grid (→ 64 bits); internal, **not** a frame size |
| `_FALLBACK_FPS` | 30.0 | Last-resort fps if OpenCV+ffprobe both fail (flagged via `fps_source`) |
| `_RAW_VIDEO_DIR` | `dataset/raw-video/` | Repo-anchored default location |

---

## Design choices & research grounding

- Follows the **"Frame selection / keyframe extraction"** entry in
  `research/RESEARCH.md` (2026-06-21): low fixed-stride decode at fps read from
  the media; perceptual-hash dedup by Hamming/popcount (near-dup d_H ≤ ~8);
  cheap board gate; PySceneDetect/deep methods reserved for dev-only.
- **dHash over pHash (DCT):** both are research-endorsed; dHash needs no DCT (so
  `cv2`+`numpy` only, no `imagehash` dependency), is robust on stable screen
  captures (encodes gradient sign → tolerant of fades/tooltip dimming), and the
  threshold is one re-tunable constant on an art swap.
- **No resolution baked:** dimensions come from the decoded media; the hash
  resizes to a fixed internal grid (proven by `test_dhash_resolution_independent`
  and `test_read_media_info_reads_resolution_and_fps`).
- **Stride from real fps:** `stride_for_fps` divides by the *measured* rate
  (proven by `test_stride_for_fps`, incl. 24/60 fps cases — never 30-assumed).

## Testing without opening the samples (hard constraint)

`tests/test_frame_select.py` never touches `dataset/raw-video/`:
- decode/stride/cascade exercised on a **tiny synthetic clip** written with
  `cv2.VideoWriter` (~12 small frames at a known fps in `tmp_path`);
- hash/dedup logic exercised on **synthetic numpy arrays** and on the
  already-extracted **PNGs** in `dataset/frames/Sample1/`.
