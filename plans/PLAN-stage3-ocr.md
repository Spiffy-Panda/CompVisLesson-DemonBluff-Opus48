# PLAN — stage3-ocr

**Goal:** read on-card and HUD text (role names, digit counts, state markers, misc) via a tiny closed-vocabulary recognizer with a mobile-OCR fallback, per the lightweight-OCR entry in `research/RESEARCH.md` (2026-06-21).

**Status:** wave-1 spec filled 2026-07-28 from direct inspection of the extracted frames (`dataset/frames/`, 1280×720 in these samples — dimensions always read from media, never assumed) and the Stage-1 localizer outputs (`dataset/pipeline-out/*.json` + `*_overlay.png`). Frames inspected: Sample1 000/003/007/010/012/015/020/023, Sample2 000/005/010/015/020/023, plus 4×-upscaled crops of plates, HUD counters, and tooltips (scrap `13_ocr_surface_crops.py`).

## Tasks

- [x] Text-surface catalog from extracted frames (regions, resolution-relative) *(2026-07-28 — §1 below; 14 surfaces, each with an owner)*
- [x] Closed vocabulary inventory (roster names + digits + misc tokens) *(2026-07-28 — §2 below; ~84 tokens)*
- [x] Rendered-crop training-set design (font approach, augmentation) *(2026-07-28 — §3 below; font ID is provisional pending mined-crop validation)*
- [ ] Recognizer build + training (wave 2)
- [ ] PaddleOCR-mobile ONNX fallback (wave 2)
- [ ] Stage 2 name-label cross-check integration (wave 2)
- [ ] *(new, wave 2)* Digit reader must be a **variable-length string head, not a 0–99 classifier** — Score reaches 4 digits (820 / 1180 / 1400 / 2510 observed); charset `{0-9, /, #}`
- [ ] *(new, wave 2)* `<Corrupted>` two-line plate variant → `Readings.state` mapping (+ skull-overlay "dead" marker, icon not text — decide owner with Stage 4)
- [ ] *(new, wave 2)* Plate-crop padding: the Stage-1 bbox sometimes clips the name plate (e.g. Scout card, `Sample1_003` JSON, h=0.14 vs typical ~0.17) — pad the read region below the bbox before cropping
- [ ] *(new, wave 2)* Bubble / tooltip region detection (near-white rounded-rect and dark-navy panel locate) to feed the fallback OCR — these are free-form and cannot be fixed-position
- [ ] *(new, wave 2)* Font validation against real mined crops once `dataset/crops/` lands (round-2 mining, in flight): render candidate fonts side-by-side vs mined plates, then lock the renderer font
- [x] *(flag, resolved)* **Roster gap "Counsellor":** not a missing role — it is the **pre-v0.390a name of `Chancellor`** (renamed 2025-10-28, wiki verified 2026-07-29). The sample footage predates that patch, so it renders the old string. **Vocabulary action (wave 2):** add `COUNSELLOR` as a 45th name string aliased to `Chancellor`; it *can* appear on a card plate in our footage. See `knowledge-base/wiki/townees/ROSTER.md` § Footage-version drift.
- [ ] *(new, wave 2)* **Footage-version drift, general:** the videos are ≤ v0.389 (Oct 2025); current wiki is v0.762a. `Rambler` (v0.610) and `Investigator` (v0.760e) **cannot appear** in our footage, and v0.730 overhauled character-type/alignment icons — do not assume mined crops match current wiki art or current UI chrome.

---

## 1 — Text-surface catalog

Conventions: **card-relative** regions are fractions of a card `bbox_rel` (Stage-1 output; `(x, y, w, h)` fractions of frame, origin top-left — `CodeDocs/sources/dbcv/localize.md`). **Frame-relative** regions are fractions of frame w/h and are *search regions to seed a local detector*, never hard pixel crops. Owner legend: **CV** = closed-vocab recognizer, **FB** = PaddleOCR-mobile fallback, **SKIP** = not read (icon / out of scope).

### On-card (card-relative)

| # | Surface | Where | Visual properties | Size | Owner |
|---|---------|-------|-------------------|------|-------|
| 1 | **Name plate** | bottom band of card: full width, y ∈ [~0.76, 1.0] of bbox; pad read-crop to x ± 0.10 w, y up to 1.15 h (plate can hang past the localizer bbox) | ALL-CAPS role name, heavy rounded geometric sans, wide tracking. Plate/ink varies by class: white-on-purple (villager), dark-olive-on-gold (outcast), orange-on-dark-red (minion/demon, low contrast) | glyph x-height ~8–10 px at 720p → mandatory 3–4× upscale | **CV** (44 name strings) |
| 2 | **Corrupted plate variant** | same band, two lines: tiny role name on top (orange caps), `<Corrupted>` below (larger, gray-lavender, literal angle brackets) | seen on DREAMER, HUNTER, ENLIGHTENED | name line is *smaller* than normal plates — hardest on-card read | **CV** (name + state token) |
| 3 | **Slot number** `# N` | centered band above the card: y ∈ [−0.32 h, −0.02 h], width ~0.6 w | white heavy rounded sans, `#` + space + 1–2 digits, N ∈ 1–10 (10 observed) | ~12 px glyphs | **CV** (digits + `#`) |
| 4 | Mark flags (green/orange/red chevrons on card edges) | right/left card edge | **no text** — player-placed marks (bottom-right legend "Mark 1/2/3, Remove marks 4") | — | **SKIP** (colour → possible `Readings.state`, Stage 4 call) |
| 5 | Skull overlay (dead), purple selection arrows, cursor | over card art | icons, not text | — | **SKIP** |

### HUD (frame-relative search regions)

| # | Surface | Where | Visual properties | Owner |
|---|---------|-------|-------------------|-------|
| 6 | **Objective banner** | top-left, x ≈ [0.01, 0.25], y ≈ [0.03, 0.12] | line 1 "Find and Execute N Evil Characters" white mixed-case rounded sans; line 2 "(N Minions and N Demons)" with class words colour-coded (red/orange) | **CV** (fixed phrase + digit slots) |
| 7 | **Left counter stack** | x ≈ [0, 0.14], rows y ≈ 0.15–0.45, row h ≈ 0.055; **two modes**: in-round (`Evils killed: N/M`, `Village: N / M`, `Ascension: N`, `Score: N`) and meta (`Current Village: N`, `Score: N`, `All Saved Villages: N`) | fixed label + colon + digits; value colour-coded (red = bad, green = score); mixed case rounded sans on dark plates | **CV** (labels fixed-vocab, values digit-string) |
| 8 | **Deck-composition strip** | top-right, x ≈ [0.845, 0.93], y ≈ [0.055, 0.11] | 4 icons (villager/outcast/minion/demon) each with one white digit 0–9 below | **CV** (single digits) |
| 9 | **Health disc** | bottom-left; locate the red disc by HSV in x ≈ [0.06, 0.19], y ≈ [0.74, 0.93], then crop | `N/10`, big outlined rounded digits, disc drains as pie-fill; hearts row below is icon-only | **CV** (digit-string with `/`) |

### Modal / overlay (frame-relative or shape-located)

| # | Surface | Where | Visual properties | Owner |
|---|---------|-------|-------------------|-------|
| 10 | Modal titles + verdicts | centered, y ≈ [0.13, 0.20] (`CURRENT DECK`, `NEW CHARACTERS UNLOCKED`); centered panel for `Village is safe!` / `All Evil characters have been executed!` | large white/green caps or mixed case | **CV** (small fixed phrase set); unseen titles → FB |
| 11 | Deck-composition line | centered, y ≈ [0.215, 0.25], modal only | `Villagers 5, Outcasts 1, Minions 2, Demons 1` — colour-coded class words + digits | **CV** |
| 12 | Buttons | bottom-center, y ≈ [0.86, 0.95] (`Next`, `Close`); `More info` label near tooltips | white rounded sans on red/dark plate | **CV** (fixed words) |
| 13 | **Claim bubbles** | white rounded-rects adjacent to cards — locate by near-white fill + rounded contour, not position | dark rounded sans, mixed case, multi-line, **free-form** ("#2 is Good", "Baa is 2 cards away from closest Evil", "I am the original Baker", "#9 could be: Lilis"); mention roles incl. the legacy name "Counsellor" (= `Chancellor` pre-v0.390a) | **FB** (flagship fallback surface) |
| 14 | Ability tooltips + class tags | dark-navy panels near hovered card; small tag plates below (`Villager`/`Good`, `Minion`/`Evil`) | free-form sentence text with colour-coded keywords; tags are fixed words | panels → **FB**; tags → **CV** |

Out of scope entirely: streamer watermark script signature bottom-center (third-party overlay — never read, never emit), "Barely a scratch!" transient combat text (FB-able but zero game-state value), mark-legend keybind text (static config UI).

## 2 — Closed vocabulary

**Role-name strings — 44** (43 recognition classes; `Minion` and `Twin Minion` are distinct *plate strings* even though they collapse to one recognition class downstream — `ROSTER.md`):

Alchemist · Architect · Baker · Bard · Bishop · Confessor · Dreamer · Druid · Empress · Enlightened · Fortune Teller · Gemcrafter · Hunter · Investigator · Jester · Judge · Knight · Knitter · Lover · Medium · Oracle · Poet · Scout · Slayer · Witness · Chancellor · Minion · Poisoner · Puppet · Puppeteer · Shaman · Twin Minion · Werewolf · Witch · Bombardier · Doppelganger · Drunk · Lycanthrope · Plague Doctor · Rambler · Wretch · Baa · Lilis · Pooka

(Plates render these ALL-CAPS. `PUPPET`, `LILIS`, `POOKA`, `BAA` confirmed on-screen in the samples.)

**Digit + punctuation glyphs — 12:** `0–9`, `/`, `#`. Observed composites: slot `#1`–`#10`; health `0–10`/`10`; objective counts 0–3; deck-strip digits 0–9; `Village N / M` with M ≤ 9 seen; **Score 0–2510 observed (4 digits)** and `All Saved Villages: 57`, `Current Village: 12` — so numeric reads are **variable-length strings, not a 0–99 class set** (checklist item added).

**Fixed UI tokens — ~28:** `Next` · `Close` · `More info` · `Mark` · `Remove marks` · `CURRENT DECK` · `NEW CHARACTERS UNLOCKED` · `Village is safe!` · `All Evil characters have been executed!` · `Villagers` · `Outcasts` · `Minions` · `Demons` · `Villager` · `Outcast` · `Minion` · `Demon` · `Good` · `Evil` · `<Corrupted>` · `Find and Execute` · `Evil Character(s)` · `Evils killed:` · `Village:` · `Current Village:` · `Ascension:` · `Score:` · `All Saved Villages:`

**Total: ~84 tokens** (44 names + 12 glyphs + ~28 UI tokens). Everything outside this set (claim bubbles, ability tooltips) is free-form by design and owned by the fallback.

**Confusable-glyph warnings (font-specific, verified in crops):** digit `1` renders as a bare vertical stem — indistinguishable from cap `I` (`Ascension: 1` reads as "Ascension: I"); digit `0` ≈ cap `O` (deck line "Demons 0" reads as "Demons O"). Context (label grammar) must disambiguate, which the closed-vocab / template-slot design gets for free — a general OCR would not.

## 3 — Rendered training-set design

**Font strategy.** One rounded geometric sans family serves the whole UI: near-monoline strokes, rounded terminals, bare-stem `1`, `O`-like `0`; plates use a heavy weight, ALL-CAPS, wide tracking; HUD/bubbles a regular weight, mixed case. Closest free matches by eye: **Fredoka (SemiBold/Bold)**, **Baloo 2**, **Quicksand Bold** — Fredoka currently looks closest on the plate caps. **Flag: this is likely a licensed/custom rounded font; do not lock a single match yet.** Hedge = render the synthetic set in *all three* candidates (font becomes an augmentation axis), then validate side-by-side against real mined plates once `dataset/crops/` exists and drop the losers (wave-2 checklist item).

**Compositing.** Render token → ink colour → plate background → frame context:
- Backgrounds: real plate-band crops cut from the 48 frames via the Stage-1 bboxes (available **now**, unlabeled — background harvesting needs no labels), plus flat/gradient fills matching observed plate palettes (purple, gold, dark-red, HUD dark-gray, bubble white).
- Ink/plate colour pairs per §1: white-on-purple, dark-on-gold, orange-on-red (deliberately low contrast — keep it), white-on-dark, dark-on-white.
- Layout jitter: tracking ±20%, baseline ±2 px, horizontal inset jitter; two-line `<Corrupted>` variant rendered explicitly.
- Occluders composited on top at low rate: cursor sprite, sparkle particles, blood splats, skull-overlay edge, neighbouring-card fringe.

**Augmentation** (consistent with the round-1 embedding fine-tune recipe in `utils/python/finetune_embedding.py` — reuse its transform stack where types match): random scale 0.5–1.5× (simulating 720p→1080p sources) with bilinear down-up resample; Gaussian blur σ ≤ 1.5; JPEG quality 40–90 (h264 artifact proxy); brightness/contrast/gamma jitter; mild hue shift; ±2° rotation + slight perspective; 1–2 px translation. **Preprocess identically at train and inference: locate-region → upscale 3–4× → grayscale → contrast-normalize (CLAHE or mean/std) — no global binarization** (research entry prescription).

**Set size.** Name/word head: ~84 classes × ~500–600 renders ≈ **45 k crops**; digit-string head: ~20 k synthetic strings (lengths 1–4, `N/M` and `# N` patterns). Trivially generated; training fits in minutes-to-an-hour on the Titan Xp (and free Colab).

**Round-2 mined crops (`dataset/crops/`, forthcoming — parallel agent; do not depend on it existing yet):** needed only for (a) font validation, (b) an optional real-data fine-tune / calibration split (a few hundred labeled plates), (c) honest held-out evaluation. The synthetic set alone is sufficient to train v1 — this ordering is exactly the cheap-retrain story for art swaps (re-render, retrain, done).

## 4 — Architecture sketch

**Recommendation: fixed-vocab whole-crop classification for words + a tiny CRNN-CTC for digit strings** — two heads, not one CRNN for everything.

- **Word head** — grayscale 32×128 (aspect-preserving pad) → depthwise-separable conv trunk (MobileNetV3-Small-style, cut to ~4 blocks) → GAP → **85 logits** (84 tokens + explicit `other`). ~0.3–1 M params, ~1–4 MB ONNX. *Why classification over CRNN for names:* the vocabulary is closed and whole-word; a classifier is smaller, trains faster, calibrates better, and its failure mode on unseen text is a clean abstain — per-character decoding would instead hallucinate plausible strings, which is precisely what we don't want feeding game state. CRNN's strength (open vocab) is the fallback's job.
- **Digit-string head** — grayscale 32×96 → 2–3 conv blocks → BiGRU(64) → CTC over `{0-9, /, #, blank}`. ~0.5 M params. Handles `# 8`, `0/3`, `10/10`, `2510` at any length — this is the CRNN-shaped sub-problem, kept to a 13-char alphabet.
- **Budget:** ≤ ~1.5 M params combined vs the ≤ 30 M runtime ceiling — ~5 % of budget; sub-ms per crop on CPU via onnxruntime (same serving pattern as the Stage-2 embedder: export ONNX, load once in `lifespan`).
- **Abstention → fallback → `Readings`:** word head abstains on softmax top1−top2 margin below threshold (mirror the Stage-2 margin-gate convention; calibrate on the round-2 labeled split) or on `other`. Abstained/free-form crops route to **PaddleOCR-mobile *recognizer-only* ONNX (~5–11 MB)** — detection stage skipped since our regions come from the localizer/HUD locate step — and the raw string lands in `Readings.text` unvalidated. Recognized names land in `Readings.text` (feeding the Stage-2 identity cross-check), numbers in `Readings.number`, `<Corrupted>` in `Readings.state` (`schema.py` 0.2.0 fields).
- **Retrain on art swap:** re-render the synthetic set with the new font/palette (minutes), retrain both heads (< 1 h Titan Xp / Colab), re-export ONNX; serving layer untouched. No real-data labeling required for v1 parity.

## 5 — Inspection surprises (recorded for wave 2 and the lesson module)

1. **Score breaks the 0–99 assumption** (4 digits observed) — hence the CTC digit head.
2. **`<Corrupted>` two-line plate variant** shrinks the role name and adds a literal angle-bracketed state token — the plan's original "name plate" surface was really two layouts.
3. **Chevron badges carry no text** — they are player-placed marks (keybind legend confirms), so the "counts/numbers on cards" hypothesis is dead: *all* on-card text is the plate + the slot number above.
4. **Bare-stem `1` / `O`-like `0`** make a general OCR structurally error-prone here; closed vocab + label grammar absorbs it.
5. **"Counsellor" is a version artifact, not a missing role** (resolved 2026-07-29) — it is `Chancellor`'s pre-v0.390a name. The find is that **the dataset is pinned to an old build** (≤ v0.389, Oct 2025) while our reference art and wiki are v0.762a. A text surface can therefore date the footage — and a name-vocabulary built from the *current* roster would silently miss a string that is on real cards.
6. **Two HUD modes** for the left counter stack (in-round vs meta) with different label sets.
7. Face-down cards have **no plate at all** — Stage 4 must not treat "no text read" as a failed read on face-down slots.
