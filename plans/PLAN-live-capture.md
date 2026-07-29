# PLAN — live-capture

**Goal:** turn the two live-frame evals (`collect_01`/`eval_01`, `collect_02`/`eval_02`, both 2026-07-29) into shipped pipeline fixes, ranked by the evals' own "updated ranked fix list."

**Status:** wave 1 (this pass) implements fix 1 (HUD/objective-text false positive) and fix 3 (classical+embedding ensemble) fully; prototypes fix 2 (gate detection) in `scrap_scripts/` only; defers fix 4 (embedding margin recalibration).

## Context

Two live-capture sessions (`DEV-LOG.md`, both dated 2026-07-29) ran the shipped pipeline against real, live-captured Demon Bluff frames for the first time (everything before that was tuned against the two recorded sample videos):

- **`collect_01`** — 48 frames, unsupported window aspect ratio (1.53:1 raw), a tutorial popup partially off-screen. First real signal that the pipeline has precision problems on live frames despite the sample-video localizer spike being 8/8 and 9/9.
- **`collect_02`** — 99 frames, the game's actual supported borderless 1280×720 (16:9) geometry, 16 distinct roles across a much more content-diverse session (7 roles in `collect_01`). This isolated "geometry" from "content diversity" as confounds and confirmed the dominant failure modes are **not** aspect-ratio artifacts — they reproduce, worse, at the correct geometry.

`collect_02`'s DEV-LOG entry gives the updated ranked fix list this plan implements:

1. HUD mask — widen to cover the ~20% text block; add robustness to the Kill-Mode red-tint variant.
2. Gate detection for dialog/win/reward/menu/tutorial screens (only deck overlays currently reliable).
3. Classical+embedding ensemble, motivated by the session's complementarity finding.
4. Embedding margin recalibration (still open from `collect_01`, still untouched by labeled live data).
5. Dark-menu signal (a subset of 2, kept separate — brightness heuristic root cause, not gate coverage).

## What this wave implements

### Fix 1 — HUD/objective-text false positive (`src/dbcv/localize.py`, `src/dbcv/frame_state.py`, `src/dbcv/pipeline.py`)

Evidence-gathering (this session, both `eval_01` and `eval_02` `summary.json`, plus direct pixel measurement on several `collect_01`/`collect_02` frames of different objective-text lengths and villages):

- The recurring "Hunter@0.42–0.50" false positive is the objective-text block ("Find and Execute N Evil Character(s)" / "(N Minions and N Demon)" / "Evils killed: X/Y" / "Village: X/7" / "Ascension: N" / "Score: NNNN"), which sits **top-left**, not full-width. Measured pixel extent across sampled frames (1280×720 and the unsupported `collect_01` geometry alike): x from ~0.02 to ~0.25 of frame width, y from ~0.06 to ~0.25 of frame height. The existing top strip only masked the top 9% of height full-width, and the existing left panel only masked the left 13% of width — the text block is both taller (up to ~25% height) *and* wider (up to ~25% width) than either existing zone alone, so it leaked through both.
- **Rejected approach: widening the full-width top band to 20%.** Verified harmful — the top-center card slot (e.g. `collect_02/050.png` card #7) has its top edge as high as y≈0.076 at x≈0.48–0.53, well inside a naive 0–20%-height full-width band. A full-width widen would blind the localizer to that slot.
- **Chosen fix:** a new **left-only** HUD zone, `(0.00, 0.00, 0.27, 0.27)` (x, y, w, h fractions), added to `HUD_ZONES` and to the Stage-1 pixel-zeroing block. Checked against every "confident real card" bbox in both evals (confidence ≥ 0.5, or any embedding non-unknown hit): the minimum x for any real card in the left column is ≈0.29, comfortably clear of the zone's x=0.27 right edge.
- **Second, independently-discovered HUD leak (not in the original brief, found while measuring the first one):** the top-right "revealed evils" thumbnail strip (small portrait badges of already-executed/found evil characters, e.g. `collect_02/050.png`'s three badges after multiple evils are found) is genuinely-matching character art in a HUD summary badge, not board content — it produced the **second-largest** false-positive cluster in `eval_02` (55 confident hits, `x≈0.876–0.921, y≈0.18, conf up to 0.94` — high confidence because it IS a small copy of the real character's face, just not a board card). Added a second new HUD zone, `(0.86, 0.00, 0.14, 0.36)`, comfortably clear of the confirmed real-card right edge (~0.74–0.76 in every non-tinted frame checked).
- **"Poisoner@0.42–0.44" recurring FP — investigated, found NOT a fixed background element.** Traced its bbox coordinates across `eval_02`'s `full_classical` JSONs; every recurring position corresponds to a real, moving board-card slot (visually confirmed against `collect_02/018.png`, `010.png`, `043.png` — a revealed **Hunter** card sits exactly at the reported box). The classical HSV matcher has a systematic Hunter→Poisoner confusion (their reference art evidently shares palette structure). Cross-checked `full_embedding` at the same frame/bbox: embedding correctly says `Hunter` (e.g. `collect_02/018.png`: classical `Poisoner@0.47`, embedding `Hunter@0.20`). **Decision: no HUD zone or saturation/size guard added for this** — it is a real card and a zone would create false negatives on genuine board content. It is left as a known classical-identifier weakness, and is exactly the kind of case the Fix-3 ensemble should help with (see disagreement-handling below) — though on raw confidence alone the ensemble would pick classical's wrong answer, which is why disagreement resolves to abstain, not "pick the higher number" (see Fix 3).
- **Kill-Mode red-tint:** confirmed as a real, distinct failure source (frame `collect_02/043.png` and neighbors) — global red-shift causes (a) a spurious "card" box around the center demon-altar decoration (a deeper localizer-mask issue, out of scope this wave) and (b) reduced identification reliability. Implemented the "at minimum" option from the brief: a cheap global tint measurement, `frame_state.measure_red_shift()` = `mean(R − max(G,B))` over the whole frame. Measured on all 147 sampled frames across both evals: tinted frames score 14.4–32.2, normal frames score −16.9 to +0.5 — a clean, wide gap. `is_red_tint()` thresholds at 10.0 (centered in the gap). Wired into `pipeline.run_pipeline` as an injectable `tint_fn`: when active, per-card confidence is discounted (`×0.7`) and any card whose discounted confidence falls below `tint_confidence_floor` (default 0.5, matching `Settings.confidence_threshold`'s documented-but-previously-unused intent) is downgraded to `("unknown", "unknown", <discounted_confidence>)`. This does not fix the altar-shaped localizer FP (still open — noted, not fixed this wave) but does raise the abstention bar on identification while tinted, and the discounted-not-dropped confidence is exactly the right shape of input for the temporal fusion layer's margin-weighted vote (`PLAN-temporal-assembly.md`) once that lands.

### Fix 3 — classical+embedding ensemble (`src/dbcv/identify.py`)

Evidence (this session, `eval_02` `full_classical`/`full_embedding`, IoU-matched per box, 917 matched card-slot pairs):

| outcome | count |
|---|---|
| agree (same identity, both non-unknown) | 78 |
| classical abstains, embedding answers | 235 |
| embedding abstains, classical answers | 99 |
| disagree (different identity, both non-unknown) | 20 |
| both abstain | 485 |

- **Agreement → boost.** `confidence = min(1.0, max(classical_conf, embedding_conf) + 0.15)`, source tag `"agree"`.
- **One abstains → adopt the other**, with its own (un-rescaled) confidence, source tag `"classical_only"` / `"embedding_only"`. This is the single biggest lever — 334 of 917 matched pairs (36%) are resolved this way, versus 78 by agreement.
- **Disagreement → abstain, not "prefer higher confidence."** All 20 disagreement examples in `eval_02` show classical's raw confidence (0.41–0.47, its histogram-correlation scale) numerically exceeding embedding's raw confidence (0.14–0.22, its top1-top2-margin scale) — but visual ground-truth on the Hunter/Poisoner cases (see Fix 1 above) confirms **classical is wrong and embedding is right** in exactly this recurring pattern. Comparing raw confidence across two identifiers with structurally different, uncalibrated scales is unsound with the data on hand — there is no labeled live-crop set to fit a fair normalization (same reason Fix 4 is deferred: don't calibrate on the same 99 frames being evaluated). Chosen: abstain on disagreement, source tag `"disagree_abstain"`, documented with the Hunter/Poisoner evidence. A future wave with labeled live crops could replace this with a calibrated normalized-confidence comparison.
- **Both abstain → unknown**, source tag `"both_unknown"`.
- Implementation: `combine_identifications(classical, embedding) -> (identity, role_class, confidence, source)` is the tested core (source-tagged, 4-tuple). `make_ensemble_identifier(classical_fn, embedding_fn)` adapts it to the existing 3-tuple pipeline/identifier contract (drops `source` — the schema (`CardRead`) has no provenance field yet; adding one is a schema-version bump left for a future wave, noted here rather than done as a drive-by change alongside two other fixes). Both `classify_crop`/`classify_crop_embedding` are untouched — the ensemble is a pure composition layer over the existing two identifiers.
- Wired as an **opt-in** identifier choice: new `Settings.identifier: Literal["embedding", "classical", "ensemble"]` (default `"embedding"`, unchanged from today) in `src/dbcv/config.py`; `api.py`'s lifespan builds the ensemble identifier when `DBCV_IDENTIFIER=ensemble` is set, falling back to embedding-only if the ONNX model is unavailable (same fallback pattern already used for the plain embedding identifier).

### Measured before/after (re-run on `collect_02`, `scrap_scripts/python/10_eval_collect02_after_fixes.py` → `out/eval_03/`, vs the original `out/eval_02/`)

| metric (full_classical arm, 93 board frames) | before (eval_02) | after (eval_03) |
|---|---|---|
| frames with the exact top-left HUD-text FP box | 24.7% | **0.0%** |
| frames with ANY "Hunter" hit (real cards + FP combined) | 82.8% | 23.7% |
| boxes / board frame | 9.86 | 8.94 |
| classical non-unknown rate | 36.3% | 25.9%* |

*The classical non-unknown rate dropping is the FP disappearing, not a
regression: the removed HUD-text hits were themselves always "non-unknown"
(Hunter@0.42-0.50 is a confident, wrong answer, never an abstention), so
removing ~86 spurious boxes mechanically pulls the raw non-unknown percentage
down while removing a pure false-positive source. Visual overlay check
(`eval_03/full_classical/050_overlay.png`, `018_overlay.png` — both viewed
directly) confirms recall on real cards is intact, including the exact
top-center-card-with-a-high-top-edge case the fix was designed not to break,
and that the top-right badge strip is now correctly unboxed too.

**Ensemble (Fix 3) non-unknown rate:** 34.1% (283/831) on `eval_03`, ahead of
classical alone (25.9%) and well ahead of embedding alone (20.6%) on the
same fixed-localizer boxes.

**Spot-check accuracy on the 6 known-revealed frames** (023, 039, 055, 071,
085, 097 — ground truth transcribed from `actions.md`, distinct-role-set
match, not per-slot): classical **17/34** (50.0%), embedding **18/34**
(52.9%), ensemble **22/34 (64.7%)** — the ensemble beats both alone.
One instructive loss: on frame 097, classical said `Bombardier`/`Poisoner`
(both wrong) and embedding correctly said `Slayer` for a different box, but
the ensemble abstained on the box where they disagreed (rather than trusting
embedding) and only adopted classical's other wrong `Poisoner` guess where
classical answered and embedding abstained — a real example of the
disagree-abstain rule trading away one correct answer to avoid rubber-stamping
a wrong one, exactly the conservative behavior it was designed for, not a bug.

### Fix 2 — gate detection prototype (scrap only, `scrap_scripts/python/09_gate_dialog_probe.py`)

Explores which cheap signal best separates `collect_02`'s 22 ground-truth non-board frames (win dialogs 014/028/044/061/075/090, objective dialogs 002/094, menus 000/001, tutorials 035/050, milestone 092, scoreboard 093, reward 015/029/045/076/091, skin 089, deck 003/004) from the 77 board frames. **Not wired into `src/` this wave** — report only, per the brief.

Six candidate signals measured (reusing `frame_state.py`'s center/ring geometry plus new ones): the existing gate ratio, the raw ring-mean ("dimmed background" statistic), whole-frame mean brightness, center-box brightness std (texture/flatness), localizer box count, and Canny edge density.

**Result: none of the six is a reliable single-signal gate on its own for most categories** — pooled board-vs-non-board ranges overlap on every signal. Per-category, only 4 of 9 non-board categories are cleanly separable by any single signal from this set: **deck** (ring_mean, gap 7.9), **milestone** (gate_ratio, gap 0.24), **scoreboard** (center_std, gap 2.4), **skin** (center_std, gap 5.4). The five categories DEV-LOG already flagged as unreliable — **win_dialog, objective_dialog, menu, reward, tutorial** — remain unseparated by any of these six signals (best available gap is negative/no-separation for all five, matching DEV-LOG's "0/6 win dialogs, 0/2 objective dialogs, 0/2 menus" finding almost exactly). This is consistent with — and quantifies — the DEV-LOG finding rather than fixing it: a usable gate for these five categories needs either a different signal family (e.g. actual panel/rectangle detection, template matching on the dialog chrome, or a tiny learned classifier) or a combination of signals, not a single cheap statistic. Left as a concrete, scoped starting point for a future gate-improvement pass. Full numeric report in the script's own output (re-run it to reproduce).

### Fix 4 — embedding margin recalibration (deferred)

`_EMBED_MARGIN_THRESHOLD = 0.12` in `src/dbcv/identify.py` remains untouched. It needs a labeled live-crop set (a round-2-style labeling pass over live frames, analogous to `PLAN-round2-dataset.md`'s sample-video mining) to calibrate honestly. Tuning it against the same 99 `collect_02` frames used to *evaluate* this wave's fixes would be circular — the project's own standing rule from the sample-video round-2 plan (don't fit calibration and evaluation on the same set). Left for a future wave once live crops are labeled.

## Tasks

- [x] Measure HUD-text-block extent from `collect_01`/`collect_02` frames; add two new `HUD_ZONES` entries + Stage-1 zeroing in `localize.py`.
- [x] Investigate the "Poisoner@0.42-0.44" recurring FP; confirm not a fixed background element; document, no zone/guard added.
- [x] Implement `measure_red_shift`/`is_red_tint` in `frame_state.py`; wire tint-aware confidence discount into `pipeline.run_pipeline`.
- [x] Implement `combine_identifications` + `make_ensemble_identifier` in `identify.py`.
- [x] Wire `Settings.identifier` (config.py) + lifespan selection (api.py).
- [x] Tests: synthetic-frame HUD-zone tests (`tests/test_localize.py`), red-tint tests (`tests/test_frame_state.py`), ensemble combiner tests with stub outputs (`tests/test_identify.py`), pipeline tint-discount test (`tests/test_pipeline.py`).
- [x] CodeDocs sync for every touched `src/` file.
- [x] Prototype fix 2 in `scrap_scripts/python/09_gate_dialog_probe.py` (report only).
- [x] Re-run the eval harness (`scrap_scripts/python/08_eval_collect02.py`, extended with a third `full_ensemble` arm) on `collect_02`; record before/after numbers in `DEV-LOG.md`.
- [x] DEV-LOG entry.
- [x] Commit.

## Promotion question for the scrap capture/eval scripts (pending, not resolved this wave)

`scrap_scripts/python/01`–`10` (capture, click, launch, frame-grab, title-bar-crop, three eval runners, and the gate-signal probe) are still gitignored scrap. None of them are yet depended on by anything other than a human/agent at the CLI (Rule 1's promotion trigger), so none are promoted this wave. Worth revisiting once a live-capture CLI workflow stabilizes enough to be worth a `utils/python/` home + `utils/README.md` entry — flagged here for the next session, not resolved.

## Open items carried forward

- The Kill-Mode red-tint localizer false positive (a spurious box around the center demon-altar decoration, caused by the red HSV mask firing on non-card red content) is identified but not fixed this wave — `is_red_tint` only raises the *identification* abstention bar, not the *localization* one.
- Fix 2's signal is a prototype/report only; wiring a real Stage-0 gate improvement is future work.
- Fix 4 needs a labeled live-crop round.
- The ensemble's `source` provenance tag is available from `combine_identifications` but is dropped by `make_ensemble_identifier`'s 3-tuple pipeline adapter — surfacing it end-to-end needs a `CardRead` schema bump, left for later.
