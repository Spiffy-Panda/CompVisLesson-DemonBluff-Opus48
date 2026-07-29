# DEV-LOG

Append-only decision log. **Newest entry on top.** Absolute dates. Git commits record *what changed*; this records *why we chose it, what we tried first, and what would surprise the next person*. Write an entry before every commit (Rule 5).

### Entry template

```
## YYYY-MM-DD — <short title>

**Context:** what prompted this.
**Options considered:** A / B / C.
**Choice:** what we picked.
**Why:** the deciding reason(s).
**Notes / risks:** anything the next person should know.
```

---

## 2026-07-29 — Second live eval: supported 16:9 geometry is hygiene, not an accuracy lever — content diversity was

**Context:** The first eval (`collect_01`, below) ran 48 frames at an unsupported window AR, with a tutorial popup rendering partially off-screen — a confound on top of the pipeline's own faults. This session captured a second, larger collection at the now-verified true borderless 1280×720 geometry (launch flags held from last session) and re-ran the eval to separate "geometry fixed" from "pipeline fixed."

**Choice:** New `scrap_scripts/python/collect_02/` session — an overnight scheduled run, 99 frames at true borderless 1280×720. The agent played through clearing Villages 2–7 plus the full Ascension 1 (final score 1104), surfacing **16 distinct roles** (9 new beyond `collect_01`'s 7) but only **2 tutorial frames** this time — a content-diversity-heavy, tutorial-light session, the opposite mix of `collect_01`. New scrap runner `08_eval_collect02.py` sweeps `collect_02` through **2 arms** — {classical, embedding} — no crop arm this time (the title-bar-chrome question was already closed last session: byte-identical raw-vs-cropped gate, and it's moot now that the launch flags ship truly borderless). Outputs land in `scrap_scripts/python/out/eval_02/`.

**Why:** Cheapest way to isolate one variable per session: last time we fixed geometry *and* got a small, tutorial-heavy sample; this time we hold geometry fixed and get a large, role-diverse sample, so any remaining failure mode can't be blamed on aspect ratio.

**Results — geometry confirmed a non-issue:** 16:9 at 1280×720 is table stakes for a valid capture (no more off-screen tutorial rendering), but fixing it did **not** move the dominant failure modes — they are exactly the same shape as `collect_01`'s, just measured more precisely now. **Headline of the session: supported geometry is hygiene, not an accuracy lever — the thing that moved the needle here was content diversity (16 roles seen vs. 7), which is what actually exposed the HUD-mask, gate, and identification gaps at real scale.**

**Results — HUD-text false positive, worse at 720p:** the "Hunter" HUD-text false positive from `collect_01` reproduces and is **worse** at true 720p: 82.8% of board frames vs. 73–80% at the old unsupported AR. Mechanism confirmed geometrically: the current mask covers ~9% of frame height, but the HUD text block occupies ~20% — the mask needs to widen, not just shift. **New this session:** a Kill-Mode red-tint UI variant hallucinates up to **5 Hunters on a single Hunter-less board** (one at confidence 0.93) — a second, higher-confidence route into the same false-positive family, tied to a color-tinted UI state not present in `collect_01`'s sample.

**Results — identification, roughly flat:** classical ident held flat at **36.3%** (vs. `collect_01`'s small-sample read); embedding ticked up slightly to **21.5%**. Neither number moved meaningfully with the larger, more diverse sample — the identification ceiling isn't a sample-size problem.

**Results — timing:** classical 18.5 ms/frame, embedding 39 ms/frame — both still comfortably real-time on this CPU-only laptop, consistent with `collect_01`.

**Results — newly quantified: non-board screens broadly gate as "board."** With 99 frames covering real menu/dialog variety for the first time, the Stage-0 gate's weak spot is now measured, not just suspected: **0/6 win dialogs, 0/2 objective dialogs, 0/2 menus, 0/2 tutorial popups** were correctly gated off the board path. Only **deck overlays** gate reliably — and re-checking `eval_01`'s smaller sample confirms the same pattern was already present there, just below the sample size needed to see it clearly.

**Results — classical/embedding complementarity, new finding:** on the new roles, classical alone correctly identifies **Wretch** and **Jester**; embedding alone correctly identifies **Judge** and **Slayer**; both whiff on **Empress**, **Knight**, and **Witch**. An ensemble of the two would cover **5 of 8** new-role identifications this session vs. **2 of 8** for either identifier alone — the clearest evidence yet that the two identifiers are catching genuinely different things, not just agreeing-or-not on the same cases. Also reconfirmed: a static **"Poisoner @0.42"** false positive recurs across unrelated frames — almost certainly a fixed background element being misread, not a genuine per-frame identification failure. Cross-checked against the gallery: **all 44 roster identities are confirmed present** in `card-art/` — every miss this session is a recognition failure, not a missing reference.

**Updated ranked fix list (supersedes `collect_01`'s):** (1) HUD mask — widen to cover the ~20% text block, and add robustness to the Kill-Mode red-tint variant; (2) gate detection for dialog/win/reward/menu/tutorial screens (only deck overlays currently reliable); (3) classical+embedding ensemble, motivated directly by this session's complementarity finding; (4) embedding margin recalibration (still open from `collect_01`, still untouched by labeled live data); (5) dark-menu signal — a subset of (2), kept as its own line since it's a distinct root cause (brightness heuristic, not gate coverage).

**Data-sufficiency note:** none of the dominant failures found so far are data-starved — every one is a mask, threshold, or gate logic issue, fixable without more frames. Raw frame count is also the wrong metric to chase: each board frame yields 5–7 card instances, and consecutive frames are highly correlated (same board, same overlay state), so 99 frames is a larger effective sample than the raw count suggests but still concentrated. Coverage plan going forward: playing the game is what unlocks new roles into view; the in-game **Compendium is spoiler-gated** — it only displays cards already encountered in a town — so a Compendium sweep is a *cataloging pass after play coverage*, not a shortcut to seeing all 44 roles. The brightness-sweep micro-session (testing the untested Brightness-slider domain-shift risk from `collect_01`) is still pending.

**Artifacts:** `scrap_scripts/python/out/eval_02/` (per-arm results, `summary.json`/`summary.csv`) + `scrap_scripts/python/08_eval_collect02.py`; source frames at `scrap_scripts/python/out/collect_02/` (99 frames + `actions.md`). All gitignored, same as `eval_01`/`collect_01`.

## 2026-07-29 — First live-frame pipeline eval: real-time on CPU, but precision/identification need work

**Context:** The previous entry got live capture + input validated and a first 48-frame collection (`scrap_scripts/python/out/collect_01/`) on the ground. Nothing had yet run the actual detection pipeline against a single live frame — every number so far was from the two recorded sample videos. This session ran the full pipeline over `collect_01` and read the results honestly.

**Choice:** New scrap runner `scrap_scripts/python/07_eval_collect01.py` sweeps all 48 frames through **4 arms** — {raw, title-bar-cropped} × {classical, embedding} — using `06_crop_titlebar.py` for the crop arm. Outputs land in `scrap_scripts/python/out/eval_01/` (per-frame JSON + overlay PNGs per arm, plus `summary.json`/`summary.csv`). Not promoted to `utils/` — one-shot diagnostic eval, not a script anything depends on yet (Rule 1's promotion trigger not met).

**Why:** Before building anything further on top of localization/identification, we needed to know whether the pipeline — tuned entirely against recorded footage — actually works on live-captured frames, and where it breaks first if not.

**Results — timing:** comfortably real-time on this CPU-only laptop: classical ~21–22 ms/frame, embedding ~31–32 ms/frame (both arms, raw and cropped, near-identical).

**Results — localization:** recall is perfect on every visually spot-checked frame — the card grid is found correctly every time. Precision is the actual problem: the live in-game objective text ("Find and Execute N Evil...") sits just below the top-9% HUD mask boundary and gets misdetected as a card slot, misidentified as **"Hunter" at confidence 0.42–0.50**, in **57% of board frames**. This is a false-positive card, not a false negative — the mask needs to be wider, not smarter.

**Results — identification:** unreliable on live frames. Classical spot-check on one fully-revealed board: **2/5 correct**. A revealed **Minion** misread as **Hunter @0.80** (high-confidence wrong read). A **dead Medium** (skull status overlay) misread as **Lilis** — a demon role not even in this session's deck — because the overlay corrupts the crop at exactly the post-execution moments we most need a correct read. The served embedding identifier abstains on most real cards (**17.6% non-unknown** vs classical's **37.5%**); its shipped 0.12 margin threshold is untouched by any live data and needs recalibration against labeled live crops before it can be trusted or even fairly compared to classical.

**Results — Stage 0 gate:** the menu branch never fired once across all 48 frames — this game's menus are dark-themed, and the gate's `brightness > 160` heuristic assumes a bright menu, so the main menu is gated as "board" instead. Dead code on this game as currently tuned. The modal gate fared better: 4/4 correct.

**Results — title-bar chrome, exonerated:** raw vs. cropped arms are byte-identical on the gate and near-identical everywhere else — the existing top-9% Stage-1 HUD mask already exceeds the ~38px OS title-bar height from the original (un-flagged) launch options, so cropping bought nothing. (This is now moot anyway — see the launch-flag fix below.)

**Results — aspect ratio:** the live window during this collection was 1.53:1 raw / 1.60:1 cropped, vs. 1.778:1 (16:9) training footage. Not proven to be the cause of the dominant failures above (the HUD-mask and status-overlay issues are geometry-independent). **Important finding to carry forward:** the old capture AR was not a properly game-supported resolution — a tutorial popup rendered **partially off-screen** during this very collection run, a classic GUI-scaling symptom of an unsupported window shape. Some share of any geometry fault here belongs to the game itself at non-16:9 ARs, not just to our pipeline. Future captures standardize on true 16:9 1280×720 via the full launch-flag set `-screen-width 1280 -screen-height 720 -popupwindow -screen-fullscreen 0`, now verified borderless and exact via both the Steam launcher and the `steam://rungameid` script path (see today's RESEARCH.md update).

**Ranked fix list for next session:** (1) widen the top HUD mask past the objective-text band; (2) make identification robust to status overlays (skull/dead, etc.) instead of reading straight through them; (3) recalibrate the embedding margin threshold on labeled live crops; (4) replace the brightness-based Stage-0 menu heuristic with a real menu signal for dark-themed UIs; (5) 16:9 standardization — hygiene item, already adopted via the launch flags.

**Notes / risks:** Eval artifacts are `scrap_scripts/python/out/eval_01/` (gitignored, per-frame JSON + overlays for all 4 arms, `summary.json`/`summary.csv`) plus the two new scrap scripts `06_crop_titlebar.py` and `07_eval_collect01.py`. A second Python environment now exists on this machine: an agent-created `.venv` (Python 3.12, runtime-only deps) alongside the global Python 3.14 environment that ran the actual test suite — worth being deliberate about which one runs what going forward.

## 2026-07-29 — Live capture + input stack validated end-to-end; first live frame collection

**Context:** Everything so far has run against the two recorded sample videos. Touching the *live, running* game needs its own capture and input path, validated before anything gets built on top of it.

**Choice:** **Windows Graphics Capture** (`windows-capture` 2.0.0, window-targeted, BGRA frames) over `DXcam`/DXGI (desktop-composited — breaks under occlusion, not a window API) or `mss`/GDI `BitBlt` (CPU-bound, unreliable against DirectX-rendered surfaces) — OBS's virtual camera works but is a heavy external dependency plus an extra encode hop, kept as a lesson-only mention. **`pydirectinput` 1.0.4** (`SendInput` + scan codes) over `pyautogui` (drives the legacy `mouse_event` API, which many DirectX games — including this one, per the discipline of actually checking — ignore).

**Why:** Both choices trace to a documented capability gap, not a hunch — new `research/RESEARCH.md` entry (2026-07-29) with the primary sources plus this session's own empirical measurement.

**Empirical validation (this machine, Windows 11, game v0.762b, Unity IL2CPP, Steam appid 3803820, launch options `-popupwindow -screen-fullscreen 0`):** WGC captured the windowed game at 1346×879 @ ~55–57 FPS. Synthetic `SendInput` clicks **are accepted** by this Unity build — verified via before/after frame diffs across a real menu transition (menu → Pick Game Mode). Launching via `steam://rungameid/3803820` + window polling works. Capture **survives a minimized RDP client session**.

**First live frame collection:** a 48-frame session (38 clean / 10 tutorial-tagged) captured 7 distinct roles (Gemcrafter, Lover, Confessor, Hunter, Enlightened, Medium, Minion), at `scrap_scripts/python/out/collect_01/` with an `actions.md` inventory.

**Honest caveats (open risks for whoever builds on this next):** `-popupwindow` did **not** strip the window chrome — frames include the OS title bar, so client-area cropping needs a **live** `GetClientRect`, never cached. Title-substring window matching is unsafe (a File Explorer window browsing the game's install folder matched "Demon Bluff") — must also verify the owning process is `Demon Bluff.exe`. A minimized (not just occluded) game window gives stale WGC frames. The in-game Brightness slider is an untested domain-shift risk — all frames so far are default-brightness. Card hit-boxes extend beyond visible art; the bottom-right action-icon row isn't positionally stable across game states. A full RDP disconnect (vs. staying minimized/backgrounded) locks the session and breaks both capture and input. DPI: the process must call `SetProcessDpiAwareness(2)` before capture/click; `pydirectinput` was used standalone (no `pyautogui` import) partly because some of PyAutoGUI's own deps call the older `SetProcessDpiAware()` on import.

**Not yet promoted:** exploration scripts stay in gitignored `scrap_scripts/python/` (`01` capture test, `02` click test, `03` launcher, `04` single-frame grab, `05` frame-space click) — candidates for `utils/` promotion under a future `PLAN-live-capture` once something depends on them (Rule 1's promotion trigger isn't met yet).

**Housekeeping:** added `_drop-off/` to `.gitignore` — a staging folder for files being migrated in from another machine, not yet triaged into a real tier.

## 2026-07-29 — Migration prep: promoted the two load-bearing scrap scripts; fresh-machine bootstrap doc

**Context:** Development moves to a machine set up to capture *current-version* gameplay (the sample footage is v0.389-era — see the footage-version-drift entry below). `scrap_scripts/` is gitignored wholesale, so anything in it dies with this machine.

**Choice:** Promoted the two scrap scripts that tracked content actually depends on — `01_wiki_harvest.py` → `utils/python/wiki_harvest.py` (regenerates the gitignored card-art gallery + tracked manifest; a fresh clone cannot build the gallery, serve, or fully test without it) and `09_embed_eval.py` → `utils/python/embed_eval.py` (`finetune_embedding.py`'s own output text directs users to it). Updated all tracked references; cataloged both in `utils/README.md`. Added a **Fresh-machine bootstrap** section to `README.md` (venv → ffmpeg → harvest → model regen-or-copy → footage → CLAUDE.local.md → pytest).

**Why:** Rule 1's promotion trigger was already met in both cases (regenerates tracked content; depended on by a tracked script); migration just made the latent violation visible.

**Notes / risks:** (1) The remaining 17 scrap scripts stay gitignored and will NOT survive the machine switch — reviewed as genuinely throwaway probes/spikes; the mined `dataset/crops/` (6,202 crops) also stays behind but is deprecated with the old footage anyway (regenerable from the tool + videos if ever needed). (2) The served `models/*.onnx` are gitignored — copy them to the new machine or regenerate (seeded). (3) New-footage capture should record the game build version at capture time — wave-2 labeling now requires art-set version per crop (see below).

**Context:** Wave 1 closed with a flagged roster desync — claim bubbles referencing a role `Counsellor` absent from `ROSTER.md`. Re-checked the live wiki to confirm the 44-role roster and settle the flag.

**Roster check:** all four role categories re-queried via the MediaWiki API. **Identical to the 2026-06-21 harvest — 25/9/7/3 = 44, no additions, removals, or renames.** Re-ran `scrap_scripts/python/01_wiki_harvest.py` (fetch-once, skips cached); it pulled exactly one new page, `Delusion`, an unimplemented evil role in `Unused Roles` (12 → 13). Manifest updated; art count unchanged at 67.

**The actual finding — `Counsellor` is not a missing role, it is a date stamp.** It is `Chancellor`'s pre-**v0.390a** name (renamed 2025-10-28). That pins the `dataset/` footage to **≤ v0.389, Oct 2025**, while the wiki and our harvested `card-art/` are **v0.762a (2026-07-14)** — a ~9-month gap we had been implicitly assuming away. Consequences now written into `ROSTER.md` § *Footage-version drift*: (1) the Stage-3 name vocabulary needs a **45th string `COUNSELLOR → Chancellor`**, else every Chancellor plate in the dataset abstains; (2) **`Rambler` (v0.610) and `Investigator` (v0.760e) cannot appear in our footage** — footage roster is effectively **42 roles**, and any mined crop identified as one is a known-bad read, i.e. a free negative check on the round-2 identifiers; (3) **v0.730** replaced the character-type/alignment icons, so footage UI chrome ≠ current chrome; (4) Halloween/Christmas **skins** already shipped for 7 villagers — the "card art can change" constraint is documented history, not a hypothetical.

**Why it matters beyond the fix:** the reference gallery in `card-art/` is the *current* art set and is therefore only a **weak prior** for footage-era cards. This is a plausible contributor to the round-2 mining result that only 44% of both-confident crops agreed — worth testing in wave 2 before attributing it all to identifier weakness.

**Notes / risks:** (1) Wave-2 labeling must record the **art-set version** per crop; a single dataset spanning two art sets, silently mixed, would poison the embedding gallery. (2) Not yet checked whether the two sample videos are from the *same* build as each other — verify before pooling their crops. (3) `PLAN-stage3-ocr.md` flag closed; a new wave-2 item covers general version drift. (4) No cross-check yet that pre-v0.390a art for the 42 in-footage roles is obtainable at all — if not, the recognizer must be fit from mined crops rather than wiki art.

## 2026-07-29 — Wave 1 of the restart roadmap: round-2 mining run, Stage-4 decision, Stage-3 spec

**Context:** After the audit resync, three parallel agents developed the next roadmap steps: round-2 dataset mining, the Stage-4 temporal design decision, and the Stage-3 OCR spec. Three new plan slugs created (`PLAN-round2-dataset`, `PLAN-stage3-ocr`, `PLAN-temporal-assembly`).

**Stage 4 decision — A2+B2, decided in-course:** The user delegated the temporal-depth/REST-contract call to the course ("you are the professor — make the choice and describe why"). Chosen: **stateless windowed fusion** (`POST /v1/snapshot/window`, sequence in → one fused snapshot out) with a **margin-weighted recency-decayed per-slot vote**, reveal latch, staleness fields, schema → 0.3.0. No tracker (fixed slots make association free); no session state (A3 = same fusion core + plumbing, bridge kept); Bayes filter taught in module 07 but deferred — it needs margin→likelihood calibration on labeled crops that don't exist yet. Full professor-voice rationale in `plans/PLAN-temporal-assembly.md`; new RESEARCH entry (FPP3, Murphy SSM, SORT, object-permanence tracking). Binding wave-2 constraints: decay by **timestamp delta, not frame count**; face-down = no observation, not a failed read.

**Round-2 mining (tool + full run):** promoted `utils/python/mine_card_crops.py` (frame_select → classical_localize → crops + dual-proposal manifest). Full pass over both videos, ~43 min: 9,909 strided decodes → 784 kept board frames → **6,202 crops** with `label: null` awaiting wave 2. Headline stats: classical identifies 26.8%, embedding 17.1% (82.9% abstention at margin 0.12), and — the surprise — **only 44% of the 330 both-confident crops agree**, so identifier agreement can seed but never substitute for labeling. Spot-check found: a systematic classical **Wretch→Architect confusion at 0.70+ confidence**; a clean in-gallery **Knitter missed by both identifiers** (direct round-2 fine-tune motivation); a localizer false-positive tail on busy frames (wave-2 labels need `not_a_card`); high embedding margin under dialog occlusion (0.355 on a dubious read) — margin ≠ correctness when occluded.

**Stage-3 OCR spec (from real frames):** 14 text surfaces cataloged with owners (closed-vocab vs PaddleOCR-fallback vs skip), ~84-token vocabulary, two-head architecture (~1M-param word classifier + ~0.5M CRNN-CTC digit-string reader, ~5% of runtime budget). Spec corrections to prior assumptions: **Score reaches 4 digits** (2510 observed) so digits are a variable-length string head, not a 0–99 classifier; all on-card text is just plate + slot number; `<Corrupted>` renders as a literal second plate line → `Readings.state`.

**Notes / risks:** (1) **Roster desync flagged, not fixed:** claim bubbles reference a role "Counsellor" absent from `ROSTER.md` — if it can appear on a card the recognizer vocabulary has a hole; user call pending. (2) The temporal agent's safety layer flagged the mid-flight decision-delegation redirect as possible instruction poisoning — false positive (the delegation is verbatim from the user in-session), noted here for provenance. (3) Wave 2 = labeling pass (manifest `label` fill, `not_a_card` value, temporal-run propagation), margin calibration, recognizer build, Stage-4 implementation per the decided contract.

**Context:** ~5 weeks idle after the 2026-06-22 handoff. A four-agent read-only audit (plans/docs, code+tests, deliverables, git/env) confirmed the code was healthy — **110/110 tests re-verified green (~23 s)**, venv matches pins exactly, no hard-coded resolutions, public-surface gate clean — but found the last two feature commits had synced CodeDocs while skipping the Lesson-Plan and site tiers, plus assorted drift.

**Choice:** Fast-forwarded `main` onto `embedding-finetune-round1` (branch's purpose — the round-1 fine-tune — was complete; round 2 is defined as dataset-building, a separate unit), then a three-agent remediation pass over disjoint tiers, then push.

**What was fixed (headline):** modules 02/08 no longer claim frame selection / ONNX serving are unbuilt; README no longer says "the CV pipeline is not built yet"; `gallery.md`'s art-swap contract now matches the adopted re-fine-tune story; `api.py` docstrings/fallback message corrected (was pointing at `export_backbone.py` — running it would NOT have fixed a missing served model; `finetune_embedding.py` is correct); dead `localize` alias + 3 dead imports removed; `__version__` → 0.2.0; line numbers resynced across all CodeDocs overviews; site status page current; PLAN-pipeline's two standing invariants moved out of the checklist (the slug was structurally un-completable); PROJECT-PITCH gained the missing Stage-0 and course-delivery decision rows; three retroactive RESEARCH entries logged (brightness-ratio gate, HSV histogram correlation → Swain & Ballard '91, ORB tiebreaker → Rublee '11 + Lowe '04) — each marked as closing a research-before-deciding violation; knowledge-base playbook finally appended (4 lessons).

**Round-1 numbers not previously captured here** (from the gitignored `models/_round1_train.log` / `finetune_round1_results.txt`, recorded before they're ever lost): hyperparams `lr_backbone=1e-4, lr_proxy=1e-2, alpha=32.0, margin=0.1, seed=0`; inter-prototype cosine median 0.857→0.412, p95 0.913→0.482, min 0.651→0.232; trainable params exactly 736,488 (79.4% of backbone); final losses Phase A ≈6.56, Phase B ≈0.29; refs-per-class min/max = 1/3.

**Correction:** the 2026-06-22 RESEARCH fine-tune entry says "1–67 reference images/class" — that misreads 67 (the *total* ref count) as a per-class max; actual per-class range is **1–3**. RESEARCH is not append-only but the entry is dated, so the correction lives here.

**Notes / risks:** (1) The `_ft`-suffixed artifact names in the old training log are a relic — current `finetune_embedding.py` defaults `--out-onnx/--out-pt` to the canonical unsuffixed served paths, so `models/README.md`'s "re-run to regenerate" is correct as written. (2) The audit's advisory on the third-party streamer handle in module 04 / observed-board-layout was left for a human call — not remediated. (3) Claude-5-generation model quality checks were explicitly deferred by the user; the audit flag lists in this session are the natural worklist when that happens.

**Context:** Stage 0 had only the board-state gate; the plan still owed the low-fixed-stride decode + perceptual-hash near-duplicate dedup that turn a ~1 h capture into the handful of frames worth analysing.

**Choice:** New `src/dbcv/frame_select.py` (cv2+numpy, no torch, **dev/batch only — explicitly NOT on the REST path**). Cascade: **strided decode (stride derived from the media's REAL fps) → dHash near-duplicate dedup (Hamming ≤ 8 vs the last *kept* frame) → reuse `classify_frame_state` gate.** `select_frames` returns metadata only (`SelectedFrame`: index, timestamp, state, dHash); `iter_selected_frames` streams pixels in constant memory.

**Why dHash over pHash:** both are research-endorsed (RESEARCH.md frame-selection entry). dHash needs no DCT (cv2+numpy only — no new `imagehash` dep), and encoding the *sign* of the horizontal brightness gradient makes it robust to the global brightness/contrast shifts (fades, tooltip dimming) between near-identical game frames while still flipping bits on real structure change. One-constant re-tune on an art swap.

**Why dedup BEFORE the gate, vs the last *kept* frame:** dedup-first means a long idle-board run costs one gate call; comparing to the last kept hash (not the immediate predecessor) collapses slow-drift runs into one keeper. Non-board frames still advance the dedup anchor so a modal→board transition isn't masked.

**Constraints honored:** never opened the sample videos (tests use a tiny synthetic `cv2.VideoWriter` clip + the already-extracted PNGs + numpy arrays); fps read from media (OpenCV → ffprobe → documented fallback, surfaced via `fps_source`); resolution read from media; `stride = round(media_fps / target_fps)`, never assumes 30.

**Did NOT promote a `utils/` CLI runner:** nothing depends on one yet (Rule 1's promotion trigger isn't met) and the old `03_sample_frames.py` does *uniform* sampling — a different job. The module is importable; a batch CLI can graduate when a consumer appears.

**Result:** 81 → **110 tests** (+29 synthetic), ~24 s, no regressions. (Dev-only PySceneDetect segmentation, the other Stage 0 item, remains intentionally deferred per the research's dev-only reservation.)

## 2026-06-22 — Embedding fine-tune (round 1): fixed the collapse; adopted as default with a margin gate

**Context:** The shipped embedder was a *frozen* ImageNet MobileNetV3-Small. On our 43 stylised card-characters its features collapsed (inter-prototype cosine 0.65–0.94), so embedding-NN over-identified and did not beat the conservative classical matcher. The prior handoff left two open decisions: do a fine-tuning round, and whether to flip the default identifier. This session resolved both.

**Research first (research-before-deciding):** New `research/RESEARCH.md` entry. The failure is a *known* phenomenon — **domain shift of frozen ImageNet features to a stylised / fine-grained domain** (Chen et al. ICLR'19; explicitly NOT "neural collapse"). Prescription: fine-tune the *same* backbone with a metric-learning **margin** loss (**Proxy-Anchor**), **LP-FT** (warm the head, then unfreeze top blocks — full FT distorts features on <1e3 images, Kumar et al. ICLR'22), strong augmentation from the clean refs, and a **margin** abstention criterion.

**What we built:** `utils/python/finetune_embedding.py` (promoted from scrap — it now generates the served artifact). Proxy-Anchor implemented inline (no new dep, teachable). LP-FT: Phase A 250 steps proxy warmup (backbone frozen, BN in eval), Phase B 600 steps with the top 4 feature blocks unfrozen (~736k params), AdamW, proxies warm-started from the frozen prototypes. Synthetic augmentation via `torchvision.transforms.v2` (RandomResizedCrop + perspective + ≤6° rotation + RandAugment + GaussianBlur + RandomErasing; **no horizontal flip** — in-game cards are never mirrored, so flipping would teach a false invariance). Batch = 43 classes × 4 views, on the Titan Xp (~minutes).

**Round-1 result (leak-proof metric = inter-prototype cosine on CLEAN refs):** mean off-diagonal **0.850 → 0.409** (max 0.939 → 0.536); synthetic top-1 79.9% → 100% with top1−top2 margin 0.031 → 0.405; torch↔ONNX parity 5.5e-6. The collapse is undone. On real frames the confident (high-margin) calls match documented content (Scout, Wretch on frame 008).

**Decision #1 (user call) — adopt the fine-tuned embedding-NN + a margin gate as default.** Fine-tuning compressed the absolute cosine scale (correct match ~0.6, unrelated ~0.4), so the old absolute threshold (0.60) over-identified 125/125. Switched `classify_crop_embedding` to abstain on the **top1−top2 margin** (`_EMBED_MARGIN_THRESHOLD = 0.12`); the reported `confidence` is now that margin (semantic change). Real-frame result: 30/125 (24%) confident IDs, 95 honest "unknown", classical↔embedding agreement 27 → 90. Conservative + high-precision, matching the project's values.

**Served-model layout (all gitignored, regenerable):** `models/mobilenetv3_small_embed.onnx` is now the **fine-tuned** served model (generator `finetune_embedding.py`; weights `mobilenetv3_small_embed.pt`). The frozen ImageNet **baseline** moved to `models/mobilenetv3_small_embed_frozen.onnx` (generator `export_backbone.py`, repointed). The runtime path is unchanged → `embed.py`/`api.py`/tests needed no path change. 81 tests green (~24 s).

**Decision #2 (user call) — stop at round 1.** The leak-proof metrics show the collapse is fixed; round 2's levers (SupCon / deeper unfreeze / class-balancing / real-mined crops) target "if round 1 underperforms," which it didn't. The one genuine open gap is real-frame generalisation *beyond the confident few*, which needs **labeled** board crops to even measure — the documented next lever (round 2 = mine + label real crops), tied to dataset-building, not a quick retrain.

**Notes / risks the next person should know:**
- **Art-swap cost changed.** Frozen backbone → art swap was *re-embed only, zero training*. The served backbone is now fine-tuned to the current 43 characters, so a *new* art set is best handled by **re-fine-tuning** (`finetune_embedding.py`, ~minutes on Titan Xp) then the gallery rebuild; a quick re-embed still works but won't separate new art as well. This is the accuracy ↔ retrain-cost tradeoff (a teachable point for Module 05).
- **Margin threshold 0.12 is provisional**, calibrated on round-1 real-frame margins (`scrap_scripts/python/11_ft_abstain_probe.py`): confident cards ≥ ~0.11, ambiguous < ~0.06. Refine once labeled / face-down crops exist. Frames 1–18 are all face-up (verified visually on frame 008), so face-down rejection is unmeasured on real data; the classical hist-gate stays available as a face-down signal if false positives appear.
- **`confidence` semantics changed** for the embedding identifier (now the top1−top2 margin, not the (cos+1)/2 remap). Stage 4 temporal smoothing should weight accordingly when built.
- Fixed a naming desync: `embed.py`/`identify.py` called identification "Stage 3"; it is **Stage 2** (Stage 3 = OCR) per PLAN-pipeline.

## 2026-06-22 — Handoff consolidation: reconcile PLAN-pipeline + publish a status page

**Context:** Context window filling up; wrapping the session for clean handoff to fresh chats. Goal: make sure the *written* record is the single source of truth (a fresh chat has only the docs + git log) and add a human-readable status page to the public site.

**Found + fixed a drift:** `plans/PLAN-pipeline.md` had gone stale — it still showed Stage 1 integration as "next" and Stage 2 embedding-NN as "deferred," because the torch-adoption + embedding-NN rounds (which updated DEV-LOG + the decisions table) didn't fully re-sync the plan. Reconciled it to reality: Stages 0/1/2 + REST marked shipped with accurate notes; the honest embedding finding folded in; and a new **"Open decisions & next steps (handoff)"** section pinned near the top so a fresh chat sees the two pending calls immediately: (1) flip the default identifier to classical until fine-tuned, (2) the fine-tuning round + approach. (This is the Rule 3 "on noticed desync, full audit" in action.)

**Public status page:** added `site/pages/status.html` — a plain-language snapshot (what works, the honest identification result, the course at 8/10 modules, what's next, and pointers to the cited repo docs). Linked from `site/index.html` (Pages) and `site/pages/notes.html` (the misc section). **Rule 6/7 gate run:** repo-wide grep for the dead name / real last name / private absolute paths → **no matches** in tracked content; the page is project-focused prose with repo-relative paths only, no identity or third-party bulk.

**State at handoff:** 81 tests green (~22 s); 14 commits this session, all on local `main`, **nothing pushed** (per the standing instruction). Entry chain for the next chat: `CLAUDE.md` → `PLAN.md` → `plans/PLAN-pipeline.md` (open decisions at the top) → `DEV-LOG.md` → `PROJECT-PITCH.md` → `CODE-DESIGN.md`/`CodeDocs/`.

## 2026-06-22 — Test suite speed: ~295 s → 23.5 s via process-level memoization

**Context:** After Stage 3 landed the 81-test suite had ballooned to ~295 s. The root cause (already diagnosed) was that the two expensive builders — `build_gallery()` (~10 s, loads 67 PNGs + computes HSV histograms + ORB descriptors) and `OnnxEmbedder` + `build_embedding_gallery()` (~2–3 s, loads ONNX session + embeds all 67 references) — were being re-run on every test module that owned a `scope="module"` fixture AND on every `TestClient(app)` that triggered the FastAPI `lifespan`. With five test modules each pulling the lifespan + fixtures, the total reached ~7+ full rebuilds.

**Fix applied:** Process-level memoization (module-level dict caches) in `src/dbcv/gallery.py` and `src/dbcv/embed.py`:
- `_GALLERY_CACHE: dict[Path, Gallery]` in `gallery.py` — `build_gallery(art_root)` returns the cached result on the second call (key = resolved `art_root`).
- `_EMBEDDER_CACHE: dict[Path, OnnxEmbedder]` in `embed.py` — new `get_onnx_embedder(onnx_path)` factory returns a cached `OnnxEmbedder` instance (key = resolved ONNX path). `api.py` lifespan updated to call `get_onnx_embedder()` instead of `OnnxEmbedder()`.
- `_EMBED_GALLERY_CACHE: dict[tuple, EmbeddingGallery]` in `gallery.py` — `build_embedding_gallery(classical_gallery, embedder, art_root)` returns the cached result (key = `(art_root, embedder._onnx_path)`). Requires `OnnxEmbedder` to store `self._onnx_path` (added to `embed.py`).

**Why memoization, not session-scoped fixtures:** The lifespan path (TestClient) bypasses pytest fixture scoping entirely — it runs its own build every time a new TestClient is created. Making the builders cheap to call a second time removes the cost regardless of whether the caller is a fixture, the lifespan, or a helper function inside a test body. The cache is also correct in production (build once at startup; an art/model swap requires a restart which clears the module-level dict naturally).

**Idempotency test:** `test_gallery_rebuild_is_idempotent` calls `build_gallery(_ART_ROOT)` twice and checks `.townee_names` equality and `.n_references` equality. With the cache the second call returns the same object, which trivially satisfies both checks — no assertion weakened.

**Result:** 81/81 pass in **23.5 s** (was ~295 s). No identification behavior changed; no thresholds or defaults touched.

## 2026-06-22 — Stage 3 embedding-NN built — honest result: frozen ImageNet does NOT beat classical

**Built it correctly:** MobileNetV3-Small (frozen, ImageNet) → exported to ONNX (**torch↔onnx parity 1.7e-6**) → **onnxruntime-CPU** runtime (verified **no `import torch` anywhere in `src/dbcv/`** — serving stays torch-free) → cosine-NN over a 576-d, re-embeddable gallery (prototypical mean per townee). Wired into the `lifespan` (load once). **81 tests pass.** ONNX is gitignored + regenerable via `utils/python/export_backbone.py` (`models/README.md` documents it).

**The finding (important, and the opposite of the easy story):** the frozen ImageNet backbone **collapses all 43 cartoon characters into one tight cluster** — inter-prototype cosines run 0.65–0.94 (e.g. Architect↔Shaman = 0.93). ImageNet features discriminate *photographs*, not cartoon art style. Consequence on the Sample1 board frames: embedding-NN names **100%** of card slots (vs classical's conservative 35% fire-rate) but is **not actually more accurate** — it overidentifies, biased toward the cluster-central prototypes (Architect, Hunter). Classical, by honestly returning "unknown" ~65% of the time, has higher *precision* when it commits. Net: **frozen embedding ≈ or slightly worse than classical in correctness**, just far less conservative. Face-down/blank crops correctly fall below the 0.60 cosine threshold → "unknown".

**What this means:** the embedding *architecture* is right (re-fit-cheap, ONNX-on-CPU, the research path) but a **frozen generic backbone is insufficient for this domain — light fine-tuning on card crops is the real fix**, and it's now feasible locally on the Titan Xp. This is a genuinely instructive result for Module 05 ("embedding doesn't magically win — you must adapt the backbone to your domain"), to be written once the fine-tuned variant exists.

**Open decision (flagged for Panda):** the build wired embedding-NN as the default identifier; I **recommend the default be the more conservative classical** identifier (honest "unknown" beats confident-wrong for a state API) **until** the backbone is fine-tuned. Both remain available on `app.state` (`identifier`, `classical_identifier`). Not flipped autonomously — it's a product call.

**Regression (now urgent):** the test suite ballooned to **~295 s** (was ~74 s) — the embedding-gallery build compounds the existing per-test-module classical-gallery rebuilds. The tracked test-speed cleanup is being done next (memoize/session-scope the heavy builds); it's independent of the default/fine-tuning decision.

## 2026-06-22 — Adopt torch + ONNX (GPU verified on the Titan Xp); embedding-NN unblocked

**Context:** Panda green-lit torch + ONNX, with **local GPU dev on the Titan Xp** (easier dev loop than round-tripping Colab). This also corrected an imprecision of mine.

**Correction worth keeping:** I had lumped `onnxruntime` in with `torch` as "heavy." Wrong. `onnxruntime` (CPU) is ~15 MB, pure inference, needs no CUDA — it's the *light* runtime dep the research already calls for. `torch` is heavy by **wheel size** (the default CUDA wheel is ~2.5–3 GB because it **bundles its own CUDA runtime**) and by being the training-tier gateway — **not** because of the system CUDA toolkit, which pip-installed torch/onnxruntime do not use at all. What gates GPU use is the **driver**, not the toolkit. ("Irreversible" was also the wrong word — pip installs are reversible.)

**What landed:** installed into `.venv` — `torch 2.7.1+cu118`, `torchvision 0.22.1+cu118` (chose the **cu118** build because the Titan Xp is **Pascal sm_61** and the newest CUDA wheels trim old archs), plus `onnx 1.22.0` + `onnxruntime 1.27.0` (CPU). Verified with a scrap smoke script that runs a **real GPU matmul** (not just `is_available()`, which can lie on a build missing sm_61 kernels): matmul succeeded → Pascal kernels present. numpy stayed 2.5.0 (no downgrade).

**GPU now recorded** (CLAUDE.local had it "unverified — record the actual card"): **NVIDIA Titan Xp, 12 GB, Pascal sm_61**; driver 582.53 (CUDA-13.0-capable); toolkit 12.2. The dev box *is* the runtime-budget anchor.

**Sync:** `requirements.txt` now pins `onnx`/`onnxruntime` as real deps and documents `torch`/`torchvision` as machine-specific dev-time installs (cu118 index command + CPU-only alternative). Decisions-table row added (supersedes the "embedding-NN deferred" note). Next: build the Stage 2 **embedding-NN identifier** — frozen MobileNetV3-Small → ONNX for CPU serving → cosine-NN over a re-embeddable gallery — evaluated head-to-head vs the classical ~40–60% baseline.

## 2026-06-22 — Lesson modules 01 (framing) + 09 (staying alive) → 8/10 authored

Authored the two "bookend" modules: `Lesson-Plan/modules/01_framing.md` (the course's front door — why this project teaches CV through one genuinely-constrained real system; the constraints-as-characters from `PROJECT-PITCH.md`; a preview of the pipeline arc; cites the compute-budget research entry) and `09_staying-alive.md` (the art-swap-cheap thesis as shipped — localization HSV re-tune, gallery rebuild with zero training, font re-render for the future OCR; why the trained classifier was rejected for production; honest that drift/health monitoring is design-not-code). **Lesson plan now 8/10 authored** (00–05, 08, 09); only **06 (on-card OCR)** and **07 (state assembly/temporal)** remain, and both correctly await their unbuilt pipeline stages. Citations all trace to existing RESEARCH entries / PROJECT-PITCH / shipped-code docstrings; no new research or desync.

## 2026-06-22 — Lesson modules 03 (resolution geometry) + 08 (REST) + Module 04 reconciliation

Authored `Lesson-Plan/modules/03_resolution-agnostic-geometry.md` (the "never bake a resolution" principle as enforced in shipped code — `Resolution` from `image.shape`, `bbox_rel` fractions, thresholds relative to `min(W,H)`, relative→pixel only at the edge via `crop_relative`; the self-correcting resolution test; the 1280×720-vs-1920×1080 sampler-downscale story) and `08_rest-serving.md` (lifespan load-once onto `app.state`, plain-`def`-in-threadpool vs event-loop-blocking `async def`, versioned `GameStateSnapshot` + `schema_version`, the 0.1.0→0.2.0 bump as versioning-in-action; cites RESEARCH entry 5). **Inventory now 6/10 authored** (00, 02, 03, 04, 05, 08; planned: 01, 06, 07, 09 — 06/07 await the unbuilt OCR/temporal stages).

**Reconciled a desync I introduced:** Module 04 was authored in parallel with the Stage 0 gate, so it claimed the state gate "is weak / no reliable gate" and that the `observed-board-layout.md` badge caveat was "not yet corrected." Both became false within the same round (the gate shipped; I added the caveat). Updated Module 04's failure-modes section to reflect that the gate is now handled upstream (`frame_state.py`) and the KB caveat is in place. Lesson: parallel authoring + implementation in one round can self-contradict — reconcile at the round's commit.

## 2026-06-22 — Lesson modules 02 (frame selection) + 05 (identification)

Authored two more course modules, each grounded in shipped+tested code and matching the established skeleton/voice: `Lesson-Plan/modules/02_frame-selection.md` (the frame-selection design space + the shipped board/modal/menu gate, including the honest "absolute brightness failed → the center-vs-ring *ratio* is the invariant" debugging story; forward-references the still-owed stride/dedup) and `05_card-identification.md` (the four identification families on the retrain-cost axis, the shipped classical baseline, the honest ~40–60% face-up result, the `compareHist(zeros,*)==1.0` bug, and the deliberately-deferred embedding-NN upgrade). Cites existing RESEARCH entries 1 and 3 (no new research needed). `LESSON-PLAN.md` inventory now **4/10 authored** (00, 02, 04, 05). No desync.

## 2026-06-22 — End-to-end CLI runner (capstone): the pipeline reads the board

Promoted a durable runner to `utils/python/run_pipeline.py` (Rule 1 promotion: descriptive name, repo-root-anchored, **row added to `utils/README.md`**). It runs the full pipeline offline (no HTTP) on sampled PNGs: builds the gallery once, then per frame does gate → localize (board only) → identify, prints a readable summary (frame_state + per-card identity@confidence), writes snapshot JSON, and (with `--overlay`) saves annotated PNGs to the gitignored `dataset/pipeline-out/`. Flags: `--frames`, `--out`, `--overlay`, `--limit`, `--no-gallery`.

**Verified end-to-end (I viewed an overlay myself):** board frames render boxes + identity labels + a `board` banner; modal frames (`Sample1_000`, `Sample2_000`) gate to `modal` with **zero** boxes; the partial-modal `Sample1_006` is correctly treated as `board` (peripheral cards still found). Real identities surface on face-up cards — `Wretch@0.80`, `Baa@0.69/0.70`, `Confessor`, `Hunter`, `Druid`, `Scout`, `Fortune_Teller`, `Doppelganger` — with face-down cards correctly `unknown`, consistent with the honest ~40–60% face-up baseline. This is the reproducible hands-on artifact the lesson modules point at.

**Notes:** 14 light CLI unit tests (helpers only, 0.31s — deliberately no gallery build, so the slow-suite issue isn't worsened). Fixed a Windows cp1252 crash (a `→` in the argparse help string; module docstring keeps the arrows since argparse doesn't print it). `dataset/pipeline-out/` added to `.gitignore`.

## 2026-06-22 — Stage 2: classical identification baseline (embedding-NN deferred)

**Context:** Cards were being localized but not named (`identify` was a stub). Built the classical identification baseline per the conservative directive — opencv+numpy only, **no torch/onnxruntime, no model download** — explicitly deferring the research-preferred embedding-NN to a later, heavier round.

**What landed:** `src/dbcv/gallery.py` — `build_gallery()` walks `knowledge-base/card-art/<class>/<role>/*.png` and builds an **in-memory** gallery (no persisted artifact → card art stays a gitignored input, Rule 6): **43 townees / 67 references** (24 have skin variants, all loaded). Directory = label (`class`→role_class, dir→identity); `Twin_Minion`→`Minion` aliased. `src/dbcv/identify.py` rewritten: `classify_crop` matches a card crop by **2-D HSV (hue×sat) histogram correlation** (primary — value excluded to survive state-tinting) with an **ORB-feature tiebreaker** when top-2 are within 0.05; confidence = clamped Pearson correlation, threshold 0.40 → else "unknown". Gallery built once in the API `lifespan` onto `app.state` (load-once pattern). **46/46 pytest green.**

**Honest result (the pedagogical point):** ~**40–60% on face-up cards**, **100% correct "unknown" on face-down cards** (a uniform card back matches no character art → low confidence, which is the right answer). A `Scout` was predicted twice in one frame (impossible in-game) — a real false match. **Verdict: classical histograms are a useful, honest *lower bound* but insufficient for production → the embedding-NN upgrade is warranted.** This is exactly the worked example for the (future) identification lesson module.

**Teaching bug found:** `cv2.compareHist(zeros, anything)` returns **1.0** (Pearson 0/0 → clamped), so a degenerate all-black crop "perfectly matched" everything. Fixed with a zero-sum guard. Good "always test degenerate inputs" lesson material.

**Known issue (tracked separately, not a blocker):** the suite now runs ~73s (was ~1.4s) because the gallery rebuilds ~7× across test files + per-frame matching in API tests. A session-scoped shared-gallery fixture (and injecting the gallery into the app for tests) would fix it. Deferred so as not to perturb the just-validated identifier under this run's budget.

**Deferred (logged per conservative directive):** embedding-NN identification — a small frozen backbone (e.g. MobileNetV3) exported to ONNX + nearest-neighbour over the gallery, served on CPU via onnxruntime. Needs `onnxruntime` (+ a one-time model export/download) — the first genuinely "heavier" dependency. Left for when Panda is back to approve the dep, or a later round.

## 2026-06-22 — Deepening round 1: foundation lesson modules + Stage 0 state gate

Two parallel sub-agents on disjoint tiers (lessons vs `src/`), both light/classical (conservative path).

**Lessons (primary deliverable, first modules authored):** `Lesson-Plan/modules/00_python-environments.md` (the user-requested **venv vs virtualenv vs pip vs conda vs uv** module — teaches the interpreter/venv/installer split, compares all five honestly, justifies this project's venv+pip choice, no fabricated benchmarks) and `Lesson-Plan/modules/04_card-localization.md` (classical vs trained-detector vs foundation-model, worked example = our real spike results, with the honest caveats — art-tuned HSV, badge-blob failure, weak state gate). `LESSON-PLAN.md` inventory populated (Modules 00–09; 2 authored / 8 planned). New `research/RESEARCH.md` entry on env tooling (official docs, trust A). Module files use `NN_slug` per the LESSON-PLAN skeleton. I read Module 00 end-to-end — accurate and well-pitched.

**Stage 0 — frame-state gate (the spike's known gap):** `src/dbcv/frame_state.py` — `classify_frame_state(image) -> "board"|"modal"|"menu"`. The winning discriminator is a **center-vs-ring brightness ratio**: a modal's bright panel sits on the *same dark starfield* as the board, so absolute center-brightness failed (the spike's 0/3), but the panel is ~3–6× brighter than the dark ring around it, where a board is ~1.0–1.1. Threshold 2.0 sits in a clean 3× gap. **7/7 labelled board+modal frames correct**; the partial-modal (`Sample1_006`, peripheral cards visible) is deliberately called "board" so the localizer can still read it. Schema → **v0.2.0** with a `frame_state` field; the pipeline now runs the gate first and **skips localization on non-board frames** (returns `cards=[]`). **24/24 pytest green** (verified by me). CodeDocs + `CODE-DESIGN.md`/`00_PROJECT.md` synced; the stale localize.py "gate held back" TODO removed.

**Also:** resolved the twice-flagged desync — `knowledge-base/lessons/observed-board-layout.md` now carries the badge implementation caveat (blob detection aliases on clue text → badges are for ordering, not anchoring) and notes the implemented state gate.

**Notes / risks:** the gate threshold (2.0) is validated on 3 modal types; a future modal with a very small bright panel could approach it (`Sample1_000` already the closest at 3.1). Frame *selection* proper (stride decode + perceptual-hash dedup) was intentionally skipped this round to keep the gate focused — still owed in Stage 0.

## 2026-06-22 — Classical localizer promoted into `src/`; approach confirmed

Integrated the spike algorithm into `src/dbcv/localize.py` as `classical_localize` (5 stages: relative HUD-exclusion → HSV segmentation → morphology → contour/geometry filter → IoU-NMS), now the pipeline/API default; `stub_localize` retained as the teaching "before" baseline. Made the API test deterministic on a known board frame (`Sample1_003`, validated 8/8) and added a direct localizer unit test. **12/12 pytest green**; I verified the suite and eyeballed the overlay myself — boxes sit cleanly on all 8 ring cards, HUD + "Benji" overlay excluded. Recorded localization + the REST contract + the env choice as **confirmed** rows in the `PROJECT-PITCH.md` decisions table (superseding the provisional localization entry). This closes the "confirm the approach works first" milestone; next is the step-back into deepening (lesson modules for the validated foundation, then Stage 0's board/modal gate and Stage 2 identification).

## 2026-06-22 — Vertical slice lands + classical localization validated

**Context:** First code in `src/`. Two parallel sub-agents: one built the end-to-end REST skeleton (stub localizer/identifier), one ran a classical-localization spike on the real sampled frames to test the project's central architectural bet.

**What landed (skeleton):** `src/dbcv/` package — `schema.py` (Pydantic `GameStateSnapshot` v0.1.0, all coords relative, resolution read from media), `localize.py`/`identify.py` (pluggable interfaces + stubs), `pipeline.py`, `assemble.py`, `config.py`, `api.py` (FastAPI, `lifespan` "load once onto app.state" pattern, `POST /v1/snapshot` accepting an uploaded frame, **plain `def`** so CPU-bound inference runs in the threadpool per RESEARCH entry 5). `tests/` + repo-root `conftest.py` (puts `src/` on path). **11/11 pytest green.** CodeDocs synced: `sources/dbcv/*.md` overviews + `io/inputs.md`/`io/outputs.md` reconciled (schema bumped 0.0.0→0.1.0, `confidence` bounded [0,1], `role_class` a validated Literal).

**What the spike found (the important part):** **Classical, layout-based localization is viable** — confidence ~0.80. On clean board frames it hit **8/8 and 9/9 cards exact, zero false positives**, using HSV colour segmentation of card regions → morphology → external contours filtered by area/aspect → relative HUD-exclusion zones → IoU-NMS. All thresholds expressed relative to `min(W,H)` (no baked resolution). This validates the decision to NOT train a detector.

**What did NOT work / open risks:**
- **Board-vs-modal state gate is weak** (0/3 on modal frames). The game's modals are *dark*-backgrounded with bright text/art, so a center-brightness threshold misreads them as "board." Needs a better signal (pentagram-absence or modal-header detection). This is Stage 0's real problem, now concrete.
- **Numbered position badges are NOT usable as primary anchors via blob detection** — card clue/ability text panels produce indistinguishable bright blobs (badge blob detection over-fired 30–60/frame and was demoted). Badges may still work for *ordering* detected boxes via targeted `#`-glyph OCR. **Flag:** `knowledge-base/lessons/observed-board-layout.md` calls badges "ideal landmarks" — true for geometry, but the implementation note that *raw blob detection on them fails* should be added when we deepen Stage 1. (Not edited yet — flagged per Rule 3.)
- HSV hue ranges are tuned to this art set; an art swap = re-tune ranges (cheap, no training) — consistent with the "cheap to re-fit" thesis, but worth teaching as the honest caveat of the classical path.

**Choice:** Commit the skeleton as a rewindable checkpoint; next integrate the spike's `localize()` into `src/dbcv/localize.py` (replacing the stub) and make the API test deterministic on a known board frame.

**Notes / risks:** Sampled frames are **1280×720** (the frame sampler downscaled the 1920×1080 source) — code reads resolution from the image, so this is transparent. Spike artifacts (script + overlays) live in gitignored `scrap_scripts/`; the real `localize()` is promoted into `src/`.

## 2026-06-22 — Repo-local venv + start of the pipeline build (overseer run)

**Context:** Long unattended "overseer" session: spawn sub-agents to build the pipeline (Stages 0–5) toward the functional + teaching goals, committing as we go, no push. User set three guardrails up front: (1) **repo-local env**, and *teach* uv vs conda vs pip as a course module; (2) **confirm the approach works end-to-end first, then step back and deepen**; (3) at forks, **prefer the conservative/lighter/classical path** and log it.

**Choice — environment:** `python -m venv .venv` at repo root (gitignored) + pinned `requirements.txt`. Installed numpy 2.5, pillow 12.2, **opencv-python-headless 4.13**, fastapi 0.138, uvicorn 0.49, pydantic 2.13, httpx 0.28, pytest 9.1. Smoke-imported all via a scrap script (Rule 1 — never `python -c`). Standard interpreter for every script/agent from here: `.venv/Scripts/python.exe`.

**Why venv+pip over uv/conda:** zero-install, universally reproducible baseline — every Python ships `venv`; a learner can follow without first installing a tool. `opencv-python-headless` (not `opencv-python`) because the pipeline is server/batch, no GUI. uv (speed) and conda (binary deps) become the *alternatives* in the owed env-management lesson module, not a runtime requirement. Did **not** install onnxruntime/torch/imagehash yet — deferred until a stage's research justifies them (conservative path).

**Build plan:** thin **vertical slice** next — load an existing sampled frame → classical localization (the riskiest assumption) → placeholder identity → `GameStateSnapshot` → `POST /v1/snapshot` — to validate the architecture, schema, and REST contract on real frames before deepening any single stage. Then reassess. Recorded in `PLAN-pipeline.md` ("Build approach").

**Notes / risks:** A fresh venv is isolated, so the global numpy/fastapi/etc. do **not** carry in — `requirements.txt` is the source of truth. The env-management lesson module is owed (tracked in `PLAN-pipeline.md` cross-cutting). Localization viability on our footage is still unproven; the slice exists to find out early.

## 2026-06-22 — Claude launcher + townee clarification

- Added `.claude/launch.json` — the Claude Code desktop launcher (`local-server` → `python utils/python/serve_site.py --port 8000`). Verified via the preview MCP that it drives the running server and the site renders. Gitignored `.claude/settings.local.json` defensively (public repo). Backed out a `.vscode/launch.json` written from a first-pass misread of the request.
- Recorded game knowledge on the three ambiguous minion entries (**source: project player; not yet cross-checked against the wiki text**): `Minion` and `Twin Minion` are functionally identical (a lone one is *usually* `Minion` — not a hard rule, due to card/mode interactions) → collapse to **one recognition class** for CV; `Puppet` is **created by the `Puppeteer`** card (distinctions live on the cached Puppeteer page). Synced into `ROSTER.md`, `PROJECT-PITCH` (still-open → clarified), `PLAN-pipeline`, and the public `site/pages/notes.html` (verified rendered).

## 2026-06-21 — serve_site.py: flush startup banner

One-line follow-up: flush the local server's startup banner so the URL prints immediately even when stdout is captured/redirected (the Claude Code preview-window case). Also confirmed the first Pages run's `startup_failure` was Pages-not-yet-enabled; after enabling the Actions source the `workflow_dispatch` run deployed cleanly and the site returns 200 at the project subdirectory.

## 2026-06-21 — git/GitHub init, local server, Pages deploy, landing site

**Context:** Put the repo under version control and stand up a web surface. Motivation for the local server: the Claude Code desktop preview window can't render complex plain local HTML — it needs a real HTTP origin.

**What landed:**
- **`utils/python/serve_site.py`** — stdlib static server, binds `0.0.0.0`, no-cache headers, prints the LAN URL for the preview window. **Local only.**
- **`.github/workflows/deploy-pages.yml`** — GitHub Pages via Actions (`configure-pages` with `enablement: true` → `upload-pages-artifact` path `site/` → `deploy-pages`). Same `site/` content as the local server; Pages uses its own static stack. Project Pages URL is a subdirectory: `https://spiffy-panda.github.io/CompVisLesson-DemonBluff-Opus48/`.
- **`site/`** — `index.html` landing page listing pages; `pages/notes.html` (scratch/misc, first entry = rough plan outline); `assets/style.css`. **All links relative** so the site is byte-identical at `/` (local) and `/<repo>/` (Pages) — the one subtlety of project-Pages subdirectory hosting.

**Options considered / why:**
- *Local server tech:* stdlib `http.server` over FastAPI — the site is static; no need to couple the dev-preview server to the (future) lesson REST API. Kept them separate.
- *Pages mechanism:* Actions artifact deploy over "serve /docs from branch" — the user asked for an Action, and the artifact flow lets the source dir be `site/` and adds no Jekyll surprises.
- *Repo visibility:* **public** — free GitHub Pages requires it, and the user explicitly asked to host on Pages. Ran the Rule 6/7 name gate first (below).

**Public-surface gate (Rule 6/7):** grep for the dead name / real last name across the repo matched **only `CLAUDE.local.md`**, which is gitignored — confirmed excluded from the commit. Published `site/` is our own prose; no third-party bulk, no identity leakage. Commit identity is the handle `Spiffy-Panda` / handle email, not a real name.

**Notes / risks:** First push triggers the workflow; `enablement: true` should turn Pages on automatically — verified the run after pushing. Pages can take a minute to go live on first deploy. The harvester/probe/sampler remain in gitignored `scrap_scripts/` (not committed); the harvester is still a `utils/` promotion candidate if we want it tracked.

## 2026-06-21 — Wiki harvest + first research pass + frame tooling

**Context:** Same session as bootstrap, continued into the chosen scope (scaffold + harvest + first research). 

**What landed:**
- **Harvest** via `scrap_scripts/python/01_wiki_harvest.py` (MediaWiki API, stdlib urllib, fetch-once, polite UA + delay): 44 role pages (25/9/7/3), 26 mechanics pages, 67 card-art files. Raw cache + art gitignored; `wiki/harvest-manifest.json` and a transcribed `wiki/townees/ROSTER.md` tracked.
- **First research pass**: six A/B-sourced `research/RESEARCH.md` entries (frame selection, localization, identification, OCR, REST serving, compute budget), produced by six parallel background subagents, each briefed with Rule 1 verbatim + the project constraints.
- **Frame tooling**: `02_probe_video_meta.py` (ffprobe; both samples are 1920×1080 h264 60 fps, ~48 m / ~62 m — read from media, *not* baked in) and `03_sample_frames.py` (fast-seek uniform sampler → `dataset/frames/`). Inspected a spread of frames *without opening the raw video*.

**Key findings driving design:**
- Localization and identification are **separable**; the UI **layout is stable across art swaps** (radial card ring + numbered position badges), so localization is a **geometry problem, not a learned detector** — classical wins on speed, labels, and art-swap robustness.
- Identification should be **embedding-NN over a per-art gallery** (re-fit with reference images, zero training) with **name-label OCR** as a cross-check; a trained classifier is explicitly rejected for its retrain cost.
- On-card/HUD text is a **closed glyph set** → tiny custom recognizer beats general OCR; PaddleOCR-mobile (ONNX) is the narrow fallback.
- Runtime budget anchor: **≤~30 M params / ≤~100 MB**; SAM/Grounding-DINO/large ViT are **dev-only**. Train FP32 on Titan XP/Colab; no mixed-precision/INT8 on Pascal.
- **Real-world wrinkle:** sample footage has **streamer overlays** ("Benji") and transient **modals that occlude the board** → a board/menu/modal state-gate and overlay tolerance are mandatory. Logged in `knowledge-base/lessons/observed-board-layout.md`.

**Options considered / why:** Background subagents over inline research — keeps six independent web investigations off the main context and runs them concurrently with the harvest. MediaWiki API over HTML scraping — cleaner page/category/image enumeration and a natural "fetch once" key.

**Sync:** `PROJECT-PITCH.md` decisions table + direction updated; `KNOWLEDGE-BASE.md` inventory updated; `CodeDocs/io/*` already describe the intended contracts (no code yet, so no `sources/` overviews). No desync found.

**Notes / risks:** Harvester writes the tracked manifest, so it's a **promotion candidate** (`utils/`) once we re-run it for an art swap — left in `scrap_scripts/` for now. Card art may include base/mechanic pages (`Minion`, `Puppet`, `Twin Minion`) that aren't distinct faces — verify when building the gallery. RESEARCH directions are **provisional** (no code yet); supersede via the decisions table as the pipeline proves them.

## 2026-06-21 — Bootstrap

**Context:** New repo for a web-based course on modern computer vision, worked through one system: extracting *Demon Bluff* game state from video frames and serving it via REST. Two ~370 MB sample videos were the only contents of the root.

**Options considered:**
- *Repo shape:* code-only vs. prose-only vs. **mixed**. The deliverable is a course (prose) but it is built around a real CV pipeline (code).
- *Sample-video placement:* in-repo gitignored dataset folder vs. out-of-tree referenced by path.
- *First-session scope:* scaffold only vs. scaffold + wiki/townee harvest vs. scaffold + harvest + first research pass.

**Choice:** Mixed repo (both the code-doc tier and the deliverable-pairing tier are live). Videos moved to `dataset/raw-video/` and gitignored. First-session scope = scaffold + harvest + first research pass (user-selected).

**Why:** The course can't be written without the pipeline existing, and the pipeline can't be reasoned about without the cached game knowledge — so both tiers earn their keep. In-repo gitignored keeps the project self-contained while guaranteeing the 740 MB never lands in history.

**Notes / risks:**
- Added four non-skill folders beyond the standard scaffold: `research/` (RESEARCH.md log of every non-Demon-Bluff thing researched), `Lesson-Plan/` (the course), `knowledge-base/` (cached wiki + learned lessons), `dataset/` (large files).
- Card art and verbatim wiki text are third-party reference inputs — gitignored where bulk/verbatim (raw caches, `card-art/`), kept where small and transformative (transcribed `.md`). Public-surface gate (Rule 6) applies before any publish.
- Git not initialized (skill says wait until asked). `.gitignore` written ahead of time so the dataset, scrap, local file, and bulk third-party caches are covered the moment git appears.
- Constraints encoded in CLAUDE.md: runtime on a mid-grade gaming PC (heavy models dev-only; runtime models must train on Titan XP / Colab), never open the sample videos directly, never bake in a resolution, research-before-deciding, card recognition must be cheap to retrain.
