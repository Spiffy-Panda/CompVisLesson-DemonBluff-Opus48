# CV-Project Playbook (reusable)

Notes for spinning up the *next* "read game/UI state out of video with computer vision" project without rewriting the brief from scratch. Append a dated note whenever a reusable lesson emerges. This is the cross-project distillation; project-specific findings stay in `research/` and the rest of `knowledge-base/`.

## The one-paragraph brief (fill the blanks)

> Build a study/lesson on **modern computer vision** applied to **`<game/app>`**. Extract **`<what state>`** from **video frames only** (no audio, no internal files) and serve it via **`<REST/other>`**. Primary deliverable is the **lesson plan**; the pipeline is the worked example. Research professional/academic guidance before adopting any technique and log it. Don't add techniques just to cover material.

## Constraints worth copying verbatim

- **State a runtime compute budget up front** ("runs on a mid-grade gaming PC; runtime models train on a Titan XP / free Colab"). Let heavy models exist only for dataset creation, labeling, and debugging. This single constraint drives most architecture choices and makes the course honest.
- **Never load the raw long video into context.** Always insert a frame-selection stage; downstream only ever sees chosen frames. Mandate it as a rule, not a preference.
- **Never bake in a resolution.** Read it from the media; express geometry relatively. Samples sharing a resolution is a trap — don't depend on it.
- **Assume the art/skin changes.** Build any appearance-based recognizer to be cheap to retrain; keep templates/training refs separated and versioned.
- **Audio is off the table** if the spec says video-only — say so explicitly so nobody "helpfully" adds it.

## Repo scaffold that worked

- Bootstrapped from the init skill (CLAUDE.md rule set, production hierarchy, sync discipline, public-surface gate, identity rules in gitignored CLAUDE.local.md). See this repo's root files for the filled-in shape.
- **Four CV-specific folders beyond the standard scaffold:**
  - `research/` + `RESEARCH.md` — log of every *outside* (non-game) subject, each with **source / authority-trust / reason / abstract**. Gates technique adoption.
  - `knowledge-base/` — `wiki/` (fetch-once cache + transformative transcriptions) and `lessons/` (this playbook + per-project lessons).
  - `Lesson-Plan/` — the course (primary deliverable), problem-driven modules each citing a research entry.
  - `dataset/` — large media, **gitignored contents, tracked READMEs**; derived `frames/ crops/ state/` are regenerable.
- **Gitignore early** (even before `git init`): the local file, scrap scripts, dataset media, and bulk/verbatim third-party caches (wiki raw, card art). Force-include the explanatory READMEs.

## Third-party data hygiene (game wikis, art)

- **Fetch once, cache locally.** Prefer the wiki's **MediaWiki API** (`api.php?action=query`) over scraping HTML — cleaner page lists, categories, and image URLs.
- **Cache = verbatim & gitignored; transcription = transformative & tracked.** Keep the bulk out of git; keep tables/summaries/extracted fields in.
- **Card/sprite art is a reference input, not redistributable** — gitignore it, never put it in a public bundle (fair-use posture: small transformative excerpts only).

## Tooling posture

- Don't force a language. For CV the professional default is **Python + the standard CV/DL stack**, but justify it from `research/` rather than assuming.
- **No inline interpreter calls** (`python -c`, `node -e`). Real work goes in a file under `scrap_scripts/<lang>/` (throwaway) and graduates to `utils/<lang>/` (durable, cataloged) once depended on. Anchor every script to the repo root.
- Likely first scrap scripts: probe video metadata (incl. resolution) with `ffprobe`; sample/scene-detect frames with `ffmpeg`/`opencv`; the wiki API harvester.

## Sequence that worked

1. Scaffold + the four CV folders + move large media into gitignored `dataset/`.
2. Harvest game knowledge (wiki API, fetch-once cache, transcriptions, art download).
3. First research pass on the CV craft (frame selection, localization, identification, OCR, REST) → `RESEARCH.md`.
4. Only then start the pipeline, one decision/module at a time, each gated by research.

---

## Dated notes

### 2026-06-21 — bootstrap
First instance of this playbook, written during the *Demon Bluff* project bootstrap. The structure above is what we stood up; revise it here as later phases prove or disprove pieces.

### 2026-07-28 — always test degenerate inputs against similarity metrics
`cv2.compareHist(zeros, anything)` with the correlation metric returns **1.0** (Pearson 0/0, clamped) — so an all-black crop "perfectly matched" every gallery entry. Any similarity/correlation metric has a degenerate input that maxes it out; write the zero/empty/uniform-input test *before* trusting the metric, and guard (here: a zero-sum check) rather than assuming real inputs are always well-formed.

### 2026-07-28 — re-validate threshold *semantics*, not just values, after any re-fit
Fine-tuning the embedding backbone compressed the absolute cosine scale (correct match ~0.6, unrelated ~0.4), so the old absolute-cosine abstention threshold (0.60) silently over-identified **125/125** real cards — the number wasn't miscalibrated, the *quantity it thresholded* had stopped meaning anything. The fix was switching to a relative signal (top1−top2 margin). Lesson: a model re-fit can invalidate what a threshold measures, not just where it sits; re-derive every downstream cutoff (and the meaning of any exported `confidence`) as part of the retrain checklist.

### 2026-07-28 — validate "ideal landmark" assumptions on real frames; demote, don't discard
The numbered position badges looked like ideal localization anchors on inspection, but raw blob detection over-fired 30–60/frame (card clue-text panels produce indistinguishable bright blobs). Instead of abandoning them, they were **demoted from anchoring to ordering** — a role the noisy signal can still serve. Lesson: an eyeballed "obvious landmark" needs an implementation-level check on real frames, and a failed primary role may still leave a useful secondary one.

### 2026-07-28 — name the failure correctly before fixing it
When the frozen-ImageNet embedder collapsed all 43 stylised characters into one cluster, the tempting label was "neural collapse" — which would have suggested the prototype-gallery architecture was wrong. The correct diagnosis (from the literature) was **domain shift of frozen ImageNet features to stylised fine-grained art**, which says the *features* need adapting and the gallery is fine. That naming directly selected the fix (LP-FT fine-tune of the same backbone, gallery unchanged) and it worked (inter-prototype cosine 0.85 → 0.41). Lesson: research the failure mode's proper name first — the wrong name points the fix at the wrong component.
