# LESSON-PLAN.md — course spec (agent-facing)

The agent-facing spec for the **primary deliverable**. Holds the course's structure, conventions, and a module inventory so an agent never has to open a module file to learn the shape.

## Audience & promise

Learners who know some programming and a little ML, and want to learn **modern computer vision the way it is actually practiced**. The promise: by the end, you can take an unglamorous real input (an hour of game video) and ship a constrained, explainable system that reads structured state out of it — and you understand *why* each technique was chosen over its alternatives.

## Non-negotiable course principles

1. **Problem-driven, not coverage-driven.** A technique is taught only because the *Demon Bluff* reader needed it. No "and here's another method for completeness."
2. **Every claim is cited.** Each module references the `research/RESEARCH.md` entry that justifies its technique choice. If a citation is missing, the research is missing — fix that first.
3. **Trade-offs are taught honestly**, including the cheaper method we *didn't* pick and why, and the failure modes of the one we did.
4. **The compute budget is a character in the story.** "Runs on a mid-grade gaming PC; trains on a Titan XP / Colab" shapes nearly every choice and the course says so out loud.
5. **Reproducible.** A learner can run the same frame-selection, localization, and recognition steps on the sample footage.

## Module skeleton

Each module is a file `Lesson-Plan/modules/<NN>_<slug>.md` (created when authored) with:

```
# Module NN — <title>

**The problem (in the pipeline):** the concrete decision this module exists to make.
**What you'll be able to do:** 2–4 learning outcomes.
**The options:** the candidate techniques, with the trade-off axis (accuracy / speed / data-need / retrain-cost / compute).
**What we chose and why:** the decision, citing `research/RESEARCH.md#<entry>`.
**Hands-on:** the step a learner runs against the sample footage (points at `utils/` scripts, never `scrap_scripts/`).
**Failure modes:** how it breaks and how we'd notice.
**Further reading:** the A/B-tier sources from research.
```

## Intended arc (subject to revision as research and pipeline land)

A draft spine, not a commitment — modules earn their place from real pipeline decisions:

1. Framing the problem: state-from-pixels under a compute budget; why video-only is hard.
2. Taming an hour of video: frame selection / keyframe strategies so nothing downstream sees raw video.
3. Resolution-agnostic geometry: why nothing is hard-coded to the sample's pixels.
4. Finding the cards: classical vs. learned localization, judged on art-swap robustness and the compute budget.
5. Naming the cards: template/embedding vs. a tiny classifier, and making retraining cheap.
6. Reading on-card text/numbers: OCR vs. a narrow recognizer.
7. From cards to state: assembling a snapshot, using temporal cues across frames.
8. Serving it: the REST contract and the game-state schema.
9. Staying alive: monitoring, retraining triggers when the art set changes, honest evaluation.

## Conventions

- **Hands-on points at `utils/`, never `scrap_scripts/`** (a published spec citing a gitignored path rots — Rule 1 / deliverable-pairing honesty rule).
- **No baked-in resolution** in any code a module shows.
- **Public-surface gate (Rule 6)** before any publish: no third-party bulk (wiki prose, card art), no identity leakage.

## Module inventory

| # | Slug | Status | Backing research |
|---|------|--------|------------------|
| 00 | `python-environments` | Authored — `modules/00_python-environments.md` | "Python environment & dependency management (venv/virtualenv/pip/conda/uv) — 2026-06-22" |
| 01 | _(framing-the-problem)_ | Planned | — |
| 02 | _(frame-selection)_ | Planned | "Frame selection / keyframe extraction from long gameplay video — 2026-06-21" |
| 03 | _(resolution-agnostic-geometry)_ | Planned | — |
| 04 | `card-localization` | Authored — `modules/04_card-localization.md` | "Card/region localization robust to art swaps under a tight compute budget — 2026-06-21" |
| 05 | _(card-identification)_ | Planned | "Card identification that is cheap to retrain when art changes — 2026-06-21" |
| 06 | _(on-card-text)_ | Planned | "Lightweight OCR for short on-card text and numbers in game UI — 2026-06-21" |
| 07 | _(game-state-assembly)_ | Planned | — |
| 08 | _(REST-serving)_ | Planned | "Serving CV inference over a REST API (Python) — 2026-06-21" |
| 09 | _(monitoring-and-retraining)_ | Planned | — |

**Placement note (Module 00):** The environment module is a prerequisite cross-cutting concern, not a CV technique in the 9-module arc. It is slotted as Module 00 so the arc (Modules 01–09) can remain intact. Learners should complete it before running any pipeline code.
