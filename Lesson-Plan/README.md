# Lesson-Plan/ — the course (human-facing)

This folder holds the **primary deliverable**: a web-based course that teaches modern computer vision by building the *Demon Bluff* state-reader as its single worked example. The agent spec is **[LESSON-PLAN.md](LESSON-PLAN.md)**.

The course's guiding principle is the same as the repo's: **techniques appear because the problem demanded them**, never to fill out a syllabus. Every module is anchored to a real decision made while building the pipeline and cites the supporting entry in `research/RESEARCH.md`.

It is LLM-authored. If it is ever published, it crosses a public surface — the Rule 6 gate (no third-party bulk, no identity leakage) applies before any deploy.

## Where it surfaces

A web-based course. The delivery stack (static site, slides, notebook-backed, etc.) is an open design question in `PROJECT-PITCH.md`, decided once enough of the pipeline exists to teach from. Until then this folder holds the **structure and content plan**, not a built site.

## Adding / extending a module

Agents: read [LESSON-PLAN.md](LESSON-PLAN.md) for the module skeleton, the citation convention, and the running inventory of modules — so you never have to open another module file to learn the shape.

Kickoff prompt (paste-to-start, scenario-specific part only):

> Draft/extend the lesson-plan module on `<topic>`. Anchor it to the real pipeline decision it teaches, cite the backing `research/RESEARCH.md` entry (add one first if it's missing), follow the module skeleton in LESSON-PLAN.md, and update the inventory there. Teach the trade-off honestly; do not include a technique we didn't actually need.
