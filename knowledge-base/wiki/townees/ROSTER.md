# Townee roster (transcribed 2026-06-21 · re-verified against the wiki 2026-07-29)

Transformative index of every townee harvested from demonbluff.wiki.gg, grouped by role class. Built from the wiki's role categories; raw page content is cached (gitignored) under `../_raw_cache/<class>/` and indexed in `../../harvest-manifest.json`. Card art per role is in `../../card-art/<class>/<role>/` (gitignored).

Per-role **mechanics** transcriptions are intentionally deferred — we transcribe a role's rules into its own `.md` here when the pipeline or a lesson actually needs them, keeping this knowledge base transformative rather than a verbatim mirror of the wiki (Rule 6). This file is the spine those entries hang off.

**Counts:** 25 villagers · 9 minions · 7 outcasts · 3 demons = **44 roles**, 67 art files.

**Verification 2026-07-29** (wiki at v0.762a, 2026-07-14): all four role categories re-queried via the MediaWiki API — **membership identical, 44/44, no additions, removals, or renames** since the harvest. Only delta anywhere in the harvest: `Unused Roles` gained **Delusion** (12 → 13 pages), an unimplemented evil/demon role with placeholder art — not part of the recognition set. Patch notes v0.390a→v0.762a add no roles beyond those already listed.

## Villagers (good) — 25

Alchemist · Architect · Baker · Bard · Bishop · Confessor · Dreamer · Druid · Empress · Enlightened · Fortune Teller · Gemcrafter · Hunter · Investigator · Jester · Judge · Knight · Knitter · Lover · Medium · Oracle · Poet · Scout · Slayer · Witness

## Minions (evil) — 9

Chancellor · Minion · Poisoner · Puppet · Puppeteer · Shaman · Twin Minion · Werewolf · Witch

## Outcasts (neutral / independent) — 7

Bombardier · Doppelganger · Drunk · Lycanthrope · Plague Doctor · Rambler · Wretch

## Demons (evil, primary target) — 3

Baa · Lilis · Pooka

---

## Recognition notes (for the CV pipeline)

- **Two identity signals per card** were confirmed in the sample footage: the **card art** (image) and a **name-label text** under each card. Identification can use either or both (visual embedding-NN + OCR cross-check) — see `research/RESEARCH.md` (identification, OCR entries).
- **Art is swappable**; the recognizer must re-fit from new reference images only. The per-role `card-art/` folders are the reference gallery, versioned per art set.
- **`Minion` and `Twin Minion` are functionally identical.** Convention: a lone evil of this type is *usually* the `Minion` and a pair are `Twin Minion` — but "usually," not a hard rule (card interactions and game modes can change it). For CV this means they may share (or nearly share) art and are **not reliably separable visually** — treat them as a **single recognition class** and let downstream game-logic/context disambiguate. *(Source: project player; not yet cross-checked against the wiki text.)*
- **`Puppet` is not a normally-dealt role** — it is **created by the `Puppeteer`** card. The distinguishing mechanics live on the Puppeteer page (cached at `../_raw_cache/minion/Puppeteer.json`); transcribe on demand when the identification stage needs it.
- **Unused Roles** (13 pages) and **Relics** (8 pages) are cached under `../_raw_cache/knowledge/` for completeness but are not part of the core 44-role recognition set.

---

## Footage-version drift (established 2026-07-29) — read before building any recognizer

The roster above is the **current** (v0.762a) roster. The `dataset/` sample videos are **not** from this build, and the gap is load-bearing for the CV pipeline.

**How we know:** claim bubbles in the footage say **"Counsellor"**. That is not a missing role — it is `Chancellor`'s former name, renamed in **v0.390a (2025-10-28)**. So the footage is **≤ v0.389, October 2025 or earlier**; the wiki and our `card-art/` gallery are ~9 months newer.

| Consequence | What it means for us |
|---|---|
| `Chancellor` renders as **`COUNSELLOR`** on plates and in bubbles | The name vocabulary needs a **45th string**, `COUNSELLOR → Chancellor`, or every Chancellor plate in the dataset reads as an abstain. Build the vocabulary from *the footage's* roster, not the current one. |
| `Rambler` added in **v0.610**, `Investigator` in **v0.760e** | Both are **absent from our footage**. Any mined crop "identified" as one of these is a misread — useful as a free negative-check on the round-2 identifiers. Footage roster is effectively **42 roles**. |
| **v0.730** shipped new character-type / alignment icons | Footage UI chrome differs from anything current. Don't calibrate HUD/icon work against present-day screenshots or wiki images. |
| Halloween (v0.390a) and Christmas (v0.398b) **skins** exist for at least Baker, Bard, Medium, Architect, Lover, Empress, Knight | Concrete instance of the project's "card art can change" constraint — alternate art sets are real and already shipped, not hypothetical. |

**Standing rule:** treat the art-set version as a first-class attribute of every reference image and every mined crop. Reference art harvested from the wiki is the *current* set; it is a **weak prior** for footage-era cards, not ground truth.
