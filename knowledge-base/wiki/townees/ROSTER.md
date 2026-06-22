# Townee roster (transcribed 2026-06-21)

Transformative index of every townee harvested from demonbluff.wiki.gg, grouped by role class. Built from the wiki's role categories; raw page content is cached (gitignored) under `../_raw_cache/<class>/` and indexed in `../../harvest-manifest.json`. Card art per role is in `../../card-art/<class>/<role>/` (gitignored).

Per-role **mechanics** transcriptions are intentionally deferred — we transcribe a role's rules into its own `.md` here when the pipeline or a lesson actually needs them, keeping this knowledge base transformative rather than a verbatim mirror of the wiki (Rule 6). This file is the spine those entries hang off.

**Counts:** 25 villagers · 9 minions · 7 outcasts · 3 demons = **44 roles**, 67 art files.

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
- **Unused Roles** (12 pages) and **Relics** (8 pages) are cached under `../_raw_cache/knowledge/` for completeness but are not part of the core 44-role recognition set.
