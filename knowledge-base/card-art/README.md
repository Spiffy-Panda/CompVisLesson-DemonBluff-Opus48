# card-art/ — townee card images (GITIGNORED contents)

Downloaded *Demon Bluff* card art, one set of images per townee, used as **recognition templates / training references** for card identification.

## Why the contents are gitignored

This is third-party game art. Under the public-surface gate (Rule 6) it is a **reference input for the pipeline, not redistributable content** — it must never land in git history or a published bundle. Only this `README.md` is tracked (it is force-included in `.gitignore`); every image here is ignored.

## Conventions

- **Fetched once** during the harvest phase from the wiki, alongside the cached pages in `../wiki/_raw_cache/`.
- **Organized by role class** then townee: `card-art/<villager|minion|outcast|demon>/<townee>/...`.
- **Versioned for art swaps.** The art set can change to a limited alternate set; keep a note of which art-set version each image belongs to so a swap is traceable and the recognizer can be cheaply re-fit.
- **Referenced, not embedded.** The per-townee `.md` in `../wiki/townees/` points here; nothing tracked embeds these images.
