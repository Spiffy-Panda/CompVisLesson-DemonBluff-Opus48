# research/ — the research log (human-facing)

This folder holds **[RESEARCH.md](RESEARCH.md)**: a log of **every non-Demon-Bluff subject we research** on the way to a design decision. Computer-vision techniques, libraries, model architectures, training tricks, compute budgets, OCR engines, REST patterns — anything we had to learn from the outside world before choosing.

It exists to honor the project's core rule: **research before deciding.** No non-trivial technique enters the pipeline or the course without a logged source, a trust rating, and an abstract of what we found. It is also what keeps the lesson plan honest — every claim in the course should trace back to an entry here.

It is LLM-authored. Each entry is rated for **authority / trust** so a reader can weigh it.

## What does *not* go here

Demon Bluff game facts (roles, mechanics, card data) go in `knowledge-base/`, not here. This log is specifically the *outside* knowledge — the CV/engineering craft — we pulled in.

## Adding an entry

Agents: read [RESEARCH.md](RESEARCH.md) for the full entry skeleton and the trust-rating rubric, then append. One entry per subject investigated; cite the real source; rate it honestly; keep the abstract to what you actually found, not what you hoped to find.

Kickoff prompt (paste-to-start, scenario-specific part only):

> Research `<subject>` for the decision `<which pipeline/course fork it informs>`. Find professional/academic guidance, prefer primary and recent sources, and append a RESEARCH.md entry using the skeleton there — including an honest authority/trust rating and an abstract of findings. Do not adopt the technique here; just report.
