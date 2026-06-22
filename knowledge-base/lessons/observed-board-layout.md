# Lesson — observed board layout (from sample footage)

What the *actual* sample frames show about the *Demon Bluff* board, derived by sampling frames (never opening the raw video) and inspecting them. This is ground truth for the CV pipeline; revise as we look at more frames. Source: `dataset/frames/Sample1/` and `Sample2/` (gitignored), 2026-06-21.

## The board

- **Cards are arranged in a radial ring** around a central pentagram/altar. Not a rectangular grid — a roughly circular/oval arrangement.
- **Each card carries a numbered position badge** (`#1`, `#2`, … `#8+`) at a consistent spot on the card. These badges are **art-independent UI chrome** — ideal landmarks for layout-based localization.
- **Board size varies between sessions** (Sample1 shows fewer cards than Sample2). The localizer must **derive the card count and positions from the layout**, never assume a fixed N. (Reinforces the no-baked-geometry rule.)
- **Each card shows:** the role **art**, a **name label** (text) beneath the art, and on revealed/acted cards a **clue/ability text** (e.g. "It's a 3", "I sense a Corruption"). Card borders/backings appear color-coded by state/alignment.

## HUD / non-card elements

- **Top-left:** the objective ("Find and Execute N Evil Characters") and running stats (Evils killed, Villagers count, Ascension, Score) — small stylized text + numbers.
- **Top-right:** resource/ability icons (cards/tokens).
- **Bottom-left:** a red circular **counter/timer** with a number (actions remaining or HP).
- **Right edge:** red demon icons.
- **Center:** transient **modal dialogs** ("Village is safe! All Evil characters have been executed", "Pick N characters") that **occlude part of the board** — the pipeline must detect and handle these states, not misread them as cards.

## Real-world robustness findings

- **Streamer overlays are present** — e.g. a handle ("Benji") rendered over the board, and likely webcam/alert regions in other clips. The runtime must tolerate **partial occlusion and arbitrary overlays**. Do not assume a clean game capture.
- Frames were inspected **downscaled to ≤1280 px wide**; the source is larger. Everything geometric must be **relative to the measured frame size** — confirms the no-baked-resolution rule end to end.

## Implications for the pipeline (cross-refs)

- **Localization** → geometry from the ring + numbered badges + UI anchors; classical, retrain-free. See `research/RESEARCH.md` localization entry, and [[cv-project-playbook]].
- **Identification** → art embedding-NN **and** name-label OCR as a cross-check; both re-fit from references on an art swap.
- **State detection** → a "board vs. menu vs. modal" gate is mandatory (modals occlude; menus aren't boards). See the frame-selection entry's board-gate recommendation.
- **On-card reading** → closed-vocabulary clue/HUD text favors a tiny custom recognizer over general OCR.
