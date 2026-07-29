# PLAN — temporal-assembly

**Goal:** resolve the last open design question (temporal-logic depth) and the REST contract it implies (stateless frame-sequence vs. server-side session), then implement Stage 4 state assembly.

**Status:** options memo + decision written 2026-07-28 (wave-1 research agent); research logged in `research/RESEARCH.md` ("Temporal aggregation of per-frame CV reads", 2026-07-28). **The user delegated the decision to the course** ("this is a lesson — the professor chooses and shows the work"), so the DECISION section below makes the call and carries the rationale module 07 will teach.

---

## Decision memo — temporal depth × fusion mechanism

### Context (what Stage 4 must solve)

- The pipeline reads game state **per frame**; the REST contract is a stateless single-frame `POST /v1/snapshot` (multipart image → `GameStateSnapshot`, schema 0.2.0).
- Across frames the raw reads exhibit: **identity flicker** (a slot's identity toggling between reads), **modal occlusion** (whole frames where `frame_state != "board"` and `cards` is empty), and **one-way reveals** (face-down → face-up over the course of a game).
- Identifier `confidence` is now a **top1−top2 cosine margin** (decisiveness), *not* a probability. Any fusion must weight by margin, not treat it as P(correct).
- Cards sit in **fixed board slots** — cross-frame association is deterministic by slot (nearest slot center / ring position), so no motion model or Hungarian association is needed. Research finding (d): full MOT is overkill here; Stage 4 reduces to **K independent per-slot label filters** plus slot matching across variable card counts.
- Constraints: mid-grade PC runtime (fusion must be numpy-cheap — all options below are), and this becomes **lesson module 07**, so pedagogical clarity is a first-class criterion.

### AXIS A — temporal depth (who holds the history?)

| | Contract | Costs | Benefits |
|---|---|---|---|
| **A1 — per-frame cold reads + client-side smoothing** | Server unchanged (`POST /v1/snapshot`, one frame). Smoothing is the *client's* problem; we ship at most a reference client helper. | Fusion logic lives outside the pipeline → every consumer reimplements it; module 07 would teach code that isn't in the served path; cross-frame semantics (staleness, reveals) never appear in the API contract. | Zero server change, zero schema change; server stays perfectly stateless and idempotent. |
| **A2 — windowed fusion (stateless sequence request)** | New `POST /v1/snapshot/window`: multipart **sequence of frames** (ordered, with per-frame `frame_index`/`timestamp_s`) → **one fused** snapshot. Server fuses internally, holds nothing between requests. | Client must buffer N frames; larger request payloads; per-request latency ≈ N × per-frame cost; belief cannot accumulate beyond one request's window. | Server stays **stateless** (same request → same answer; trivially testable); fusion is a **pure function** `fuse(list[per-frame reads]) → fused state` — ideal teaching shape; single-frame endpoint kept as the degenerate N=1 case. |
| **A3 — stateful session** | `POST /v1/session` → id; `POST /v1/session/{id}/frame` streams frames; response is the session's current fused state. Server holds per-slot beliefs across requests. | Session lifecycle (create/expire/evict), concurrency (locking per session), memory management, harder tests, breaks statelessness — the largest code and lesson surface; most of that surface is web-service plumbing, not CV. | Truest to the live use case (a bot polling a running game); belief accumulates over the whole game; O(1) payload per request. |

**Key structural fact:** with fixed slots, per-slot belief state is tiny (a label→weight dict + staleness counter per slot). A3's fusion core and A2's fusion core are the **same pure function**; A3 only adds session plumbing around it. So choosing A2 now does not burn the bridge to A3 — it builds A3's engine.

### AXIS B — fusion mechanism (conditioned on A2/A3)

| | Mechanism | Costs | Benefits |
|---|---|---|---|
| **B1 — margin-weighted vote** | Per slot, each frame's read votes for its identity with weight = margin; abstentions ("unknown") vote nothing. Winner = max accumulated weight; fused confidence = winner share (top1−top2 of *accumulated* weights, keeping margin semantics end-to-end). | Equal weight to old and new frames → slow to accept a genuine change within the window. | Uses the margin exactly as it is (decisiveness weight — no probability pretense, no calibration needed); one pass, no parameters beyond the window. |
| **B2 — recency-weighted (exponentially decayed) belief** | Same vote, but each frame's weight is margin × decay^(age). One knob (decay/α); O(1) memory per slot under A3. | One more parameter to justify/tune; still not a probability model. | Responds faster to sustained change (reveals!) while smoothing flicker; the standard streaming-smoothing tool (FPP3, RESEARCH entry family (b)). |
| **B3 — discrete Bayes filter (per-slot HMM forward pass)** | Belief vector over {43 identities} ∪ {face_down, unknown}; sticky transition matrix (identity self-loop ≈ 1, face_down→face_up allowed, reverse ≈ 0); observation likelihood from the identifier; modal frames skip the update step. | **Requires an observation model** — margin must be calibrated to P(correct) on *labeled real board crops*, which do not exist yet (couples to round-2 dataset-building). Uncalibrated, B3 collapses into a ceremonial B2. | The principled endpoint; occlusion handled with zero special cases; the transition matrix makes the reveal asymmetry explicit and teachable. |

**Reveals and gaps in the snapshot (any of B1–B3):**

- **Reveal latch:** face-down→face-up is (near) one-way. Once a slot's identity is confidently established (accumulated support over a threshold), hold it; require *k* consecutive confident contrary reads to flip (hysteresis). Represent as per-card `revealed: bool`.
- **Modal gaps = missing observations:** hold each slot's last fused state; do **not** delete (the slot never leaves the board — SORT's `max_age` deletion half does not transfer). Expose the gap honestly: per-card `last_seen_frame: int` + `staleness_frames: int` (frames since the slot was last read on a board frame).
- **Schema bump → 0.3.0** (additive): per-card `revealed`, `last_seen_frame`, `staleness_frames`, `support` (accumulated fused weight); snapshot-level `fused_from: int` (count of contributing board frames) and `window: {first_frame, last_frame}`. Single-frame responses set `fused_from = 1`, staleness 0 — one schema serves both endpoints.

---

## DECISION (2026-07-28)

*(The user delegated this call to the course: "this is a lesson, you are the professor — make the choice and describe why in the course." What follows is the decision and its rationale in the voice module 07 will teach it in; the module author should be able to lift it nearly verbatim.)*

**Chosen: A2 + B2.** Stage 4 is a **stateless windowed-fusion endpoint** — `POST /v1/snapshot/window` accepts an ordered sequence of frames and returns **one fused snapshot** — whose core is a pure per-slot fusion function implementing a **margin-weighted, recency-decayed vote** with a **reveal latch** and **explicit staleness fields**, under **schema 0.3.0**. No multi-object tracker. No server-side session. The discrete Bayes filter is taught, not shipped — deferred until margins can be calibrated to likelihoods.

### How the professor chooses (module 07 rationale)

Start where every temporal-fusion problem should start: **ask what actually varies over time, and what doesn't.** In our footage, three things vary — the reported identity of a card flickers between reads, whole frames disappear behind modals, and face-down cards reveal. One thing does *not* vary: **where the cards are.** They sit in fixed board slots that the localizer finds fresh in every frame. That single observation eliminates half the design space, so take the candidates in order of how much machinery they'd have us buy.

**Why not a tracker — the option most tutorials reach for first.** Multi-object tracking (SORT, ByteTrack, DeepSORT) is machinery for **data association under motion**: which detection in frame *t+1* is the same object as this detection in frame *t*? Kalman motion models, Hungarian matching, re-identification embeddings — all of it exists to answer that question. Our association question is answered by the game itself: a card read in slot 3 *is* the card in slot 3. When association is free, a "tracker" degenerates into K independent per-slot label filters — which is what we should build directly. The lesson generalizes: **buy machinery for the problem you have, not the problem the literature usually has.** In a different system — physical cards sliding on a table, a handheld camera — tracking-by-detection would be exactly the right answer, and SORT's own finding (detector quality dominates; keep the tracker minimal) would be the guide for how much of it to buy.

**Why not client-side smoothing (A1).** We could keep the server untouched and tell every consumer to smooth the flicker themselves. But the *semantics* of fused state — "this identity has 40 frames of support," "this slot hasn't been seen since the modal opened" — belong in the API contract, and a raw per-frame contract cannot express them. Push fusion to the client and every consumer reimplements it, each slightly differently, and the course's own worked example would omit the very stage the module teaches. **Put the honest semantics where the contract is.** A1 is the right answer in a different system: when consumers have wildly different latency/stability needs and the server must stay a dumb sensor.

**Why not a stateful session (A3).** The seductive option — it models the "real" live-bot use case. But walk through what it adds: session creation and expiry, per-session locking, memory eviction, restart semantics. All of that is web-service plumbing, none of it is computer vision, and — the decisive fact — **the fusion core inside A3 is the identical pure function A2 needs.** A3 is A2 plus plumbing. When extra state buys no new capability for the deliverable at hand, statelessness wins by default: same request, same answer; trivial tests; nothing to leak. If a live bot ever polls a running game, we wrap the same function in session endpoints that day and lose nothing by having waited. A3 is the right answer in a different system: a long-running deployment where clients genuinely cannot buffer frames and belief must accumulate across minutes, not seconds.

**Why not a flat vote (B1).** Within a window, equal weights make the oldest frame exactly as loud as the newest — which fights the one transition that is legitimately real and directional: the reveal. A face-up card should not have to out-vote its own face-down history at parity. Recency decay is the one-parameter fix, and it is the standard streaming-smoothing tool for a reason.

**Why not the Bayes filter (B3) — yet.** The discrete Bayes filter is the *principled* formulation, and module 07 teaches it as such: the transition matrix states the reveal asymmetry explicitly (face-down→face-up allowed, the reverse ~forbidden), and a modal frame is handled with no special case at all — a missing observation simply skips the update step. But a Bayes filter is only as honest as its **observation model**, P(observed label | true state), and our identifier's confidence is a **top1−top2 cosine margin — a decisiveness score, not a probability.** Calibrating margin → likelihood requires labeled real board crops, which do not exist yet (that is precisely the round-2 dataset-building work). Deploy the filter uncalibrated and the math *looks* principled while numerically collapsing into the weighted vote we could have defended directly. **A course should ship the simplest mechanism that honestly solves the problem — and say out loud that that is what it is doing.** The margin-weighted vote uses the margin exactly as what it is (a weight), needs zero calibration, and leaves a clean upgrade: once round-2 labels exist, swap the vote for a forward pass and keep everything else.

**The decision procedure, made portable.** A student facing a different game should re-run these questions and be free to land elsewhere:

1. Is cross-frame association free (fixed regions/slots)? → per-region label filters, no tracker. Objects move? → tracking-by-detection.
2. Is per-frame confidence a calibrated probability? → Bayes filter. A ranking/margin score? → weighted vote now, calibrate later if the stakes justify it.
3. Must fused state outlive one request? → session state. Otherwise → stateless window.
4. Are some state transitions one-way (reveals, deaths, unlocks)? → encode the asymmetry explicitly (latch/hysteresis, or a transition matrix when you graduate to B3).

### Bound to wave 2 (implementation notes, not re-decisions)

- Schema **0.3.0** as specified above (per-card `revealed`, `last_seen_frame`, `staleness_frames`, `support`; snapshot `fused_from`, `window`). Single-frame `POST /v1/snapshot` stays, returns `fused_from = 1`.
- **Decay by timestamp delta, not frame count** — clients sample at arbitrary strides, and the no-baked-assumptions rule applies to time exactly as it does to resolution.
- Window length, decay constant, and latch thresholds are calibrated on sample footage in wave 2 and documented as constants with provenance (like `_EMBED_MARGIN_THRESHOLD`), never silently hard-coded.

---

## Tasks

- [x] RESEARCH.md entry: temporal smoothing / multi-frame state fusion for per-frame CV reads *(2026-07-28 — "Temporal aggregation of per-frame CV reads")*
- [x] Options memo: temporal depth × REST contract, with recommendation *(2026-07-28 — this file)*
- [x] Decision: user delegated the call to the course (2026-07-28) → decided in-memo (A2 + B2) → PROJECT-PITCH decisions row added
- [ ] Stage 4 implementation (wave 2, post-decision)
