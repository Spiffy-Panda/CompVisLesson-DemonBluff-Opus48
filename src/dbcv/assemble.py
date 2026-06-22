"""
dbcv/assemble.py — Assemble per-card reads into a GameStateSnapshot.

The assembly stage is the last pure-data step: it takes the measured resolution,
a list of localized bounding boxes, and a parallel list of identification
results, then packages them into the versioned ``GameStateSnapshot`` that the
REST API returns.

In the full pipeline this stage will also handle temporal cues (smoothing
across frames, handling modal occlusions), but the vertical-slice version is
deliberately stateless: one frame → one snapshot.

Research grounding
------------------
Stateless assembly was the recommended starting point from research/RESEARCH.md
entry 5 ("Serving CV inference over a REST API") — keep the data path simple
until temporal logic is justified by footage analysis.
"""

from __future__ import annotations

from dbcv.schema import (
    CardRead,
    GameStateSnapshot,
    Readings,
    Resolution,
    Source,
)


def assemble(
    source: Source,
    resolution: Resolution,
    boxes: list[tuple[float, float, float, float]],
    identities: list[tuple[str, str, float]],
) -> GameStateSnapshot:
    """Build a ``GameStateSnapshot`` from pipeline outputs.

    Parameters
    ----------
    source:
        Provenance metadata for the frame (video ID, frame index, timestamp).
    resolution:
        Frame dimensions measured from the decoded image — never assumed.
    boxes:
        List of relative bounding boxes (x, y, w, h) in [0, 1], one per card,
        as returned by the localizer.
    identities:
        Parallel list of (identity, role_class, confidence) tuples as returned
        by the identifier.  Must have the same length as ``boxes``.

    Returns
    -------
    GameStateSnapshot
        A fully-formed snapshot ready for the REST response.

    Raises
    ------
    ValueError
        If ``boxes`` and ``identities`` have different lengths (indicates a
        logic error in the pipeline; the caller should fix it, not suppress it).
    """
    if len(boxes) != len(identities):
        raise ValueError(
            f"assemble(): boxes ({len(boxes)}) and identities ({len(identities)}) "
            "must have the same length — each box needs an identification result."
        )

    cards: list[CardRead] = []
    for bbox_rel, (identity, role_class, confidence) in zip(boxes, identities):
        # Validate role_class: the schema's Literal only accepts known values.
        # Anything outside the set is normalised to "unknown" here rather than
        # crashing — the identifier stub returns "unknown" anyway, and a real
        # identifier may occasionally return something unexpected during dev.
        valid_roles = {"villager", "minion", "outcast", "demon", "unknown"}
        safe_role = role_class if role_class in valid_roles else "unknown"

        cards.append(
            CardRead(
                bbox_rel=bbox_rel,
                role_class=safe_role,  # type: ignore[arg-type]  # validated above
                identity=identity,
                readings=Readings(),   # on-card readings are a later stage
                confidence=confidence,
            )
        )

    return GameStateSnapshot(
        source=source,
        resolution=resolution,
        cards=cards,
    )
