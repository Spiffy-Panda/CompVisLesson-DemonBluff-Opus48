"""
dbcv/schema.py — Pydantic v2 data models for the Demon Bluff CV pipeline.

These models define the contract between every pipeline stage and the REST API.
The schema is versioned (schema_version) so clients can detect breaking changes.

Design notes
------------
- ``Resolution`` is populated from the decoded image's shape at runtime.
  No pixel dimensions are ever hard-coded here or anywhere in the pipeline.
- ``CardRead.bbox_rel`` uses *relative* coordinates — fractions of the frame
  width and height — so the schema is resolution-agnostic.  Representation is
  (x, y, w, h) with origin at the top-left corner of the frame.
- ``Readings`` captures optional on-card text/numbers/state; all three fields
  are None until a real reader fills them in.
- ``role_class`` uses a closed Literal set that mirrors the Demon Bluff role
  taxonomy from knowledge-base/wiki.  ``"unknown"`` is the stub default.

Research grounding
------------------
The overall shape (resolution-relative boxes, per-card confidences, schema
version in the payload) was driven by research/RESEARCH.md entry 5
("Serving CV inference over a REST API") and the io/outputs.md draft.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Provenance: which video and which frame produced this snapshot."""

    video: str = Field(description="Identifier for the source video file (stem, not a path).")
    frame_index: int = Field(description="Zero-based frame index within the video.")
    timestamp_s: float = Field(description="Timestamp in seconds of this frame within the video.")


class Resolution(BaseModel):
    """Frame dimensions measured at runtime — never hard-coded.

    Both fields are populated by reading the decoded image's shape, so the
    pipeline stays correct even if the footage resolution changes.
    """

    w: int = Field(description="Frame width in pixels, read from the decoded image.")
    h: int = Field(description="Frame height in pixels, read from the decoded image.")


class Readings(BaseModel):
    """On-card text, number, and state-marker readings.

    All fields are optional; a real OCR/closed-vocab recognizer fills them in.
    Until then they are None (i.e. 'not yet read').
    """

    text: str | None = Field(
        default=None,
        description="Free-form or role-name text found on the card (e.g. the townee name).",
    )
    number: int | None = Field(
        default=None,
        description="A numeric reading from the card (e.g. ability count, HUD digit).",
    )
    state: str | None = Field(
        default=None,
        description="A discrete state marker on the card (e.g. 'poisoned', 'dead').",
    )


class CardRead(BaseModel):
    """Everything the pipeline knows about one localized card.

    bbox_rel
        Bounding box as (x, y, w, h) fractions of the frame's width and height
        respectively, origin top-left.  All values must be in [0.0, 1.0].
        Keeping coordinates relative lets the rest of the system stay
        resolution-agnostic: convert to pixels only when you need pixels.

    role_class
        The high-level Demon Bluff role family (villager/minion/outcast/demon).
        ``"unknown"`` is used by the stub and whenever confidence is too low.

    identity
        The specific townee name within the role class (e.g. "Alchemist").
        ``"unknown"`` until a real identifier runs.

    confidence
        A float in [0.0, 1.0] representing the identifier's certainty.
        Allows downstream consumers (the lesson plan, a confidence gate) to
        decide how to treat uncertain reads.
    """

    bbox_rel: tuple[float, float, float, float] = Field(
        description=(
            "Bounding box as (x, y, w, h) relative to the frame "
            "(fractions in [0, 1], origin top-left)."
        )
    )
    role_class: Literal["villager", "minion", "outcast", "demon", "unknown"] = Field(
        description="High-level role family."
    )
    identity: str = Field(
        description="Specific townee name, or 'unknown' if not identified."
    )
    readings: Readings = Field(
        default_factory=Readings,
        description="Optional on-card text/number/state readings.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Identifier confidence in [0.0, 1.0].",
    )


class GameStateSnapshot(BaseModel):
    """The top-level output of the pipeline — one complete board read.

    Returned by POST /v1/snapshot and optionally written to dataset/state/
    for debugging.  The ``schema_version`` field lets clients detect when the
    contract changes; bump it on any breaking change.
    """

    source: Source = Field(description="Provenance: video ID, frame index, timestamp.")
    resolution: Resolution = Field(
        description="Frame dimensions read from the decoded image — never assumed."
    )
    cards: list[CardRead] = Field(
        default_factory=list,
        description="One entry per card found by the localizer.",
    )
    schema_version: str = Field(
        default="0.1.0",
        description=(
            "Semver string for this schema.  Clients should check this field "
            "before parsing.  Bump on any breaking change to the snapshot shape."
        ),
    )
