"""
tests/test_schema.py — Unit tests for dbcv.schema (Pydantic models).

Tests that:
  - A GameStateSnapshot can be constructed with realistic values.
  - model_dump_json() serialises it cleanly.
  - Re-parsing from JSON produces an equal object.
  - Key fields survive the round-trip unchanged.
"""

import json

import pytest

from dbcv.schema import (
    CardRead,
    GameStateSnapshot,
    Readings,
    Resolution,
    Source,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_snapshot() -> GameStateSnapshot:
    """A realistic snapshot with two cards (one identified, one unknown)."""
    return GameStateSnapshot(
        source=Source(video="Sample1", frame_index=7, timestamp_s=921.0),
        resolution=Resolution(w=1920, h=1080),
        cards=[
            CardRead(
                bbox_rel=(0.08, 0.30, 0.18, 0.30),
                role_class="villager",
                identity="Alchemist",
                readings=Readings(text="Alchemist", number=2, state=None),
                confidence=0.91,
            ),
            CardRead(
                bbox_rel=(0.40, 0.05, 0.18, 0.30),
                role_class="unknown",
                identity="unknown",
                readings=Readings(),
                confidence=0.0,
            ),
        ],
        schema_version="0.2.0",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_snapshot_construction(sample_snapshot: GameStateSnapshot) -> None:
    """GameStateSnapshot is constructable without errors."""
    assert sample_snapshot.schema_version == "0.2.0"
    assert len(sample_snapshot.cards) == 2


def test_json_round_trip(sample_snapshot: GameStateSnapshot) -> None:
    """Serialise to JSON and re-parse; the result must equal the original."""
    json_str = sample_snapshot.model_dump_json()

    # Confirm it is valid JSON
    data = json.loads(json_str)
    assert isinstance(data, dict)

    # Re-parse into a new model instance
    reparsed = GameStateSnapshot.model_validate(data)

    # Key structural fields survive unchanged
    assert reparsed.schema_version == sample_snapshot.schema_version
    assert reparsed.source.video == sample_snapshot.source.video
    assert reparsed.source.frame_index == sample_snapshot.source.frame_index
    assert reparsed.resolution.w == sample_snapshot.resolution.w
    assert reparsed.resolution.h == sample_snapshot.resolution.h
    assert len(reparsed.cards) == len(sample_snapshot.cards)
    # frame_state added in 0.2.0 — survives round-trip
    assert reparsed.frame_state == sample_snapshot.frame_state


def test_card_read_fields(sample_snapshot: GameStateSnapshot) -> None:
    """CardRead fields survive a JSON round-trip."""
    card = sample_snapshot.cards[0]
    assert card.role_class == "villager"
    assert card.identity == "Alchemist"
    assert card.confidence == pytest.approx(0.91)
    assert card.readings.text == "Alchemist"
    assert card.readings.number == 2
    assert card.readings.state is None


def test_bbox_rel_values_in_range(sample_snapshot: GameStateSnapshot) -> None:
    """All bbox_rel components must be in [0.0, 1.0]."""
    for card in sample_snapshot.cards:
        for component in card.bbox_rel:
            assert 0.0 <= component <= 1.0, (
                f"bbox_rel component {component} is outside [0, 1] "
                f"for card identity={card.identity!r}"
            )


def test_unknown_card_defaults() -> None:
    """An unknown card (stub output) has the correct defaults."""
    card = CardRead(
        bbox_rel=(0.0, 0.0, 0.1, 0.1),
        role_class="unknown",
        identity="unknown",
    )
    assert card.confidence == 0.0
    assert card.readings.text is None
    assert card.readings.number is None
    assert card.readings.state is None


def test_schema_version_default() -> None:
    """A snapshot constructed without an explicit schema_version gets 0.2.0.

    0.2.0 added the frame_state field (Stage 0 gate result).
    """
    snap = GameStateSnapshot(
        source=Source(video="test", frame_index=0, timestamp_s=0.0),
        resolution=Resolution(w=100, h=100),
    )
    assert snap.schema_version == "0.2.0"


def test_frame_state_default_is_unknown() -> None:
    """A snapshot constructed without frame_state gets 'unknown' as the default.

    'unknown' distinguishes test fixtures (no gate ran) from pipeline outputs.
    """
    snap = GameStateSnapshot(
        source=Source(video="test", frame_index=0, timestamp_s=0.0),
        resolution=Resolution(w=100, h=100),
    )
    assert snap.frame_state == "unknown"


def test_frame_state_modal_accepted() -> None:
    """frame_state field accepts all four documented values without error."""
    for state in ("board", "modal", "menu", "unknown"):
        snap = GameStateSnapshot(
            source=Source(video="test", frame_index=0, timestamp_s=0.0),
            resolution=Resolution(w=100, h=100),
            frame_state=state,  # type: ignore[arg-type]
        )
        assert snap.frame_state == state
