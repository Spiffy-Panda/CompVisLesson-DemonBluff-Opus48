"""
tests/test_pipeline.py — Unit tests for dbcv.pipeline.run_pipeline.

Focused on the 2026-07-29 Kill-Mode red-tint abstention wiring (
plans/PLAN-live-capture.md, Fix 1's "at minimum" tint item) — everything
else about run_pipeline (gate-first, crop_relative, assembly) is already
exercised indirectly via tests/test_frame_state.py and tests/test_identify.py's
API integration tests. These tests use stub localizer/identifier/frame_state_fn
callables (no gallery, no ONNX) so they run fast and unconditionally.

Rule 1 compliance: no inline interpreter calls. No ``import`` on the command
line. All code runs through pytest.
"""

from __future__ import annotations

import numpy as np

from dbcv.pipeline import crop_relative, run_pipeline
from dbcv.schema import Source

_SOURCE = Source(video="synthetic", frame_index=0, timestamp_s=0.0)


def _frame(w: int = 640, h: int = 360) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _one_box_localizer(image: np.ndarray, resolution) -> list[tuple[float, float, float, float]]:
    return [(0.3, 0.3, 0.2, 0.2)]


# ---------------------------------------------------------------------------
# crop_relative sanity (not previously covered by a dedicated test module)
# ---------------------------------------------------------------------------


def test_crop_relative_returns_expected_shape() -> None:
    img = _frame(w=100, h=200)
    crop = crop_relative(img, (0.1, 0.2, 0.3, 0.4))
    # crop_relative converts each edge (x0, x1=x0+w) to pixels independently,
    # then takes the difference -- match that exactly rather than assuming
    # round(w * W) == round((x + w) * W) - round(x * W) (not true in general).
    assert crop.shape[1] == round((0.1 + 0.3) * 100) - round(0.1 * 100)
    assert crop.shape[0] == round((0.2 + 0.4) * 200) - round(0.2 * 200)


# ---------------------------------------------------------------------------
# Kill-Mode red-tint abstention wiring
# ---------------------------------------------------------------------------


def test_tint_inactive_leaves_confidence_untouched() -> None:
    """tint_fn reporting False must not alter the identifier's raw output."""
    snapshot = run_pipeline(
        image=_frame(),
        source=_SOURCE,
        localizer=_one_box_localizer,
        identifier=lambda crop: ("Wretch", "outcast", 0.80),
        frame_state_fn=lambda _img: "board",
        tint_fn=lambda _img: False,
    )
    assert len(snapshot.cards) == 1
    assert snapshot.cards[0].identity == "Wretch"
    assert snapshot.cards[0].confidence == 0.80


def test_tint_active_discounts_confidence_above_floor() -> None:
    """A confidence that stays above the floor after discounting keeps its identity."""
    snapshot = run_pipeline(
        image=_frame(),
        source=_SOURCE,
        localizer=_one_box_localizer,
        identifier=lambda crop: ("Wretch", "outcast", 0.80),
        frame_state_fn=lambda _img: "board",
        tint_fn=lambda _img: True,
        tint_confidence_discount=0.7,
        tint_confidence_floor=0.5,
    )
    card = snapshot.cards[0]
    assert card.identity == "Wretch"   # 0.80 * 0.7 = 0.56, still >= 0.5 floor
    assert card.confidence == 0.56


def test_tint_active_reabstains_below_floor() -> None:
    """A confidence that drops below the floor after discounting is forced to unknown."""
    snapshot = run_pipeline(
        image=_frame(),
        source=_SOURCE,
        localizer=_one_box_localizer,
        identifier=lambda crop: ("Hunter", "villager", 0.55),
        frame_state_fn=lambda _img: "board",
        tint_fn=lambda _img: True,
        tint_confidence_discount=0.7,
        tint_confidence_floor=0.5,
    )
    card = snapshot.cards[0]
    # 0.55 * 0.7 = 0.385, below the 0.5 floor -> re-abstained
    assert card.identity == "unknown"
    assert card.role_class == "unknown"
    assert card.confidence == 0.385


def test_tint_none_disables_discount_entirely() -> None:
    """Passing tint_fn=None skips the tint step even for a would-be-tinted frame."""
    snapshot = run_pipeline(
        image=_frame(),
        source=_SOURCE,
        localizer=_one_box_localizer,
        identifier=lambda crop: ("Hunter", "villager", 0.55),
        frame_state_fn=lambda _img: "board",
        tint_fn=None,
    )
    card = snapshot.cards[0]
    assert card.identity == "Hunter"
    assert card.confidence == 0.55


def test_tint_discount_is_identifier_agnostic() -> None:
    """The discount applies uniformly regardless of which identifier produced the result --
    it operates on the returned tuple, not on identifier internals."""
    calls = []

    def _tracking_identifier(crop: np.ndarray) -> tuple[str, str, float]:
        calls.append(1)
        return ("Judge", "villager", 0.9)

    snapshot = run_pipeline(
        image=_frame(),
        source=_SOURCE,
        localizer=_one_box_localizer,
        identifier=_tracking_identifier,
        frame_state_fn=lambda _img: "board",
        tint_fn=lambda _img: True,
    )
    assert len(calls) == 1   # identifier still called exactly once per box
    assert snapshot.cards[0].confidence == round(0.9 * 0.7, 4)


def test_non_board_frame_never_calls_tint_fn() -> None:
    """Modal/menu frames skip localization+identification entirely, so tint_fn
    (which would otherwise run on every board frame) is never invoked either --
    there are no identities to discount."""
    calls = []

    def _tracking_tint(image: np.ndarray) -> bool:
        calls.append(1)
        return True

    snapshot = run_pipeline(
        image=_frame(),
        source=_SOURCE,
        localizer=_one_box_localizer,
        identifier=lambda crop: ("Wretch", "outcast", 0.8),
        frame_state_fn=lambda _img: "modal",
        tint_fn=_tracking_tint,
    )
    assert snapshot.frame_state == "modal"
    assert snapshot.cards == []
    assert calls == []
