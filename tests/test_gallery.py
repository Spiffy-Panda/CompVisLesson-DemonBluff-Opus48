"""
tests/test_gallery.py — Unit tests for the Stage 2 reference gallery builder.

Tests that:
  - build_gallery() succeeds without error.
  - The expected number of townees (unique identities) are loaded.
  - The expected number of reference images (>= townees, due to skins) are loaded.
  - All four role classes are present.
  - Every entry has a valid identity string and a role_class from the allowed set.
  - The alias rule holds: "Twin_Minion" dir → identity "Minion".
  - Rebuilding with a fresh call returns a consistent result (idempotent).

Rule 1 compliance: no inline interpreter calls.  No ``import`` on the command
line.  All code runs through pytest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dbcv.gallery import Gallery, GalleryEntry, build_gallery

# ---------------------------------------------------------------------------
# Paths — anchored to repo root
# ---------------------------------------------------------------------------

# conftest.py puts src/ on sys.path; __file__ is tests/test_gallery.py.
# parents[0] = tests/
# parents[1] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ART_ROOT = _REPO_ROOT / "knowledge-base" / "card-art"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gallery() -> Gallery:
    """Build the gallery once per test module."""
    if not _ART_ROOT.exists():
        pytest.skip(f"Card-art directory not found: {_ART_ROOT}")
    return build_gallery(_ART_ROOT)


# ---------------------------------------------------------------------------
# Gallery structure tests
# ---------------------------------------------------------------------------


def test_gallery_builds_without_error(gallery: Gallery) -> None:
    """build_gallery() completes and returns a non-empty Gallery."""
    assert isinstance(gallery, Gallery)
    assert gallery.n_references > 0
    assert gallery.n_townees > 0


def test_gallery_has_expected_townee_count(gallery: Gallery) -> None:
    """Gallery contains the expected number of distinct townee identities.

    67 PNGs across the tree; after alias resolution (Twin_Minion → Minion),
    the number of unique identities should be between 25 and 45.
    The exact count may shift as card art is added/removed, so we bound it.
    """
    assert 25 <= gallery.n_townees <= 45, (
        f"Expected 25–45 distinct townees; got {gallery.n_townees}. "
        f"Townees: {gallery.townee_names}"
    )


def test_gallery_has_expected_reference_count(gallery: Gallery) -> None:
    """Gallery contains at least as many references as townees (due to skins).

    Total PNGs in the tree is ~67.  Some may be filtered (unreadable files,
    etc.), so we require at least 50 and at most 80.
    """
    assert 50 <= gallery.n_references <= 80, (
        f"Expected 50–80 reference images; got {gallery.n_references}. "
        "Check knowledge-base/card-art/ for missing or added files."
    )
    # References >= townees because of skin variants
    assert gallery.n_references >= gallery.n_townees, (
        "More townees than references — impossible (each townee needs >= 1 image)."
    )


def test_gallery_has_all_four_role_classes(gallery: Gallery) -> None:
    """Gallery must contain exactly the four Demon Bluff role classes."""
    expected = {"villager", "minion", "outcast", "demon"}
    actual = set(gallery.role_classes)
    assert actual == expected, (
        f"Expected role classes {expected!r}; got {actual!r}. "
        "Check knowledge-base/card-art/ directory names."
    )


def test_gallery_role_class_derivable_from_path(gallery: Gallery) -> None:
    """Every entry's role_class is one of the four valid strings."""
    valid_roles = {"villager", "minion", "outcast", "demon"}
    for entry in gallery.entries:
        assert entry.role_class in valid_roles, (
            f"Entry {entry.file_stem!r} has role_class {entry.role_class!r}, "
            f"which is not in {valid_roles!r}."
        )


def test_gallery_twin_minion_aliased_to_minion(gallery: Gallery) -> None:
    """Twin_Minion directory entries must have identity='Minion' (alias applied)."""
    # Find all entries derived from the Twin_Minion dir by checking file_stem
    twin_entries = [
        e for e in gallery.entries if "Twin_Minion" in e.file_stem
    ]
    if not twin_entries:
        pytest.skip(
            "No Twin_Minion entries found in gallery (dir may not exist or be empty)."
        )
    for entry in twin_entries:
        assert entry.identity == "Minion", (
            f"Twin_Minion entry {entry.file_stem!r} has identity={entry.identity!r}; "
            "expected 'Minion' (alias rule)."
        )


def test_gallery_entries_have_valid_numpy_arrays(gallery: Gallery) -> None:
    """Every GalleryEntry must have non-empty numpy arrays for thumb and hsv_hist."""
    for entry in gallery.entries:
        assert isinstance(entry.thumb, np.ndarray), (
            f"Entry {entry.file_stem!r}: thumb is not a numpy array."
        )
        assert entry.thumb.size > 0, (
            f"Entry {entry.file_stem!r}: thumb array is empty."
        )
        assert isinstance(entry.hsv_hist, np.ndarray), (
            f"Entry {entry.file_stem!r}: hsv_hist is not a numpy array."
        )
        assert entry.hsv_hist.size > 0, (
            f"Entry {entry.file_stem!r}: hsv_hist is empty."
        )


def test_gallery_rebuild_is_idempotent(gallery: Gallery) -> None:
    """A second call to build_gallery() returns the same townee names."""
    gallery2 = build_gallery(_ART_ROOT)
    assert gallery2.townee_names == gallery.townee_names, (
        "Second build_gallery() call returned different townee names — "
        "the gallery builder is not idempotent."
    )
    assert gallery2.n_references == gallery.n_references, (
        "Second build_gallery() call returned different reference count."
    )


def test_gallery_skins_increase_reference_count(gallery: Gallery) -> None:
    """Characters with skin variants must appear in the gallery more than once.

    Alchemist, Empress, and Medium are known to have skin variants in the
    current art tree.  We check that each appears at least twice.
    """
    # Count references per identity
    counts: dict[str, int] = {}
    for entry in gallery.entries:
        counts[entry.identity] = counts.get(entry.identity, 0) + 1

    multi_skin_known = ["Alchemist", "Empress", "Medium"]
    for name in multi_skin_known:
        if name in counts:
            assert counts[name] >= 2, (
                f"{name} is known to have skin variants but only {counts[name]} "
                "reference(s) found in the gallery."
            )
