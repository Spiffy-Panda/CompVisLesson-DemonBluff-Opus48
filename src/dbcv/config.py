"""
dbcv/config.py — Runtime configuration via pydantic-settings.

Settings are read from environment variables (or a .env file) and validated
at startup.  No resolution is baked in here or anywhere else.

Usage
-----
    from dbcv.config import get_settings

    settings = get_settings()
    frames_dir = settings.frames_dir

``get_settings()`` is cached via ``@lru_cache`` so the same Settings object
is returned on every call — environment variables are read exactly once.

Adding new settings
-------------------
Add a class attribute with a type annotation and a default (or no default to
make it required).  pydantic-settings will look for an env var with the same
name (uppercased by default).  Never add a hard-coded resolution here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Repo-root anchor
# ---------------------------------------------------------------------------

# This file lives at src/dbcv/config.py.
# parents[0] = src/dbcv
# parents[1] = src
# parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Project-wide runtime settings.

    All path fields are resolved relative to the repo root so the code runs
    correctly regardless of the working directory at invocation time (Rule 1:
    anchor to the repo root, never assume CWD).

    Fields
    ------
    frames_dir:
        Directory containing the pre-sampled frame PNGs.  Defaults to
        ``dataset/frames`` under the repo root.
    confidence_threshold:
        Minimum identification confidence to accept a card read as valid.
        Below this threshold the card is still included in the snapshot but
        its role_class / identity are "unknown".  Placeholder for future use.
    identifier:
        Which Stage 2 identifier the API lifespan should build and serve.
        "embedding" (default, unchanged) -- the fine-tuned embedding-NN,
        margin-gated.  "classical" -- the HSV-histogram baseline.
        "ensemble" -- the opt-in classical+embedding composition layer
        (``dbcv.identify.make_ensemble_identifier``), added 2026-07-29 per
        plans/PLAN-live-capture.md Fix 3.  Set via ``DBCV_IDENTIFIER``.
    """

    model_config = SettingsConfigDict(
        env_prefix="DBCV_",   # env vars must start with DBCV_ (e.g. DBCV_FRAMES_DIR)
        env_file=".env",       # optional .env file at CWD; silently ignored if absent
        env_file_encoding="utf-8",
        extra="ignore",        # ignore unknown env vars rather than raising
    )

    frames_dir: Path = Field(
        default=_REPO_ROOT / "dataset" / "frames",
        description=(
            "Root directory for pre-sampled frames.  Sub-directories are "
            "named by sample set (e.g. Sample1/, Sample2/).  "
            "Never bake a resolution assumption into this path."
        ),
    )

    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence score for a card read to be treated as "
            "identified.  Below this threshold the read is kept in the "
            "snapshot but identity / role_class remain 'unknown'."
        ),
    )

    identifier: Literal["embedding", "classical", "ensemble"] = Field(
        default="embedding",
        description=(
            "Which Stage 2 identifier the API lifespan builds and serves. "
            "'embedding' (default) is unchanged from before 2026-07-29. "
            "'ensemble' opts into the classical+embedding composition layer "
            "from plans/PLAN-live-capture.md Fix 3."
        ),
    )


# ---------------------------------------------------------------------------
# Cached accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (constructed once from env)."""
    return Settings()
