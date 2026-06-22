"""
conftest.py — repo-root pytest configuration.

Prepends ``src/`` to sys.path so that ``import dbcv`` works in every test
without an editable install or pyproject.toml entry.  This is intentional for
the vertical-slice phase: packaging is kept light.

The Path anchoring below follows Rule 1 of CLAUDE.md: every script anchors
to the repo root via ``Path(__file__).resolve().parents[N]``, never via CWD.
"""

import sys
from pathlib import Path

# This file is at the repo root; parents[0] is the repo root itself.
_SRC = Path(__file__).resolve().parent / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
