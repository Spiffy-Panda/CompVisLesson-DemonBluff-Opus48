"""
dbcv — Demon Bluff CV pipeline package.

Public surface
--------------
The recommended imports for external code (tests, scripts, the lesson plan
worked examples) are:

    from dbcv.schema import GameStateSnapshot, Source, Resolution, CardRead, Readings
    from dbcv.pipeline import run_pipeline, crop_relative
    from dbcv.localize import localize, stub_localize
    from dbcv.identify import identify
    from dbcv.assemble import assemble
    from dbcv.config import get_settings
    from dbcv.api import app   # FastAPI application object

The package does not re-export everything from every sub-module at the top
level to keep the namespace clean.  Import from the specific module when you
need something not listed above.
"""

# Package version mirrors schema_version so they stay in sync.
__version__ = "0.1.0"
