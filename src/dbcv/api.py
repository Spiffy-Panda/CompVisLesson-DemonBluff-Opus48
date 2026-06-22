"""
dbcv/api.py — FastAPI application for the Demon Bluff CV pipeline.

Exposes a single endpoint:
    POST /v1/snapshot

Accepts an uploaded frame image and optional source metadata, runs the full
pipeline, and returns a versioned GameStateSnapshot as JSON.

Design decisions (all grounded in research/RESEARCH.md entry 5)
---------------------------------------------------------------
1. Model loading in lifespan, not at import time.
   The ``lifespan`` async context manager runs once at startup and once at
   shutdown.  All heavy resources (models, galleries, ONNX sessions) are stored
   on ``app.state`` so every request sees the same already-loaded objects.
   In this slice there is nothing heavy to load — the stub localizer and
   identifier are plain functions — but the pattern is laid down explicitly so
   the lesson plan can demonstrate it and the real implementation slots in
   without restructuring the app.

2. Inference endpoint as plain ``def``, not ``async def``.
   FastAPI runs plain ``def`` path operations in an external threadpool
   (via Starlette's ``run_in_threadpool``), so a slow CV frame does not block
   the event loop and freeze every other concurrent request.
   Using ``async def`` for CPU-bound inference would block the event loop —
   this is the counter-intuitive but clearly documented behaviour in the
   FastAPI concurrency docs (research/RESEARCH.md entry 5, source 2).

3. Resolution is read from the decoded image, never from the request.
   The client uploads pixels; the server measures their dimensions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from dbcv.config import get_settings
from dbcv.pipeline import run_pipeline
from dbcv.schema import GameStateSnapshot, Source


# ---------------------------------------------------------------------------
# Lifespan — load (or prepare) resources once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load shared resources once at startup; clean up at shutdown.

    This is the recommended pattern for model loading in FastAPI
    (research/RESEARCH.md entry 5, source 1 — FastAPI lifespan docs).
    Using the deprecated ``@app.on_event("startup")`` decorator is
    intentionally avoided here.

    In the vertical-slice version nothing heavy is loaded — the localizer
    and identifier are stubs.  When the real ONNX models are ready, load
    them here, store on ``application.state``, and pass them into
    ``run_pipeline`` via the ``localizer`` / ``identifier`` parameters.

    Example (future):
        application.state.ort_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        # Set thread count to physical cores for best throughput
        # (research/RESEARCH.md entry 5, source 4)
    """
    # --- startup ---
    settings = get_settings()
    application.state.settings = settings
    # Future: load ONNX InferenceSession(s) here
    application.state.localizer = None   # None → pipeline uses default stub
    application.state.identifier = None  # None → pipeline uses default stub

    yield  # <-- application is live here

    # --- shutdown ---
    # Future: close sessions, release GPU memory, etc.


# ---------------------------------------------------------------------------
# App object
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Demon Bluff CV — snapshot API",
    version="0.1.0",
    description=(
        "Accepts a raw game frame and returns a structured GameStateSnapshot "
        "describing the board as seen in the image.  Resolution is read from "
        "the uploaded image; no resolution is ever assumed or hard-coded."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# POST /v1/snapshot
# ---------------------------------------------------------------------------


@app.post(
    "/v1/snapshot",
    response_model=GameStateSnapshot,
    summary="Analyse a game frame and return a structured board snapshot.",
)
def snapshot(
    file: UploadFile = File(..., description="PNG/JPEG game frame to analyse."),
    video: str = Form(default="unknown", description="Source video identifier (stem only, no path)."),
    frame_index: int = Form(default=0, description="Zero-based frame index within the video."),
    timestamp_s: float = Form(default=0.0, description="Timestamp in seconds of this frame."),
) -> GameStateSnapshot:
    """Decode the uploaded frame, run the CV pipeline, return a snapshot.

    This endpoint is declared as a plain ``def`` (not ``async def``) so that
    FastAPI runs it in an external threadpool.  CV inference is CPU-bound; if
    it ran in an ``async def`` it would occupy the event loop and block every
    other concurrent request for the duration of the inference.
    (research/RESEARCH.md entry 5, source 2 and 5.)

    Parameters
    ----------
    file:
        The uploaded image file.  Accepts any format that OpenCV can decode
        (PNG, JPEG, BMP, etc.).
    video:
        Identifier for the source video — used only for provenance metadata,
        not for any filesystem lookup.
    frame_index:
        The frame's position within the video (zero-based).
    timestamp_s:
        The frame's timestamp in seconds.

    Returns
    -------
    GameStateSnapshot
        Versioned JSON snapshot.  The ``resolution`` field reflects the actual
        dimensions of the uploaded image, not any assumed value.
    """
    # --- Decode the uploaded bytes into a numpy array ---
    raw_bytes = file.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # cv2.imdecode returns None on failure (unsupported format, truncated file)
    image_array = cv2.imdecode(
        np.frombuffer(raw_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,   # decode as BGR — consistent with the rest of the pipeline
    )

    if image_array is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not decode the uploaded file as an image.  "
                "Supported formats: PNG, JPEG, BMP, TIFF, WebP."
            ),
        )

    # --- Assemble provenance metadata ---
    source = Source(
        video=video,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
    )

    # --- Run the pipeline ---
    # The pipeline reads resolution from image_array.shape — never from a constant.
    # Localizer and identifier come from app.state when real models exist;
    # for the slice, passing None causes run_pipeline to use its defaults (stubs).
    snapshot_result = run_pipeline(
        image=image_array,
        source=source,
        # Future: localizer=request.app.state.localizer or stub_localize,
        #         identifier=request.app.state.identifier or identify,
    )

    return snapshot_result
