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
from dbcv.embed import OnnxEmbedder
from dbcv.gallery import build_embedding_gallery, build_gallery
from dbcv.identify import make_embedding_identifier, make_gallery_identifier
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

    # Build the classical reference gallery once at startup.
    # The gallery loads ~67 small PNGs from knowledge-base/card-art/ and
    # precomputes HSV histograms + ORB descriptors entirely in-memory (~100 ms).
    # It is stored on app.state so every request shares the same pre-built object.
    #
    # On an art swap: restart the server (or call build_gallery() again).
    # No training required — the gallery is rebuilt purely from the new PNGs.
    # This is the "load once" pattern from research/RESEARCH.md entry 5.
    gallery = build_gallery()
    application.state.gallery = gallery
    # Stage 2 classical identifier — kept as a fallback / selectable baseline
    application.state.classical_identifier = make_gallery_identifier(gallery)

    # Build the Stage 3 embedding gallery (onnxruntime, CPU, no torch).
    # OnnxEmbedder loads the ONNX session once; build_embedding_gallery embeds
    # all ~67 reference PNGs and computes prototypical mean embeddings (~1-2 s).
    # Stored on app.state so every request reuses the same pre-built objects.
    #
    # If the ONNX file is missing (not yet exported), fall back gracefully to
    # the classical identifier with a warning.  Run export_backbone.py to generate.
    try:
        embedder = OnnxEmbedder()
        embed_gallery = build_embedding_gallery(gallery, embedder)
        application.state.embedder = embedder
        application.state.embed_gallery = embed_gallery
        # Embedding-NN is the default identifier (Stage 3)
        application.state.identifier = make_embedding_identifier(embedder, embed_gallery)
    except FileNotFoundError as exc:
        # ONNX file not generated yet — fall back to classical
        import warnings
        warnings.warn(
            f"Embedding-NN model not found ({exc}). "
            "Falling back to classical identifier. "
            "Run: .venv\\Scripts\\python.exe utils\\python\\export_backbone.py",
            stacklevel=1,
        )
        application.state.embedder = None
        application.state.embed_gallery = None
        application.state.identifier = application.state.classical_identifier

    yield  # <-- application is live here

    # --- shutdown ---
    # Gallery and ONNX session are in-memory only; nothing explicit to close.
    # onnxruntime sessions are released when the Python object is garbage-collected.


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
    # The classical gallery identifier was built in lifespan and stored on app.state.
    # We access it via the module-level ``app`` object (the lifespan already ran).
    # If for any reason the identifier is absent, run_pipeline falls back to the
    # stub identifier via its default argument.
    identifier_fn = getattr(app.state, "identifier", None)
    snapshot_result = run_pipeline(
        image=image_array,
        source=source,
        **({"identifier": identifier_fn} if identifier_fn is not None else {}),
    )

    return snapshot_result
