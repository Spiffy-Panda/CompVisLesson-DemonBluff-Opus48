# io/inputs — formats the code consumes

Update this whenever a consumed format changes (Rule 3).

| Input | Location | Format | Notes |
|-------|----------|--------|-------|
| Sample footage | `dataset/raw-video/*.mp4` (gitignored) | video container | ~370 MB, ~1 h each. **Never opened directly** — only via the frame-selection stage. Resolution read from the media, never assumed. |
| Selected frames | `dataset/frames/` (gitignored) | PNG image files | Output of frame selection; what all downstream stages actually see.  Named `<Set>_<NNN>_t<SSSSS>s.png`. |
| **API frame upload** | `POST /v1/snapshot` (multipart) | image bytes (PNG/JPEG/BMP/etc.) | **New in slice (2026-06-22).** The REST endpoint accepts any format that OpenCV (`cv2.imdecode`) can decode.  Resolution is read from the decoded `np.ndarray` — never assumed from the request. |
| Card-art templates | `knowledge-base/card-art/` (gitignored) | PNG image files | Reference art per townee for identification; assumed swappable, so kept easy to re-fit.  Not yet consumed by code (Stage 2). |
| Cached game facts | `knowledge-base/wiki/*.md` | markdown | Transcribed/transformed wiki content (roles, mechanics). The raw fetch cache (`wiki/_raw_cache/`) is gitignored. |
| Run config | `src/dbcv/config.py` via env vars | pydantic-settings / `.env` | `Settings` class with `DBCV_` prefix.  `frames_dir` defaults to `dataset/frames/` anchored to repo root.  No resolution baked in. |

---

## API frame upload — detail

Endpoint: `POST /v1/snapshot`

Accepted fields (multipart/form-data):
| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `file` | `UploadFile` (image bytes) | Yes | — | Any format `cv2.imdecode` supports |
| `video` | `str` (Form) | No | `"unknown"` | Source video stem (no path) |
| `frame_index` | `int` (Form) | No | `0` | Zero-based frame index |
| `timestamp_s` | `float` (Form) | No | `0.0` | Seconds from video start |

Resolution is read from the decoded image (`image_array.shape[:2]`) inside
`run_pipeline` — never passed as a request parameter and never assumed.

## Open contract questions

- Frame-selection output: naming convention `<Set>_<NNN>_t<SSSSS>s.png` is
  observed in `dataset/frames/`; a sidecar format (JSON or CSV) for metadata
  has not yet been defined (Stage 0).
- Card-art template manifest format (how identities map to files, versioned
  for art swaps) — not yet defined (Stage 2).
