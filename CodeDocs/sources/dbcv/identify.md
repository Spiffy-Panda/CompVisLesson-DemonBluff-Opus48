# CodeDocs/sources/dbcv/identify.py

**Status:** slice/stub — always returns ("unknown", "unknown", 0.0).

**Purpose:** Defines the card-identification interface and a stub.  Given a
cropped card image, returns (identity, role_class, confidence).

**Who uses it:**
- `dbcv/pipeline.py` — imports `identify` as the default `identifier` argument

---

## Key signatures (with line numbers)

### `identify(card_crop) -> (str, str, float)` — line 44
```python
def identify(card_crop: np.ndarray) -> tuple[str, str, float]:
```
**Parameters:**
- `card_crop` — cropped card region, numpy array (HxWxC, BGR from cv2)

**Returns:** `(identity, role_class, confidence)` where
- `identity` — townee name string or "unknown"
- `role_class` — one of "villager" | "minion" | "outcast" | "demon" | "unknown"
- `confidence` — float in [0.0, 1.0]

**Stub behaviour:** discards `card_crop`, returns `("unknown", "unknown", 0.0)`.

---

## Replacement guide

The real identifier (research/RESEARCH.md entry 3) will:
1. Preprocess the crop (resize to backbone input size, normalize).
2. Run a forward pass through a small frozen backbone (MobileNetV3-Small).
3. Compute cosine distance to every entry in the reference gallery
   (built from `knowledge-base/card-art/`).
4. Return the closest match's name/role and `1 - distance` as confidence.

On an art swap: re-embed the new reference images.  No gradient steps needed.

Pass a real identifier as `identifier=my_real_identifier` to `run_pipeline`.
The function signature must remain identical.
