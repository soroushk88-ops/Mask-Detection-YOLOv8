"""
main.py
-------
A FastAPI web service that wraps the MaskDetector.

Run it locally from the project root with:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs in your browser to try it interactively.
"""

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from app.detector import MaskDetector

# A small place to hold the loaded model.
# It stays empty until the server starts up (see `lifespan` below).
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything BEFORE `yield` runs ONCE, when the server starts.
    # This is where we load the heavy model into memory — exactly once —
    # so that no single request ever has to pay the loading cost.
    ml_models["mask_detector"] = MaskDetector(weights_path="models/best.pt")
    yield
    # Everything AFTER `yield` runs ONCE, when the server shuts down.
    ml_models.clear()


app = FastAPI(
    title="Mask Detection API",
    description="Detects whether people in an image are wearing a mask, using a YOLOv8 model.",
    version="1.0.0",
    lifespan=lifespan,
)


def _read_image(file_bytes: bytes) -> Image.Image:
    """Turn raw uploaded bytes into a PIL image, or fail with a clean error."""
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")


@app.get("/")
def health_check():
    """A simple endpoint to confirm the service is alive and running."""
    return {"status": "ok", "message": "Mask Detection API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded image and return the detections as JSON.

    Example response:
        {
          "filename": "photo.jpg",
          "count": 2,
          "detections": [
            {"label": "mask", "confidence": 0.94, "box": [x1, y1, x2, y2]},
            ...
          ]
        }
    """
    image = _read_image(await file.read())
    detector = ml_models["mask_detector"]
    detections = detector.predict(image)
    return {
        "filename": file.filename,
        "count": len(detections),
        "detections": detections,
    }


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    """
    Accept an uploaded image and return the SAME image with the
    detection boxes drawn on it (as a JPEG).
    """
    image = _read_image(await file.read())
    detector = ml_models["mask_detector"]
    annotated = detector.annotate(image)

    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")
