# Mask Detection Service

A real-time face-mask detection service built around a fine-tuned **YOLOv8** model,
served through a **FastAPI** web API and packaged as a portable **Docker** image.

This directory turns a trained detection model into a deployable inference service:
you send it an image over HTTP, and it returns the detected faces, each labelled
`mask` or `no-mask` with a confidence score and bounding box.

---

## How it works

The service is organised around a clean separation between the model logic and the
web layer:

- **`app/detector.py`** — a self-contained `MaskDetector` class that loads the YOLOv8
  weights once and exposes a simple `predict()` method returning structured results.
  It has no dependency on notebooks, Colab, or any specific environment.
- **`app/main.py`** — a FastAPI application that loads the model a single time at
  startup and wraps it behind HTTP endpoints.
- **`Dockerfile`** — packages the whole service (Python, dependencies, model, code)
  into one image that runs identically anywhere.

The model is loaded exactly once, at service startup, so no individual request pays
the (slow) model-loading cost.

---

## Tech stack

- **Model:** YOLOv8 (Ultralytics), fine-tuned on a mask-wearing dataset via transfer learning
- **Serving:** FastAPI + Uvicorn
- **Packaging:** Docker (CPU-only PyTorch build for a lean image)

---

## Project structure

```
Mask Detector/
├── app/
│   ├── __init__.py
│   ├── detector.py       # MaskDetector class (core inference logic)
│   └── main.py           # FastAPI application
├── models/
│   └── best.pt           # trained weights (not tracked in git — see below)
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

---

## The model weights

The trained weights file (`models/best.pt`) is intentionally **not** committed to the
repository — model binaries are kept out of version control to keep the repo light.

To run the service you need to place a `best.pt` file inside the `models/` directory.
You can regenerate it by training a YOLOv8 model on the mask-wearing dataset (see the
`project/` directory in the repository root for the training notebook), then copying
the resulting `runs/detect/train/weights/best.pt` into `models/`.

---

## Running with Docker (recommended)

From inside this directory:

```bash
# Build the image
docker build -t mask-detector .

# Run the container, mapping port 8000 to your machine
docker run -p 8000:8000 mask-detector
```

Then open the interactive API docs in your browser:

```
http://localhost:8000/docs
```

---

## Running locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs`.

---

## API endpoints

| Method | Endpoint             | Description                                              |
|--------|----------------------|---------------------------------------------------------|
| GET    | `/`                  | Health check — confirms the service is running.         |
| POST   | `/predict`           | Upload an image; returns detections as JSON.             |
| POST   | `/predict/annotated` | Upload an image; returns the image with boxes drawn on. |

Example JSON response from `/predict`:

```json
{
  "filename": "photo.jpg",
  "count": 2,
  "detections": [
    { "label": "mask", "confidence": 0.94, "box": [x1, y1, x2, y2] }
  ]
}
```

---

## Model performance

Evaluated on the validation set after fine-tuning `yolov8m` for 50 epochs:

| Class    | mAP@50 |
|----------|--------|
| mask     | ~0.91  |
| no-mask  | ~0.74  |
| **all**  | ~0.83  |

The `mask` class performs strongly; `no-mask` is lower, reflecting the smaller number
of `no-mask` examples in the training data.

---

## License

Released under the MIT License.
