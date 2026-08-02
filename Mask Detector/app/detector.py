"""
detector.py
-----------
Core inference logic for the mask-detection model.

This module has NO dependency on Google Colab, Google Drive, or any notebook.
It exposes a single class, `MaskDetector`, that:
  1. loads the trained YOLOv8 weights ONCE (when the object is created), and
  2. runs detection on any image you give it.

Loading the model once and reusing it is important: loading weights is slow,
so we never want to do it per-request inside a web service.
"""

from pathlib import Path
from typing import List, Dict, Union

from PIL import Image
from ultralytics import YOLO


class MaskDetector:
    def __init__(self, weights_path: Union[str, Path] = "models/best.pt"):
        """
        Load the trained model into memory.

        weights_path: path to the trained YOLOv8 weights file (best.pt).
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'. "
                "Put your trained best.pt there first."
            )
        # This single line loads the network + weights into memory.
        self.model = YOLO(str(weights_path))

    def predict(self, image: Union[str, Path, Image.Image], conf: float = 0.25) -> List[Dict]:
        """
        Run detection on one image and return a clean, structured result.

        image: a file path OR a PIL Image.
        conf:  minimum confidence threshold (detections below this are dropped).

        Returns a list of detections, each like:
            {"label": "mask", "confidence": 0.94, "box": [x1, y1, x2, y2]}
        """
        # ultralytics accepts a path or a PIL image directly.
        results = self.model.predict(source=image, conf=conf, verbose=False)

        # `predict` returns a list (one entry per image). We sent one image.
        result = results[0]

        detections: List[Dict] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                {
                    "label": self.model.names[class_id],          # "mask" / "no-mask"
                    "confidence": round(float(box.conf[0]), 4),
                    "box": [round(float(v), 1) for v in box.xyxy[0]],  # [x1,y1,x2,y2]
                }
            )
        return detections

    def annotate(self, image: Union[str, Path, Image.Image], conf: float = 0.25) -> Image.Image:
        """
        Same as predict(), but returns the image with boxes drawn on it
        (as a PIL Image) instead of raw numbers. Useful for a quick visual check.
        """
        results = self.model.predict(source=image, conf=conf, verbose=False)
        # .plot() gives a numpy array in BGR order; convert to a normal RGB PIL image.
        annotated_bgr = results[0].plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]  # BGR -> RGB
        return Image.fromarray(annotated_rgb)


# This block only runs when you execute the file directly:
#   python -m app.detector path/to/image.jpg
# It's a tiny built-in test so you can check the class works before we wrap it in a web API.
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.detector <image_path> [weights_path]")
        raise SystemExit(1)

    image_path = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else "models/best.pt"

    detector = MaskDetector(weights_path=weights)
    found = detector.predict(image_path)

    print(f"\nDetections in '{image_path}':")
    if not found:
        print("  (nothing detected above the confidence threshold)")
    for d in found:
        print(f"  - {d['label']:8s}  conf={d['confidence']:.2f}  box={d['box']}")

    # Also save an annotated copy next to the input so you can look at it.
    out = detector.annotate(image_path)
    out_path = "annotated_output.jpg"
    out.save(out_path)
    print(f"\nAnnotated image saved to: {out_path}\n")
