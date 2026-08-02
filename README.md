# Mask-Detection-YOLOv8

Real-time **mask-wearing detection** built with **YOLOv8**. The project covers the full object-detection workflow: dataset preprocessing, transfer-learning-based training, and evaluation with precision, recall, and PR curves.

## Overview

**Objective:** Develop an AI model that detects whether people are wearing masks, in real time.

**Problem:** Monitoring mask-wearing compliance automatically to support public-safety use cases.

**Approach:** Fine-tune a pretrained YOLOv8 object detector on a labeled mask-wearing dataset, then evaluate its detection quality.

## Dataset

- **Source:** Mask Wearing (v4-raw) dataset
- **Images:** 149, annotated with bounding boxes around each person, labeled by mask / no-mask
- **Formats supported:** COCO JSON, Pascal VOC XML, YOLO Darknet TXT

## Preprocessing

- Converted COCO and Pascal VOC annotations into YOLO format
- Normalized bounding-box coordinates
- Organized the data into train / test splits

## Model & Training

| Setting | Value |
|---|---|
| Model | YOLOv8m (medium variant) |
| Pretrained weights | `yolov8m.pt` (transfer learning) |
| Config | `data.yaml` |
| Epochs | 50 |
| Batch size | 8 |
| Image size | 640 × 640 |
| Hardware | GPU |

**Why YOLOv8m:** real-time inference speed, strong detection accuracy, and an efficient architecture that balances performance and cost.

## Evaluation

The trained model was evaluated using standard object-detection metrics, with results visualized as:
- **P-curve** (precision vs confidence)
- **R-curve** (recall vs confidence)
- **PR-curve** (precision vs recall)

Qualitative results on sample images are also included, showing the model detecting mask / no-mask on unseen inputs.

**Outcome:** the model reached high detection accuracy with real-time performance, successfully distinguishing mask-wearing from non-mask-wearing individuals.

## Tech Stack

- Python
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- PyTorch
- OpenCV / NumPy for image handling

## Repository Contents

- `project/` — training code, configuration, and outputs
- `PowerPoint/` — project presentation with methodology, curves, and sample detections

## How to Use

# install dependencies
pip install ultralytics

# train
yolo detect train data=data.yaml model=yolov8m.pt epochs=50 imgsz=640 batch=8

# predict on an image
yolo detect predict model=path/to/best.pt source=path/to/image.jpg

---

*Computer-vision project by Soroush Karami, MSc Computer Science (AI & Data Engineering), Ca' Foscari University of Venice.*
