"""Model loading and inference helpers for the YOLO11n-seg receipt (bonnetje) app.

Kept free of Streamlit imports so it can be unit-tested and reused from scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

HF_REPO_ID = os.getenv("HF_REPO_ID", "ciCic/yolo11n-seg-bonnetjes")
HF_FILENAME = os.getenv("HF_FILENAME", "yolo11n-seg-bonnetjes.pt")
LOCAL_MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

MAX_SIDE = 2000  # downscale huge phone photos to keep memory low on small hosts


def resolve_model_path() -> str:
    """Return a path to the weights.

    Priority:
      1. $MODEL_PATH environment variable
      2. ./models/best.pt  (drop your own fine-tuned weights here)
      3. Download from the Hugging Face Hub (cached after the first run)
    """
    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    if LOCAL_MODEL_PATH.is_file():
        return str(LOCAL_MODEL_PATH)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)


def load_model(path: Optional[str] = None):
    from ultralytics import YOLO

    return YOLO(path or resolve_model_path())


def load_image(file_or_path) -> np.ndarray:
    """Load an image as an RGB uint8 array, honouring EXIF rotation and limiting size."""
    img = Image.open(file_or_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


@dataclass
class Detection:
    index: int
    label: str
    confidence: float
    box_xyxy: tuple
    mask_area_pct: Optional[float] = None
    polygon: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class PredictionResult:
    annotated_rgb: np.ndarray
    detections: List[Detection]
    inference_ms: float


def predict(model, image_rgb: np.ndarray, conf: float = 0.25, iou: float = 0.45,
            imgsz: int = 640) -> PredictionResult:
    """Run segmentation on an RGB image and return the annotated image + detections."""
    # Ultralytics treats numpy arrays as BGR (OpenCV convention).
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(source=bgr, conf=conf, iou=iou, imgsz=imgsz,
                            device="cpu", verbose=False)
    r = results[0]
    annotated_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]
    detections: List[Detection] = []
    if r.boxes is not None and len(r.boxes) > 0:
        polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        for i, box in enumerate(r.boxes):
            poly = polys[i] if i < len(polys) else None
            area_pct = None
            if poly is not None and len(poly) >= 3:
                area_pct = float(cv2.contourArea(poly.astype(np.float32)) / (h * w) * 100)
            detections.append(Detection(
                index=i + 1,
                label=str(model.names[int(box.cls[0])]),
                confidence=float(box.conf[0]),
                box_xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                mask_area_pct=area_pct,
                polygon=poly,
            ))
    speed = getattr(r, "speed", None) or {}
    ms = float(sum(speed.get(k, 0.0) for k in ("preprocess", "inference", "postprocess")))
    return PredictionResult(annotated_rgb, detections, ms)


def crop_box(image_rgb: np.ndarray, det: Detection, pad: int = 8) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = det.box_xyxy
    x1, y1 = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
    x2, y2 = min(int(x2) + pad, w), min(int(y2) + pad, h)
    return image_rgb[y1:y2, x1:x2]


def deskew_crop(image_rgb: np.ndarray, det: Detection) -> np.ndarray:
    """Straighten the detected receipt using the minimum-area rectangle of its mask.

    Falls back to a plain bounding-box crop when no polygon is available.
    """
    poly = det.polygon
    if poly is None or len(poly) < 3:
        return crop_box(image_rgb, det)

    rect = cv2.minAreaRect(poly.astype(np.float32))
    (_, _), (rw, rh), _ = rect
    if rw < 5 or rh < 5:
        return crop_box(image_rgb, det)

    box = cv2.boxPoints(rect)  # 4 corners, any order
    # order: top-left, top-right, bottom-right, bottom-left
    s = box.sum(axis=1)
    d = np.diff(box, axis=1).ravel()
    ordered = np.array([box[np.argmin(s)], box[np.argmin(d)],
                        box[np.argmax(s)], box[np.argmax(d)]], dtype=np.float32)

    width = int(round(max(np.linalg.norm(ordered[0] - ordered[1]),
                          np.linalg.norm(ordered[3] - ordered[2]))))
    height = int(round(max(np.linalg.norm(ordered[0] - ordered[3]),
                           np.linalg.norm(ordered[1] - ordered[2]))))
    if width < 5 or height < 5:
        return crop_box(image_rgb, det)

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    out = cv2.warpPerspective(image_rgb, M, (width, height), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    # Receipts are tall: rotate so the long side is vertical.
    if out.shape[1] > out.shape[0]:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out


def to_png_bytes(image_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
