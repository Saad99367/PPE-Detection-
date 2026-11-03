"""Thread-safe, in-memory PPE image inference and command-line interface."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps

from visualization import Appearance, draw_detections

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "best.pt"


@dataclass(frozen=True)
class Settings:
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640

    def __post_init__(self) -> None:
        if not 0 <= self.conf <= 1 or not 0 <= self.iou <= 1:
            raise ValueError("Confidence and IoU must be between 0 and 1.")
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("Image size must be a positive multiple of 32.")


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    xyxy: list[float]


@dataclass
class Prediction:
    image_bgr: np.ndarray
    detections: list[Detection]
    counts: dict[str, int]
    inference_ms: float


class Detector:
    """Own one model and serialize all predictor access, including GPU work."""

    def __init__(self, path: str | Path = DEFAULT_MODEL) -> None:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}. Place your trained best.pt in models/best.pt.")
        from ultralytics import YOLO
        import torch

        self.lock = Lock()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(path), task="detect")
        names = self.model.names
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))

    def predict(self, image_bgr: np.ndarray, settings: Settings = Settings(), appearance: Appearance = Appearance()) -> Prediction:
        """Accept uint8 BGR; return a new annotated BGR image and plain data."""
        if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
            raise ValueError("Expected a non-empty uint8 BGR image with three channels.")
        with self.lock:
            start = perf_counter()
            result = self.model.predict(image_bgr, conf=settings.conf, iou=settings.iou,
                                        imgsz=settings.imgsz, device=self.device,
                                        save=False, save_txt=False, save_crop=False, verbose=False)[0]
            rows = result.boxes.data.cpu().tolist() if result.boxes is not None else []
            elapsed = (perf_counter() - start) * 1000
            detections = [Detection(int(row[5]), self.names[int(row[5])], float(row[4]),
                                    [float(v) for v in row[:4]]) for row in rows]
        counts = {name: 0 for name in self.names.values()}
        for detection in detections:
            counts[detection.label] += 1
        return Prediction(draw_detections(image_bgr, detections, appearance), detections, counts, elapsed)


def load_model(path: str | Path = DEFAULT_MODEL) -> Detector:
    """Load only the supplied local checkpoint; never substitute a model."""
    return Detector(path)


def read_image(source) -> np.ndarray:
    """Decode a path or file-like image, applying EXIF orientation and RGB→BGR."""
    try:
        with Image.open(source) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc


def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--show", action="store_true")
    add_inference_arguments(parser)
    args = parser.parse_args()
    try:
        settings = Settings(args.conf, args.iou, args.imgsz)
        image = read_image(args.source)
        result = load_model(args.model).predict(image, settings)
        print(json.dumps({"counts": result.counts, "inference_ms": result.inference_ms,
                          "detections": [asdict(d) for d in result.detections]}, indent=2))
        if args.show:
            try:
                cv2.imshow("PPE Detection — press any key to close", result.image_bgr)
                cv2.waitKey(0)
            finally:
                cv2.destroyAllWindows()
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\nFor --show, install opencv-python instead of opencv-python-headless.\n")


if __name__ == "__main__":
    main()
