"""Resolution-aware OpenCV annotations shared by every inference mode."""
from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2
import numpy as np


class DetectionLike(Protocol):
    label: str
    confidence: float
    xyxy: list[float]


@dataclass(frozen=True)
class Appearance:
    """Thickness and font scale are specified at a reference width of 1280px."""
    box_thickness: int = 5
    font_scale: float = 1.0
    show_confidence: bool = True
    show_labels: bool = True


def draw_detections(frame: np.ndarray, detections: Sequence[DetectionLike],
                    appearance: Appearance = Appearance()) -> np.ndarray:
    """Return one annotated copy; never mutate source pixels or prediction data."""
    output = frame.copy()
    height, width = output.shape[:2]
    ratio = width / 1280
    thickness = max(2, round(appearance.box_thickness * ratio))
    scale = max(0.45, appearance.font_scale * ratio)
    stroke = max(1, round(2 * ratio))
    padding = max(3, round(7 * ratio))
    font = cv2.FONT_HERSHEY_SIMPLEX
    occupied = []
    labels = []
    for item in detections:
        x1, y1, x2, y2 = map(round, item.xyxy)
        x1, x2 = [int(v) for v in np.clip([x1, x2], 0, width - 1)]
        y1, y2 = [int(v) for v in np.clip([y1, y2], 0, height - 1)]
        name = item.label.casefold()
        color = ((60, 200, 60) if name in {"hardhat", "mask", "safety vest"} else
                 (50, 50, 255) if name in {"no-hardhat", "no-mask", "no-safety vest"} else (255, 190, 40))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if not appearance.show_labels:
            continue
        text = item.label + (f" {item.confidence:.0%}" if appearance.show_confidence else "")
        fs, ts = scale, stroke
        # Fit the complete label even on narrow frames; tiny frames have physical limits.
        for _ in range(40):
            (tw, th), baseline = cv2.getTextSize(text, font, fs, ts)
            lw, lh = tw + 2 * padding + ts, th + baseline + 2 * padding + ts
            if lw <= width and lh <= height:
                break
            fs *= 0.85
            ts = max(1, round(ts * 0.85))
        else:
            continue
        left = max(0, min(x1, width - lw))
        candidates = []
        for top in [y1 - lh - 2, y1 + 2, y2 + 2] + [y1 + n * lh for n in range(1, 5)]:
            top = max(0, min(top, height - lh))
            candidates.append((left, top, left + lw, top + lh))
        def overlap(rect):
            return sum(max(0, min(rect[2], r[2]) - max(rect[0], r[0])) *
                       max(0, min(rect[3], r[3]) - max(rect[1], r[1])) for r in occupied)
        rect = min(candidates, key=overlap)
        occupied.append(rect)
        labels.append((rect, text, fs, ts, th, color))
    # Labels drawn last cannot be crossed out by a later box border.
    for (left, top, right, bottom), text, fs, ts, th, color in labels:
        cv2.rectangle(output, (left, top), (right - 1, bottom - 1), color, -1)
        luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        ink = (0, 0, 0) if luminance > 140 else (255, 255, 255)
        cv2.putText(output, text, (left + padding, top + padding + th), font,
                    fs, ink, ts, cv2.LINE_AA)
    return output
