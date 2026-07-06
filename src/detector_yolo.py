"""YOLO-based Pokemon detector -- a drop-in replacement for detector.propose().

Same interface as the classical detector: `propose(img, phone) -> Optional[Target]`,
selecting the box nearest the avatar within the central search region. So the
screen-state safety guards (center region, gym/stop close-X bail, verify-and-
recover) all still apply unchanged; only the "where is a Pokemon" step is swapped
from hand-tuned CV to a trained model that generalizes to VFX-heavy scenes.

`ultralytics` is imported lazily so the bot only depends on it when YOLO
detection is actually enabled (config `detector: "yolo"`). Train a model with
training/train_yolo.py; point `yolo_model_path` at the resulting best.pt (or the
exported .onnx).
"""
import math
from typing import Optional

from src.detector import (
    AVATAR_RATIO,
    SEARCH_X_HIGH,
    SEARCH_X_LOW,
    SEARCH_Y_HIGH,
    SEARCH_Y_LOW,
    Target,
)


class YoloDetector:
    def __init__(self, model_path: str, conf: float = 0.35, imgsz: int = 1280):
        from ultralytics import YOLO  # lazy: only needed when YOLO is enabled

        self._model = YOLO(model_path)
        self._conf = conf
        self._imgsz = imgsz

    def propose(self, img, phone) -> Optional[Target]:
        h, w = img.shape[:2]
        result = self._model.predict(
            img, conf=self._conf, imgsz=self._imgsz, verbose=False
        )[0]

        ax, ay = AVATAR_RATIO[0] * w, AVATAR_RATIO[1] * h
        best, best_dist = None, None
        for box in result.boxes:
            if int(box.cls) != 0:
                continue  # class 1 = "avoid" (panel objects); never tap those
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # Same central-region guard as the CV detector: off the UI strips and
            # the top gym/raid band, so a stray box there can't trigger a tap.
            if not (
                SEARCH_X_LOW * w <= cx <= SEARCH_X_HIGH * w
                and SEARCH_Y_LOW * h <= cy <= SEARCH_Y_HIGH * h
            ):
                continue
            dist = math.hypot(cx - ax, cy - ay)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                # Tap the sprite's UPPER THIRD, not its centre: a Pokemon
                # standing on/near a PokeStop has the stop's generous ground-
                # level hit region right at its box centre, and the stop was
                # stealing the tap (live audit: stop panel opened on mon taps).
                best = Target(
                    x=round(cx),
                    y=round(y1 + 0.32 * (y2 - y1)),
                    bbox=(round(x1), round(y1), round(x2 - x1), round(y2 - y1)),
                    src="yolo",
                )
        return best
