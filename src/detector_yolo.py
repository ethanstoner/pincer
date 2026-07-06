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

    @staticmethod
    def _rect_dist(px, py, rect):
        """Distance from a point to an (x1,y1,x2,y2) rectangle (0 if inside)."""
        x1, y1, x2, y2 = rect
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return math.hypot(dx, dy)

    def _tap_point(self, box, avoid_rects):
        """Pick the tap point INSIDE the pokemon box that keeps the most
        distance from every detected avoid object (gym/stop/dynamax hitboxes
        steal taps -- live review: correct detections, stolen clicks). Falls
        back to the upper-third point when nothing threatens."""
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        candidates = [
            (x1 + 0.50 * bw, y1 + 0.32 * bh),   # upper third (default)
            (x1 + 0.50 * bw, y1 + 0.50 * bh),
            (x1 + 0.30 * bw, y1 + 0.40 * bh),
            (x1 + 0.70 * bw, y1 + 0.40 * bh),
            (x1 + 0.50 * bw, y1 + 0.18 * bh),   # head
        ]
        if not avoid_rects:
            return candidates[0]
        return max(candidates,
                   key=lambda p: min(self._rect_dist(p[0], p[1], r)
                                     for r in avoid_rects))

    def propose(self, img, phone) -> Optional[Target]:
        h, w = img.shape[:2]
        result = self._model.predict(
            img, conf=self._conf, imgsz=self._imgsz, verbose=False
        )[0]

        ax, ay = AVATAR_RATIO[0] * w, AVATAR_RATIO[1] * h
        best_box, best_dist = None, None
        avoid_rects = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            if int(box.cls) != 0:
                # class 1 = "avoid": never tapped, but its rect is used to
                # steer the tap point AWAY from its hitbox.
                avoid_rects.append((x1, y1, x2, y2))
                continue
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
                best_box = (x1, y1, x2, y2)

        if best_box is None:
            return None
        tx, ty = self._tap_point(best_box, avoid_rects)
        x1, y1, x2, y2 = best_box
        return Target(
            x=round(tx), y=round(ty),
            bbox=(round(x1), round(y1), round(x2 - x1), round(y2 - y1)),
            src="yolo",
        )
