"""Route A: classical OpenCV Pokemon detector for the overworld MAP screen.

`propose()` does NOT decide whether a tap actually opens an encounter -- the
catch_loop taps the proposed pixel and re-checks screen_state afterward; only
a confirmed ENCOUNTER counts. This module's only job is to point at a pixel
that plausibly has a wild Pokemon on it.

Thresholds below were measured empirically against tests/fixtures/map.png
(1080x2388, an extremely crowded lure-party map). See comments at each
constant for the measured values that motivated them.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.config import Phone


@dataclass
class Target:
    x: int
    y: int
    bbox: tuple  # (x, y, w, h)


# --- UI exclusion ratios (regions of the screen that are never a Pokemon) ---
# Given/known regions:
LEFT_UI_RATIO = 0.10          # the client menu + nearby-list strip
BOTTOM_UI_RATIO = 0.85        # avatar/ball/binoculars button bar
TOP_UI_RATIO = 0.07           # status/clock bar
# Measured on the fixture: a fixed the client joystick/quick-action cluster
# (compass arrow + red/white "recenter" pokeball button) sits at roughly
# ratio (0.92, 0.12) and (0.92, 0.24) -- both are small, solid, saturated
# circular icons that otherwise pass the color+shape filters below just like
# a real Pokemon blob would. Exclude that corner.
TOPRIGHT_UI_X_RATIO = 0.90
TOPRIGHT_UI_Y_RATIO = 0.30

# Avatar reference point (used only for ranking candidates by proximity).
AVATAR_RATIO = (0.42, 0.55)

# --- Color thresholds ---
# Measured interior (non-UI) saturation on the fixture: median ~140,
# 75th pct ~157, 90th pct ~178 (0-255 scale). The map background itself is
# fairly saturated blue/teal (hue ~90-130), so saturation alone can't
# separate background from Pokemon -- hue must also be restricted.
SAT_THRESHOLD = 150
# Hue band (OpenCV 0-179 scale) covering the desaturated-but-still-quite-
# saturated blue/teal map/road color; Pokemon models are excluded from this
# band far more often than they fall inside it. Measured: ~70% of
# high-saturation interior pixels on the fixture fall in hue 90-120.
BG_HUE_LOW = 85
BG_HUE_HIGH = 135

# --- Morphology ---
# 5x5 open removes tiny specks (cherry-blossom lure particles, sparkles).
# 15x15 close merges a Pokemon model's fragmented color patches into one blob.
OPEN_KERNEL = np.ones((5, 5), np.uint8)
CLOSE_KERNEL = np.ones((15, 15), np.uint8)

# --- Contour filters ---
# Measured on the fixture: plausible Pokemon blobs land at areas
# ~2200-4700 px^2. Below ~1200 px^2 survivors were UI/PokeStop fragments or
# ambiguous partial occlusions; above ~20000 nothing plausible was observed
# (the crowded PokeStop disc itself is ~6800 px^2 but is rejected below by
# its shape, not its area).
MIN_AREA = 1200
MAX_AREA = 20000
# extent = contour area / bbox area, solidity = contour area / hull area.
# Both are LOW for rings, radial light rays, and the humanoid player avatar
# (measured extent ~0.19-0.35, solidity ~0.37-0.62) and HIGH for a Pokemon's
# roughly-solid body silhouette (measured extent ~0.58-0.66, solidity
# ~0.86-0.90). This is what actually separates real Pokemon blobs from the
# PokeStop disc, the avatar model, and spinning UI icons -- color alone
# could not.
MIN_EXTENT = 0.5
MIN_SOLIDITY = 0.75


def _build_search_mask(width: int, height: int) -> np.ndarray:
    mask = np.full((height, width), 255, np.uint8)
    mask[:, : int(LEFT_UI_RATIO * width)] = 0
    mask[int(BOTTOM_UI_RATIO * height) :, :] = 0
    mask[: int(TOP_UI_RATIO * height), :] = 0
    mask[: int(TOPRIGHT_UI_Y_RATIO * height), int(TOPRIGHT_UI_X_RATIO * width) :] = 0
    return mask


def propose(img: np.ndarray, phone: Phone) -> Optional[Target]:
    height, width = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]

    search_mask = _build_search_mask(width, height)
    not_bg_hue = ((hue < BG_HUE_LOW) | (hue > BG_HUE_HIGH)).astype(np.uint8) * 255
    high_sat = (sat > SAT_THRESHOLD).astype(np.uint8) * 255
    colorful = cv2.bitwise_and(not_bg_hue, high_sat)
    colorful = cv2.bitwise_and(colorful, colorful, mask=search_mask)

    clean = cv2.morphologyEx(colorful, cv2.MORPH_OPEN, OPEN_KERNEL)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    avatar_x = AVATAR_RATIO[0] * width
    avatar_y = AVATAR_RATIO[1] * height

    best_target = None
    best_dist = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        extent = area / (w * h)
        if extent < MIN_EXTENT:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        dist = math.hypot(cx - avatar_x, cy - avatar_y)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_target = Target(x=round(cx), y=round(cy), bbox=(x, y, w, h))

    return best_target


def _next_label_index(dataset_dir: str) -> int:
    highest = -1
    for name in os.listdir(dataset_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".png" or not stem.startswith("pokemon_"):
            continue
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        highest = max(highest, idx)
    return highest + 1


def save_label(img: np.ndarray, target: Target, dataset_dir: str) -> str:
    os.makedirs(dataset_dir, exist_ok=True)

    index = _next_label_index(dataset_dir)
    stem = f"pokemon_{index:06d}"
    png_path = os.path.join(dataset_dir, f"{stem}.png")
    txt_path = os.path.join(dataset_dir, f"{stem}.txt")

    x, y, w, h = target.bbox
    crop = img[y : y + h, x : x + w]
    cv2.imwrite(png_path, crop)

    with open(txt_path, "w") as f:
        f.write(f"{x} {y} {w} {h}\n")

    return png_path
