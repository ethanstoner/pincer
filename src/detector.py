"""Route A: classical OpenCV Pokemon detector for the overworld MAP screen.

`propose()` does NOT decide whether a tap actually opens an encounter -- the
catch_loop taps the proposed pixel and re-checks screen_state afterward; only
a confirmed ENCOUNTER counts. This module's only job is to point at a pixel
that plausibly has a *catchable wild Pokemon* on it -- never a gym, PokeStop,
the player avatar, or a UI button.

Design decisions (from user QA on tests/fixtures/map.png, a permanent
mega-spawn / lure-party map that is ALWAYS this crowded):
  - Pure center-region CV, no radar. There is basically always a real Pokemon
    right next to the avatar, so we search a central band around it.
  - Gyms/raids cluster in the TOP third; catchable Pokemon cluster in the
    CENTER around the avatar. So the search region starts BELOW the top band.
  - Gyms/PokeStops carry a bright WHITE spinning concentric-ring photodisc;
    Pokemon models do not. That white-disc fraction is a measured discriminator.

All thresholds below were measured empirically against map.png (1080x2388).
See comments at each constant for the measured values + margins that motivate
them.
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


# --- Central detection region (ratios of width/height) ---
# x: left edge past the the client menu / nearby-list strip (< 0.10 w); right edge
#    trimmed at 0.95 w. y: TOP boundary 0.40 h drops the whole top gym/raid band
#    AND both fixed UI buttons (the "recenter" Poke Ball at y-ratio ~0.24 and
#    the rocket/compass arrow at y-ratio ~0.12 -- measured cand_09/cand_11);
#    BOTTOM boundary 0.85 h drops the avatar/ball/binoculars button bar.
# Verified on map.png: all four QA-labeled gyms sit at y-ratio 0.47-0.82 (inside
# this band, so they are rejected by the shape+photodisc filters below, not by
# the region), while every labeled real Pokemon centroid falls inside the band.
SEARCH_X_LOW = 0.10
SEARCH_X_HIGH = 0.95
SEARCH_Y_LOW = 0.40
SEARCH_Y_HIGH = 0.85

# --- Avatar reference + exclusion box ---
# Proximity anchor (given): the player avatar sits ~ ratio (0.42, 0.55).
AVATAR_RATIO = (0.42, 0.55)
# The avatar MODEL (a humanoid trainer standing on the PokeStop) occupies a
# tight box around screen-center (measured cand_06 bbox = (487,1412,103,106),
# centroid ratio ~ (0.50, 0.61)). A candidate whose centroid falls inside this
# box is the avatar itself and is rejected. Kept tight so the real Pokemon
# immediately to its right (measured centroid ratio ~ (0.58, 0.61)) survives.
AVATAR_EXCL_X = (0.44, 0.56)
AVATAR_EXCL_Y = (0.57, 0.66)

# --- Color thresholds ---
# Measured interior saturation on the fixture: median ~140, 75th pct ~157 (0-255
# scale). The map background is itself fairly saturated blue/teal (hue ~90-130),
# so saturation alone can't separate it from Pokemon -- hue must also be
# restricted. Measured: ~70% of high-saturation interior pixels fall in hue
# 90-120 (the blue/teal map + roads), so we exclude that hue band.
SAT_THRESHOLD = 150
BG_HUE_LOW = 85
BG_HUE_HIGH = 135

# --- Morphology ---
# 5x5 open removes tiny specks (cherry-blossom lure particles, sparkles).
# 15x15 close merges a Pokemon model's fragmented color patches into one blob.
OPEN_KERNEL = np.ones((5, 5), np.uint8)
CLOSE_KERNEL = np.ones((15, 15), np.uint8)

# --- Contour shape filters ---
# Measured plausible Pokemon blob area on the fixture: ~2200-4700 px^2. Below
# ~1200 survivors were UI/PokeStop fragments or partial occlusions; nothing
# plausible was observed above ~20000.
MIN_AREA = 1200
MAX_AREA = 20000
# extent = area / bbox area, solidity = area / hull area. Both are LOW for the
# gym photodiscs, radial light rays, and the humanoid avatar (measured
# extent 0.19-0.35, solidity 0.37-0.62) and HIGH for a Pokemon's roughly-solid
# body silhouette (measured extent 0.58-0.66, solidity 0.86-0.90).
MIN_EXTENT = 0.5
MIN_SOLIDITY = 0.75

# --- Gym / PokeStop photodisc discriminator ---
# Gyms and PokeStops render a bright WHITE spinning concentric-ring photodisc;
# Pokemon models do not. Measured fraction of bbox pixels that are bright AND
# desaturated (V > 200 and S < 60):
#     gyms/stops with a photodisc: cand_00=0.298, cand_01=0.456, cand_07=0.349
#     "recenter" Poke Ball UI button:            cand_09=0.224
#     real Pokemon (max observed):               cand_05=0.133
#     other real Pokemon:  cand_02=0.043 cand_04=0.013 cand_08=0.022 cand_10=0.014
# Clean margin between the highest real Pokemon (0.133) and the lowest
# photodisc blob (0.224). Threshold 0.20 rejects every photodisc gym/stop and
# the Poke Ball UI while keeping every real Pokemon (0.067 headroom below).
WHITE_V_MIN = 200
WHITE_S_MAX = 60
WHITE_DISC_MAX = 0.20


def _in_range(val, lo_ratio, hi_ratio, dim):
    return lo_ratio * dim <= val <= hi_ratio * dim


def propose(img: np.ndarray, phone: Phone) -> Optional[Target]:
    height, width = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Restrict the search to the central region (below top gym/raid band + UI,
    # above bottom button bar, off the left menu strip).
    search_mask = np.zeros((height, width), np.uint8)
    y0, y1 = int(SEARCH_Y_LOW * height), int(SEARCH_Y_HIGH * height)
    x0, x1 = int(SEARCH_X_LOW * width), int(SEARCH_X_HIGH * width)
    search_mask[y0:y1, x0:x1] = 255

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

        # Reject gym / PokeStop photodiscs (and the Poke Ball UI button) by
        # their bright, desaturated concentric-ring signature.
        region_v = val[y : y + h, x : x + w]
        region_s = sat[y : y + h, x : x + w]
        white_frac = np.count_nonzero(
            (region_v > WHITE_V_MIN) & (region_s < WHITE_S_MAX)
        ) / float(w * h)
        if white_frac >= WHITE_DISC_MAX:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        # Reject the player avatar's own box (screen-center trainer model).
        if _in_range(cx, *AVATAR_EXCL_X, width) and _in_range(cy, *AVATAR_EXCL_Y, height):
            continue

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
