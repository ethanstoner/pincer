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

# --- Gym / PokeStop photodisc discriminator (STRUCTURAL, not a raw pixel count) ---
# Gyms/PokeStops render a bright white concentric-ring PHOTODISC. A raw
# white-pixel FRACTION does NOT generalize: live maps have persistent white/
# purple spinning ring-VFX (event/lure radius circles) that streak across
# Pokemon sprites and inflate a raw fraction to gym-like levels (measured: a
# real Pokemon under VFX at radar0 (611,1497,68,68) has raw white frac 0.226,
# radar1 (615,1495,71,63) has 0.227 -- both above any usable raw threshold).
#
# The real difference is STRUCTURAL: a photodisc is one LARGE, COMPACT white
# blob; a VFX streak is THIN / ELONGATED. So we threshold white pixels
# (V>200 & S<60), take the LARGEST connected component, and reject only if it
# is BOTH large AND compact. Measured on the largest white component:
#   feature = component_area / bbox_area (fill) ; and min-area-rect long/short (aspect)
#     GYM  cand_00: fill 0.269 aspect 2.30 | cand_01: 0.317 1.16 | cand_07: 0.347 1.60
#     POKE-UNDER-VFX radar0: fill 0.173 aspect 6.24 | radar1: 0.078 aspect 3.36
#     POKE  cand_05: 0.103 3.49 | cand_08: 0.010 3.75 | cand_02: 0.014 6.99
# Both features separate independently:
#   fill  -> gyms >= 0.269, real <= 0.173  (threshold 0.22, gap 0.096)
#   aspect-> gyms <= 2.30,  real >= 3.36   (threshold 2.80, gap 1.06)
# Requiring BOTH (large AND compact) means a Pokemon crossed by a thin streak
# fails at least one test and is KEPT; only a solid concentric disc trips both.
WHITE_V_MIN = 200
WHITE_S_MAX = 60
PHOTODISC_FILL_MIN = 0.22    # largest white component must cover >= this share of bbox
PHOTODISC_ASPECT_MAX = 2.80  # ...and be compact (min-area-rect long/short <= this)


def is_gym_photodisc(crop: np.ndarray) -> bool:
    """True if a candidate crop (BGR) is a gym/PokeStop photodisc that must be
    rejected. Structural: the largest bright-desaturated connected component
    must be BOTH large (disc-sized) AND compact (not a thin VFX streak)."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    white = ((val > WHITE_V_MIN) & (sat < WHITE_S_MAX)).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n <= 1:  # only background -> no white blob
        return False
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    fill = stats[largest, cv2.CC_STAT_AREA] / float(w * h)
    if fill < PHOTODISC_FILL_MIN:  # thin streak / small patch -> not a disc
        return False

    comp = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    aspect = max(rw, rh) / max(1.0, min(rw, rh))
    return aspect <= PHOTODISC_ASPECT_MAX  # large AND compact -> reject


def _in_range(val, lo_ratio, hi_ratio, dim):
    return lo_ratio * dim <= val <= hi_ratio * dim


def propose(img: np.ndarray, phone: Phone) -> Optional[Target]:
    height, width = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]

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

        # Reject gym / PokeStop photodiscs by their structural signature (one
        # large, compact white disc), NOT by a raw white-pixel count -- a thin
        # spinning VFX ring crossing a Pokemon must not trip this.
        if is_gym_photodisc(img[y : y + h, x : x + w]):
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
