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
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.config import Phone

# Serializes save_label across per-phone worker threads: without it, two phones
# catching at once could read the same _next_label_index and overwrite each
# other's frame. The critical section (index -> write) is tiny.
_LABEL_LOCK = threading.Lock()


@dataclass
class Target:
    x: int
    y: int
    bbox: tuple  # (x, y, w, h)
    src: str = "cv"  # which detector proposed it ("cv" / "yolo") -- for audit


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
# Right edge 0.87: the fixed right-side UI column (pause, calendar, binoculars/
# Campfire, rocket radar) sits at x-ratio ~0.92 -- at 0.95 the detector tapped
# the binoculars rim and opened Campfire (live audit catch).
SEARCH_X_HIGH = 0.87
# Top edge was 0.40 when the ONLY gym defense was this region cut; live recall
# eval showed real catchable Pokemon walking just above it. Now that gyms,
# badges and tower-toppers have dedicated semantic rejectors, the band opens
# to 0.33 (still below the top raid/status strip and both fixed UI buttons).
SEARCH_Y_LOW = 0.33
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
# restricted. The FIXED band below covers the night-blue map; the in-game DAY
# cycle turns terrain GREEN (hue ~40-70), which flooded the mask with grass
# blobs (live: 121 empty taps in 14 min). propose() therefore ALSO excludes the
# frame's DOMINANT hue bands measured per frame (_dynamic_bg_bands): terrain
# always dominates pixel count, Pokemon never do, so this adapts to day/night/
# storm palettes automatically. The static band stays as a floor.
SAT_THRESHOLD = 150
BG_HUE_LOW = 85
BG_HUE_HIGH = 135
BG_DYNAMIC_MODES = 3     # exclude the top-N dominant hues of the search region
                         # (3: the DAY palette has light+dark grass AND paths;
                         # 2 modes left grass-speck junk -> 511 empty taps/30min)
BG_DYNAMIC_TOL = 12      # +/- band around each dominant hue

# --- Morphology ---
# 5x5 open removes tiny specks (cherry-blossom lure particles, sparkles).
# 15x15 close merges a Pokemon model's fragmented color patches into one blob.
OPEN_KERNEL = np.ones((5, 5), np.uint8)
CLOSE_KERNEL = np.ones((15, 15), np.uint8)

# --- Contour shape filters ---
# Measured plausible Pokemon blob area on the fixture: ~2200-4700 px^2, but
# the live camera TILTS while walking and distant Pokemon shrink well below
# the old 1200 floor (recall eval: most "small" rejects in tilted frames were
# real Pokemon). 800 recovers them; sub-800 specks are petals/cube fragments.
MIN_AREA = 800
MAX_AREA = 20000
# extent = area / bbox area, solidity = area / hull area. Both are LOW for the
# gym photodiscs, radial light rays, and the humanoid avatar (measured
# extent 0.19-0.35, solidity 0.37-0.62) and HIGH for a Pokemon's roughly-solid
# body silhouette (measured extent 0.58-0.66 for compact bodies -- but live
# recall eval caught a real DRAGONITE at extent <0.5: wings/limbs leave the
# bbox mostly empty). 0.40 keeps winged/limbed Pokemon while staying above the
# 0.35 ceiling measured for disc rays / the avatar. Solidity: junk measured
# <= 0.62, compact Pokemon >= 0.86; 0.68 recovers VFX-crossed bodies (live
# recall eval: a Fearow and a Krabby died at 0.75) with margin over junk.
MIN_EXTENT = 0.40
MIN_SOLIDITY = 0.68
# Pokemon bodies are roughly compact: measured bbox aspect (long/short side) is
# ~1.0-1.1 on the radar fixtures and stays <~1.6 across the map fixture. Thin
# high-saturation STREAKS -- spinning ring edges, lure beams, name-label bars --
# are very elongated (measured live mis-detections: 21x88 = 4.2, 152x15 = 10.1).
# Reject anything too elongated to be a Pokemon body. 2.5 leaves headroom for
# genuinely tall sprites while killing the streaks.
MAX_ASPECT = 2.5

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

# --- Gym CONTEXT check: the disc is BELOW the tapped blob, not inside it ---
# Live click-audit review (dataset/clicks) showed most wasted taps were gym
# paraphernalia whose own bbox contains no white disc: red/orange tower-top
# fragments and defender Pokemon standing ON towers. The gym's white
# photodisc/ribbon system sits DIRECTLY BELOW them, outside the candidate bbox.
# So each candidate is ALSO checked with the same structural photodisc test on
# a context crop extended DOWNWARD 2.5x the bbox height (padded w/4 sideways).
# Measured: flags live defender-on-tower (113x99) and tower-top fragment
# (39x57) boxes; flags 0/25 ground-truth Pokemon boxes (23 confirmed-catch
# labels + 2 live-frame Pokemon) -- a Pokemon stands on plain map ground, never
# on a white disc.
GYM_CONTEXT_HEIGHT_MULT = 2.5
GYM_CONTEXT_PAD_DIV = 4

# --- Max-battle / raid ORANGE badge rejector (color + white-glyph structure) ---
# Power spots & raids hang an ORANGE disc badge with a big WHITE monster-face
# glyph over the map ("N players" pill below). It kept surviving the purity
# check because pink VFX petals inside the bbox dilute hue purity. Structure
# separates cleanly (measured on 10 live badge mis-click crops vs all
# ground-truth Pokemon): badge orange-fraction 0.49-0.60 with largest-white
# fill 0.071-0.213; ORANGE Pokemon (Growlithe-likes, Hisuian Voltorb) have
# orange up to 0.74 but largest-white fill <= 0.029 (no big white glyph).
ORANGE_HUE = (5, 22)
ORANGE_SAT_MIN = 150
MAX_BADGE_ORANGE_MIN = 0.35
MAX_BADGE_WHITE_FILL_MIN = 0.05

# --- Pink ELITE-raid badge rejector (template at the candidate's location) ---
# The pink "elite raid" disc badge partially-boxed escapes every color rule
# (partial boxes mix background), and a white-glyph rule would false-reject a
# real pink Pokemon (measured wfill 0.069 vs badge 0.042-0.051 -- overlaps).
# Template matching is exact: badge_pink.png (cropped from the live fixture),
# searched in a +/-45 px window around the candidate centre. IMPORTANT: a high
# score alone is NOT enough -- a Pokemon STANDING NEXT TO a badge also gets a
# window hit (measured 0.89 on a real Pokemon box). Reject only when the
# candidate's CENTRE lies inside the matched badge rectangle: measured badge
# and badge-slice candidates 1.00 (centre inside), neighbouring-Pokemon boxes
# keep their tap because their centre is outside the matched rect.
PINK_BADGE_SCORE_MIN = 0.80
PINK_BADGE_SEARCH_TOL = 45

# --- Flat-UI badge rejector (raid badge / elite-raid badge / timer pill) ---
# Raid eggs+bosses hang an ORANGE badge and elite raids a PINK "233 days"
# badge/timer pill on the map; all are FLAT single-hue UI discs/pills, while a
# Pokemon is a shaded 3D model with hue variation. Measured hue purity (share
# of colored [sat>120] pixels within +/-8 of the mode hue), on ground truth:
#   25 real Pokemon (23 confirmed-catch boxes + green + orange Voltorb): <= 0.88
#   flat UI badges (pink elite badge 0.93/0.94, pink timer pill 0.97, orange
#   tower frag 1.00, yellow spin-arrow 0.91): >= 0.91
# Threshold 0.90 splits the measured gap. Requires >= 30 colored pixels.
BADGE_HUE_PURITY_MIN = 0.90
BADGE_HUE_TOL = 8
BADGE_MIN_COLORED_PX = 30
BADGE_SAT_MIN = 120


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


_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
# The map camera tilts, so the badge renders at different perspectives; one
# template per observed tilt (same multi-template pattern as the close-X).
# badge_pink2.png was cropped from a live mis-click audit frame (tilted view;
# the flat template scored only ~0.6 there). Measured per-template max over
# all confirmed-catch Pokemon boxes: 0.65 / 0.59 -- threshold 0.80 is safe.
_PINK_BADGE_TEMPLATES = [
    t for t in (
        cv2.imread(os.path.join(_TEMPLATE_DIR, "badge_pink.png")),
        cv2.imread(os.path.join(_TEMPLATE_DIR, "badge_pink2.png")),
    ) if t is not None
]


def _largest_white_fill(crop: np.ndarray) -> float:
    """Area fraction of the largest bright-desaturated connected component."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] > WHITE_V_MIN) & (hsv[:, :, 1] < WHITE_S_MAX)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n <= 1:
        return 0.0
    h, w = crop.shape[:2]
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / float(w * h)


def is_max_badge(crop: np.ndarray) -> bool:
    """True if a candidate crop is the orange Max-battle / raid badge: a large
    orange fraction PLUS a big compact white glyph (the monster face). Orange
    Pokemon pass because they have no large white component (<= 0.029 measured
    vs badge >= 0.071)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    orange = float(((hue >= ORANGE_HUE[0]) & (hue <= ORANGE_HUE[1])
                    & (sat > ORANGE_SAT_MIN)).mean())
    if orange < MAX_BADGE_ORANGE_MIN:
        return False
    return _largest_white_fill(crop) >= MAX_BADGE_WHITE_FILL_MIN


def is_pink_badge(img: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """True if the candidate at bbox (x,y,w,h) IS the pink elite-raid badge:
    the badge template matches near the candidate AND the candidate's centre
    falls inside the matched badge rectangle (a Pokemon merely standing next
    to a badge keeps its tap)."""
    H, W = img.shape[:2]
    cx, cy = x + w // 2, y + h // 2
    for templ in _PINK_BADGE_TEMPLATES:
        th, tw = templ.shape[:2]
        y0 = max(0, cy - th // 2 - PINK_BADGE_SEARCH_TOL)
        y1 = min(H, cy + th // 2 + PINK_BADGE_SEARCH_TOL)
        x0 = max(0, cx - tw // 2 - PINK_BADGE_SEARCH_TOL)
        x1 = min(W, cx + tw // 2 + PINK_BADGE_SEARCH_TOL)
        region = img[y0:y1, x0:x1]
        if region.shape[0] < th or region.shape[1] < tw:
            continue
        res = cv2.matchTemplate(region, templ, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < PINK_BADGE_SCORE_MIN:
            continue
        mx, my = x0 + loc[0], y0 + loc[1]  # matched badge rect top-left, frame coords
        if mx <= cx <= mx + tw and my <= cy <= my + th:
            return True
    return False


# --- Max-battle spawn detector: the red "N players" pill ----------------------
# Dynamax/max-battle spawns (giant Chansey etc.) are Pokemon MODELS, so no
# body-shape rule separates them (measured: contiguous-component size gave NO
# gap vs dense-scene wild spawns). Their unique tell is UI: a COREL-RED rounded
# "N players" pill hangs on/over the spawn. Measured pill pixels: hue ~1,
# sat ~160, val ~235 -- a flat wide bar (w>=80, aspect>=1.8, fill>=0.5).
# Candidate has one in/near its box -> it's part of a max spawn -> reject.
# Measured: flags both live giant-Chansey picks; 4/162 confirmed catches were
# near enough to a pill to be skipped (acceptable recall tax vs ~2 panel
# round-trips per minute near a spawn).
_PILL_SAT = (120, 210)
_PILL_VAL_MIN = 200
_PILL_MIN_W = 80
_PILL_MIN_H = 28
_PILL_MIN_ASPECT = 1.8
_PILL_MIN_FILL = 0.5


def has_max_pill_near(img: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """True if a red max-battle player-count pill sits in / near the candidate
    bbox (searched 1.5*w sideways, 2.5*h up/down -- the pill floats over the
    spawn's body, which the candidate is usually a fragment of)."""
    H, W = img.shape[:2]
    x0, y0 = max(0, x - int(1.5 * w)), max(0, y - int(2.5 * h))
    x1, y1 = min(W, x + w + int(1.5 * w)), min(H, y + h + int(2.5 * h))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # hue <= 8 ONLY (no wrap-around branch): the player pill is pure red
    # (measured hue ~1); the magenta "233 days" countdown pill sits at hue
    # ~178 and must NOT trip this -- it also hangs near ordinary gyms.
    red = ((hue <= 8)
           & (sat > _PILL_SAT[0]) & (sat < _PILL_SAT[1])
           & (val > _PILL_VAL_MIN)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    for i in range(1, n):
        cw, ch, ca = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if (cw >= _PILL_MIN_W and ch >= _PILL_MIN_H
                and cw / max(1, ch) >= _PILL_MIN_ASPECT and ca >= _PILL_MIN_FILL * cw * ch):
            return True
    return False


def is_flat_badge(crop: np.ndarray) -> bool:
    """True if a candidate crop (BGR) is a flat single-hue UI element -- a raid
    badge, elite-raid badge, countdown pill, or spin-arrow -- that must be
    rejected. A Pokemon is a shaded 3D model: its colored pixels spread across
    hues; a UI badge is one flat tint (measured gap: Pokemon <= 0.88 purity,
    badges >= 0.91; threshold 0.90)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0][hsv[:, :, 1] > BADGE_SAT_MIN].astype(int)
    if hue.size < BADGE_MIN_COLORED_PX:
        return False
    mode = int(np.bincount(hue, minlength=180).argmax())
    circ_dist = np.minimum(np.abs(hue - mode), 180 - np.abs(hue - mode))
    return float((circ_dist <= BADGE_HUE_TOL).mean()) >= BADGE_HUE_PURITY_MIN


def has_gym_below(img: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """True if the candidate at bbox (x,y,w,h) sits ON TOP of a gym: the same
    structural photodisc test, but run on a context crop extended DOWNWARD --
    tower-top fragments and defender Pokemon have the gym's white disc/ribbon
    system directly below their own bbox, real wild Pokemon have map ground."""
    H, W = img.shape[:2]
    pad = w // GYM_CONTEXT_PAD_DIV
    ext = img[y : min(H, y + int(GYM_CONTEXT_HEIGHT_MULT * h)),
              max(0, x - pad) : min(W, x + w + pad)]
    return is_gym_photodisc(ext)


def _dynamic_bg_bands(hue: np.ndarray, region_mask: np.ndarray) -> list:
    """Top-N dominant hues of the search region (the terrain), to be excluded
    from the foreground mask. Suppresses +/-(2*tol) around each found mode
    before finding the next so both modes aren't the same band."""
    hues = hue[region_mask > 0]
    if hues.size == 0:
        return []
    hist = np.bincount(hues.ravel(), minlength=180).astype(float)
    bands, idx = [], np.arange(180)
    for _ in range(BG_DYNAMIC_MODES):
        mode = int(hist.argmax())
        if hist[mode] <= 0:
            break
        bands.append(mode)
        d = np.minimum(np.abs(idx - mode), 180 - np.abs(idx - mode))
        hist[d <= 2 * BG_DYNAMIC_TOL] = 0
    return bands


def _in_range(val, lo_ratio, hi_ratio, dim):
    return lo_ratio * dim <= val <= hi_ratio * dim


def propose(img: np.ndarray, phone: Phone, exclude=None) -> Optional[Target]:
    """Propose the best tap target. `exclude` is an optional list of
    (x, y, radius) no-tap zones -- the catch loop blacklists spots whose tap
    just yielded nothing/a panel (giant raid bosses on gyms, the walking buddy,
    inert icons), so we pick the next-best candidate instead of re-tapping the
    same object every tick."""
    height, width = img.shape[:2]
    exclude = exclude or []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]

    # Restrict the search to the central region (below top gym/raid band + UI,
    # above bottom button bar, off the left menu strip).
    search_mask = np.zeros((height, width), np.uint8)
    y0, y1 = int(SEARCH_Y_LOW * height), int(SEARCH_Y_HIGH * height)
    x0, x1 = int(SEARCH_X_LOW * width), int(SEARCH_X_HIGH * width)
    search_mask[y0:y1, x0:x1] = 255

    not_bg = (hue < BG_HUE_LOW) | (hue > BG_HUE_HIGH)
    for mode in _dynamic_bg_bands(hue, search_mask):
        d = np.minimum(np.abs(hue.astype(int) - mode), 180 - np.abs(hue.astype(int) - mode))
        not_bg &= d > BG_DYNAMIC_TOL
    not_bg_hue = not_bg.astype(np.uint8) * 255
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
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > MAX_ASPECT:  # thin streak (ring edge / lure beam / label) -> not a Pokemon
            continue

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

        # A striped power-spot / gym tower whose disc straddles the bbox edge
        # can dodge the tight-bbox test: re-run it 25% expanded. Measured: 0/26
        # ground-truth Pokemon flagged; VFX-ring Pokemon (radar0/1) stay kept.
        ph, pw = int(h * 0.25), int(w * 0.25)
        if is_gym_photodisc(img[max(0, y - ph) : y + h + ph, max(0, x - pw) : x + w + pw]):
            continue

        # Reject anything standing ON a gym (tower-top fragments, defenders):
        # same disc test on the context BELOW the bbox.
        if has_gym_below(img, x, y, w, h):
            continue

        # Reject flat single-hue UI badges (raid / elite-raid / timer pills).
        if is_flat_badge(img[y : y + h, x : x + w]):
            continue

        # Reject the orange Max-battle/raid badge (orange disc + white face).
        if is_max_badge(img[y : y + h, x : x + w]):
            continue

        # Reject the pink elite-raid badge (template match at this location).
        if is_pink_badge(img, x, y, w, h):
            continue

        # Reject anything under a red max-battle player pill (dynamax spawn).
        if has_max_pill_near(img, x, y, w, h):
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        # Reject the player avatar's own box (screen-center trainer model).
        if _in_range(cx, *AVATAR_EXCL_X, width) and _in_range(cy, *AVATAR_EXCL_Y, height):
            continue

        # Skip blacklisted no-tap zones (recent failed taps).
        if any(math.hypot(cx - ex, cy - ey) <= er for ex, ey, er in exclude):
            continue

        dist = math.hypot(cx - avatar_x, cy - avatar_y)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_target = Target(x=round(cx), y=round(cy), bbox=(x, y, w, h))

    return best_target


def _next_label_index(images_dir: str, prefix: str = "pokemon_") -> int:
    if not os.path.isdir(images_dir):
        return 0
    highest = -1
    for name in os.listdir(images_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".png" or not stem.startswith(prefix):
            continue
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        highest = max(highest, idx)
    return highest + 1


# Label tightening: the segmentation CLOSE morphology can merge a Pokemon with
# an ADJACENT flat-magenta UI badge (dynamax/elite-raid disc + countdown pill)
# into one blob -- the tap still lands and the catch confirms, but the saved
# YOLO box then spans badge+Pokemon (bad supervision). Before writing a label,
# shrink the box to the largest colorful component that is NOT badge-magenta.
# Magenta only: the badge pink is a narrow flat band; orange is NOT excluded
# here (orange Pokemon are common, the orange badge rarely adjoins spawns).
_BADGE_MAGENTA_HUE = (158, 176)
_BADGE_MAGENTA_SAT_MIN = 180
_TIGHTEN_MIN_KEEP = 0.20  # tightened part must keep >=20% of box area, else keep box


def tighten_bbox(img: np.ndarray, bbox: tuple) -> tuple:
    """Shrink a label bbox to the largest non-badge colorful component. Returns
    the original bbox when nothing meaningful remains (e.g. an actually-pink
    Pokemon whose body IS magenta-ish)."""
    x, y, w, h = bbox
    crop = img[y : y + h, x : x + w]
    if crop.size == 0:
        return bbox
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    colorful = (((hue < BG_HUE_LOW) | (hue > BG_HUE_HIGH)) & (sat > SAT_THRESHOLD))
    badge = ((hue >= _BADGE_MAGENTA_HUE[0]) & (hue <= _BADGE_MAGENTA_HUE[1])
             & (sat >= _BADGE_MAGENTA_SAT_MIN))
    keep = (colorful & ~badge).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(keep, connectivity=8)
    if n <= 1:
        return bbox
    li = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by = stats[li, cv2.CC_STAT_LEFT], stats[li, cv2.CC_STAT_TOP]
    bw, bh = stats[li, cv2.CC_STAT_WIDTH], stats[li, cv2.CC_STAT_HEIGHT]
    if bw * bh < _TIGHTEN_MIN_KEEP * w * h:
        return bbox  # too little left -> the box was probably fine (pink mon)
    return (x + bx, y + by, bw, bh)


def save_label(img: np.ndarray, target: Target, dataset_dir: str) -> str:
    """Save a YOLO training example from a CONFIRMED catch: the FULL map frame
    plus a one-box label (class 0 = wild Pokemon) at the target. Called on every
    confirmed encounter, so normal operation accumulates real, correctly-labeled
    training data. Layout: `<dataset>/images/<stem>.png` + `<dataset>/labels/
    <stem>.txt` (YOLO format `class cx cy w h`, all normalized 0-1). Returns the
    image path.

    (The confirmed box is ground truth -- the tap DID open an encounter -- so
    every saved frame has at least one correct Pokemon box. Other Pokemon in the
    same frame stay unlabeled; enough frames average that sparsity out.)
    """
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    h, w = img.shape[:2]
    bx, by, bw, bh = tighten_bbox(img, target.bbox)  # drop merged-in UI badges
    cx = (bx + bw / 2.0) / w
    cy = (by + bh / 2.0) / h

    with _LABEL_LOCK:  # atomic index+write so 2 phones can't collide
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        index = _next_label_index(images_dir)
        stem = f"pokemon_{index:06d}"
        img_path = os.path.join(images_dir, f"{stem}.png")
        lbl_path = os.path.join(labels_dir, f"{stem}.txt")
        cv2.imwrite(img_path, img)
        with open(lbl_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}\n")

    return img_path


def save_negative_label(img: np.ndarray, target: Target, dataset_dir: str) -> str:
    """Save a HARD-NEGATIVE training example from a tap that opened a closable
    panel (gym / stop / power spot / Rocket / Route...): the full map frame plus
    a one-box label of class 1 ("avoid") at the tapped bbox. YOLO then learns
    panel-openers as an explicit class -- far stronger avoidance signal than
    leaving them as implicit background -- and the live detector acts ONLY on
    class-0 (pokemon) boxes. Stems are `avoid_*` beside the `pokemon_*` frames.

    (Only `panel` outcomes are saved: the tap PROVABLY hit an interactable
    non-Pokemon object. `nothing` outcomes stay unlabeled -- they are usually
    motion-drift misses of REAL Pokemon, which must not be taught as avoid.)
    """
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    h, w = img.shape[:2]
    bx, by, bw, bh = target.bbox
    cx = (bx + bw / 2.0) / w
    cy = (by + bh / 2.0) / h

    with _LABEL_LOCK:
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        index = _next_label_index(images_dir, prefix="avoid_")
        stem = f"avoid_{index:06d}"
        img_path = os.path.join(images_dir, f"{stem}.png")
        with open(os.path.join(labels_dir, f"{stem}.txt"), "w") as f:
            f.write(f"1 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}\n")
        cv2.imwrite(img_path, img)

    return img_path


# Padding around the target bbox in a click-audit crop: enough surrounding map
# context to tell WHAT the detector latched onto (a stop's disc, a route bubble,
# VFX) without saving the whole frame.
_CLICK_PAD = 100


def save_click_debug(img: np.ndarray, target: Target, outcome: str, dataset_dir: str,
                     result_img: np.ndarray = None) -> str:
    """Audit trail of EVERY tap the bot attempts: a padded crop of the map frame
    around the proposed target (detector bbox in red) and, when available, the
    RESULTING screen pasted alongside -- so each image reads "clicked this ->
    got this" (a PokeStop panel, a gym, a Rocket grunt, an encounter, ...).
    Filed by outcome:

        <dataset>/clicks/encounter/  tap opened an encounter (real Pokemon - correct)
        <dataset>/clicks/panel/      tap opened a closable panel (gym / stop /
                                     power spot / Rocket / Route - a WRONG click)
        <dataset>/clicks/nothing/    still on the map afterwards (empty scenery
                                     or an inert icon)
        <dataset>/clicks/timeout/    screen never resolved within the timeout

    The clicks/ tree is CLEARED at every runner start (see runner.py) so each
    run's folder holds exactly that run's taps for review.
    """
    h, w = img.shape[:2]
    bx, by, bw, bh = target.bbox
    x0, y0 = max(0, bx - _CLICK_PAD), max(0, by - _CLICK_PAD)
    x1, y1 = min(w, bx + bw + _CLICK_PAD), min(h, by + bh + _CLICK_PAD)
    crop = img[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (bx - x0, by - y0), (bx - x0 + bw, by - y0 + bh), (0, 0, 255), 3)
    cv2.putText(crop, getattr(target, "src", "?"), (4, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if result_img is not None:
        ch = crop.shape[0]
        rw = max(1, int(result_img.shape[1] * ch / result_img.shape[0]))
        result_small = cv2.resize(result_img, (rw, ch))
        crop = np.hstack([crop, np.full((ch, 6, 3), 255, np.uint8), result_small])

    out_dir = os.path.join(dataset_dir, "clicks", outcome)
    with _LABEL_LOCK:  # atomic index+write so 2 phones can't collide
        os.makedirs(out_dir, exist_ok=True)
        index = _next_label_index(out_dir, prefix="click_")
        path = os.path.join(out_dir, f"click_{index:06d}.png")
        cv2.imwrite(path, crop)
    return path
