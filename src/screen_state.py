from enum import Enum, auto

import cv2
import numpy as np


class ScreenState(Enum):
    MAP = auto()
    ENCOUNTER = auto()
    POKESTOP = auto()  # no fixture yet; detection not implemented -> currently unreachable
    GYM = auto()        # no fixture yet; detection not implemented -> currently unreachable
    UNKNOWN = auto()


# --- Measured feature values (mean HSV, OpenCV ranges H:0-179 S:0-255 V:0-255) ---
# Regions measured on the real 1080x2388 BGR fixtures via cv2.cvtColor(..., COLOR_BGR2HSV):
#
#   region "full"        = entire frame
#   region "mid-center"  = rows 35%-65%, cols 20%-80% (AR background band, avoids
#                           top status icons and bottom ball/button row)
#
#                     full H   full S   mid H    mid S
#   map.png:          104.18   127.73   104.77   132.45
#   encounter.png:      90.74   190.38    88.81   216.05
#   map_after_catch:   105.80   129.98   107.02   135.53
#
# Encounter's AR grass background is both a distinctly lower hue (~89-91, greenish)
# and a much higher saturation (~190-216, vivid) than either map fixture's blue
# satellite background (hue ~104-107, saturation ~128-136).
#   Hue gap:        max(map hues) 107.02 vs encounter max 90.74  -> margin ~16.3
#   Saturation gap: min(encounter sat) 190.38 vs max(map sat) 135.53 -> margin ~54.9
#
# Both regions must agree (AND) before we trust either classification, so a
# single-region fluke (e.g. a UI overlay skewing one region) can't flip the result.

_ENCOUNTER_HUE_MAX = 97      # encounter hues sit ~89-91; maps sit ~104-107 -> midpoint w/ margin
_ENCOUNTER_SAT_MIN = 170     # encounter sat sits ~190-216; maps sit ~128-136 -> huge margin

_MAP_HUE_MIN = 98
_MAP_HUE_MAX = 120
_MAP_SAT_MAX = 160           # maps sit ~128-136, well under this; encounter sits ~190-216


def _region_hsv_mean(img, y0, y1, x0, x1):
    h, w = img.shape[:2]
    region = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)  # (H, S, V)


def classify(img: np.ndarray) -> ScreenState:
    """Classify a phone screenshot into a ScreenState.

    Safety-first ordering: ENCOUNTER is only returned when confidently detected
    (the catch loop throws balls only on ENCOUNTER), then MAP is checked, and
    anything that doesn't confidently match either falls back to UNKNOWN rather
    than being forced into MAP or ENCOUNTER. POKESTOP/GYM are not detectable yet
    (no fixtures exist) and will present as UNKNOWN until implemented.
    """
    full_h, full_s, _ = _region_hsv_mean(img, 0.0, 1.0, 0.0, 1.0)
    mid_h, mid_s, _ = _region_hsv_mean(img, 0.35, 0.65, 0.2, 0.8)

    is_encounter = (
        full_h < _ENCOUNTER_HUE_MAX and full_s > _ENCOUNTER_SAT_MIN
        and mid_h < _ENCOUNTER_HUE_MAX and mid_s > _ENCOUNTER_SAT_MIN
    )
    if is_encounter:
        return ScreenState.ENCOUNTER

    is_map = (
        _MAP_HUE_MIN < full_h < _MAP_HUE_MAX and full_s < _MAP_SAT_MAX
        and _MAP_HUE_MIN < mid_h < _MAP_HUE_MAX and mid_s < _MAP_SAT_MAX
    )
    if is_map:
        return ScreenState.MAP

    return ScreenState.UNKNOWN
