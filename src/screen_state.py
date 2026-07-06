import os
from enum import Enum, auto

import cv2
import numpy as np


class ScreenState(Enum):
    MAP = auto()
    ENCOUNTER = auto()
    POKESTOP = auto()  # no fixture yet; detection not implemented -> currently unreachable
    GYM = auto()        # no fixture yet; detection not implemented -> currently unreachable
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# ENCOUNTER detection: background-INVARIANT UI anchors (cv2.matchTemplate).
#
# The earlier version keyed ENCOUNTER on the AR background color (green grass
# hue<97 AND sat>170). That was WRONG: encounter backgrounds vary (day grass,
# dusk city, night, water) because they are live AR scenery. On the dusk-city
# fixture the saturation sat right on the threshold, so frame-to-frame jitter
# (particles, the Pokemon bobbing) FLICKERED ENCOUNTER<->UNKNOWN and the throw
# loop bailed without throwing.
#
# The encounter *UI* is pixel-identical regardless of background. We anchor on
# two fixed-position UI buttons that are ABSENT/different on the map:
#   - berry button, bottom-left   ~ratio (0.12, 0.90)  -> src/templates/enc_berry.png
#   - ball-select button, bottom-right ~ratio (0.87, 0.90) -> src/templates/enc_ballselect.png
# (The far-left the client menu strip is deliberately avoided — it appears on both
# map and encounter.) Templates were cropped from encounter.png and ship in
# src/templates/ so the module does not depend on tests/.
#
# Measured TM_CCOEFF_NORMED match scores (max over a +/-45px search window at
# each anchor's fixed home box), on all six fixtures:
#
#                     enc_berry   enc_ballselect
#   encounter          1.0000       1.0000
#   encounter_dusk     0.9997       0.9998   <- non-grass bg, still ~1.0 (invariant)
#   map                0.0010       0.2561
#   map_after_catch    0.2676       0.2116
#   radar0             0.2131       0.3191
#   radar1             0.2223       0.3185
#
# Both encounter backgrounds score >= 0.9997 on BOTH anchors; every map fixture
# scores <= 0.3191 on both. We require BOTH anchors above threshold (AND) so a
# false ENCOUNTER (which would trigger a ball throw) needs two independent UI
# buttons to match. Threshold 0.60 sits in the wide empty gap:
#   margin above worst map score:      0.60 - 0.3191 = 0.28
#   margin below worst encounter score: 0.9997 - 0.60 = 0.40
# ---------------------------------------------------------------------------

_MATCH_THRESHOLD = 0.60
_SEARCH_TOL = 45  # px slack around each anchor's home box, absorbs alignment jitter

# (template filename, center-x ratio, center-y ratio, half-size px used at crop time)
_ANCHORS = [
    ("enc_berry.png", 0.12, 0.90, 70),
    ("enc_ballselect.png", 0.87, 0.90, 70),
]

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def _load_templates():
    loaded = []
    for fname, cxr, cyr, half in _ANCHORS:
        path = os.path.join(_TEMPLATE_DIR, fname)
        templ = cv2.imread(path)
        if templ is None:
            raise FileNotFoundError(f"ENCOUNTER anchor template missing: {path}")
        loaded.append((templ, cxr, cyr, half))
    return loaded


_TEMPLATES = _load_templates()


# --- MAP detection (unchanged): map backgrounds ARE color-consistent -----------
# Blue satellite imagery. Measured mean HSV (H:0-179 S:0-255):
#                  full H  full S   mid H   mid S
#   map.png:       104.18  127.73  104.77  132.45
#   map_after:     105.80  129.98  107.02  135.53
# Encounter backgrounds vary, so this is NOT used for encounter — only to
# positively confirm MAP. Anything failing both ENCOUNTER and MAP -> UNKNOWN.
_MAP_HUE_MIN = 98
_MAP_HUE_MAX = 120
_MAP_SAT_MAX = 160


def _anchor_score(img, templ, cxr, cyr, half):
    h, w = img.shape[:2]
    cx, cy = int(cxr * w), int(cyr * h)
    y0, y1 = max(0, cy - half - _SEARCH_TOL), min(h, cy + half + _SEARCH_TOL)
    x0, x1 = max(0, cx - half - _SEARCH_TOL), min(w, cx + half + _SEARCH_TOL)
    region = img[y0:y1, x0:x1]
    th, tw = templ.shape[:2]
    if region.shape[0] < th or region.shape[1] < tw:
        return 0.0  # image too small / wrong shape -> not an encounter
    res = cv2.matchTemplate(region, templ, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def encounter_scores(img: np.ndarray) -> list:
    """Return the match score for each ENCOUNTER anchor (exposed for tests/debug)."""
    return [_anchor_score(img, t, cxr, cyr, half) for t, cxr, cyr, half in _TEMPLATES]


def in_encounter(img: np.ndarray) -> bool:
    """Cheap boolean the catch loop POLLS: True iff both ENCOUNTER UI anchors
    match above threshold. This is exactly classify()'s ENCOUNTER decision but
    exposed on its own so the loop can spin on it without paying for the MAP
    hue/sat work every poll. Ball throws are gated on this, so it requires BOTH
    anchors (AND) -- a single false match can never green-light a throw."""
    return all(s >= _MATCH_THRESHOLD for s in encounter_scores(img))


# --- closable-panel (gym / PokeStop / menu) detection --------------------------
# Gyms, PokeStops and full-screen menus all show a circular X close button at the
# bottom-centre that the map and encounter screens do NOT have. Detecting it lets
# the catch loop bail out of a mis-tapped panel immediately, instead of waiting
# out the whole encounter-load timeout. Measured close_x match scores:
#   gym / pokestop screens : ~1.000 (button is pixel-identical across panels)
#   map / encounter screens: <= 0.64
# The template is a TIGHT crop of the X GLYPH ONLY (not the surrounding panel):
# a wider crop drags in the panel background, which differs by theme (dark gym vs
# light-purple PokeStop) and tanked the score on PokeStops (0.57 -> stuck). The
# tight glyph matches on BOTH: measured PokeStop 1.00, gym 0.85, map/encounter
# <= 0.39. Threshold 0.60 sits in the gap (margin >= 0.21 either side).
_CLOSE_THRESHOLD = 0.60
_CLOSE_ANCHOR = ("close_x.png", 0.498, 0.9527, 42)  # (file, cx ratio, cy ratio, half px)


def _load_close_template():
    path = os.path.join(_TEMPLATE_DIR, _CLOSE_ANCHOR[0])
    templ = cv2.imread(path)
    if templ is None:
        raise FileNotFoundError(f"close-button template missing: {path}")
    return templ


_CLOSE_TEMPLATE = _load_close_template()


def close_button_score(img: np.ndarray) -> float:
    """Match score for the bottom-centre X close button (exposed for tests/debug)."""
    _, cxr, cyr, half = _CLOSE_ANCHOR
    return _anchor_score(img, _CLOSE_TEMPLATE, cxr, cyr, half)


def has_close_button(img: np.ndarray) -> bool:
    """True iff the X close button is present -- i.e. we're on a gym / PokeStop /
    menu panel (NOT the map and NOT an encounter). Lets the loop exit a mis-tapped
    panel at once."""
    return close_button_score(img) >= _CLOSE_THRESHOLD


def _region_hsv_mean(img, y0, y1, x0, x1):
    h, w = img.shape[:2]
    region = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)  # (H, S, V)


def classify(img: np.ndarray) -> ScreenState:
    """Classify a phone screenshot into a ScreenState.

    Safety-first ordering: ENCOUNTER is only returned when BOTH background-
    invariant UI anchors match confidently (the catch loop throws balls only on
    ENCOUNTER, so a false positive is dangerous). Then MAP is checked by its
    stable background color. Anything matching neither falls back to UNKNOWN
    rather than being forced into MAP/ENCOUNTER. POKESTOP/GYM are not detectable
    yet (no fixtures) and will present as UNKNOWN until implemented.
    """
    scores = encounter_scores(img)
    if all(s >= _MATCH_THRESHOLD for s in scores):
        return ScreenState.ENCOUNTER

    full_h, full_s, _ = _region_hsv_mean(img, 0.0, 1.0, 0.0, 1.0)
    mid_h, mid_s, _ = _region_hsv_mean(img, 0.35, 0.65, 0.2, 0.8)
    is_map = (
        _MAP_HUE_MIN < full_h < _MAP_HUE_MAX and full_s < _MAP_SAT_MAX
        and _MAP_HUE_MIN < mid_h < _MAP_HUE_MAX and mid_s < _MAP_SAT_MAX
    )
    if is_map:
        return ScreenState.MAP

    return ScreenState.UNKNOWN
