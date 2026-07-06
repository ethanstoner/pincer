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
# The SAT range needs BOTH bounds: measured map mean sat is ~127-136, but the
# WHITE encounter-loading flash has sat ~0-25 with an arbitrary hue mean that
# can land inside the hue window — with no lower bound it classified as MAP,
# so _await_encounter bailed out of real encounters mid-load ("nothing" audit
# entries whose result frame is white), wasting a recover+rescan per catch.
_MAP_HUE_MIN = 98
_MAP_HUE_MAX = 120
_MAP_SAT_MIN = 60
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


# --- closable-panel (gym / PokeStop / menu / Rocket) detection ------------------
# Gyms, PokeStops, full-screen menus and Team GO Rocket grunt dialogs all show a
# circular X close button at the bottom-centre that the map and encounter screens
# do NOT have. Detecting it lets the catch loop bail out of a mis-tapped panel
# immediately, instead of waiting out the whole encounter-load timeout.
# The X comes in DIFFERENT THEMES, so this is a multi-template match (max score):
#   - close_x.png: grey/white X glyph -- gym, PokeStop, menus. TIGHT crop of the
#     glyph ONLY (a wider crop drags in the panel background, which differs by
#     theme -- dark gym vs light-purple PokeStop -- and tanked the score on
#     PokeStops, 0.57 -> stuck). Measured: PokeStop 1.00, gym 0.85,
#     map/encounter <= 0.39.
#   - close_x_rocket.png: white X on a filled teal disc -- Team GO Rocket grunt
#     dialog. The grey template scores only 0.31 on it (got stuck on an invaded
#     stop), AND the grunt screen's purple background fooled the MAP hue check,
#     so classify() ran the detector on it. Cropped from a real stuck frame
#     (tests/fixtures/rocket_grunt.png). Measured: rocket 1.0000, all other
#     fixtures <= 0.17.
#   - close_x_route.png: teal ring + teal X on a LIGHT card -- the Route detail
#     screen (grey scored 0.20, rocket 0.47 there -> got stuck on a route
#     marker). Cropped from a real stuck frame (tests/fixtures/route_screen.png).
#     Measured: route 1.0000, all other fixtures <= 0.42.
# Threshold 0.60 sits in the gap for ALL templates (margin >= 0.18 either side).
# NOTE: _recover also blind-taps this same bottom-centre spot when a screen
# stays un-playable with NO template match, so an unseen theme delays the bot a
# couple seconds instead of trapping it -- but add a template (crop the real
# frame, measure margins on all fixtures) whenever a new theme shows up.
_CLOSE_THRESHOLD = 0.60
# (file, cx ratio, cy ratio, half px) -- one entry per close-button THEME
_CLOSE_ANCHORS = [
    ("close_x.png", 0.498, 0.9527, 42),
    ("close_x_rocket.png", 0.500, 0.9455, 42),
    ("close_x_route.png", 0.500, 0.9410, 42),
]


def _load_close_templates():
    loaded = []
    for fname, cxr, cyr, half in _CLOSE_ANCHORS:
        path = os.path.join(_TEMPLATE_DIR, fname)
        templ = cv2.imread(path)
        if templ is None:
            raise FileNotFoundError(f"close-button template missing: {path}")
        loaded.append((templ, cxr, cyr, half))
    return loaded


_CLOSE_TEMPLATES = _load_close_templates()


def close_button_scores(img: np.ndarray) -> list:
    """Per-theme match scores for the bottom-centre X (exposed for tests/debug)."""
    return [_anchor_score(img, t, cxr, cyr, half) for t, cxr, cyr, half in _CLOSE_TEMPLATES]


def close_button_score(img: np.ndarray) -> float:
    """Best match score across all close-button themes."""
    return max(close_button_scores(img))


def has_close_button(img: np.ndarray) -> bool:
    """True iff the X close button is present -- i.e. we're on a gym / PokeStop /
    menu panel (NOT the map and NOT an encounter). Lets the loop exit a mis-tapped
    panel at once."""
    return close_button_score(img) >= _CLOSE_THRESHOLD


# --- overworld pokeball menu button (bottom-centre) ----------------------------
# The main-menu pokeball button is ALWAYS visible on the overworld map and is
# REPLACED by the close-X on every panel. It sits right next to the universal
# X spot, so _recover's blind tap could open the main menu whenever a real map
# frame mis-classified as UNKNOWN (petal-dense scenes throw off the hue check).
# Guard: pokeball visible => we're on the map => blind tap FORBIDDEN.
# Measured (threshold 0.85): all map/radar fixtures 1.000 (one 0.24 outlier
# where a spawn model occludes the button -- blind-tapping there would hit that
# Pokemon, which is fine); panels/encounters <= 0.69.
_MAP_POKEBALL_THRESHOLD = 0.85
_MAP_POKEBALL_ANCHOR = ("map_pokeball.png", 0.500, 0.9401, 42)


def _load_map_pokeball_template():
    path = os.path.join(_TEMPLATE_DIR, _MAP_POKEBALL_ANCHOR[0])
    templ = cv2.imread(path)
    if templ is None:
        raise FileNotFoundError(f"map pokeball template missing: {path}")
    return templ


_MAP_POKEBALL_TEMPLATE = _load_map_pokeball_template()


def has_map_pokeball(img: np.ndarray) -> bool:
    """True iff the overworld pokeball menu button is visible bottom-centre --
    i.e. this frame is the map, whatever the hue check said."""
    _, cxr, cyr, half = _MAP_POKEBALL_ANCHOR
    return _anchor_score(img, _MAP_POKEBALL_TEMPLATE, cxr, cyr, half) >= _MAP_POKEBALL_THRESHOLD


# --- OK-button dialogs (bonus popups / notices) --------------------------------
# Some full-screen dialogs have NO close-X at all -- just one wide teal-green
# "OK" pill (live catch: the "Groudon Primal Reversion Bonus" popup trapped a
# phone). Detection is color+structure: hue 50-90, sat 60-190, val>190
# (measured on the live pill: hue 55-81, sat 82-175, val 208-218), one
# component at least 45% of screen width, wide (aspect>=3), solid (fill>=0.55),
# in the lower 45% of the screen. Two-button confirm dialogs use ~35%-width
# side-by-side pills, so the width floor keeps this from ever "accepting"
# anything -- it only fires on single-button info dialogs.
_OK_HUE = (50, 90)
_OK_SAT = (60, 190)
_OK_VAL_MIN = 190
_OK_MIN_WIDTH_RATIO = 0.45
_OK_MIN_ASPECT = 3.0
_OK_MIN_FILL = 0.55
_OK_Y_LOW = 0.55


def find_ok_button(img: np.ndarray):
    """Centre (x, y) of a single wide OK pill in the lower screen, or None."""
    h, w = img.shape[:2]
    y0 = int(_OK_Y_LOW * h)
    crop = img[y0:, :]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((hue >= _OK_HUE[0]) & (hue <= _OK_HUE[1])
            & (sat >= _OK_SAT[0]) & (sat <= _OK_SAT[1])
            & (val >= _OK_VAL_MIN)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        cw, ch, ca = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if (cw >= _OK_MIN_WIDTH_RATIO * w and cw / max(1, ch) >= _OK_MIN_ASPECT
                and ca >= _OK_MIN_FILL * cw * ch):
            cx, cy = centroids[i]
            return int(cx), int(cy) + y0
    return None


def is_screen_off(img: np.ndarray) -> bool:
    """True if the frame is a blank/near-black display-off screen.

    When the phone display sleeps, `screencap` returns an all-black image, which
    otherwise classifies as UNKNOWN and makes the loop spin uselessly in recovery.
    A slept screen is ~uniformly black (mean intensity ~0); a genuine dark night
    scene still has bright UI (ball/berry icons, HUD), so its mean is far higher.
    """
    return float(img.mean()) < 6.0


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

    # A closable panel is checked BEFORE the map hue test: the Team GO Rocket
    # grunt dialog's purple background lands inside the map hue/sat window, so
    # hue alone mis-classified it as MAP and the loop kept running the detector
    # on (and tapping) a dialog. Any bottom-centre X means "panel", never MAP.
    if has_close_button(img):
        return ScreenState.UNKNOWN

    # The overworld pokeball button is the STRONGEST map signal (measured
    # 1.000 on every map fixture, <= 0.69 on panels): event weather (purple
    # raid-storm sky) shifts the map hue outside the window below, which froze
    # BOTH phones in an UNKNOWN->wait loop on a perfectly good map. Panels
    # were already returned above, so this cannot fire on one.
    if has_map_pokeball(img):
        return ScreenState.MAP

    full_h, full_s, _ = _region_hsv_mean(img, 0.0, 1.0, 0.0, 1.0)
    mid_h, mid_s, _ = _region_hsv_mean(img, 0.35, 0.65, 0.2, 0.8)
    is_map = (
        _MAP_HUE_MIN < full_h < _MAP_HUE_MAX and _MAP_SAT_MIN < full_s < _MAP_SAT_MAX
        and _MAP_HUE_MIN < mid_h < _MAP_HUE_MAX and _MAP_SAT_MIN < mid_s < _MAP_SAT_MAX
    )
    if is_map:
        return ScreenState.MAP

    return ScreenState.UNKNOWN
