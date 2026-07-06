import cv2
import numpy as np
import pytest
from src.screen_state import (
    classify, ScreenState, encounter_scores, in_encounter, _MATCH_THRESHOLD,
    has_close_button, close_button_score, is_screen_off,
)


def test_map_pokeball_visible_on_map_absent_on_panels():
    from src.screen_state import has_map_pokeball
    for name in ["map.png", "map_after_catch.png", "radar0.png", "radar1.png",
                 "map_gym_badges.png", "map_dynamax.png"]:
        assert has_map_pokeball(_load(name)) is True, name
    for name in ["gym.png", "pokestop.png", "rocket_grunt.png", "route_screen.png",
                 "encounter.png", "encounter_dusk.png"]:
        assert has_map_pokeball(_load(name)) is False, name


def test_find_ok_button_on_dialog_none_on_map_screens():
    from src.screen_state import find_ok_button
    pt = find_ok_button(_load("dialog_ok.png"))    # Groudon bonus popup (live)
    assert pt is not None
    x, y = pt
    assert abs(x - 525) < 60 and abs(y - 2040) < 60
    for name in ["map.png", "map_purple_storm.png", "encounter.png",
                 "gym.png", "pokestop.png", "map_dynamax.png"]:
        assert find_ok_button(_load(name)) is None, name


def test_is_screen_off_true_on_black():
    assert is_screen_off(np.zeros((2388, 1080, 3), np.uint8)) is True

@pytest.mark.parametrize("name", ["map.png", "map_after_catch.png", "encounter.png", "encounter_dusk.png", "gym.png", "pokestop.png"])
def test_is_screen_off_false_on_real_screens(name):
    assert is_screen_off(cv2.imread(f"tests/fixtures/{name}")) is False

def _load(name): return cv2.imread(f"tests/fixtures/{name}")

# --- in_encounter(): the CHEAP boolean the catch loop polls. Must agree with
#     classify()'s ENCOUNTER decision but skip the MAP hue/sat work. ---
@pytest.mark.parametrize("name", ["encounter.png", "encounter_dusk.png"])
def test_in_encounter_true_for_encounter_fixtures(name):
    assert in_encounter(_load(name)) is True

@pytest.mark.parametrize("name", ["map.png", "map_after_catch.png", "radar0.png", "radar1.png"])
def test_in_encounter_false_for_map_fixtures(name):
    assert in_encounter(_load(name)) is False

def test_in_encounter_false_for_black_image():
    assert in_encounter(np.zeros((2388, 1080, 3), np.uint8)) is False

# --- ENCOUNTER: must hold across DIFFERENT backgrounds (the flicker regression) ---
@pytest.mark.parametrize("name", ["encounter.png", "encounter_dusk.png"])
def test_encounter_fixtures_classify_as_encounter(name):
    # encounter_dusk.png has a dusk/city background (not grass); keying on the
    # background color used to flicker ENCOUNTER<->UNKNOWN here. UI anchors fix it.
    assert classify(_load(name)) == ScreenState.ENCOUNTER

# --- MAP fixtures must classify as MAP and NEVER as ENCOUNTER (dangerous) ---
# map_purple_storm.png: live frame under event weather -- the purple raid-storm
# sky pushes the map hue outside the hue/sat window, which froze both phones in
# an UNKNOWN->wait loop; the pokeball-button signal now classifies it MAP.
@pytest.mark.parametrize("name", ["map.png", "map_after_catch.png", "radar0.png", "radar1.png", "map_purple_storm.png"])
def test_map_fixtures_classify_as_map(name):
    result = classify(_load(name))
    assert result != ScreenState.ENCOUNTER  # never throw balls on a map
    assert result == ScreenState.MAP

# --- Safety fallback: images matching neither must land in UNKNOWN ---
def test_solid_black_classifies_as_unknown():
    assert classify(np.zeros((2388, 1080, 3), np.uint8)) == ScreenState.UNKNOWN

def test_solid_gray_classifies_as_unknown():
    assert classify(np.full((2388, 1080, 3), 128, np.uint8)) == ScreenState.UNKNOWN

def test_white_loading_flash_is_not_map():
    # The white encounter-loading flash has near-zero saturation but its hue
    # MEAN is arbitrary and landed inside the map hue window -> classified MAP
    # -> _await_encounter bailed out of REAL loading encounters (live audit
    # frames nothing/5+18: clean Pokemon taps whose result frame was white).
    # A near-white frame (slight blue tint puts hue in the map band) must be
    # UNKNOWN so the encounter poll keeps waiting.
    img = np.full((2388, 1080, 3), 245, np.uint8)
    img[:, :, 0] = 252  # faint blue cast -> hue lands ~map band, sat stays tiny
    assert classify(img) == ScreenState.UNKNOWN

# --- Assert the anchor-score margin directly (proves it is not on the edge) ---
def test_encounter_anchor_score_margin():
    # Both encounter backgrounds score well above threshold on ALL anchors...
    for name in ["encounter.png", "encounter_dusk.png"]:
        for s in encounter_scores(_load(name)):
            assert s >= 0.95, f"{name} anchor score {s} unexpectedly low"
    # ...and every map fixture scores well below threshold on at least one anchor.
    for name in ["map.png", "map_after_catch.png", "radar0.png", "radar1.png"]:
        assert min(encounter_scores(_load(name))) < _MATCH_THRESHOLD - 0.15


# --- has_close_button(): detects a mis-tapped gym / PokeStop / menu so the loop
#     can bail instead of waiting out the encounter timeout. Must fire on BOTH
#     gym (dark theme) and PokeStop (light theme) -- a gym-only template got stuck
#     on PokeStops. ---
@pytest.mark.parametrize("name", ["gym.png", "pokestop.png", "rocket_grunt.png", "route_screen.png"])
def test_has_close_button_true_on_closable_panels(name):
    assert has_close_button(_load(name)) is True

@pytest.mark.parametrize("name", ["map.png", "map_after_catch.png", "encounter.png", "encounter_dusk.png"])
def test_has_close_button_false_off_panel(name):
    assert has_close_button(_load(name)) is False

def test_close_button_score_margin():
    # all closable panels score well above threshold; map/encounter well below.
    assert close_button_score(_load("pokestop.png")) >= 0.90
    assert close_button_score(_load("gym.png")) >= 0.75
    assert close_button_score(_load("rocket_grunt.png")) >= 0.90  # teal Rocket X theme
    assert close_button_score(_load("route_screen.png")) >= 0.90  # Route detail X theme
    for name in ["map.png", "map_after_catch.png", "encounter.png", "encounter_dusk.png"]:
        assert close_button_score(_load(name)) < 0.50


# --- Panel screens that trapped the bot live (each is a real stuck frame) -------
# rocket_grunt.png: Team GO Rocket grunt dialog on an invaded stop. TWO failures
# compounded: (1) the grey close_x template scored only 0.31 on the teal Rocket
# X, so recovery never closed it; (2) the purple dialog background sits inside
# the MAP hue window, so classify() said MAP and the loop kept running the
# detector on (and tapping!) the dialog -- dangerously close to BATTLE.
# route_screen.png: Route detail sheet (tapped a route marker). Its teal-ring X
# on a light card scored 0.20/0.47 on the earlier templates -> recovery spun.
@pytest.mark.parametrize("name", ["rocket_grunt.png", "route_screen.png"])
def test_panel_screens_are_unknown_never_map_or_encounter(name):
    result = classify(_load(name))
    assert result != ScreenState.ENCOUNTER   # never throw balls at a panel
    assert result != ScreenState.MAP         # never run the map detector on one
    assert result == ScreenState.UNKNOWN     # -> _recover, which taps the X
