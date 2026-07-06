import cv2
import numpy as np
import pytest
from src.screen_state import (
    classify, ScreenState, encounter_scores, in_encounter, _MATCH_THRESHOLD,
    has_close_button, close_button_score,
)

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
@pytest.mark.parametrize("name", ["map.png", "map_after_catch.png", "radar0.png", "radar1.png"])
def test_map_fixtures_classify_as_map(name):
    result = classify(_load(name))
    assert result != ScreenState.ENCOUNTER  # never throw balls on a map
    assert result == ScreenState.MAP

# --- Safety fallback: images matching neither must land in UNKNOWN ---
def test_solid_black_classifies_as_unknown():
    assert classify(np.zeros((2388, 1080, 3), np.uint8)) == ScreenState.UNKNOWN

def test_solid_gray_classifies_as_unknown():
    assert classify(np.full((2388, 1080, 3), 128, np.uint8)) == ScreenState.UNKNOWN

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
#     can bail instead of waiting out the encounter timeout. ---
def test_has_close_button_true_on_gym():
    assert has_close_button(_load("gym.png")) is True

@pytest.mark.parametrize("name", ["map.png", "map_after_catch.png", "encounter.png", "encounter_dusk.png"])
def test_has_close_button_false_off_panel(name):
    assert has_close_button(_load(name)) is False

def test_close_button_score_margin():
    # gym scores ~1.0; map/encounter score well under the 0.82 threshold.
    assert close_button_score(_load("gym.png")) >= 0.95
    for name in ["map.png", "map_after_catch.png", "encounter.png", "encounter_dusk.png"]:
        assert close_button_score(_load(name)) < 0.72
