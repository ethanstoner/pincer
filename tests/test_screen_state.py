import cv2
import numpy as np
from src.screen_state import classify, ScreenState

def _load(name): return cv2.imread(f"tests/fixtures/{name}")

def test_map_fixture_classifies_as_map():
    assert classify(_load("map.png")) == ScreenState.MAP

def test_encounter_fixture_classifies_as_encounter():
    assert classify(_load("encounter.png")) == ScreenState.ENCOUNTER

def test_map_after_catch_classifies_as_map():
    assert classify(_load("map_after_catch.png")) == ScreenState.MAP

def test_solid_black_classifies_as_unknown():
    # Safety fallback: an image matching neither encounter nor map must land in
    # UNKNOWN, never be forced into ENCOUNTER/MAP (the catch loop only throws on
    # ENCOUNTER). Solid black has saturation 0 -> fails both feature bands.
    assert classify(np.zeros((2388, 1080, 3), np.uint8)) == ScreenState.UNKNOWN

def test_solid_gray_classifies_as_unknown():
    # Solid mid-gray also has saturation 0 and hue 0 -> fails both bands.
    assert classify(np.full((2388, 1080, 3), 128, np.uint8)) == ScreenState.UNKNOWN
