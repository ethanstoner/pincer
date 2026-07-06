import cv2
from src.screen_state import classify, ScreenState

def _load(name): return cv2.imread(f"tests/fixtures/{name}")

def test_map_fixture_classifies_as_map():
    assert classify(_load("map.png")) == ScreenState.MAP

def test_encounter_fixture_classifies_as_encounter():
    assert classify(_load("encounter.png")) == ScreenState.ENCOUNTER

def test_map_after_catch_classifies_as_map():
    assert classify(_load("map_after_catch.png")) == ScreenState.MAP
