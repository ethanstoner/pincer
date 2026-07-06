import cv2
import numpy as np

from src.config import load_config
from src.detector import propose


def test_proposes_target_on_real_map():
    cfg = load_config("config.json")
    phone = cfg.phones[0]
    img = cv2.imread("tests/fixtures/map.png")
    t = propose(img, phone)
    assert t is not None
    assert t.x > 0.10 * phone.width      # not in left UI strip
    assert t.y < 0.85 * phone.height     # not in bottom bar
    assert t.y > 0.07 * phone.height     # not in top status bar


def test_returns_none_on_empty_scene():
    cfg = load_config("config.json")
    phone = cfg.phones[0]
    blank = np.full((2388, 1080, 3), (120, 70, 40), np.uint8)  # flat desaturated blue, low saturation
    assert propose(blank, phone) is None
