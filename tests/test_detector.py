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


# --- Ground-truth bboxes (x, y, w, h) measured on tests/fixtures/map.png and
# QA-labeled by the user. These lock in "never proposes a gym / avatar / UI". ---
GYM_BBOXES = [
    (228, 1373, 192, 148),  # cand_00: gym (white photodisc + red pillar)
    (108, 1878, 176, 151),  # cand_01: gym
    (237, 1118, 81, 121),   # cand_03: gym (red light-ray pillar)
    (225, 1562, 142, 89),   # cand_07: gym
]
AVATAR_BBOX = (487, 1412, 103, 106)  # cand_06: screen-center player trainer
UI_BBOXES = [
    (952, 549, 70, 49),  # cand_09: "recenter" Poke Ball button
    (950, 270, 32, 27),  # cand_11: red rocket / compass arrow
]


def _inside(px, py, box):
    x, y, w, h = box
    return x <= px <= x + w and y <= py <= y + h


def test_target_in_central_region_and_not_gym_avatar_ui():
    cfg = load_config("config.json")
    phone = cfg.phones[0]
    img = cv2.imread("tests/fixtures/map.png")
    t = propose(img, phone)
    assert t is not None
    # (a) inside the central detection region (below top gym/raid band,
    #     above bottom button bar, off the left menu strip)
    assert 0.10 * phone.width <= t.x <= 0.95 * phone.width
    assert 0.40 * phone.height <= t.y <= 0.85 * phone.height
    # (b) not inside any known gym / avatar / UI bbox
    for box in GYM_BBOXES + [AVATAR_BBOX] + UI_BBOXES:
        assert not _inside(t.x, t.y, box), f"target landed in excluded box {box}"
