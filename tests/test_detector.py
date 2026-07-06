import os

import cv2
import numpy as np

from src.config import load_config
from src.detector import is_gym_photodisc, propose, save_label, Target


def test_save_label_writes_full_frame_and_yolo_label(tmp_path):
    img = np.zeros((2388, 1080, 3), np.uint8)
    target = Target(x=440, y=1150, bbox=(432, 1140, 108, 120))  # center (486,1200)
    ds = str(tmp_path / "dataset")

    path = save_label(img, target, ds)

    # full frame saved (not a crop)
    saved = cv2.imread(path)
    assert saved.shape == (2388, 1080, 3)
    assert path.replace("\\", "/").endswith("images/pokemon_000000.png")

    # YOLO label: class cx cy w h, normalized 0-1
    lbl = os.path.join(ds, "labels", "pokemon_000000.txt")
    parts = open(lbl).read().split()
    assert parts[0] == "0"
    cls, cx, cy, w, h = parts
    assert abs(float(cx) - 486 / 1080) < 1e-3
    assert abs(float(cy) - 1200 / 2388) < 1e-3
    assert abs(float(w) - 108 / 1080) < 1e-3
    assert abs(float(h) - 120 / 2388) < 1e-3

    # index advances on the next save
    p2 = save_label(img, target, ds)
    assert p2.replace("\\", "/").endswith("images/pokemon_000001.png")


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


def _crop(img, box):
    x, y, w, h = box
    return img[y : y + h, x : x + w]


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


def test_photodisc_helper_rejects_gym_keeps_pokemon_under_vfx():
    """Locks the live false-rejection fix: the structural gym/stop filter must
    reject a solid photodisc but KEEP a real Pokemon crossed by a thin VFX ring.
    """
    m = cv2.imread("tests/fixtures/map.png")
    r0 = cv2.imread("tests/fixtures/radar0.png")
    r1 = cv2.imread("tests/fixtures/radar1.png")

    # solid gym photodiscs -> must be rejected
    assert is_gym_photodisc(_crop(m, (228, 1373, 192, 148))) is True   # cand_00
    assert is_gym_photodisc(_crop(m, (108, 1878, 176, 151))) is True   # cand_01
    assert is_gym_photodisc(_crop(m, (225, 1562, 142, 89))) is True    # cand_07

    # real Pokemon under a white/purple spinning VFX ring -> must be KEPT.
    # (raw white-fraction of these crops is 0.226 / 0.227 -- the exact live
    #  false-rejection this structural filter fixes.)
    assert is_gym_photodisc(_crop(r0, (611, 1497, 68, 68))) is False
    assert is_gym_photodisc(_crop(r1, (615, 1495, 71, 63))) is False


def test_propose_on_live_radar_scenes():
    """Dense live scenes (heavy VFX overlays) must still yield a real Pokemon,
    inside the central region and never a gym/stop photodisc."""
    cfg = load_config("config.json")
    phone = cfg.phones[0]
    # (fixture, a known raid/gym disc bbox the target must avoid)
    cases = [
        ("radar0.png", (850, 546, 113, 113)),
        ("radar1.png", (853, 543, 111, 115)),
    ]
    for fname, gym_bbox in cases:
        img = cv2.imread(f"tests/fixtures/{fname}")
        t = propose(img, phone)
        assert t is not None, f"{fname}: dense scene yielded no target"
        assert 0.10 * phone.width <= t.x <= 0.95 * phone.width
        assert 0.40 * phone.height <= t.y <= 0.85 * phone.height
        assert not _inside(t.x, t.y, gym_bbox), f"{fname}: target landed on gym"
        # the tapped crop is structurally NOT a photodisc
        assert is_gym_photodisc(_crop(img, t.bbox)) is False
