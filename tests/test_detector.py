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


def test_flat_badge_rejects_ui_badges_keeps_pokemon():
    """Locks the live gym/raid mis-click fix: flat single-hue UI badges (elite
    raid badge, countdown pill, spin-arrow, tower fragment) must be rejected;
    shaded 3D Pokemon (incl. the flat-ISH orange Hisuian Voltorb) must be kept.
    Boxes are ground-truthed live-frame crops (dataset/clicks review)."""
    from src.detector import is_flat_badge
    g = cv2.imread("tests/fixtures/map_gym_badges.png")
    v = cv2.imread("tests/fixtures/map_voltorb_tower.png")

    # flat UI elements -> reject
    assert is_flat_badge(_crop(g, (904, 1435, 96, 104))) is True   # pink "233 days" elite badge
    assert is_flat_badge(_crop(g, (791, 1352, 39, 57))) is True    # orange tower fragment
    assert is_flat_badge(_crop(v, (108, 990, 58, 50))) is True     # pink countdown pill

    # real Pokemon -> keep
    assert is_flat_badge(_crop(g, (350, 1775, 103, 124))) is False  # green Pokemon
    assert is_flat_badge(_crop(v, (608, 1520, 82, 79))) is False    # orange Hisuian Voltorb


def test_gym_below_rejects_tower_toppers_keeps_ground_pokemon():
    """Candidates standing ON a gym tower (defenders / tower-top fragments)
    have the gym's white disc system below the bbox -> rejected by context.
    Wild Pokemon stand on plain map ground -> kept."""
    from src.detector import has_gym_below
    g = cv2.imread("tests/fixtures/map_gym_badges.png")
    v = cv2.imread("tests/fixtures/map_voltorb_tower.png")

    assert has_gym_below(v, 770, 1081, 113, 99) is True   # defender on tower top
    assert has_gym_below(g, 791, 1352, 39, 57) is True    # tower-top fragment

    assert has_gym_below(g, 350, 1775, 103, 124) is False  # green Pokemon on ground
    assert has_gym_below(v, 608, 1520, 82, 79) is False    # Voltorb on ground


def test_propose_avoids_gym_and_badges_on_live_gym_dense_frame():
    """End-to-end on the live gym-dense frame that produced the wasted-tap
    streak: whatever propose() picks, it must not be a badge, a timer pill, or
    anything on a gym tower."""
    cfg = load_config("config.json")
    phone = cfg.phones[0]
    bad_boxes = {
        "map_gym_badges.png": [
            (904, 1435, 96, 104),   # pink elite badge
            (791, 1352, 39, 57),    # orange tower fragment
            (858, 890, 180, 260),   # right gym tower + defender
            (600, 530, 260, 400),   # upper gym towers
        ],
        "map_voltorb_tower.png": [
            (108, 990, 58, 50),     # pink countdown pill
            (770, 1081, 113, 99),   # defender on tower
        ],
    }
    for fname, boxes in bad_boxes.items():
        img = cv2.imread(f"tests/fixtures/{fname}")
        t = propose(img, phone)
        if t is None:
            continue  # no proposal at all is acceptable (never a bad tap)
        for box in boxes:
            assert not _inside(t.x, t.y, box), f"{fname}: picked excluded box {box}"


def test_max_badge_rejects_orange_badge_keeps_orange_pokemon():
    """Orange Max-battle/raid badge = orange disc + big white face glyph.
    Orange Pokemon (Hisuian Voltorb) have no large white component -> kept."""
    from src.detector import is_max_badge

    # synth badge: orange disc with a big compact white glyph in the middle
    badge = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(badge, (50, 50), 48, (30, 100, 235), -1)   # orange (BGR)
    cv2.circle(badge, (50, 50), 20, (255, 255, 255), -1)  # white glyph
    assert is_max_badge(badge) is True

    v = cv2.imread("tests/fixtures/map_voltorb_tower.png")
    assert is_max_badge(_crop(v, (608, 1520, 82, 79))) is False  # orange Voltorb


def test_pink_badge_template_rejects_badge_not_neighbours():
    """The pink elite-raid badge is template-matched AT the candidate location:
    the badge box and a partial slice of it are rejected; a Pokemon box merely
    near a badge keeps its tap (its centre is outside the matched rect)."""
    from src.detector import is_pink_badge
    g = cv2.imread("tests/fixtures/map_gym_badges.png")
    assert is_pink_badge(g, 904, 1435, 96, 104) is True   # the badge itself
    assert is_pink_badge(g, 904, 1490, 96, 44) is True    # partial bottom slice
    assert is_pink_badge(g, 350, 1775, 103, 124) is False  # green Pokemon far away


def test_clear_click_audit_removes_only_clicks(tmp_path):
    from src.runner import clear_click_audit
    d = tmp_path / "dataset"
    (d / "clicks" / "panel").mkdir(parents=True)
    (d / "clicks" / "panel" / "click_000000.png").write_bytes(b"x")
    (d / "images").mkdir()
    (d / "images" / "pokemon_000000.png").write_bytes(b"x")
    clear_click_audit(str(d))
    assert not (d / "clicks").exists()                     # audit wiped
    assert (d / "images" / "pokemon_000000.png").exists()  # training data kept


def test_bottom_ui_strip_is_unreachable_even_for_vivid_blobs():
    """The pokeball menu button (bottom-centre, ~y-ratio 0.94) and the whole
    bottom UI bar sit BELOW the search region (SEARCH_Y_HIGH=0.85): even a
    perfectly Pokemon-like vivid blob painted there must never be proposed."""
    cfg = load_config("config.json")
    phone = cfg.phones[0]
    img = cv2.imread("tests/fixtures/map.png").copy()
    cv2.circle(img, (540, 2245), 45, (0, 0, 230), -1)   # vivid red blob on the pokeball
    t = propose(img, phone)
    if t is not None:
        assert t.y <= 0.85 * phone.height   # whatever it picked, not bottom UI
        assert abs(t.x - 540) > 60 or abs(t.y - 2245) > 60


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
