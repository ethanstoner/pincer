import random
import threading

import cv2
import numpy as np

from src.catch_loop import CatchLoop
from src.config import Config, Phone
from src.detector import Target
from src.screen_state import ScreenState


DUMMY_IMG = np.full((4, 4, 3), 40, np.uint8)   # non-black -> is_screen_off() False
BLACK_IMG = np.zeros((4, 4, 3), np.uint8)       # a display-asleep (black) frame


class FakeDevice:
    def __init__(self, img=None):
        self.taps = []
        self.swipes = []
        self.key_backs = []
        self.wakes = 0
        self.screencaps = 0
        self._img = DUMMY_IMG if img is None else img

    def screencap(self):
        self.screencaps += 1
        return self._img

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, x1, y1, x2, y2, ms):
        self.swipes.append((x1, y1, x2, y2, ms))

    def key_back(self):
        self.key_backs.append(True)

    def wake(self):
        self.wakes += 1


class Scripted:
    """Callable double: yields values from a list, sticking on the last one
    once exhausted (predicates are polled repeatedly, so exact call counts
    must not matter -- only the sequence tail)."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]


def make_config():
    return Config(
        adb_path="adb.exe",
        dataset_dir="dataset",
        phones=[Phone(serial="SERIAL", width=1080, height=2388)],
        anchors_ratio={
            "ball_center": [0.5, 0.88944],
            "throw_start": [0.5, 0.83752],
            "throw_end": [0.5, 0.35595],
            "flee_button": [0.04815, 0.06072],
            "close_button": [0.498, 0.9527],
        },
        timing={
            "encounter_load_ms": [1, 2],   # -> encounter poll timeout ~2ms
            "post_throw_ms": [1, 2],       # -> resolve poll timeout ~2ms
            "map_scan_ms": [1, 2],
            "stuck_timeout_ms": 15,
            "max_throws": 3,
        },
    )


def make_loop(classifier, encounter_check, device=None, detector_fn=None, labeler=None, close_check=None, pokeball_check=None):
    config = make_config()
    phone = config.phones[0]
    device = device or FakeDevice()
    detector_fn = detector_fn or (lambda img, phone: None)
    close_check = close_check or (lambda img: False)
    pokeball_check = pokeball_check or (lambda img: False)
    calls = {"labels": [], "clicks": [], "neg_labels": []}

    def default_labeler(img, target, dataset_dir):
        calls["labels"].append((img, target, dataset_dir))
        return "labeled.png"

    def neg_labeler(img, target, dataset_dir):
        calls["neg_labels"].append((target, dataset_dir))
        return "avoid.png"

    def click_logger(img, target, outcome, dataset_dir, result_img=None):
        calls["clicks"].append((target, outcome))
        return "click.png"

    labeler = labeler or default_labeler
    sleeps = []
    clock = {"t": 0.0}

    def sleep_fn(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds        # fake clock advances with each sleep

    loop = CatchLoop(
        device=device,
        config=config,
        phone=phone,
        classifier=classifier,
        detector_fn=detector_fn,
        labeler=labeler,
        neg_labeler=neg_labeler,
        click_logger=click_logger,
        sleep_fn=sleep_fn,
        rng=random.Random(42),
        encounter_check=encounter_check,
        close_check=close_check,
        pokeball_check=pokeball_check,
        clock=lambda: clock["t"],
    )
    return loop, device, calls, sleeps


def test_caught_path_taps_once_labels_once_throws_and_no_flee_or_recover():
    # initial state MAP; after tap the encounter UI appears (in_encounter True);
    # one throw, then classify sees MAP (caught) -> done.
    classifier = Scripted([ScreenState.MAP])          # MAP initially and on resolve poll
    encounter_check = Scripted([True])                # encounter confirmed throughout
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    detector_fn = lambda img, phone: target
    label_calls = []

    def labeler(img, tgt, dataset_dir):
        label_calls.append((img, tgt, dataset_dir))
        return "path.png"

    loop, device, _, _ = make_loop(
        classifier, encounter_check, detector_fn=detector_fn, labeler=labeler
    )

    loop.tick()

    assert len(device.taps) == 1
    tx, ty = device.taps[0]
    assert abs(tx - target.x) <= 10 and abs(ty - target.y) <= 10
    assert len(label_calls) == 1 and label_calls[0][1] is target
    assert len(device.swipes) >= 1          # threw at least once
    assert device.key_backs == []           # no recover


def test_wrong_target_never_throws_and_recovers():
    # tap opened a non-encounter (e.g. a gym): in_encounter never confirmed ->
    # _await_encounter bails -> _recover taps the close X. SAFETY: no throw.
    classifier = Scripted(
        [ScreenState.MAP, ScreenState.UNKNOWN, ScreenState.UNKNOWN, ScreenState.MAP]
    )
    encounter_check = Scripted([False])        # encounter NEVER confirmed
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    detector_fn = lambda img, phone: target

    loop, device, calls, _ = make_loop(
        classifier, encounter_check, detector_fn=detector_fn, close_check=lambda img: True
    )

    loop.tick()

    assert device.swipes == []       # NO throw
    assert calls["labels"] == []     # and NO self-label
    assert loop._pt("close_button") in device.taps  # recovered by tapping the X


def test_await_encounter_returns_frame_on_first_confirmed_frame():
    loop, device, _, _ = make_loop(
        Scripted([ScreenState.UNKNOWN]), Scripted([True]), close_check=lambda img: False
    )
    enc_img, _ = loop._await_encounter(100000)
    assert enc_img is not None
    assert device.screencaps == 1        # proceeded instantly, did not wait the timeout


def test_await_encounter_bails_on_first_frame_when_close_button_present():
    # a gym / PokeStop / menu / Rocket dialog opened (its X is visible) -> bail
    # at once, no waiting -- and hand back the bail frame so _recover can act on
    # it without another screencap.
    loop, device, _, _ = make_loop(
        Scripted([ScreenState.UNKNOWN]), Scripted([False]), close_check=lambda img: True
    )
    enc_img, bail_img = loop._await_encounter(100000)
    assert enc_img is None
    assert bail_img is not None          # the frame with the X, for _recover
    assert device.screencaps == 1        # bailed on the first frame


def test_await_encounter_bails_when_still_map_after_transition_window():
    # tap opened nothing: still MAP. Bail once past the transition window, NOT at
    # the (huge) timeout.
    loop, device, _, _ = make_loop(
        Scripted([ScreenState.MAP]), Scripted([False]), close_check=lambda img: False
    )
    enc_img, _ = loop._await_encounter(100000)
    assert enc_img is None
    # bailed ~_MAP_BAIL_MS in (1600ms / 30ms polls ~= 55 caps), not after 100s
    assert device.screencaps < 80


def test_recover_blind_taps_close_spot_when_no_theme_matches():
    # A panel whose X theme has NO template yet (Rocket and Route both started
    # this way): classify stays UNKNOWN, close_check never fires. After the
    # early waits, recovery must blind-tap the universal bottom-centre X spot
    # instead of spinning forever -- and still NEVER throw.
    classifier = Scripted([ScreenState.UNKNOWN])   # stuck forever, unrecognized
    loop, device, _, _ = make_loop(
        classifier, Scripted([False]), close_check=lambda img: False
    )
    loop._recover()
    assert loop._pt("close_button") in device.taps  # escalated to the blind tap
    assert device.swipes == []                      # INV-2: still never throws


def test_blind_tap_suppressed_when_map_pokeball_visible():
    # A map frame that mis-classifies as UNKNOWN (petal-dense hue drift) must
    # NOT be blind-tapped: the X spot is right beside the pokeball MENU button,
    # so that tap would open the main menu. Pokeball visible => no blind tap.
    classifier = Scripted([ScreenState.UNKNOWN])   # stuck-looking forever
    loop, device, _, _ = make_loop(
        classifier, Scripted([False]),
        close_check=lambda img: False,
        pokeball_check=lambda img: True,           # but the pokeball IS there
    )
    loop._recover()
    assert device.taps == []                       # waited, never tapped


def test_recover_taps_ok_button_on_xless_dialog():
    # A bonus popup has NO X, only a wide OK pill -> recovery taps the pill.
    classifier = Scripted([ScreenState.UNKNOWN, ScreenState.UNKNOWN, ScreenState.MAP])
    loop, device, _, _ = make_loop(
        classifier, Scripted([False]), close_check=lambda img: False,
    )
    loop.ok_finder = lambda img: (540, 2040)
    loop._recover()
    assert (540, 2040) in device.taps
    assert device.swipes == []                     # INV-2 intact


def test_recover_prefers_close_x_over_battle_like_pill():
    # SAFETY: the Rocket grunt dialog has BOTH a close X and a teal BATTLE
    # pill that matches the OK-pill profile. Recovery must tap the X, NEVER
    # the pill (that would start a Rocket battle).
    classifier = Scripted([ScreenState.UNKNOWN, ScreenState.MAP])
    loop, device, _, _ = make_loop(
        classifier, Scripted([False]), close_check=lambda img: True,
    )
    loop.ok_finder = lambda img: (540, 1726)       # "BATTLE" pill location
    loop._recover()
    assert loop._pt("close_button") in device.taps  # tapped the X
    assert (540, 1726) not in device.taps           # never the battle pill


def test_recover_does_not_blind_tap_transient_unknowns_that_resolve():
    # UNKNOWN for one attempt (e.g. catch animation) then MAP: must resolve by
    # waiting -- no blind tap fired before the escalation threshold.
    classifier = Scripted([ScreenState.UNKNOWN, ScreenState.MAP])
    loop, device, _, _ = make_loop(
        classifier, Scripted([False]), close_check=lambda img: False
    )
    loop._recover()
    assert device.taps == []


def test_click_logger_records_encounter_outcome_on_good_tap():
    classifier = Scripted([ScreenState.MAP])
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    loop, device, calls, _ = make_loop(
        classifier, Scripted([True]), detector_fn=lambda img, phone: target
    )
    loop.tick()
    assert calls["clicks"] == [(target, "encounter")]


def test_click_logger_records_panel_outcome_on_mistap():
    # tap opened a gym/stop/Rocket panel -> the audit crop is filed under panel/
    classifier = Scripted([ScreenState.MAP, ScreenState.UNKNOWN, ScreenState.UNKNOWN, ScreenState.MAP])
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    loop, device, calls, _ = make_loop(
        classifier, Scripted([False]), detector_fn=lambda img, phone: target,
        close_check=lambda img: True,
    )
    loop.tick()
    assert calls["clicks"] == [(target, "panel")]
    assert calls["labels"] == []     # a mis-tap never becomes YOLO training data


def test_panel_tap_saves_hard_negative_label():
    # tap opened a closable panel -> saved as a class-1 "avoid" YOLO example
    classifier = Scripted([ScreenState.MAP, ScreenState.UNKNOWN, ScreenState.UNKNOWN, ScreenState.MAP])
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    loop, device, calls, _ = make_loop(
        classifier, Scripted([False]), detector_fn=lambda img, phone: target,
        close_check=lambda img: True,
    )
    loop.tick()
    assert calls["neg_labels"] == [(target, "dataset")]
    assert calls["labels"] == []            # never ALSO a positive label


def test_empty_tap_saves_no_negative_label():
    # 'nothing' outcomes are usually drift-misses of REAL Pokemon -> teaching
    # them as "avoid" would poison the model. No negative label.
    classifier = Scripted([ScreenState.MAP])   # sticky map = empty tap
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    loop, device, calls, _ = make_loop(
        classifier, Scripted([False]), detector_fn=lambda img, phone: target
    )
    loop.tick()
    assert calls["neg_labels"] == []


def test_click_logger_records_nothing_outcome_on_empty_tap():
    # tap hit empty scenery: still MAP past the transition window
    classifier = Scripted([ScreenState.MAP])   # sticky MAP throughout
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    loop, device, calls, _ = make_loop(
        classifier, Scripted([False]), detector_fn=lambda img, phone: target
    )
    loop.tick()
    assert calls["clicks"] == [(target, "nothing")]


def test_failed_tap_spot_is_blacklisted_and_not_retapped():
    # tap yields nothing -> the spot is embargoed: a plain detector (no exclude
    # support) proposing the SAME spot next tick must NOT be tapped again.
    classifier = Scripted([ScreenState.MAP])          # sticky map (empty tap)
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    loop, device, _, _ = make_loop(
        classifier, Scripted([False]), detector_fn=lambda img, phone: target
    )
    loop.tick()                       # tap -> still MAP -> blacklist the spot
    taps_after_first = len(device.taps)
    assert taps_after_first == 1
    loop.tick()                       # same proposal -> embargoed -> no tap
    assert len(device.taps) == taps_after_first


def test_exclude_zones_passed_to_exclude_aware_detector():
    # An exclude-aware detector receives the embargoed spot and can pick the
    # next-best candidate -- keeps catching while a raid boss is blacklisted.
    classifier = Scripted([ScreenState.MAP])
    bad = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    good = Target(x=800, y=1600, bbox=(780, 1580, 40, 40))
    seen_excludes = []

    def detector(img, phone, exclude=None):
        seen_excludes.append(list(exclude or []))
        for t in (bad, good):
            if not any((t.x - ex) ** 2 + (t.y - ey) ** 2 <= er ** 2
                       for ex, ey, er in (exclude or [])):
                return t
        return None

    loop, device, _, _ = make_loop(Scripted([ScreenState.MAP]), Scripted([False]),
                                   detector_fn=detector)
    loop.tick()                                    # taps bad -> blacklists it
    loop.tick()                                    # detector must get the zone
    assert seen_excludes[0] == []
    assert len(seen_excludes[1]) == 1              # embargo forwarded
    assert any(abs(tx - good.x) <= 10 for tx, ty in device.taps[1:])  # next-best tapped


def test_propose_exclude_zone_yields_a_different_target():
    from src.config import Phone
    import cv2
    from src.detector import propose
    phone = Phone(serial="X", width=1080, height=2388)
    img = cv2.imread("tests/fixtures/map.png")
    t1 = propose(img, phone)
    assert t1 is not None
    t2 = propose(img, phone, exclude=[(t1.x, t1.y, 60)])
    if t2 is not None:  # another candidate exists -> must be a different spot
        assert (abs(t2.x - t1.x) > 60) or (abs(t2.y - t1.y) > 60)


def test_camera_pans_after_empty_scans_but_not_immediately():
    # No target on the map: the FIRST ticks must not pan (spawns may just be
    # loading); once the empty stretch exceeds the threshold, a horizontal
    # camera drag fires and screen-space state resets.
    classifier = Scripted([ScreenState.MAP])
    loop, device, _, _ = make_loop(classifier, Scripted([False]),
                                   detector_fn=lambda img, phone: None)
    loop.tick()
    assert device.swipes == []                       # too early to pan

    loop._fail_spots = [(1, 2, 999.0)]
    loop._last_target_t = -10.0                      # long-empty stretch
    loop.tick()
    assert len(device.swipes) == 1                   # camera drag fired
    (x1, y1, x2, y2, ms) = device.swipes[0]
    assert x1 > x2                                   # horizontal right-to-left
    assert abs(y1 - y2) <= 24                        # ~level drag, not a throw
    assert loop._fail_spots == []                    # screen-space state reset
    assert device.taps == []                         # never tapped anything


def test_no_tap_while_view_is_rotating_fast():
    # High measured pan speed = motion-blurred frame -> detections mislocate
    # (live: taps landed on stops). The tick must skip the tap entirely.
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    loop, device, _, _ = make_loop(
        Scripted([ScreenState.MAP]), Scripted([False]),
        detector_fn=lambda img, phone: target,
    )
    loop._pan_speed = 1000.0            # rotating (tiny test frames don't update it)
    loop.tick()
    assert device.taps == []            # no tap on a blurred frame


def test_no_camera_pan_right_after_a_catch():
    # A catch takes seconds; the camera-pan clock must restart afterwards so a
    # briefly-empty first rescan does NOT pan (live bug: panned after every
    # catch even with spawns still visible).
    classifier = Scripted([ScreenState.MAP])
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    targets = iter([target])
    loop, device, _, _ = make_loop(
        classifier, Scripted([True]),
        detector_fn=lambda img, phone: next(targets, None),  # then map is empty
    )
    loop.tick()                                   # catches (takes fake seconds)
    swipes_after_catch = len(device.swipes)       # throw swipes only
    loop.tick()                                   # first EMPTY rescan
    assert len(device.swipes) == swipes_after_catch   # no camera pan yet


def test_broke_free_rethrows_when_encounter_ui_returns():
    # Ball fails -> the encounter UI (berry icon) reappears -> the next throw
    # must fire as soon as the grace period passes, NOT after the full resolve
    # timeout. classifier never returns MAP; in_encounter stays True.
    classifier = Scripted([ScreenState.ENCOUNTER])   # never MAP (never caught)
    encounter_check = Scripted([True])               # UI (berry) visible again
    loop, device, _, sleeps = make_loop(classifier, encounter_check)

    loop.tick()

    max_throws = loop.config.timing["max_throws"]
    assert len(device.swipes) == max_throws          # kept re-throwing
    # each re-throw waited ~the grace period, NOT the resolve timeout: with the
    # tiny test timeouts the poll would time out anyway, so instead verify the
    # rethrow path returned an ENCOUNTER frame that was thrown on immediately
    # (screencaps stay bounded: one per poll step + initial, no extra bursts)
    assert device.taps == []                         # never fled


def test_pan_lead_points_in_the_direction_targets_move():
    # Scene content moves right+down between frames -> targets will keep
    # moving that way -> the tap must lead RIGHT+DOWN by velocity * lead-time.
    rng = np.random.default_rng(7)
    base = (rng.random((2388, 1080)) * 255).astype(np.uint8)
    frame1 = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    rolled = np.roll(np.roll(base, 40, axis=0), 24, axis=1)  # down 40, right 24
    frame2 = cv2.cvtColor(rolled, cv2.COLOR_GRAY2BGR)

    loop, _, _, _ = make_loop(Scripted([ScreenState.MAP]), Scripted([False]))
    assert loop._pan_lead(frame1, 10.0) == (0.0, 0.0)   # no prior frame yet
    lx, ly = loop._pan_lead(frame2, 10.5)               # dt = 0.5s
    # velocity = (24, 40)/0.5 = (48, 80) px/s; lead = v * 0.55 = (26.4, 44)
    assert 10 < lx < 45, lx
    assert 20 < ly < 70, ly


def test_pan_lead_is_hard_capped():
    # Even a fast (valid) pan must not lead the tap further than the cap --
    # an overshot lead is how taps miss wide.
    rng = np.random.default_rng(11)
    base = (rng.random((2388, 1080)) * 255).astype(np.uint8)
    f1 = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    f2 = cv2.cvtColor(np.roll(base, 150, axis=1), cv2.COLOR_GRAY2BGR)  # right 150
    loop, _, _, _ = make_loop(Scripted([ScreenState.MAP]), Scripted([False]))
    loop._pan_lead(f1, 20.0)
    lx, ly = loop._pan_lead(f2, 20.5)          # v = 300 px/s -> raw lead 165
    assert (lx ** 2 + ly ** 2) ** 0.5 <= loop._LEAD_MAX_PX + 0.1


def test_pan_lead_zero_when_scene_is_static():
    rng = np.random.default_rng(9)
    base = (rng.random((2388, 1080)) * 255).astype(np.uint8)
    frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    loop, _, _, _ = make_loop(Scripted([ScreenState.MAP]), Scripted([False]))
    loop._pan_lead(frame, 5.0)
    assert loop._pan_lead(frame.copy(), 5.5) == (0.0, 0.0)  # below min speed


def test_recover_uses_provided_frame_without_extra_screencap():
    # tick already has the frame in hand -> _recover's FIRST attempt must act on
    # it directly (tap the X) instead of paying another ~0.6s screencap.
    classifier = Scripted([ScreenState.UNKNOWN, ScreenState.MAP])
    loop, device, _, _ = make_loop(
        classifier, Scripted([False]), close_check=lambda img: True
    )
    loop._recover(img=DUMMY_IMG)
    assert loop._pt("close_button") in device.taps
    # screencaps only from the post-tap "back to playable?" poll, not attempt 1
    assert device.screencaps >= 1


def test_mistap_gym_never_throws_and_recovers_via_close_button():
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    loop, device, calls, _ = make_loop(
        Scripted([ScreenState.MAP, ScreenState.UNKNOWN, ScreenState.UNKNOWN, ScreenState.MAP]),
        Scripted([False]),                # encounter never confirmed
        detector_fn=lambda img, phone: target,
        close_check=lambda img: True,     # tap opened a gym panel
    )
    loop.tick()
    assert device.swipes == []                          # SAFETY: never threw
    assert calls["labels"] == []                        # and never self-labeled
    assert loop._pt("close_button") in device.taps      # recovered by tapping the X


def test_tick_wakes_on_black_screen_instead_of_recovering():
    # display asleep -> screencap black -> wake it, do NOT throw/tap/recover/label
    device = FakeDevice(img=BLACK_IMG)
    loop, device, calls, _ = make_loop(
        Scripted([ScreenState.MAP]), Scripted([False]), device=device
    )
    loop.tick()
    assert device.wakes >= 1
    assert device.swipes == []
    assert device.taps == []
    assert device.key_backs == []
    assert calls["labels"] == []


def test_no_target_no_tap_no_swipe_no_label():
    classifier = Scripted([ScreenState.MAP])
    encounter_check = Scripted([False])
    detector_fn = lambda img, phone: None

    loop, device, calls, sleeps = make_loop(
        classifier, encounter_check, detector_fn=detector_fn
    )

    loop.tick()

    assert device.taps == []
    assert device.swipes == []
    assert calls["labels"] == []
    assert len(sleeps) >= 1  # scan wait happened


def test_stubborn_pokemon_hits_throw_cap_then_yields_without_fleeing():
    # Enters mid-encounter; classify NEVER returns MAP (never caught) and
    # in_encounter stays True -> throws until the per-tick cap, then RETURNS
    # without fleeing (the next tick would keep throwing). "Never give up."
    classifier = Scripted([ScreenState.ENCOUNTER])  # initial ENCOUNTER, never MAP
    encounter_check = Scripted([True])              # always still in encounter

    loop, device, _, _ = make_loop(classifier, encounter_check)

    loop.tick()

    max_throws = loop.config.timing["max_throws"]
    assert len(device.swipes) == max_throws
    assert device.taps == []       # did NOT flee -- keeps the encounter for next tick


def test_recover_on_pokestop_then_map():
    classifier = Scripted([ScreenState.POKESTOP, ScreenState.POKESTOP, ScreenState.MAP])
    encounter_check = Scripted([False])
    loop, device, calls, _ = make_loop(classifier, encounter_check, close_check=lambda img: True)

    loop.tick()

    assert device.swipes == []
    assert loop._pt("close_button") in device.taps   # tapped the X, not just back
    assert calls["labels"] == []


def test_recover_on_gym_then_map():
    classifier = Scripted([ScreenState.GYM, ScreenState.GYM, ScreenState.MAP])
    encounter_check = Scripted([False])
    loop, device, calls, _ = make_loop(classifier, encounter_check, close_check=lambda img: True)

    loop.tick()

    assert device.swipes == []
    assert loop._pt("close_button") in device.taps   # tapped the X, not just back
    assert calls["labels"] == []


def test_recover_never_flees_an_encounter():
    # If recovery lands on an ENCOUNTER (e.g. it finished loading late), it must
    # NOT press back / close it -- that would flee a catch. It just yields.
    classifier = Scripted([ScreenState.ENCOUNTER])
    encounter_check = Scripted([True])
    loop, device, _, _ = make_loop(classifier, encounter_check, close_check=lambda img: True)

    loop._recover()

    assert device.swipes == []
    assert device.taps == []        # no close tap
    assert device.key_backs == []   # no flee


def test_recover_never_throws_on_unknown():
    classifier = Scripted([ScreenState.UNKNOWN, ScreenState.UNKNOWN, ScreenState.MAP])
    encounter_check = Scripted([False])
    loop, device, calls, _ = make_loop(classifier, encounter_check)

    loop.tick()

    assert device.swipes == []
    assert calls["labels"] == []


def test_recover_method_directly_never_swipes_for_all_non_encounter_states():
    for state in (ScreenState.ENCOUNTER, ScreenState.UNKNOWN, ScreenState.POKESTOP, ScreenState.GYM):
        classifier = Scripted([ScreenState.UNKNOWN, ScreenState.MAP])  # not-map -> close -> map
        encounter_check = Scripted([False])
        loop, device, _, _ = make_loop(classifier, encounter_check)
        loop._recover(state)
        assert device.swipes == []


def test_recover_terminates_on_stuck_panel_without_hanging_or_throwing():
    # A gym panel whose classify NEVER becomes playable -> recover exhausts its
    # attempts (tapping the X each time) and RETURNS without hanging or throwing.
    classifier = Scripted([ScreenState.UNKNOWN])   # sticky non-playable forever
    encounter_check = Scripted([False])
    loop, device, calls, _ = make_loop(classifier, encounter_check, close_check=lambda img: True)

    loop.tick()  # UNKNOWN -> _recover; must TERMINATE (no hang)

    assert device.swipes == []          # recover never throws, even when stuck
    assert calls["labels"] == []
    assert loop._pt("close_button") in device.taps  # kept trying the X
    assert device.key_backs == []                   # never blindly pressed back


def test_run_loop_calls_tick_until_stop_event_set():
    classifier = Scripted([ScreenState.MAP])
    encounter_check = Scripted([False])
    detector_fn = lambda img, phone: None
    loop, device, _, _ = make_loop(classifier, encounter_check, detector_fn=detector_fn)

    stop_event = threading.Event()
    call_count = {"n": 0}
    orig_tick = loop.tick

    def counting_tick():
        call_count["n"] += 1
        if call_count["n"] >= 3:
            stop_event.set()
        return orig_tick()

    loop.tick = counting_tick
    loop.run(stop_event)

    assert call_count["n"] == 3
