import random
import threading

import numpy as np
import pytest

from src.catch_loop import CatchLoop
from src.config import Config, Phone
from src.detector import Target
from src.screen_state import ScreenState


DUMMY_IMG = np.zeros((4, 4, 3), np.uint8)


class FakeDevice:
    def __init__(self, screencaps=None):
        self.taps = []
        self.swipes = []
        self.key_backs = []
        self._screencaps = list(screencaps) if screencaps else None

    def screencap(self):
        if self._screencaps is not None and self._screencaps:
            return self._screencaps.pop(0)
        return DUMMY_IMG

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, x1, y1, x2, y2, ms):
        self.swipes.append((x1, y1, x2, y2, ms))

    def key_back(self):
        self.key_backs.append(True)


class ScriptedClassifier:
    """Returns states from a preset list in order; repeats the last state
    forever once exhausted (handy for "many ENCOUNTER calls" tests)."""

    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        if self.states:
            if len(self.states) == 1:
                return self.states[0]
            return self.states.pop(0)
        return ScreenState.UNKNOWN


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
        },
        timing={
            "encounter_load_ms": [1, 2],
            "post_throw_ms": [1, 2],
            "map_scan_ms": [1, 2],
            "stuck_timeout_ms": 15,
            "max_throws": 3,
        },
    )


def make_loop(classifier, device=None, detector_fn=None, labeler=None, rng=None):
    config = make_config()
    phone = config.phones[0]
    device = device or FakeDevice()
    detector_fn = detector_fn or (lambda img, phone: None)
    calls = {"labels": []}

    def default_labeler(img, target, dataset_dir):
        calls["labels"].append((img, target, dataset_dir))
        return "labeled.png"

    labeler = labeler or default_labeler
    rng = rng or random.Random(42)
    sleeps = []

    def sleep_fn(seconds):
        sleeps.append(seconds)

    loop = CatchLoop(
        device=device,
        config=config,
        phone=phone,
        classifier=classifier,
        detector_fn=detector_fn,
        labeler=labeler,
        sleep_fn=sleep_fn,
        rng=rng,
    )
    return loop, device, calls, sleeps


def test_caught_path_taps_once_labels_once_throws_and_no_flee_or_recover():
    classifier = ScriptedClassifier(
        [ScreenState.MAP, ScreenState.ENCOUNTER, ScreenState.ENCOUNTER, ScreenState.MAP]
    )
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    detector_fn = lambda img, phone: target
    label_calls = []

    def labeler(img, tgt, dataset_dir):
        label_calls.append((img, tgt, dataset_dir))
        return "path.png"

    loop, device, _, _ = make_loop(classifier, detector_fn=detector_fn, labeler=labeler)

    loop.tick()

    # exactly one detect tap near the target (jitter allowed within a few px)
    assert len(device.taps) == 1
    tx, ty = device.taps[0]
    assert abs(tx - target.x) <= 10
    assert abs(ty - target.y) <= 10

    assert len(label_calls) == 1
    assert label_calls[0][1] is target

    # at least one throw swipe recorded
    assert len(device.swipes) >= 1

    # ends without flee tap and no key_back
    flee_x, flee_y = 52, 145  # resolve_point(flee_button, phone) approx
    assert device.key_backs == []


def test_wrong_target_never_throws_and_recovers():
    # tap did not open an encounter -> classifier stays MAP
    classifier = ScriptedClassifier([ScreenState.MAP, ScreenState.MAP])
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    detector_fn = lambda img, phone: target

    loop, device, calls_holder, _ = make_loop(classifier, detector_fn=detector_fn)

    loop.tick()

    # detect tap happened
    assert len(device.taps) == 1

    # NO throw swipe recorded
    assert device.swipes == []

    # labeler NOT called
    assert calls_holder["labels"] == []

    # a recover action (key_back) occurred since state is MAP/UNKNOWN-ish;
    # per spec MAP after failed tap goes through _recover(), and since state
    # is neither ENCOUNTER nor POKESTOP/GYM, _recover treats it as the
    # "not ENCOUNTER, not POKESTOP/GYM" -> UNKNOWN-style key_back path.
    assert len(device.key_backs) >= 1


def test_no_target_no_tap_no_swipe_no_label():
    classifier = ScriptedClassifier([ScreenState.MAP])
    detector_fn = lambda img, phone: None

    loop, device, calls_holder, sleeps = make_loop(classifier, detector_fn=detector_fn)

    loop.tick()

    assert device.taps == []
    assert device.swipes == []
    assert calls_holder["labels"] == []
    assert len(sleeps) >= 1  # scan wait happened


def test_stubborn_pokemon_hits_throw_cap_then_flees():
    # Enters via ENCOUNTER directly (mid-encounter), stays ENCOUNTER forever
    classifier = ScriptedClassifier([ScreenState.ENCOUNTER])

    loop, device, _, _ = make_loop(classifier)

    loop.tick()

    max_throws = loop.config.timing["max_throws"]
    assert len(device.swipes) == max_throws

    flee_point = loop._pt("flee_button")
    assert device.taps == [flee_point]


def test_recover_on_pokestop_then_map():
    classifier = ScriptedClassifier([ScreenState.POKESTOP, ScreenState.MAP])
    loop, device, calls_holder, _ = make_loop(classifier)

    loop.tick()

    assert device.swipes == []
    assert len(device.key_backs) >= 1
    assert calls_holder["labels"] == []


def test_recover_on_gym_then_map():
    classifier = ScriptedClassifier([ScreenState.GYM, ScreenState.MAP])
    loop, device, calls_holder, _ = make_loop(classifier)

    loop.tick()

    assert device.swipes == []
    assert len(device.key_backs) >= 1
    assert calls_holder["labels"] == []


def test_recover_never_throws_on_unknown():
    classifier = ScriptedClassifier([ScreenState.UNKNOWN, ScreenState.UNKNOWN, ScreenState.MAP])
    loop, device, calls_holder, _ = make_loop(classifier)

    loop.tick()

    assert device.swipes == []
    assert calls_holder["labels"] == []


def test_recover_method_directly_never_swipes_for_all_non_encounter_states():
    for state in (ScreenState.UNKNOWN, ScreenState.POKESTOP, ScreenState.GYM):
        classifier = ScriptedClassifier([ScreenState.MAP])
        loop, device, _, _ = make_loop(classifier)
        loop._recover(state)
        assert device.swipes == []


def test_run_loop_calls_tick_until_stop_event_set():
    classifier = ScriptedClassifier([ScreenState.MAP])
    detector_fn = lambda img, phone: None
    loop, device, _, sleeps = make_loop(classifier, detector_fn=detector_fn)

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
