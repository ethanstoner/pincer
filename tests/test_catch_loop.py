import random
import threading

import numpy as np

from src.catch_loop import CatchLoop
from src.config import Config, Phone
from src.detector import Target
from src.screen_state import ScreenState


DUMMY_IMG = np.zeros((4, 4, 3), np.uint8)


class FakeDevice:
    def __init__(self):
        self.taps = []
        self.swipes = []
        self.key_backs = []
        self.screencaps = 0

    def screencap(self):
        self.screencaps += 1
        return DUMMY_IMG

    def tap(self, x, y):
        self.taps.append((x, y))

    def swipe(self, x1, y1, x2, y2, ms):
        self.swipes.append((x1, y1, x2, y2, ms))

    def key_back(self):
        self.key_backs.append(True)


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


def make_loop(classifier, encounter_check, device=None, detector_fn=None, labeler=None, close_check=None):
    config = make_config()
    phone = config.phones[0]
    device = device or FakeDevice()
    detector_fn = detector_fn or (lambda img, phone: None)
    close_check = close_check or (lambda img: False)
    calls = {"labels": []}

    def default_labeler(img, target, dataset_dir):
        calls["labels"].append((img, target, dataset_dir))
        return "labeled.png"

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
        sleep_fn=sleep_fn,
        rng=random.Random(42),
        encounter_check=encounter_check,
        close_check=close_check,
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
    # encounter poll times out -> _recover taps the close X. SAFETY: no throw.
    classifier = Scripted(
        [ScreenState.MAP, ScreenState.UNKNOWN, ScreenState.UNKNOWN, ScreenState.MAP]
    )
    encounter_check = Scripted([False])        # encounter NEVER confirmed
    target = Target(x=500, y=1200, bbox=(480, 1180, 40, 40))
    detector_fn = lambda img, phone: target

    loop, device, calls, _ = make_loop(
        classifier, encounter_check, detector_fn=detector_fn
    )

    loop.tick()

    assert device.swipes == []       # NO throw
    assert calls["labels"] == []     # and NO self-label
    assert loop._pt("close_button") in device.taps  # recovered by tapping the X


def test_await_encounter_returns_frame_on_first_confirmed_frame():
    loop, device, _, _ = make_loop(
        Scripted([ScreenState.UNKNOWN]), Scripted([True]), close_check=lambda img: False
    )
    assert loop._await_encounter(100000) is not None
    assert device.screencaps == 1        # proceeded instantly, did not wait the timeout


def test_await_encounter_bails_on_first_frame_when_close_button_present():
    # a gym / PokeStop / menu opened (its X is visible) -> bail at once, no waiting
    loop, device, _, _ = make_loop(
        Scripted([ScreenState.UNKNOWN]), Scripted([False]), close_check=lambda img: True
    )
    assert loop._await_encounter(100000) is None
    assert device.screencaps == 1        # bailed on the first frame


def test_await_encounter_bails_when_still_map_after_transition_window():
    # tap opened nothing: still MAP. Bail once past the transition window, NOT at
    # the (huge) timeout.
    loop, device, _, _ = make_loop(
        Scripted([ScreenState.MAP]), Scripted([False]), close_check=lambda img: False
    )
    assert loop._await_encounter(100000) is None
    assert device.screencaps < 30        # bailed ~_MAP_BAIL_MS in, not after 100s


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
    loop, device, calls, _ = make_loop(classifier, encounter_check)

    loop.tick()

    assert device.swipes == []
    assert loop._pt("close_button") in device.taps   # tapped the X, not just back
    assert calls["labels"] == []


def test_recover_on_gym_then_map():
    classifier = Scripted([ScreenState.GYM, ScreenState.GYM, ScreenState.MAP])
    encounter_check = Scripted([False])
    loop, device, calls, _ = make_loop(classifier, encounter_check)

    loop.tick()

    assert device.swipes == []
    assert loop._pt("close_button") in device.taps   # tapped the X, not just back
    assert calls["labels"] == []


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


def test_recover_terminates_on_stuck_timeout_without_hanging_or_throwing():
    # classifier NEVER returns MAP -> recover exhausts its attempts (tap close +
    # key_back fallback each time) and RETURNS without hanging or throwing.
    classifier = Scripted([ScreenState.UNKNOWN])   # sticky UNKNOWN forever
    encounter_check = Scripted([False])
    loop, device, calls, _ = make_loop(classifier, encounter_check)

    loop.tick()  # UNKNOWN -> _recover; must TERMINATE (no hang)

    assert device.swipes == []          # recover never throws, even on timeout
    assert calls["labels"] == []
    assert loop._pt("close_button") in device.taps  # tried the X
    assert len(device.key_backs) >= 1               # and the back fallback


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
