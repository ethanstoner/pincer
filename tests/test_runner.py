import os
import threading

from unittest.mock import MagicMock, patch

import numpy as np

from src.catch_loop import CatchLoop
from src.config import load_config
from src.runner import DryRunDevice, main, _build_loops


SMALL_IMG = np.zeros((100, 100, 3), np.uint8)

TWO_PHONE_CONFIG = os.path.join(
    os.path.dirname(__file__), "fixtures", "config_2phones.json"
)


def _make_device_mock():
    device = MagicMock()
    device.screencap.return_value = SMALL_IMG
    return device


def test_once_dry_run_builds_one_loop_per_phone_and_ticks():
    n_phones = len(load_config("config.json").phones)
    with patch("src.runner.Device") as MockDevice, patch("src.runner.CatchLoop") as MockCatchLoop:
        MockDevice.side_effect = lambda serial, adb_path: _make_device_mock()
        loop_instance = MagicMock()
        MockCatchLoop.return_value = loop_instance

        main(["--config", "config.json", "--dry-run", "--once"])

        # one Device + one CatchLoop per configured phone (dry-run skips the
        # adb-connected filter, so every configured phone is built)
        assert MockDevice.call_count == n_phones
        assert MockCatchLoop.call_count == n_phones
        assert loop_instance.tick.call_count == n_phones


def test_phone_filter_selects_only_matching_serial():
    with patch("src.runner.Device") as MockDevice, patch("src.runner.CatchLoop") as MockCatchLoop:
        main(["--config", "config.json", "--phone", "NONEXISTENT", "--once", "--dry-run"])

        assert MockDevice.call_count == 0
        assert MockCatchLoop.call_count == 0


def test_dry_run_device_does_not_forward_actions():
    real_device = MagicMock()
    real_device.serial = "SERIAL"
    real_device.screencap.return_value = SMALL_IMG

    dry = DryRunDevice(real_device)

    dry.tap(1, 2)
    dry.swipe(1, 2, 3, 4, 150)
    dry.key_back()

    real_device.tap.assert_not_called()
    real_device.swipe.assert_not_called()
    real_device.key_back.assert_not_called()

    result = dry.screencap()
    real_device.screencap.assert_called_once()
    assert result is SMALL_IMG


def test_threaded_fanout_starts_one_loop_per_phone_and_runs_each_once():
    with patch("src.runner.Device") as MockDevice, patch("src.runner.CatchLoop") as MockCatchLoop:
        MockDevice.side_effect = lambda serial, adb_path: _make_device_mock()
        loop_mocks = [MagicMock(), MagicMock()]
        MockCatchLoop.side_effect = loop_mocks

        # NOT --once -> takes the threaded path. run() is mocked to return
        # immediately, so the worker threads finish and main() returns.
        main(["--config", TWO_PHONE_CONFIG, "--dry-run"])

        assert MockDevice.call_count == 2
        assert MockCatchLoop.call_count == 2
        for lp in loop_mocks:
            lp.run.assert_called_once()
            # run() receives the shared stop_event as its sole positional arg
            (arg,) = lp.run.call_args.args
            assert isinstance(arg, threading.Event)


def test_build_loops_fans_out_for_two_phones():
    with patch("src.runner.Device") as MockDevice:
        MockDevice.side_effect = lambda serial, adb_path: _make_device_mock()
        config = load_config(TWO_PHONE_CONFIG)

        loops = _build_loops(config, config.phones, dry_run=True)

        assert len(loops) == 2
        assert MockDevice.call_count == 2
        serials = {lp.phone.serial for lp in loops}
        assert serials == {"PHONE_A", "PHONE_B"}
        # dry_run=True -> each loop's device wraps the real device, never forwards
        for lp in loops:
            assert isinstance(lp.device, DryRunDevice)


def test_run_survives_tick_exception_and_exits_on_stop():
    config = load_config(TWO_PHONE_CONFIG)
    phone = config.phones[0]
    device = MagicMock()
    device.screencap.return_value = SMALL_IMG

    sleeps = []
    loop = CatchLoop(
        device=device,
        config=config,
        phone=phone,
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    stop_event = threading.Event()
    calls = {"n": 0}

    def bad_tick():
        calls["n"] += 1
        if calls["n"] >= 2:
            stop_event.set()  # let the loop exit after the 2nd raising tick
        raise RuntimeError("transient adb hiccup")

    loop.tick = bad_tick

    # Must NOT propagate the exception; must exit cleanly once stop_event is set.
    loop.run(stop_event)

    assert calls["n"] == 2  # first raise was swallowed, loop continued to 2nd
