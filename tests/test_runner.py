from unittest.mock import MagicMock, patch

import numpy as np

from src.runner import DryRunDevice, main


SMALL_IMG = np.zeros((100, 100, 3), np.uint8)


def _make_device_mock():
    device = MagicMock()
    device.screencap.return_value = SMALL_IMG
    return device


def test_once_dry_run_builds_one_loop_per_phone_and_ticks():
    with patch("src.runner.Device") as MockDevice, patch("src.runner.CatchLoop") as MockCatchLoop:
        MockDevice.side_effect = lambda serial, adb_path: _make_device_mock()
        loop_instance = MagicMock()
        MockCatchLoop.return_value = loop_instance

        main(["--config", "config.json", "--dry-run", "--once"])

        # config.json currently defines exactly one phone (DEVICE_SERIAL_B)
        assert MockDevice.call_count == 1
        assert MockCatchLoop.call_count == 1
        loop_instance.tick.assert_called_once()


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
