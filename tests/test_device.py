import numpy as np, cv2
from unittest.mock import patch, MagicMock
from src.device import Device

def test_tap_builds_correct_adb_command():
    dev = Device("SERIAL", "adb.exe")
    with patch("src.device.subprocess.run") as run:
        dev.tap(540, 2124)
        args = run.call_args[0][0]
        assert args == ["adb.exe","-s","SERIAL","shell","input","tap","540","2124"]

def test_swipe_builds_correct_adb_command():
    dev = Device("SERIAL", "adb.exe")
    with patch("src.device.subprocess.run") as run:
        dev.swipe(540,2000,540,850,150)
        args = run.call_args[0][0]
        assert args == ["adb.exe","-s","SERIAL","shell","input","swipe","540","2000","540","850","150"]

def test_key_back_builds_correct_adb_command():
    dev = Device("SERIAL", "adb.exe")
    with patch("src.device.subprocess.run") as run:
        dev.key_back()
        args = run.call_args[0][0]
        assert args == ["adb.exe","-s","SERIAL","shell","input","keyevent","4"]

def test_screencap_decodes_png_bytes():
    png = cv2.imencode(".png", np.zeros((4,4,3), np.uint8))[1].tobytes()
    dev = Device("SERIAL", "adb.exe")
    with patch("src.device.subprocess.run") as run:
        run.return_value = MagicMock(stdout=png)
        img = dev.screencap()
        assert img.shape == (4,4,3)

def test_screencap_retries_on_truncated_png():
    # Live "libpng error: PNG input buffer is incomplete" -> imdecode None.
    # First read returns truncated bytes (decodes to None), second read is a
    # valid PNG. screencap must retry and return the valid image.
    truncated = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8   # header only -> imdecode None
    valid = cv2.imencode(".png", np.zeros((4,4,3), np.uint8))[1].tobytes()
    dev = Device("SERIAL", "adb.exe")
    with patch("src.device.subprocess.run") as run, patch("src.device.time.sleep"):
        run.side_effect = [MagicMock(stdout=truncated), MagicMock(stdout=valid)]
        img = dev.screencap()
        assert img is not None
        assert img.shape == (4,4,3)
        assert run.call_count == 2   # proves it retried once

def test_screencap_returns_none_if_all_retries_fail():
    truncated = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    dev = Device("SERIAL", "adb.exe")
    with patch("src.device.subprocess.run") as run, patch("src.device.time.sleep"):
        run.return_value = MagicMock(stdout=truncated)
        assert dev.screencap() is None
