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

def test_decode_raw_parses_rgba_with_16byte_header():
    import struct
    w, h = 2, 2
    header = struct.pack("<IIII", w, h, 1, 1)            # 16-byte header (Android 12+)
    pixels = bytes([10, 20, 30, 255] * (w * h))          # RGBA
    dev = Device("SERIAL", "adb.exe")
    with open(dev._local_cap, "wb") as fh:
        fh.write(header + pixels)
    img = dev._decode_raw()
    assert img.shape == (h, w, 3)
    assert list(img[0, 0]) == [30, 20, 10]               # RGBA(10,20,30) -> BGR(30,20,10)

def test_decode_raw_returns_none_on_truncated_buffer():
    import struct
    dev = Device("SERIAL", "adb.exe")
    with open(dev._local_cap, "wb") as fh:
        fh.write(struct.pack("<IIII", 100, 100, 1, 1) + b"\x00" * 16)  # far too short
    assert dev._decode_raw() is None

def test_screencap_returns_image_when_capture_succeeds():
    dev = Device("SERIAL", "adb.exe")
    valid = np.zeros((4, 4, 3), np.uint8)
    with patch("src.device.subprocess.run"), \
         patch.object(Device, "_decode_raw", return_value=valid):
        assert dev.screencap().shape == (4, 4, 3)

def test_screencap_retries_on_failed_decode():
    # A truncated pull decodes to None; screencap must retry and return the next
    # valid image.
    dev = Device("SERIAL", "adb.exe")
    valid = np.zeros((4, 4, 3), np.uint8)
    with patch("src.device.subprocess.run"), \
         patch.object(Device, "_decode_raw", side_effect=[None, valid]) as dec, \
         patch("src.device.time.sleep"):
        img = dev.screencap()
        assert img is not None
        assert dec.call_count == 2   # proves it retried once

def test_screencap_returns_none_if_all_retries_fail():
    dev = Device("SERIAL", "adb.exe")
    with patch("src.device.subprocess.run"), \
         patch.object(Device, "_decode_raw", return_value=None), \
         patch("src.device.time.sleep"):
        assert dev.screencap() is None

def test_screencap_survives_hung_adb_via_timeout():
    # A hung adb call raises TimeoutExpired; screencap must swallow it and return
    # None instead of propagating (this is the freeze-the-whole-loop bug).
    import subprocess as sp
    dev = Device("SERIAL", "adb.exe")
    with patch("src.device.subprocess.run", side_effect=sp.TimeoutExpired("adb", 3)), \
         patch("src.device.time.sleep"):
        assert dev.screencap() is None   # returns None, does NOT raise
