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
