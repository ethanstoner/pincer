import subprocess

import cv2
import numpy as np


class Device:
    def __init__(self, serial, adb_path):
        self.serial = serial
        self.adb_path = adb_path

    def _run(self, *args):
        subprocess.run([self.adb_path, "-s", self.serial, *args])

    def screencap(self) -> np.ndarray:
        proc = subprocess.run(
            [self.adb_path, "-s", self.serial, "exec-out", "screencap", "-p"],
            capture_output=True,
        )
        return cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)

    def tap(self, x, y):
        self._run("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1, y1, x2, y2, ms):
        self._run("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(ms))

    def key_back(self):
        self._run("shell", "input", "keyevent", "4")
