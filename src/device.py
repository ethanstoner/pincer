import subprocess
import time

import cv2
import numpy as np


class Device:
    def __init__(self, serial, adb_path):
        self.serial = serial
        self.adb_path = adb_path

    def _run(self, *args):
        subprocess.run([self.adb_path, "-s", self.serial, *args])

    def screencap(self):
        """Grab a screenshot as a BGR ndarray.

        adb's `exec-out screencap -p` occasionally returns a truncated PNG
        ("libpng error: PNG input buffer is incomplete") and cv2.imdecode then
        returns None, which used to crash the catch loop and waste a whole tick.
        Retry up to 3 more times with a short sleep before giving up. Returns
        the decoded image, or None if every attempt fails.
        """
        for attempt in range(4):
            proc = subprocess.run(
                [self.adb_path, "-s", self.serial, "exec-out", "screencap", "-p"],
                capture_output=True,
            )
            img = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
            if attempt < 3:
                time.sleep(0.03)
        return None

    def tap(self, x, y):
        self._run("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1, y1, x2, y2, ms):
        self._run("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(ms))

    def key_back(self):
        self._run("shell", "input", "keyevent", "4")
