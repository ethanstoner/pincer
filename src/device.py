import os
import struct
import subprocess
import tempfile
import time

import cv2
import numpy as np


class Device:
    # adb calls occasionally HANG (flaky WiFi link, busy USB). We never capture a
    # binary stdout pipe: `exec-out screencap` streams through a pipe the adb
    # server can hold open, so on Windows a stuck call hangs forever even WITH a
    # timeout. Instead every call sends stdout/stderr to DEVNULL and is bounded by
    # a timeout, so a hung call is actually killed. We also capture RAW (not PNG):
    # PNG-encoding a heavy animated scene on the phone takes ~1.9s, while raw is
    # ~0.25s + a fast pull -- 3x faster end to end.
    _SCREENCAP_TIMEOUT_S = 4.0
    _CMD_TIMEOUT_S = 4.0
    _REMOTE_CAP = "/sdcard/_pogo_cap.raw"

    def __init__(self, serial, adb_path):
        self.serial = serial
        self.adb_path = adb_path
        self._local_cap = os.path.join(tempfile.gettempdir(), f"_pogo_cap_{serial}.raw")

    def _adb(self, *args, timeout):
        """Run an adb command with no captured pipes so a hung call can be killed
        by the timeout. Returns True if it completed, False if it timed out."""
        try:
            subprocess.run(
                [self.adb_path, "-s", self.serial, *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            return True
        except subprocess.TimeoutExpired:
            return False

    def _run(self, *args):
        self._adb(*args, timeout=self._CMD_TIMEOUT_S)

    def _decode_raw(self):
        """Decode the pulled `screencap` raw buffer into a BGR ndarray.

        Layout: header of uint32 little-endian [width, height, format, (colorspace)]
        then width*height*4 bytes of RGBA_8888. The header is 12 or 16 bytes
        depending on Android version, so we derive its size from the file length
        (self-describing -- no need to know the resolution in advance). Returns
        None on any short/corrupt read (e.g. a WiFi-truncated pull) so the caller
        retries.
        """
        try:
            with open(self._local_cap, "rb") as fh:
                data = fh.read()
            if len(data) < 12:
                return None
            w, h = struct.unpack("<II", data[:8])
            header = len(data) - w * h * 4
            if w <= 0 or h <= 0 or header not in (12, 16):
                return None
            arr = np.frombuffer(data, np.uint8, count=w * h * 4, offset=header)
            arr = arr.reshape(h, w, 4)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        except (OSError, ValueError, struct.error):
            return None

    def screencap(self):
        """Grab a screenshot as a BGR ndarray via on-device raw capture + pull.

        Screencap to a raw file on the device, then `pull` it (no captured binary
        pipe), which lets the timeout actually kill a hung call. Retry a few times
        to ride out truncated pulls on a flaky link. Returns the image, or None.
        """
        for attempt in range(3):
            ok = self._adb(
                "shell", "screencap", self._REMOTE_CAP,
                timeout=self._SCREENCAP_TIMEOUT_S,
            )
            if ok:
                ok = self._adb(
                    "pull", self._REMOTE_CAP, self._local_cap,
                    timeout=self._SCREENCAP_TIMEOUT_S,
                )
            img = self._decode_raw() if ok else None
            if img is not None:
                return img
            if attempt < 2:
                time.sleep(0.03)
        return None

    def tap(self, x, y):
        self._run("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1, y1, x2, y2, ms):
        self._run("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(ms))

    def key_back(self):
        self._run("shell", "input", "keyevent", "4")
