import os
import struct
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np


class _StreamCapture:
    """Continuous frame stream: `adb exec-out screenrecord --output-format=h264`
    decoded by PyAV in a daemon thread. The newest BGR frame is always in
    memory, so screencap() costs ~0 instead of a ~600ms capture+pull round
    trip (measured live: ~27 fps at full 1080x2388). screenrecord hard-stops
    at 3 minutes, so the thread respawns it forever. Requires `av` (present in
    the training venv the bot runs under); the caller falls back to pull-based
    capture whenever no fresh frame is available (startup, respawn gap, or a
    dead stream)."""

    _MAX_FRAME_AGE_S = 2.0   # older than this -> treat the stream as stale
    _SEGMENT_S = 175         # respawn before screenrecord's 180s hard limit

    def __init__(self, serial, adb_path):
        self.serial = serial
        self.adb_path = adb_path
        self._lock = threading.Lock()
        self._frame = None
        self._frame_t = 0.0
        self._proc = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import av  # lazy: only needed when streaming is active

        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    [self.adb_path, "-s", self.serial, "exec-out", "screenrecord",
                     "--output-format=h264", "--bit-rate=16000000",
                     f"--time-limit={self._SEGMENT_S}", "-"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                container = av.open(self._proc.stdout, format="h264", mode="r")
                for frame in container.decode(video=0):
                    if self._stop.is_set():
                        break
                    img = frame.to_ndarray(format="bgr24")
                    with self._lock:
                        self._frame = img
                        self._frame_t = time.monotonic()
            except Exception:
                pass  # decode/link hiccup -> fall through to respawn
            finally:
                if self._proc is not None:
                    try:
                        self._proc.kill()
                    except OSError:
                        pass
            if not self._stop.is_set():
                time.sleep(0.5)  # brief backoff, then respawn the segment

    def latest(self):
        """Newest frame (BGR, copy) if fresh enough, else None."""
        with self._lock:
            if self._frame is None:
                return None
            if time.monotonic() - self._frame_t > self._MAX_FRAME_AGE_S:
                return None
            return self._frame.copy()

    def close(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass


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

    # Capture-age hint for the catch loop's tap-lead: a streamed frame is
    # ~0.2-0.3s old (encode+transfer) vs ~0.5s+ for capture+pull.
    TAP_LEAD_STREAM_S = 0.35
    TAP_LEAD_PULL_S = 0.55

    def __init__(self, serial, adb_path, stream=False):
        self.serial = serial
        self.adb_path = adb_path
        self._local_cap = os.path.join(tempfile.gettempdir(), f"_pogo_cap_{serial}.raw")
        self._stream = _StreamCapture(serial, adb_path) if stream else None
        self.tap_lead_s = self.TAP_LEAD_STREAM_S if stream else self.TAP_LEAD_PULL_S
        self._input_proc = None
        self._input_lock = threading.Lock()

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
        """Grab a screenshot as a BGR ndarray.

        With streaming enabled, the newest streamed frame is returned instantly
        (~0ms); the pull path below is the fallback for stream startup/hiccups.
        Pull path: screencap to a raw file on the device, then `pull` it (no
        captured binary pipe), which lets the timeout actually kill a hung
        call. Retries ride out truncated pulls on a flaky link.
        """
        if self._stream is not None:
            img = self._stream.latest()
            if img is not None:
                return img
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

    # --- input: persistent `adb shell` session -----------------------------
    # Spawning a fresh adb process per tap costs ~150-250ms; writing the input
    # command into one long-lived shell costs ~20ms. On any write failure the
    # session is respawned and the command falls back to a one-shot call.
    def _input(self, cmd):
        with self._input_lock:
            for _ in range(2):
                try:
                    if self._input_proc is None or self._input_proc.poll() is not None:
                        self._input_proc = subprocess.Popen(
                            [self.adb_path, "-s", self.serial, "shell"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            text=True,
                        )
                    self._input_proc.stdin.write(cmd + "\n")
                    self._input_proc.stdin.flush()
                    return
                except (OSError, ValueError, AttributeError):
                    try:
                        self._input_proc.kill()
                    except Exception:
                        pass
                    self._input_proc = None
        self._run("shell", *cmd.split())  # last resort: one-shot call

    def tap(self, x, y):
        self._input(f"input tap {x} {y}")

    def swipe(self, x1, y1, x2, y2, ms):
        self._input(f"input swipe {x1} {y1} {x2} {y2} {ms}")

    def key_back(self):
        self._input("input keyevent 4")

    def wake(self):
        """Turn the display back on (screencap returns black when it's asleep)."""
        self._input("input keyevent KEYCODE_WAKEUP")

    def set_stay_awake(self):
        """Keep the display on while powered (USB), so it never sleeps mid-run."""
        self._run("shell", "svc", "power", "stayon", "true")
