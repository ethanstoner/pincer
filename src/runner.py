"""Multi-phone entry point.

Builds one CatchLoop per configured phone and either ticks each once
(`--once`, for inspection) or runs each loop forever on its own thread
until a Ctrl-C / SIGINT triggers a clean, joined shutdown.

`--dry-run` wraps each real Device in a DryRunDevice so screenshots are
still captured (and therefore still classified/detected) but taps,
swipes, and back-presses are only logged, never sent to the phone.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading

from src.catch_loop import CatchLoop
from src.config import load_config
from src.device import Device


class DryRunDevice:
    """Wraps a real Device: screencap() delegates for real; every action
    that would touch the phone is logged instead of executed."""

    def __init__(self, real_device):
        self._real = real_device

    def screencap(self):
        return self._real.screencap()

    def tap(self, x, y):
        print(f"[dry-run {self._real.serial}] tap ({x}, {y})")

    def swipe(self, x1, y1, x2, y2, ms):
        print(f"[dry-run {self._real.serial}] swipe ({x1}, {y1}) -> ({x2}, {y2}) {ms}ms")

    def key_back(self):
        print(f"[dry-run {self._real.serial}] key_back")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="pogo-catcher multi-phone runner")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--phone", action="append", default=None)
    return parser.parse_args(argv)


def _select_phones(config, serials):
    if not serials:
        return list(config.phones)
    return [p for p in config.phones if p.serial in serials]


def _connected_serials(adb_path):
    """Serials currently in the 'device' state per `adb devices` (authorized &
    online). Empty on any adb error."""
    try:
        out = subprocess.run(
            [adb_path, "devices"], capture_output=True, text=True, timeout=10
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return set()
    connected = set()
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            connected.add(parts[0])
    return connected


def clear_click_audit(dataset_dir):
    """Start every run with an empty <dataset>/clicks/ tree so the folder holds
    exactly THIS run's taps for review (Ethan critiques each run's mis-clicks).
    Training data (images/ + labels/) is never touched -- it accumulates."""
    clicks = os.path.join(dataset_dir, "clicks")
    shutil.rmtree(clicks, ignore_errors=True)


def _make_detector_fn(config):
    """Return the detector callable (img, phone) -> Optional[Target]. Defaults to
    the classical CV detector; uses the trained YOLO model when config selects it.
    One model instance is shared across phones."""
    if config.detector == "yolo" and config.yolo_model_path:
        from src.detector_yolo import YoloDetector
        return YoloDetector(config.yolo_model_path).propose
    return None  # None -> CatchLoop uses its default CV propose


def _build_loops(config, phones, dry_run):
    detector_fn = _make_detector_fn(config)
    loops = []
    for phone in phones:
        device = Device(phone.serial, config.adb_path)
        if dry_run:
            device = DryRunDevice(device)
        else:
            device.set_stay_awake()  # keep the display on for the whole run
        kwargs = {"detector_fn": detector_fn} if detector_fn else {}
        loops.append(CatchLoop(device, config, phone, **kwargs))
    return loops


def main(argv):
    args = parse_args(argv)
    config = load_config(args.config)
    phones = _select_phones(config, args.phone)

    # On a real run, only spin up phones that are actually connected, so a config
    # listing both phones works whether one or both are plugged in.
    if not args.dry_run:
        connected = _connected_serials(config.adb_path)
        for p in phones:
            if p.serial not in connected:
                print(f"skip {p.serial}: not connected to adb")
        phones = [p for p in phones if p.serial in connected]
        if not phones:
            print("no configured phones are connected; nothing to run.")
            return
        print("running phones: " + ", ".join(p.serial for p in phones))

    clear_click_audit(config.dataset_dir)  # fresh per-run click review folder
    loops = _build_loops(config, phones, args.dry_run)

    if args.once:
        for loop in loops:
            loop.tick()
        return

    stop_event = threading.Event()

    def _handle_sigint(signum, frame):
        stop_event.set()

    previous_handler = signal.signal(signal.SIGINT, _handle_sigint)

    threads = [
        threading.Thread(target=loop.run, args=(stop_event,), daemon=True)
        for loop in loops
    ]
    for t in threads:
        t.start()

    try:
        while not stop_event.is_set() and any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        # Bounded-latency shutdown: stop_event is checked between ticks so most
        # threads exit immediately; give any mid-wait tick up to 5s, then move
        # on (daemon threads are reaped at process exit).
        for t in threads:
            t.join(timeout=5)
            if t.is_alive():
                print("[runner] a phone worker is still finishing; exiting anyway")
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    main(sys.argv[1:])
