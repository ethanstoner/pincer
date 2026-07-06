"""Multi-phone entry point.

Builds one CatchLoop per configured phone and either ticks each once
(`--once`, for inspection) or runs each loop forever on its own thread
until a Ctrl-C / SIGINT triggers a clean, joined shutdown.

`--dry-run` wraps each real Device in a DryRunDevice so screenshots are
still captured (and therefore still classified/detected) but taps,
swipes, and back-presses are only logged, never sent to the phone.
"""

import argparse
import signal
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


def _build_loops(config, phones, dry_run):
    loops = []
    for phone in phones:
        device = Device(phone.serial, config.adb_path)
        if dry_run:
            device = DryRunDevice(device)
        loops.append(CatchLoop(device, config, phone))
    return loops


def main(argv):
    args = parse_args(argv)
    config = load_config(args.config)
    phones = _select_phones(config, args.phone)
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
