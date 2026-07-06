"""Per-phone catch loop state machine.

Safety contract (structural, not incidental):
  - A ball is only ever thrown from inside `_run_throw_loop`, which is only
    ever entered after `classifier(...)` has returned `ScreenState.ENCOUNTER`
    for a freshly captured screenshot (either because we just tapped a
    detected target and re-confirmed ENCOUNTER, or because the loop started
    mid-encounter).
  - `_recover` NEVER calls `_throw` / `device.swipe` under any branch. It only
    taps `flee_button` (a UI escape, not a throw) or presses back.
  - Any tap on a detected map target is verified by re-classifying the screen
    afterward; if it did not open an ENCOUNTER, control goes straight to
    `_recover` and no throw is attempted.
"""

import random
import time

from src.config import resolve_point
from src.detector import propose
from src.detector import save_label as _save_label
from src.screen_state import ScreenState
from src.screen_state import classify as _classify


class CatchLoop:
    _POLL_INTERVAL_MS = 500
    _TAP_JITTER_PX = 5
    _SWIPE_JITTER_PX = 5
    _THROW_DURATION_MS_RANGE = (120, 180)
    _ERROR_BACKOFF_MS = (1000, 1000)  # brief pause after a crashed tick

    def __init__(
        self,
        device,
        config,
        phone,
        classifier=_classify,
        detector_fn=propose,
        labeler=_save_label,
        sleep_fn=time.sleep,
        rng=None,
    ):
        self.device = device
        self.config = config
        self.phone = phone
        self.classifier = classifier
        self.detector_fn = detector_fn
        self.labeler = labeler
        self.sleep_fn = sleep_fn
        self.rng = rng if rng is not None else random.Random()

    def _sleep(self, ms_range):
        lo, hi = ms_range
        self.sleep_fn(self.rng.uniform(lo, hi) / 1000.0)

    def _pt(self, name):
        return resolve_point(self.config.anchors_ratio[name], self.phone)

    def _jitter(self, value, spread):
        return value + self.rng.randint(-spread, spread)

    def _throw(self):
        x1, y1 = self._pt("throw_start")
        x2, y2 = self._pt("throw_end")
        duration = int(self.rng.uniform(*self._THROW_DURATION_MS_RANGE))
        self.device.swipe(
            self._jitter(x1, self._SWIPE_JITTER_PX),
            self._jitter(y1, self._SWIPE_JITTER_PX),
            self._jitter(x2, self._SWIPE_JITTER_PX),
            self._jitter(y2, self._SWIPE_JITTER_PX),
            duration,
        )

    def _poll_until_map(self):
        """Poll classify(screencap()) until MAP or stuck_timeout_ms elapses.

        Never throws; only observes. Bounded so it can't spin forever even
        if the phone is genuinely stuck (or in tests, when the scripted
        classifier never yields MAP).
        """
        stuck_timeout_ms = self.config.timing["stuck_timeout_ms"]
        elapsed = 0
        while True:
            img = self.device.screencap()
            if self.classifier(img) == ScreenState.MAP:
                return True
            if elapsed >= stuck_timeout_ms:
                return False
            self._sleep([self._POLL_INTERVAL_MS, self._POLL_INTERVAL_MS])
            elapsed += self._POLL_INTERVAL_MS

    def _recover(self, state):
        """Bring the phone back to a known MAP state. NEVER throws."""
        if state == ScreenState.ENCOUNTER:
            self.device.tap(*self._pt("flee_button"))
            self._poll_until_map()
        elif state in (ScreenState.POKESTOP, ScreenState.GYM):
            self.device.key_back()
            self._poll_until_map()
        else:  # UNKNOWN (or any other non-confirmed state)
            self.device.key_back()
            if not self._poll_until_map():
                self.device.key_back()

    def _run_throw_loop(self):
        max_throws = self.config.timing["max_throws"]
        throws = 0
        while throws < max_throws:
            img = self.device.screencap()
            if self.classifier(img) != ScreenState.ENCOUNTER:
                return  # caught or fled -> done
            self._throw()
            throws += 1
            self._sleep(self.config.timing["post_throw_ms"])

        # Cap hit and still in ENCOUNTER -> give up on this Pokemon.
        if self.classifier(self.device.screencap()) == ScreenState.ENCOUNTER:
            self.device.tap(*self._pt("flee_button"))
            self._poll_until_map()

    def tick(self):
        img = self.device.screencap()
        state = self.classifier(img)

        if state == ScreenState.MAP:
            target = self.detector_fn(img, self.phone)
            if target is None:
                self._sleep(self.config.timing["map_scan_ms"])
                return

            self.device.tap(
                self._jitter(target.x, self._TAP_JITTER_PX),
                self._jitter(target.y, self._TAP_JITTER_PX),
            )
            self._sleep(self.config.timing["encounter_load_ms"])

            img2 = self.device.screencap()
            state2 = self.classifier(img2)
            if state2 != ScreenState.ENCOUNTER:
                self._recover(state2)  # wrong tap -> back out, NEVER throw
                return

            self.labeler(img, target, self.config.dataset_dir)
            self._run_throw_loop()

        elif state == ScreenState.ENCOUNTER:
            self._run_throw_loop()  # entered mid-encounter

        else:  # POKESTOP, GYM, or UNKNOWN
            self._recover(state)

    def run(self, stop_event):
        while not stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - one bad tick must not kill the phone
                # Transient adb hiccup or a weird screen: log, back off briefly,
                # and keep going. The loop only exits when stop_event is set.
                print(f"[{self.phone.serial}] tick error: {exc}")
                self._sleep(self._ERROR_BACKOFF_MS)
                continue
            self._sleep(self.config.timing["map_scan_ms"])
