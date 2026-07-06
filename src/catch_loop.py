"""Per-phone catch loop state machine (event-driven / polling).

Speed model: instead of blind fixed sleeps, the loop POLLS a cheap predicate
and proceeds the instant the screen changes -- so a catch resolves in ~one
network round-trip, not a worst-case timing constant. The timing config values
are reinterpreted as poll *timeouts* (upper bound), not durations to wait out.

Safety contract (structural, not incidental):
  - INV-1: a throw (device.swipe via _throw) only ever happens after
    `self.encounter_check(img)` returned True for a screenshot taken in that
    same loop iteration. `_run_throw_loop`'s while-body screencaps, bails if the
    image is None or not an encounter, and only THEN throws. The MAP-branch
    only enters `_run_throw_loop` after `_poll(encounter_check, ...)` confirmed
    the encounter UI appeared; the wrong-tap path (poll fails) routes to
    `_recover` and returns before any throw.
  - INV-2: `_recover` NEVER throws / swipes. It only taps `flee_button` (a UI
    escape, not a ball throw) or presses back, then polls for MAP.
"""

import random
import time

from src.config import resolve_point
from src.detector import propose
from src.detector import save_label as _save_label
from src.screen_state import ScreenState
from src.screen_state import classify as _classify
from src.screen_state import in_encounter as _in_encounter


class CatchLoop:
    _POLL_INTERVAL_MS = 80    # how often we re-check while polling (rapid-fire)
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
        encounter_check=_in_encounter,
    ):
        self.device = device
        self.config = config
        self.phone = phone
        self.classifier = classifier
        self.detector_fn = detector_fn
        self.labeler = labeler
        self.sleep_fn = sleep_fn
        self.rng = rng if rng is not None else random.Random()
        self.encounter_check = encounter_check

    # --- small helpers -----------------------------------------------------
    def _sleep(self, ms_range):
        lo, hi = ms_range
        self.sleep_fn(self.rng.uniform(lo, hi) / 1000.0)

    def _pt(self, name):
        return resolve_point(self.config.anchors_ratio[name], self.phone)

    def _timeout(self, key):
        """Poll timeout (ms) from a timing key. Ranges use their UPPER bound."""
        v = self.config.timing[key]
        return v[1] if isinstance(v, (list, tuple)) else v

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

    # --- polling core ------------------------------------------------------
    def _poll(self, predicate, timeout_ms, interval_ms=None):
        """Poll until predicate(screencap()) is True or timeout_ms elapses.

        Returns (img, ok). `img` is the frame that satisfied the predicate, or
        the last non-None frame seen on timeout. Elapsed time is counted in
        fixed interval steps so a no-op sleep_fn still terminates deterministically.
        Screencap None (truncated PNG) is skipped -- never fed to the predicate.
        """
        interval_ms = interval_ms if interval_ms is not None else self._POLL_INTERVAL_MS
        elapsed = 0
        last_img = None
        while True:
            img = self.device.screencap()
            if img is not None:
                last_img = img
                if predicate(img):
                    return img, True
            if elapsed >= timeout_ms:
                return last_img, False
            self.sleep_fn(interval_ms / 1000.0)
            elapsed += interval_ms

    def _poll_until_map(self):
        _, ok = self._poll(
            lambda im: self.classifier(im) == ScreenState.MAP,
            self.config.timing["stuck_timeout_ms"],
        )
        return ok

    # --- recovery (NEVER throws) ------------------------------------------
    def _recover(self, state):
        """Bring the phone back to a known MAP state. INV-2: never throws."""
        if state == ScreenState.ENCOUNTER:
            self.device.tap(*self._pt("flee_button"))
            self._poll_until_map()
        elif state in (ScreenState.POKESTOP, ScreenState.GYM):
            self.device.key_back()
            self._poll_until_map()
        else:  # UNKNOWN / any non-confirmed state
            self.device.key_back()
            if not self._poll_until_map():
                self.device.key_back()

    # --- throw loop (INV-1 lives here) ------------------------------------
    def _run_throw_loop(self):
        max_throws = self.config.timing["max_throws"]
        resolve_timeout_ms = self._timeout("post_throw_ms")

        throws = 0
        while throws < max_throws:
            img = self.device.screencap()
            if img is None or not self.encounter_check(img):
                return  # resolved (caught / fled) or the encounter is gone

            self._throw()  # SAFE: guarded by encounter_check(img) True just above
            throws += 1

            # Wait for the result: caught -> back to MAP; broke free -> still an
            # encounter once the shake animation ends. Poll instead of sleeping.
            _, got_map = self._poll(
                lambda im: self.classifier(im) == ScreenState.MAP,
                resolve_timeout_ms,
            )
            if got_map:
                return  # caught / returned to map

        # Cap reached and still in an encounter -> give up on this Pokemon.
        img = self.device.screencap()
        if img is not None and self.encounter_check(img):
            self.device.tap(*self._pt("flee_button"))
            self._poll_until_map()

    # --- one iteration -----------------------------------------------------
    def tick(self):
        img = self.device.screencap()
        if img is None:
            self._sleep(self.config.timing["map_scan_ms"])
            return

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

            # Proceed the instant the encounter UI appears; bail if it never does.
            img2, ok = self._poll(self.encounter_check, self._timeout("encounter_load_ms"))
            if not ok:
                # Tap did not open an encounter -> back out. INV-1: no throw.
                fallback = self.classifier(img2) if img2 is not None else ScreenState.UNKNOWN
                self._recover(fallback)
                return

            self.labeler(img, target, self.config.dataset_dir)  # confirmed -> self-label
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
            self._sleep(self.config.timing["map_scan_ms"])  # snappy map->detect scan
