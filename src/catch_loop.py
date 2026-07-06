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
from src.detector import save_click_debug as _save_click_debug
from src.detector import save_label as _save_label
from src.screen_state import ScreenState
from src.screen_state import classify as _classify
from src.screen_state import has_close_button as _has_close_button
from src.screen_state import in_encounter as _in_encounter
from src.screen_state import is_screen_off as _is_screen_off


class CatchLoop:
    _POLL_INTERVAL_MS = 80    # how often we re-check while polling (rapid-fire)
    _TAP_JITTER_PX = 5
    _SWIPE_JITTER_PX = 5
    _THROW_DURATION_MS_RANGE = (120, 180)
    _ERROR_BACKOFF_MS = (1000, 1000)  # brief pause after a crashed tick
    _RECOVER_ATTEMPTS = 4
    _RECOVER_POLL_MS = 1500           # short per-attempt poll (PoGo close is ~instant)
    _BLIND_CLOSE_AFTER = 2            # attempts w/o a recognized X before blind-tapping the X spot
    _MAP_BAIL_MS = 1200               # after this, still-on-MAP means the tap opened nothing

    def __init__(
        self,
        device,
        config,
        phone,
        classifier=_classify,
        detector_fn=propose,
        labeler=_save_label,
        click_logger=_save_click_debug,
        sleep_fn=time.sleep,
        rng=None,
        encounter_check=_in_encounter,
        close_check=_has_close_button,
        clock=time.monotonic,
    ):
        self.device = device
        self.config = config
        self.phone = phone
        self.classifier = classifier
        self.detector_fn = detector_fn
        self.labeler = labeler
        self.click_logger = click_logger
        self.sleep_fn = sleep_fn
        self.rng = rng if rng is not None else random.Random()
        self.encounter_check = encounter_check
        self.close_check = close_check
        self.clock = clock

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
        """Poll until predicate(screencap()) is True or timeout_ms of REAL time
        elapses.

        Returns (img, ok). `img` is the frame that satisfied the predicate, or
        the last non-None frame seen on timeout. Elapsed is measured on the
        wall-clock (self.clock) -- NOT by counting fixed interval steps -- because
        a single screencap can take ~0.6s, far longer than interval_ms; counting
        steps made a 1.9s timeout run ~16s of real time. Screencap None (truncated
        pull) is skipped -- never fed to the predicate.
        """
        interval_ms = interval_ms if interval_ms is not None else self._POLL_INTERVAL_MS
        start = self.clock()
        last_img = None
        while True:
            img = self.device.screencap()
            if img is not None:
                last_img = img
                if predicate(img):
                    return img, True
            if (self.clock() - start) * 1000.0 >= timeout_ms:
                return last_img, False
            self.sleep_fn(interval_ms / 1000.0)

    def _await_encounter(self, timeout_ms):
        """After tapping a map target, wait for the encounter UI -- but BAIL the
        instant it's clear no encounter will open, instead of burning the whole
        timeout. Returns (enc_img, last_img): `enc_img` is the confirmed
        encounter frame or None to back out; `last_img` is the frame seen at the
        decision point, so the caller can hand it straight to _recover instead
        of paying another ~0.6s screencap.

        Bail-fast cases (the common mis-tap costs):
          - a closable panel opened (gym / stop / menu / Rocket) -> its X is visible;
          - we're still on the MAP past the transition window -> tap hit nothing.
        Neither can ever be a loading encounter, so leaving early is safe. INV-1
        is preserved: a None enc_img routes the caller to _recover, never a throw.
        """
        start = self.clock()
        last_img = None
        while True:
            img = self.device.screencap()
            if img is not None:
                last_img = img
                if self.encounter_check(img):
                    return img, img  # encounter UI confirmed -> the ONLY throw path
                if self.close_check(img):
                    return None, img  # gym / stop / menu opened -> bail now
                elapsed = (self.clock() - start) * 1000.0
                if elapsed >= self._MAP_BAIL_MS and self.classifier(img) == ScreenState.MAP:
                    return None, img  # transition window passed, still map -> tap opened nothing
            if (self.clock() - start) * 1000.0 >= timeout_ms:
                return None, last_img
            self.sleep_fn(self._POLL_INTERVAL_MS / 1000.0)

    def _bail_outcome(self, bail_img):
        """Name what a failed tap actually opened, for the click-audit trail:
        'panel' (gym/stop/Rocket/Route...), 'nothing' (still on the map), or
        'timeout' (never resolved / no frame)."""
        if bail_img is None:
            return "timeout"
        if self.close_check(bail_img):
            return "panel"
        if self.classifier(bail_img) == ScreenState.MAP:
            return "nothing"
        return "timeout"

    # --- recovery (NEVER throws) ------------------------------------------
    def _recover(self, state=None, img=None):
        """Get back to a playable screen after a mis-tap. INV-2: never throws.

        `img`, when given, is a frame the caller ALREADY captured (tick's frame
        or _await_encounter's bail frame) -- the first attempt acts on it
        directly instead of paying another ~0.6s screencap.

        Rules, in order, per attempt:
          - MAP or ENCOUNTER  -> done. We NEVER disrupt an encounter (no flee):
            an encounter is a catch opportunity, so recovery just yields and the
            next tick's throw loop handles it.
          - close button visible (gym / stop / menu / Rocket) -> tap the
            on-screen X (the game ignores the Android back button) and poll
            for a playable screen.
          - otherwise (UNKNOWN, no X: an encounter still loading, or a catch
            animation) -> wait briefly and re-check. Deliberately NO key_back
            here: pressing back on a mid-load encounter would flee it.
        """
        playable = (ScreenState.MAP, ScreenState.ENCOUNTER)
        for attempt in range(self._RECOVER_ATTEMPTS):
            frame = img if img is not None else self.device.screencap()
            img = None  # a caller-provided frame is only current for attempt 1
            if frame is None:
                self._sleep((150, 300))
                continue
            if self.classifier(frame) in playable:
                return
            if self.close_check(frame) or attempt >= self._BLIND_CLOSE_AFTER:
                # Recognized X theme -> tap it. OR: the screen stayed
                # un-playable through the early waits with NO template match --
                # every closable PoGo panel (gym / stop / menu / Rocket / Route)
                # puts its X at this same bottom-centre spot, so tap it BLINDLY:
                # an unseen panel theme self-heals in a couple seconds instead
                # of trapping the loop forever (Rocket and Route both did that
                # before their templates existed). Still INV-2: only ever a
                # close tap, never a throw/swipe; and playable screens returned
                # above, so this never fires on MAP or an encounter.
                self.device.tap(*self._pt("close_button"))
                _, ok = self._poll(
                    lambda im: self.classifier(im) in playable,
                    self._RECOVER_POLL_MS,
                )
                if ok:
                    return
            else:
                self._sleep((300, 500))  # transient; wait, don't flee

    # --- throw loop (INV-1 lives here) ------------------------------------
    def _run_throw_loop(self, first_frame=None):
        """Throw until the encounter resolves. `first_frame`, when given, is an
        ALREADY-CONFIRMED encounter frame from _await_encounter -- we throw on it
        immediately instead of paying another ~0.6s screencap, so the ball flies
        the instant the berry/ball UI appears. INV-1 still holds: encounter_check
        is re-verified on every frame (including first_frame) before any throw."""
        max_throws = self.config.timing["max_throws"]
        resolve_timeout_ms = self._timeout("post_throw_ms")

        throws = 0
        img = first_frame
        while throws < max_throws:
            if img is None:
                img = self.device.screencap()
            if img is None or not self.encounter_check(img):
                return  # resolved (caught / fled) or the encounter is gone

            self._throw()  # SAFE: guarded by encounter_check(img) True just above
            throws += 1
            img = None  # force a fresh screencap for the next iteration's re-check

            # Wait for the result: caught -> back to MAP; broke free -> still an
            # encounter once the shake animation ends. Poll instead of sleeping.
            _, got_map = self._poll(
                lambda im: self.classifier(im) == ScreenState.MAP,
                resolve_timeout_ms,
            )
            if got_map:
                return  # caught / returned to map

        # Cap reached but still an encounter: do NOT flee. Just yield this tick.
        # The next tick re-detects the same encounter and keeps throwing, so a
        # stubborn Pokemon is thrown at until it's caught (or runs on its own).
        # max_throws is only a per-tick bound so run() can check stop_event.

    # --- one iteration -----------------------------------------------------
    def tick(self):
        img = self.device.screencap()
        if img is None:
            self._sleep(self.config.timing["map_scan_ms"])
            return

        if _is_screen_off(img):
            # Display slept -> screencap is black. Wake it instead of spinning in
            # recovery on an all-UNKNOWN screen. INV-1/2 unaffected (no throw/tap
            # into game UI; only a WAKEUP keyevent).
            self.device.wake()
            self._sleep((400, 700))
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

            # Proceed the instant the encounter UI appears; bail fast otherwise.
            enc_img, bail_img = self._await_encounter(self._timeout("encounter_load_ms"))
            if enc_img is None:
                # Tap did not open an encounter -> back out on the frame we
                # already have (no extra screencap). INV-1: no throw. The audit
                # gets the RESULT frame too: "clicked this -> got this panel".
                self.click_logger(img, target, self._bail_outcome(bail_img),
                                  self.config.dataset_dir, result_img=bail_img)
                self._recover(img=bail_img)
                return

            # Throw at once on the confirmed frame; self-label AFTER so the PNG
            # write never delays the ball.
            self._run_throw_loop(first_frame=enc_img)
            self.labeler(img, target, self.config.dataset_dir)
            self.click_logger(img, target, "encounter", self.config.dataset_dir,
                              result_img=enc_img)

        elif state == ScreenState.ENCOUNTER:
            self._run_throw_loop()  # entered mid-encounter

        else:  # POKESTOP, GYM, or UNKNOWN
            self._recover(state, img=img)  # reuse tick's frame: no extra screencap

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
