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

import inspect
import math
import random
import time

import cv2
import numpy as np

from src.config import resolve_point
from src.detector import propose
from src.detector import save_click_debug as _save_click_debug
from src.detector import save_label as _save_label
from src.detector import save_negative_label as _save_negative_label
from src.screen_state import ScreenState
from src.screen_state import classify as _classify
from src.screen_state import find_ok_button as _find_ok_button
from src.screen_state import has_close_button as _has_close_button
from src.screen_state import has_map_pokeball as _has_map_pokeball
from src.screen_state import in_encounter as _in_encounter
from src.screen_state import is_screen_off as _is_screen_off


class CatchLoop:
    _POLL_INTERVAL_MS = 30    # how often we re-check while polling. Frames are
                              # ~free with the stream (1.3ms vs 600ms pulls), so
                              # fast polling notices every transition ~sooner.
    _TAP_JITTER_PX = 5
    _SWIPE_JITTER_PX = 5
    _THROW_DURATION_MS_RANGE = (120, 180)
    _ERROR_BACKOFF_MS = (1000, 1000)  # brief pause after a crashed tick
    _RECOVER_ATTEMPTS = 4
    _RECOVER_POLL_MS = 1500           # short per-attempt poll (PoGo close is ~instant)
    _BLIND_CLOSE_AFTER = 2            # attempts w/o a recognized X before blind-tapping the X spot
    # Failed-tap blacklist: a spot whose tap yielded nothing / a panel is
    # embargoed so the detector picks the NEXT-best candidate instead of
    # re-tapping the same giant raid boss / walking buddy / inert icon every
    # tick (live audit showed 4+ consecutive taps on one Groudon raid boss).
    # 20s/170px (was 8s/130px): giant raid bosses (Gigantamax Necrozma etc.)
    # offer many distinct tappable fragments across a wide body, so short/small
    # embargoes still allowed ~2.5 panel taps per minute near one.
    _FAIL_SPOT_TTL_S = 20.0
    _FAIL_SPOT_RADIUS = 170
    # Tap-lead (motion compensation): autowalk pans the map continuously, so a
    # target detected on a ~0.5s-old frame has MOVED by tap time -- live audit
    # showed streaks of near-miss taps ~40-80px behind real Pokemon. Pan
    # velocity is measured by phase-correlating consecutive downscaled map
    # frames; the tap leads the target by velocity * _TAP_LEAD_S.
    _TAP_LEAD_S = 0.55            # capture transfer + detect + adb tap latency
    _PAN_DOWNSCALE = 4
    _PAN_BAND = (0.35, 0.78, 0.08, 0.92)   # (y0r, y1r, x0r, x1r) HUD-free band
    _PAN_MIN_SPEED = 12.0         # px/s -- below this, don't bother leading
    _PAN_MAX_SPEED = 320.0        # px/s -- above this, correlation is garbage
    _PAN_MIN_RESPONSE = 0.03      # confidence floor: feature-poor day grass
                                  # correlates weakly (<0.1) yet the shift is
                                  # real; the speed window bounds bad leads
    _RETHROW_GRACE_MS = 700       # post-throw window where encounter UI still
                                  # showing does NOT mean broke-free yet
    # Camera pan: when the visible spawns are exhausted (no target for a few
    # seconds), rotate the camera with a horizontal drag -- spawns hide behind
    # gyms/towers and off-angle (Ethan's request). A drag never taps anything.
    _CAMERA_PAN_AFTER_S = 3.0
    _CAMERA_PAN_Y = 0.55          # drag height (mid-map, clear of all UI)
    _CAMERA_PAN_X = (0.78, 0.22)  # right-to-left sweep rotates ~a third turn
    _CAMERA_PAN_MS = (220, 300)
    _CAMERA_PAN_SETTLE_MS = (250, 400)
    _BLUR_SPEED_MAX = 260         # px/s: above this the H.264 frame is motion-
                                  # blurred (camera rotating) -> don't trust
                                  # detections, skip the tap this tick. Walking
                                  # pan measures ~150-190 px/s and stays OK.
    _PAN_EMA_ALPHA = 0.5          # velocity smoothing: one noisy correlation
                                  # (low-texture frame) must not shove the tap
                                  # sideways (live: a tap missed wide)
    _LEAD_MAX_PX = 110            # hard cap on how far a tap may be led
    _MAP_BAIL_MS = 1600               # after this, still-on-MAP means the tap opened nothing
                                      # (1.2s misfiled slow day encounter loads as 'nothing')

    def __init__(
        self,
        device,
        config,
        phone,
        classifier=_classify,
        detector_fn=propose,
        labeler=_save_label,
        neg_labeler=_save_negative_label,
        click_logger=_save_click_debug,
        sleep_fn=time.sleep,
        rng=None,
        encounter_check=_in_encounter,
        close_check=_has_close_button,
        pokeball_check=_has_map_pokeball,
        ok_finder=_find_ok_button,
        clock=time.monotonic,
        monitor=None,
    ):
        self.device = device
        self.config = config
        self.phone = phone
        self.classifier = classifier
        self.detector_fn = detector_fn
        self.labeler = labeler
        self.neg_labeler = neg_labeler
        self.click_logger = click_logger
        self.sleep_fn = sleep_fn
        self.rng = rng if rng is not None else random.Random()
        self.encounter_check = encounter_check
        self.close_check = close_check
        self.pokeball_check = pokeball_check
        self.ok_finder = ok_finder
        self.clock = clock
        self.monitor = monitor  # live-UI publisher (PhoneMonitor) or None
        self._fail_spots = []  # [(x, y, expires_at)] recent failed-tap embargo
        self._prev_pan = None  # (downscaled gray band, timestamp) for tap-lead
        self._last_target_t = None  # last time the detector found anything
        self._pan_speed = 0.0       # last measured scene pan speed (px/s)
        self._pan_v = None          # smoothed (EMA) pan velocity, px/s
        # Streamed frames are fresher than pulled ones -> shorter tap lead.
        self._tap_lead_s = getattr(device, "tap_lead_s", self._TAP_LEAD_S)
        try:
            self._detector_takes_exclude = (
                "exclude" in inspect.signature(detector_fn).parameters
            )
        except (TypeError, ValueError):
            self._detector_takes_exclude = False

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
                if self.monitor is not None:
                    self.monitor.publish_raw(img)  # keep the live feed smooth
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

    def _pan_lead(self, img, now):
        """(dx, dy) px to ADD to a tap so it lands where the target will BE.

        Phase-correlates a downscaled HUD-free band of consecutive map frames
        to measure the autowalk pan velocity, then multiplies by the fixed
        capture-to-tap latency. Returns (0, 0) whenever the measurement is
        implausible (no prior frame, stale prior, low correlation confidence,
        or speed outside the sane range) -- a bad lead is worse than none.
        """
        h, w = img.shape[:2]
        y0r, y1r, x0r, x1r = self._PAN_BAND
        band = img[int(y0r * h):int(y1r * h), int(x0r * w):int(x1r * w)]
        if band.shape[0] < 128 or band.shape[1] < 128:
            return (0.0, 0.0)  # tiny/degenerate frame -> nothing to correlate
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        s = 1.0 / self._PAN_DOWNSCALE
        small = cv2.resize(gray, None, fx=s, fy=s).astype(np.float32)

        lead = (0.0, 0.0)
        if self._prev_pan is not None:
            prev_small, prev_t = self._prev_pan
            dt = now - prev_t
            if 0.05 < dt < 3.0 and prev_small.shape == small.shape:
                (dx, dy), response = cv2.phaseCorrelate(prev_small, small)
                # (dx, dy) = how far the scene content moved prev -> current
                # (in downscaled px); targets move WITH the scene.
                vx = dx * self._PAN_DOWNSCALE / dt
                vy = dy * self._PAN_DOWNSCALE / dt
                speed = math.hypot(vx, vy)
                if response >= self._PAN_MIN_RESPONSE:
                    self._pan_speed = speed
                if (response >= self._PAN_MIN_RESPONSE
                        and self._PAN_MIN_SPEED < speed < self._PAN_MAX_SPEED):
                    # EMA-smooth the velocity: a single noisy correlation on a
                    # low-texture frame must not shove the tap sideways.
                    if self._pan_v is None:
                        self._pan_v = (vx, vy)
                    else:
                        a = self._PAN_EMA_ALPHA
                        self._pan_v = (a * vx + (1 - a) * self._pan_v[0],
                                       a * vy + (1 - a) * self._pan_v[1])
                    lx = self._pan_v[0] * self._tap_lead_s
                    ly = self._pan_v[1] * self._tap_lead_s
                    mag = math.hypot(lx, ly)
                    if mag > self._LEAD_MAX_PX:  # hard cap on lead distance
                        lx, ly = (lx * self._LEAD_MAX_PX / mag,
                                  ly * self._LEAD_MAX_PX / mag)
                    lead = (lx, ly)
        self._prev_pan = (small, now)
        return lead

    def _camera_pan(self):
        """Rotate the camera with a horizontal drag to reveal spawns hidden
        behind gyms/towers or off-angle. A drag is never a tap (nothing can be
        opened) and never a throw (INV-1 concerns only the throw gesture inside
        the throw loop). Screen-space state is invalidated: the failed-tap
        embargoes and the pan-correlation basis both refer to the old view."""
        w, h = self.phone.width, self.phone.height
        y = int(self._CAMERA_PAN_Y * h)
        x1, x2 = int(self._CAMERA_PAN_X[0] * w), int(self._CAMERA_PAN_X[1] * w)
        self.device.swipe(
            self._jitter(x1, 12), self._jitter(y, 12),
            self._jitter(x2, 12), self._jitter(y, 12),
            int(self.rng.uniform(*self._CAMERA_PAN_MS)),
        )
        self._prev_pan = None
        self._pan_v = None   # rotation changes the walk's screen direction
        self._fail_spots = []
        self._sleep(self._CAMERA_PAN_SETTLE_MS)

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
            # Some dialogs have NO X at all -- just one wide OK pill (bonus
            # popups). If one is on screen, tap IT: the X spot would miss.
            ok_pt = self.ok_finder(frame)
            if ok_pt is not None and not self.close_check(frame):
                self.device.tap(*ok_pt)
                _, ok = self._poll(
                    lambda im: self.classifier(im) in playable,
                    self._RECOVER_POLL_MS,
                )
                if ok:
                    return
                continue

            # Blind tap only when this is NOT secretly the map: the overworld
            # pokeball button sits right beside the X spot, so blind-tapping a
            # map frame that mis-classified as UNKNOWN (petal-dense hue drift)
            # would open the MAIN MENU. Pokeball visible => map => never blind.
            blind_ok = (attempt >= self._BLIND_CLOSE_AFTER
                        and not self.pokeball_check(frame))
            if self.close_check(frame) or blind_ok:
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

            # Wait for EITHER resolution: caught -> back to MAP; broke free ->
            # the encounter UI (berry/ball buttons) reappears once the shake
            # animation ends -- and the NEXT ball flies on that very frame
            # instead of waiting out the timeout (Ethan: "the second the berry
            # icon appears you can throw another one"). The grace period stops
            # the pre-throw UI from being mistaken for a broke-free.
            start = self.clock()

            def _resolved(im):
                if self.classifier(im) == ScreenState.MAP:
                    return True
                return ((self.clock() - start) * 1000.0 >= self._RETHROW_GRACE_MS
                        and self.encounter_check(im))

            frame, ok = self._poll(_resolved, resolve_timeout_ms)
            if ok and frame is not None:
                if self.classifier(frame) == ScreenState.MAP:
                    return  # caught / returned to map
                img = frame  # broke free, UI is back -> re-throw on this frame

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
        if self.monitor is not None:
            self.monitor.publish(img, state, fail_spots=self._fail_spots,
                                 pan_speed=self._pan_speed)

        if state == ScreenState.MAP:
            now = self.clock()
            lead_x, lead_y = self._pan_lead(img, now)  # motion compensation
            if self._pan_speed > self._BLUR_SPEED_MAX:
                # View is rotating fast -> this H.264 frame is motion-blurred
                # and detections on it mislocate (live: taps landed on stops).
                # Skip the tick; the view settles within a frame or two.
                self._sleep(self.config.timing["map_scan_ms"])
                return
            self._fail_spots = [s for s in self._fail_spots if s[2] > now]
            exclude = [(fx, fy, self._FAIL_SPOT_RADIUS) for fx, fy, _ in self._fail_spots]
            if self._detector_takes_exclude:
                target = self.detector_fn(img, self.phone, exclude=exclude)
            else:
                target = self.detector_fn(img, self.phone)
                # Detector can't skip embargoed spots itself -> at least never
                # RE-tap one (skip this tick; the map pans / the TTL expires).
                if target is not None and any(
                    math.hypot(target.x - fx, target.y - fy) <= self._FAIL_SPOT_RADIUS
                    for fx, fy, _ in self._fail_spots
                ):
                    target = None
            if target is None:
                # Visible spawns exhausted? Rotate the camera to look around --
                # spawns hide behind gyms/towers and outside the current angle.
                if self._last_target_t is None:
                    self._last_target_t = now
                elif now - self._last_target_t >= self._CAMERA_PAN_AFTER_S:
                    self._camera_pan()
                    self._last_target_t = self.clock()
                    return
                self._sleep(self.config.timing["map_scan_ms"])
                return
            self._last_target_t = now

            tap_x = self._jitter(target.x + int(lead_x), self._TAP_JITTER_PX)
            tap_y = self._jitter(target.y + int(lead_y), self._TAP_JITTER_PX)
            if self.monitor is not None:
                self.monitor.publish(img, state, target=target,
                                     fail_spots=self._fail_spots,
                                     pan_speed=self._pan_speed,
                                     tap=(tap_x, tap_y))
            self.device.tap(tap_x, tap_y)

            # Proceed the instant the encounter UI appears; bail fast otherwise.
            enc_img, bail_img = self._await_encounter(self._timeout("encounter_load_ms"))
            if enc_img is None:
                # Tap did not open an encounter -> back out on the frame we
                # already have (no extra screencap). INV-1: no throw. The audit
                # gets the RESULT frame too: "clicked this -> got this panel".
                # Embargo the spot so the next ticks try OTHER candidates.
                self._fail_spots.append(
                    (target.x, target.y, self.clock() + self._FAIL_SPOT_TTL_S)
                )
                outcome = self._bail_outcome(bail_img)
                if self.monitor is not None:
                    self.monitor.bump(outcome)
                if outcome == "panel":
                    # The tap provably hit an interactable non-Pokemon object:
                    # save it as a class-1 "avoid" YOLO example (hard negative).
                    self.neg_labeler(img, target, self.config.dataset_dir)
                self.click_logger(img, target, outcome,
                                  self.config.dataset_dir, result_img=bail_img)
                self._recover(img=bail_img)
                return

            # Throw at once on the confirmed frame; self-label AFTER so the PNG
            # write never delays the ball.
            self._run_throw_loop(first_frame=enc_img)
            self.labeler(img, target, self.config.dataset_dir)
            self.click_logger(img, target, "encounter", self.config.dataset_dir,
                              result_img=enc_img)
            # The catch took seconds: restart the camera-pan clock, else the
            # first briefly-empty rescan pans even with spawns still visible.
            self._last_target_t = self.clock()
            if self.monitor is not None:
                self.monitor.bump("encounter")

        elif state == ScreenState.ENCOUNTER:
            self._run_throw_loop()  # entered mid-encounter
            self._last_target_t = None  # camera-pan clock restarts on the map

        else:  # POKESTOP, GYM, or UNKNOWN
            self._recover(state, img=img)  # reuse tick's frame: no extra screencap
            self._last_target_t = None  # camera-pan clock restarts on the map

    def run(self, stop_event):
        while not stop_event.is_set():
            if self.monitor is not None and self.monitor.pause_event.is_set():
                # Paused from the live UI: keep publishing a heartbeat frame so
                # the dashboard stays live, but touch nothing on the phone.
                img = self.device.screencap()
                if img is not None:
                    self.monitor.publish(img, "PAUSED")
                self.sleep_fn(0.4)
                continue
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - one bad tick must not kill the phone
                # Transient adb hiccup or a weird screen: log, back off briefly,
                # and keep going. The loop only exits when stop_event is set.
                print(f"[{self.phone.serial}] tick error: {exc}")
                self._sleep(self._ERROR_BACKOFF_MS)
                continue
            self._sleep(self.config.timing["map_scan_ms"])  # snappy map->detect scan
