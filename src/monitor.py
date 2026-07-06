"""Live monitoring + control UI for the catch bot.

A tiny threaded HTTP server (stdlib only) on http://127.0.0.1:<port>/ showing,
per phone: the live frame the loop last processed (annotated with the state,
the proposed target box, the tap point, and blacklist zones) plus counters --
and Pause / Resume buttons that gate that phone's loop between ticks.

Endpoints:
    GET  /                     the dashboard page
    GET  /frame/<serial>.jpg   latest annotated frame (JPEG)
    GET  /status               JSON: per-phone state/counters/paused
    POST /control/<serial>/pause
    POST /control/<serial>/resume
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


_PAGE = """<!DOCTYPE html>
<html><head><title>PoGo Catcher — Live</title>
<style>
  body { background:#111; color:#ddd; font-family:Segoe UI,Arial,sans-serif;
         margin:0; padding:16px; }
  h1 { font-size:18px; margin:0 0 12px; color:#7fd3a0; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .phone { background:#1b1b1b; border:1px solid #333; border-radius:10px;
           padding:12px; width:360px; }
  .phone h2 { font-size:14px; margin:0 0 8px; color:#9ecbff; }
  .phone img { width:100%; border-radius:6px; background:#000; }
  .stats { font-size:12px; margin:8px 0; line-height:1.6; color:#bbb;
           white-space:pre-line; }
  button { background:#2b6cb0; color:#fff; border:0; border-radius:6px;
           padding:8px 18px; margin-right:8px; cursor:pointer; font-size:13px; }
  button.pause { background:#b03a2b; }
  .paused-badge { color:#ff7b6b; font-weight:bold; }
</style></head>
<body>
<h1>PoGo Catcher — live monitor</h1>
<div class="row" id="phones"></div>
<script>
async function refresh() {
  const r = await fetch('/status'); const st = await r.json();
  const root = document.getElementById('phones');
  for (const [serial, s] of Object.entries(st)) {
    let card = document.getElementById('card-' + serial);
    if (!card) {
      card = document.createElement('div');
      card.className = 'phone'; card.id = 'card-' + serial;
      card.innerHTML = `<h2>${serial} <span class="paused-badge" id="pb-${serial}"></span></h2>
        <img id="img-${serial}" src="/stream/${serial}.mjpg">
        <div class="stats" id="stats-${serial}"></div>
        <button onclick="ctl('${serial}','resume')">Resume</button>
        <button class="pause" onclick="ctl('${serial}','pause')">Pause</button>`;
      root.appendChild(card);
    }
    document.getElementById('pb-' + serial).textContent = s.paused ? ' — PAUSED' : '';
    document.getElementById('stats-' + serial).textContent =
      `state: ${s.state}   detector: ${s.src}\\n` +
      `catches: ${s.catches}   wasted: ${s.wasted}   panels: ${s.panels}\\n` +
      `pan: ${s.pan_speed} px/s   last event: ${s.note}`;
  }
}
async function ctl(serial, action) { await fetch('/control/' + serial + '/' + action, {method:'POST'}); refresh(); }
setInterval(refresh, 600); refresh();
</script>
</body></html>"""


_REVIEW_PAGE = """<!DOCTYPE html>
<html><head><title>PoGo Catcher — Click Review</title>
<style>
  body { background:#111; color:#ddd; font-family:Segoe UI,Arial,sans-serif;
         margin:0; padding:16px; }
  h1 { font-size:18px; margin:0 0 4px; color:#7fd3a0; }
  a { color:#9ecbff; }
  .grid { display:flex; flex-wrap:wrap; gap:12px; margin-top:12px; }
  .card { background:#1b1b1b; border:1px solid #333; border-radius:10px;
          padding:8px; width:300px; }
  .card img { width:100%; border-radius:6px; background:#000; }
  .meta { font-size:11px; color:#999; margin:6px 0; }
  .badge { display:inline-block; padding:1px 8px; border-radius:8px;
           font-size:11px; font-weight:bold; }
  .b-encounter { background:#1c5c34; } .b-panel { background:#7a2b1e; }
  .b-nothing { background:#5c5320; } .b-timeout { background:#444; }
  button { border:0; border-radius:6px; padding:6px 12px; margin:2px;
           cursor:pointer; font-size:12px; color:#fff; }
  .good { background:#227a46; } .bad { background:#a33; }
  .reason { background:#444; }
  .voted { opacity:0.45; }
</style></head>
<body>
<h1>Click review</h1>
<div><a href="/">&larr; live monitor</a> — vote each click; bad votes with an
object reason become avoid-training labels, good votes on misses become
pokemon labels.</div>
<div class="grid" id="grid"></div>
<script>
const REASONS = ["blank/nothing","gym","gym pokemon","dynamax","raid icon",
                 "pokestop","rocket","ui element","other"];
async function load() {
  const r = await fetch('/review/list'); const items = await r.json();
  const grid = document.getElementById('grid'); grid.innerHTML = '';
  for (const it of items) {
    const card = document.createElement('div');
    card.className = 'card' + (it.vote ? ' voted' : '');
    let controls = '';
    if (!it.vote) {
      controls = `<button class="good" onclick="vote('${it.id}','good',null)">Good</button>
        <button class="bad" onclick="showReasons('${it.id}')">Bad…</button>
        <span id="rs-${it.id}" style="display:none">` +
        REASONS.map(x => `<button class="reason" onclick="vote('${it.id}','bad','${x}')">${x}</button>`).join('') +
        `</span>`;
    } else {
      controls = `<b>${it.vote}</b>${it.reason ? ' — ' + it.reason : ''}`;
    }
    card.innerHTML = `<img src="/review/img/${it.id}.jpg" loading="lazy">
      <div class="meta"><span class="badge b-${it.outcome}">${it.outcome}</span>
      &nbsp;detector: ${it.src} &nbsp;${it.id}</div>${controls}`;
    grid.appendChild(card);
  }
}
function showReasons(id) { document.getElementById('rs-' + id).style.display = 'inline'; }
async function vote(id, v, reason) {
  await fetch('/review/vote', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id, vote: v, reason})});
  load();
}
load(); setInterval(load, 15000);
</script>
</body></html>"""


class PhoneMonitor:
    """Shared state for one phone: the loop publishes, the server reads."""

    def __init__(self, serial):
        self.serial = serial
        self.lock = threading.Lock()
        self.pause_event = threading.Event()
        self.frame = None          # annotated BGR
        self.state = "-"
        self.src = "-"
        self.note = "-"
        self.pan_speed = 0.0
        self.catches = 0
        self.wasted = 0
        self.panels = 0

    def publish(self, img, state, target=None, fail_spots=(), pan_speed=0.0,
                note=None, tap=None):
        vis = img.copy()
        if target is not None:
            bx, by, bw, bh = target.bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 0, 255), 4)
            self.src = getattr(target, "src", "?")
        if tap is not None:
            cv2.circle(vis, (int(tap[0]), int(tap[1])), 26, (0, 255, 0), 5)
        for fx, fy, _exp in fail_spots:
            cv2.circle(vis, (int(fx), int(fy)), 40, (0, 200, 255), 3)
        cv2.putText(vis, str(state), (12, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (80, 240, 160), 3)
        small = cv2.resize(vis, (vis.shape[1] // 3, vis.shape[0] // 3))
        with self.lock:
            self.frame = small
            self.state = str(state)
            self.pan_speed = round(pan_speed, 1)
            if note:
                self.note = note

    def publish_raw(self, img):
        """Frame-only update (keeps the last state text) -- called from inside
        polling loops so the live feed stays smooth during encounters."""
        small = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
        with self.lock:
            self.frame = small

    def bump(self, outcome):
        if outcome == "encounter":
            self.catches += 1
        elif outcome == "panel":
            self.panels += 1
        else:
            self.wasted += 1
        self.note = outcome

    def jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            ok, buf = cv2.imencode(".jpg", self.frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    def status(self):
        with self.lock:
            return {
                "state": self.state, "src": self.src, "note": self.note,
                "pan_speed": self.pan_speed, "paused": self.pause_event.is_set(),
                "catches": self.catches, "wasted": self.wasted,
                "panels": self.panels,
            }


class MonitorServer:
    def __init__(self, port=8750, review_store=None):
        self.port = port
        self.phones = {}  # serial -> PhoneMonitor
        self.review = review_store

    def register(self, serial):
        pm = PhoneMonitor(serial)
        self.phones[serial] = pm
        return pm

    def start(self):
        phones = self.phones
        review = self.review

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence request spam
                pass

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", _PAGE.encode())
                elif path == "/review" and review is not None:
                    self._send(200, "text/html; charset=utf-8",
                               _REVIEW_PAGE.encode())
                elif path == "/review/list" and review is not None:
                    self._send(200, "application/json",
                               json.dumps(review.recent()).encode())
                elif (path.startswith("/review/img/") and path.endswith(".jpg")
                        and review is not None):
                    rid = path[len("/review/img/"):-len(".jpg")]
                    p = review.image_path(rid)
                    try:
                        with open(p, "rb") as f:
                            self._send(200, "image/jpeg", f.read())
                    except (OSError, TypeError):
                        self._send(404, "text/plain", b"gone")
                elif path == "/status":
                    body = json.dumps({s: p.status() for s, p in phones.items()})
                    self._send(200, "application/json", body.encode())
                elif path.startswith("/frame/") and path.endswith(".jpg"):
                    serial = path[len("/frame/"):-len(".jpg")]
                    pm = phones.get(serial)
                    data = pm.jpeg() if pm else None
                    if data is None:
                        self._send(404, "text/plain", b"no frame yet")
                    else:
                        self._send(200, "image/jpeg", data)
                elif path.startswith("/stream/") and path.endswith(".mjpg"):
                    # MJPEG push: one connection, ~15fps, no client polling.
                    serial = path[len("/stream/"):-len(".mjpg")]
                    pm = phones.get(serial)
                    if pm is None:
                        self._send(404, "text/plain", b"unknown phone")
                        return
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while True:
                            data = pm.jpeg()
                            if data is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(
                                    f"Content-Length: {len(data)}\r\n\r\n".encode())
                                self.wfile.write(data)
                                self.wfile.write(b"\r\n")
                            time.sleep(1 / 15)
                    except (ConnectionError, OSError):
                        return  # viewer closed the tab
                else:
                    self._send(404, "text/plain", b"not found")

            def do_POST(self):
                if self.path == "/review/vote" and review is not None:
                    try:
                        n = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(n))
                        ok = review.vote(body["id"], body["vote"],
                                         body.get("reason"))
                        self._send(200 if ok else 404, "text/plain",
                                   b"ok" if ok else b"unknown id")
                    except (ValueError, KeyError):
                        self._send(400, "text/plain", b"bad request")
                    return
                parts = self.path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "control":
                    pm = phones.get(parts[1])
                    if pm is not None and parts[2] in ("pause", "resume"):
                        if parts[2] == "pause":
                            pm.pause_event.set()
                        else:
                            pm.pause_event.clear()
                        self._send(200, "text/plain", b"ok")
                        return
                self._send(404, "text/plain", b"not found")

        server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
