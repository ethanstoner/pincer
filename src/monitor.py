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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


_PAGE = """<!DOCTYPE html>
<html><head><title>PoGo Catcher — Live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{--cyan:#22d3ee;--green:#34d399;--amber:#fbbf24;--red:#f87171;
        --ink:#dbe4ee;--muted:#6b7d94}
  body{font-family:'Segoe UI',system-ui,sans-serif;color:var(--ink);
    min-height:100vh;padding:22px 26px;
    background:radial-gradient(1200px 760px at 18% -12%,#12283f 0%,#080d15 55%,#04060a 100%);
    background-attachment:fixed}
  .hdr{display:flex;align-items:center;gap:16px;margin-bottom:22px;flex-wrap:wrap}
  .logo{width:36px;height:36px;border-radius:10px;display:grid;place-items:center;
    font-size:19px;background:linear-gradient(135deg,var(--cyan),var(--green));
    box-shadow:0 0 20px rgba(34,211,238,.45)}
  .brand h1{font-size:16px;font-weight:700;letter-spacing:.4px}
  .brand .sub{font-size:10px;color:var(--muted);letter-spacing:2.5px;text-transform:uppercase}
  .live{display:inline-flex;align-items:center;gap:7px;font-size:10.5px;color:var(--green);
    text-transform:uppercase;letter-spacing:1.5px;font-weight:700}
  .live .d{width:8px;height:8px;border-radius:50%;background:var(--green);
    animation:pulse 1.8s infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.55)}
    70%{box-shadow:0 0 0 9px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}
  .spacer{margin-left:auto}
  .kpi{display:flex;gap:20px;align-items:center;margin-right:6px}
  .kpi .k{text-align:right}
  .kpi .kn{font-size:20px;font-weight:700;font-family:ui-monospace,'Cascadia Mono',monospace;
    line-height:1}
  .kpi .kl{font-size:9.5px;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px}
  .kpi .catch{color:var(--green)} .kpi .waste{color:var(--amber)}
  .navbtn{text-decoration:none;color:var(--ink);font-size:12.5px;font-weight:600;
    padding:9px 16px;border-radius:10px;display:inline-flex;align-items:center;gap:8px;
    border:1px solid rgba(120,160,200,.2);background:rgba(30,45,66,.5);transition:.15s}
  .navbtn:hover{border-color:var(--cyan);color:#fff}
  .navbtn .count{background:var(--cyan);color:#04121a;border-radius:20px;
    padding:1px 8px;font-size:11px;font-weight:800}
  .navbtn.label{border-color:rgba(52,211,153,.25)}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(345px,1fr));gap:20px}
  .card{border-radius:16px;padding:15px;
    background:linear-gradient(180deg,rgba(21,32,48,.72),rgba(11,17,27,.75));
    backdrop-filter:blur(9px);border:1px solid rgba(90,130,170,.16);
    box-shadow:0 12px 42px rgba(0,0,0,.42),inset 0 1px 0 rgba(255,255,255,.045);
    transition:border-color .2s}
  .card.paused{border-color:rgba(248,113,113,.4)}
  .ctop{display:flex;align-items:center;gap:10px;margin-bottom:12px}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--green);
    box-shadow:0 0 10px var(--green);animation:pulse 1.8s infinite}
  .dot.paused{background:var(--red);box-shadow:0 0 10px var(--red);animation:none}
  .serial{font-size:13px;font-weight:600;font-family:ui-monospace,monospace;letter-spacing:.4px}
  .batt{margin-left:auto;display:flex;align-items:center;gap:7px;font-size:12px}
  .batt .shell{width:34px;height:15px;border:1.5px solid #4a5c72;border-radius:3px;
    position:relative;padding:1.5px}
  .batt .shell:after{content:'';position:absolute;right:-4px;top:4px;width:2.5px;height:6px;
    background:#4a5c72;border-radius:0 2px 2px 0}
  .batt .fill{display:block;height:100%;border-radius:1px;transition:width .6s,background .6s}
  .batt .pct{font-family:ui-monospace,monospace;font-weight:700}
  .batt .bolt{color:var(--amber);filter:drop-shadow(0 0 4px rgba(251,191,36,.7))}
  .feed{width:100%;border-radius:11px;background:#000;display:block;
    border:1px solid rgba(255,255,255,.06)}
  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:13px 0 11px}
  .stat{background:rgba(255,255,255,.028);border:1px solid rgba(255,255,255,.05);
    border-radius:11px;padding:9px 6px;text-align:center}
  .stat .num{font-size:22px;font-weight:700;font-family:ui-monospace,monospace;line-height:1.1}
  .stat .lbl{font-size:9.5px;color:var(--muted);text-transform:uppercase;
    letter-spacing:1px;margin-top:3px}
  .stat.catch .num{color:var(--green)} .stat.waste .num{color:var(--amber)}
  .meta{font-size:11px;color:#8aa0ba;font-family:ui-monospace,monospace;
    display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  .chip{background:rgba(34,211,238,.13);color:var(--cyan);padding:2px 8px;border-radius:6px;
    font-size:10px;text-transform:uppercase;letter-spacing:.6px;font-weight:600}
  .ctrls{display:flex;gap:8px}
  .btn{flex:1;border:0;border-radius:10px;padding:10px;cursor:pointer;font-size:12px;
    font-weight:700;letter-spacing:.4px;transition:.15s}
  .btn.resume{background:linear-gradient(135deg,#059669,var(--green));color:#04140c}
  .btn.pause{background:rgba(248,113,113,.13);color:var(--red);
    border:1px solid rgba(248,113,113,.32)}
  .btn.mir{background:rgba(148,163,184,.14);color:#cbd5e1;
    border:1px solid rgba(148,163,184,.30)}
  .btn:hover{filter:brightness(1.12)}
</style></head>
<body>
<div class="hdr">
  <div class="logo">🎯</div>
  <div class="brand">
    <h1>PoGo Catcher</h1><div class="sub">Autonomous Capture Grid</div>
  </div>
  <span class="live"><span class="d"></span>Live</span>
  <span class="spacer"></span>
  <div class="kpi">
    <div class="k"><div class="kn catch" id="tcatch">0</div><div class="kl">catches</div></div>
    <div class="k"><div class="kn waste" id="twaste">0</div><div class="kl">wasted</div></div>
    <div class="k"><div class="kn" id="nphones">0</div><div class="kl">phones</div></div>
  </div>
  <a class="navbtn" href="/review">📝 Review<span class="count" id="ungraded">0</span></a>
  <a class="navbtn label" href="/label">🎯 Dense label</a>
</div>
<div class="grid" id="grid"></div>
<script>
function battHtml(s){
  if(s.battery==null) return '';
  const l=s.battery, col=l>50?'var(--green)':l>20?'var(--amber)':'var(--red)';
  return `<span class="shell"><span class="fill" style="width:${l}%;background:${col}"></span></span>`
    +`<span class="pct" style="color:${col}">${l}%</span>`
    +(s.charging?'<span class="bolt">⚡</span>':'');
}
function shell(serial){
  const d=document.createElement('div'); d.id='c-'+serial; d.className='card';
  d.innerHTML=`
    <div class="ctop"><span class="dot" id="dot-${serial}"></span>
      <span class="serial">${serial}</span>
      <span class="batt" id="batt-${serial}"></span></div>
    <img class="feed" src="/stream/${serial}.mjpg">
    <div class="stats">
      <div class="stat catch"><div class="num" id="ca-${serial}">0</div><div class="lbl">catches</div></div>
      <div class="stat waste"><div class="num" id="wa-${serial}">0</div><div class="lbl">wasted</div></div>
      <div class="stat"><div class="num" id="pa-${serial}">0</div><div class="lbl">panels</div></div>
    </div>
    <div class="meta" id="meta-${serial}"></div>
    <div class="ctrls">
      <button class="btn resume" onclick="ctl('${serial}','resume')">▶ Resume</button>
      <button class="btn pause" onclick="ctl('${serial}','pause')">⏸ Pause</button>
      <button class="btn mir" id="mir-${serial}"
              onclick="toggleMirror('${serial}')">🔌 Screen off</button>
    </div>`;
  return d;
}
async function refresh(){
  let st; try{ st=await (await fetch('/status')).json(); }catch(e){ return; }
  const grid=document.getElementById('grid'); let tc=0,tw=0,n=0;
  for(const [serial,s] of Object.entries(st)){
    tc+=s.catches; tw+=s.wasted; n++;
    if(!document.getElementById('c-'+serial)) grid.appendChild(shell(serial));
    document.getElementById('c-'+serial).className='card'+(s.paused?' paused':'');
    document.getElementById('dot-'+serial).className='dot'+(s.paused?' paused':'');
    document.getElementById('ca-'+serial).textContent=s.catches;
    document.getElementById('wa-'+serial).textContent=s.wasted;
    document.getElementById('pa-'+serial).textContent=s.panels;
    document.getElementById('batt-'+serial).innerHTML=battHtml(s);
    mirrorState[serial]=s.mirror;
    const mb=document.getElementById('mir-'+serial);
    if(mb) mb.textContent = s.mirror ? '🔌 Screen off' : '📺 Screen on';
    const state=(s.state||'').replace('ScreenState.','');
    document.getElementById('meta-'+serial).innerHTML=
      `<span class="chip">${state}</span><span>${s.src}</span>`
      +`<span>· ${s.pan_speed}px/s</span><span>· ${s.note}</span>`;
  }
  document.getElementById('tcatch').textContent=tc;
  document.getElementById('twaste').textContent=tw;
  document.getElementById('nphones').textContent=n;
}
async function refreshReview(){
  try{ const q=await (await fetch('/review/queue')).json();
    document.getElementById('ungraded').textContent=q.ungraded; }catch(e){}
}
const mirrorState={};
async function ctl(serial,action){ await fetch('/control/'+serial+'/'+action,{method:'POST'}); refresh(); }
function toggleMirror(serial){ ctl(serial, mirrorState[serial]===false ? 'mirror_on' : 'mirror_off'); }
setInterval(refresh,500); refresh();
setInterval(refreshReview,3000); refreshReview();
</script>
</body></html>"""


_REVIEW_PAGE = """<!DOCTYPE html>
<html><head><title>PoGo Catcher — Click Review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing:border-box; }
  body { background:#0e0f12; color:#e6e6e6; font-family:Segoe UI,Arial,sans-serif;
         margin:0; padding:16px; display:flex; flex-direction:column;
         align-items:center; min-height:100vh; }
  .bar { width:100%; max-width:720px; display:flex; align-items:center;
         gap:14px; margin-bottom:12px; }
  h1 { font-size:18px; margin:0; color:#7fd3a0; }
  a { color:#9ecbff; text-decoration:none; }
  .progress { margin-left:auto; font-size:13px; color:#bbb;
              font-variant-numeric:tabular-nums; }
  .progress b { color:#fff; }
  .card { width:100%; max-width:720px; background:#17181c;
          border:1px solid #2a2c31; border-radius:14px; padding:16px; }
  .card img { width:100%; border-radius:10px; background:#000; display:block; }
  .info { display:flex; gap:10px; align-items:center; margin:12px 2px 4px;
          font-size:12px; color:#999; }
  .badge { display:inline-block; padding:2px 10px; border-radius:9px;
           font-size:12px; font-weight:bold; color:#fff; }
  .b-encounter { background:#1c7a44; } .b-panel { background:#a33a26; }
  .b-nothing { background:#8a7a20; } .b-timeout { background:#555; }
  .actions { display:flex; gap:10px; margin-top:14px; }
  .actions button { flex:1; border:0; border-radius:9px; padding:16px;
    cursor:pointer; font-size:16px; font-weight:600; color:#fff; }
  .good { background:#1f9d55; } .good:hover { background:#25b463; }
  .bad  { background:#c0392b; } .bad:hover { background:#d8452f; }
  .skip { background:#3a3d44; flex:0 0 90px !important; }
  .reasons { margin-top:12px; display:none; grid-template-columns:1fr 1fr 1fr;
             gap:8px; }
  .reasons.show { display:grid; }
  .reasons button { border:0; border-radius:8px; padding:12px 8px;
    cursor:pointer; font-size:13px; color:#eee; background:#2c2f36; text-align:left; }
  .reasons button:hover { background:#3a3e47; }
  .reasons .k { color:#7fd3a0; font-weight:700; margin-right:6px; }
  .hint { margin-top:14px; font-size:12px; color:#777; text-align:center;
          max-width:720px; }
  .empty { text-align:center; padding:60px 20px; color:#7fd3a0; font-size:18px; }
</style></head>
<body>
<div class="bar">
  <h1>📝 Click review</h1>
  <a href="/">&larr; live</a>
  <a href="/label">🎯 dense label</a>
  <span class="progress" id="progress"></span>
</div>
<div id="stage" class="card"></div>
<div class="hint">Keys: <b>←</b> good · <b>→</b> bad, then <b>1–9</b> pick reason ·
  <b>↑ ↓</b> skip &nbsp;(also G / B / S). Bad + an object reason trains an
  "avoid" label; good on a missed spawn trains a "pokemon" label.</div>
<script>
const REASONS = ["player","pokestop","gym pokemon","gym","dynamax","raid icon",
                 "rocket","ui element","blank/nothing","other"];
let queue = [], cur = null, showingReasons = false;

async function fetchQueue() {
  const r = await fetch('/review/queue'); const q = await r.json();
  queue = q.items;
  document.getElementById('progress').innerHTML =
    `<b>${q.ungraded}</b> to grade · ${q.graded} done`;
  return q;
}
function render() {
  const stage = document.getElementById('stage');
  showingReasons = false;
  if (!cur) {
    stage.className = '';
    stage.innerHTML = '<div class="empty">✓ All caught up — no clicks to grade.<br>' +
      'New clicks appear here as the bot runs.</div>';
    return;
  }
  stage.className = 'card';
  stage.innerHTML =
    `<img src="/review/img/${cur.id}.jpg">
     <div class="info"><span class="badge b-${cur.outcome}">${cur.outcome}</span>
       <span>detector: ${cur.src}</span><span>${cur.id}</span></div>
     <div class="actions">
       <button class="good" onclick="vote('good',null)">✓ Good (←)</button>
       <button class="bad" onclick="toggleReasons()">✗ Bad (→)</button>
       <button class="skip" onclick="next()">Skip (↑↓)</button>
     </div>
     <div class="reasons" id="reasons">` +
       REASONS.map((x,i) => `<button onclick="vote('bad','${x}')">` +
         `<span class="k">${i+1}</span>${x}</button>`).join('') +
     `</div>`;
}
function toggleReasons() {
  showingReasons = !showingReasons;
  document.getElementById('reasons').classList.toggle('show', showingReasons);
}
function next() {
  queue.shift();
  cur = queue[0] || null;
  render();
  if (queue.length <= 2) fetchQueue().then(() => { if (!cur) { cur = queue[0]||null; render(); } });
}
async function vote(v, reason) {
  if (!cur) return;
  const id = cur.id;
  next();  // advance instantly; POST in the background for snappy grading
  fetch('/review/vote', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id, vote: v, reason})});
}
document.addEventListener('keydown', e => {
  if (!cur) return;
  const k = e.key;
  if (k === 'ArrowLeft' || k === 'g' || k === 'G') { e.preventDefault(); vote('good', null); }
  else if (k === 'ArrowRight' || k === 'b' || k === 'B') { e.preventDefault(); toggleReasons(); }
  else if (k === 'ArrowUp' || k === 'ArrowDown' || k === 's' || k === 'S') { e.preventDefault(); next(); }
  else if (/^[1-9]$/.test(k)) { e.preventDefault(); const i = +k - 1;
    if (i < REASONS.length) vote('bad', REASONS[i]); }
});
(async () => { await fetchQueue(); cur = queue[0] || null; render(); })();
setInterval(async () => { const before = queue.length;
  await fetchQueue(); if (!cur && queue.length) { cur = queue[0]; render(); } }, 5000);
</script>
</body></html>"""


_LABEL_PAGE = """<!DOCTYPE html>
<html><head><title>PoGo Catcher — Dense Label</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing:border-box; }
  body { background:#0e0f12; color:#e6e6e6; font-family:Segoe UI,Arial,sans-serif;
         margin:0; padding:16px; display:flex; flex-direction:column; align-items:center; }
  .bar { width:100%; max-width:900px; display:flex; align-items:center; gap:14px; }
  h1 { font-size:18px; margin:0; color:#7fd3a0; }
  a { color:#9ecbff; text-decoration:none; }
  .progress { margin-left:auto; font-size:13px; color:#bbb; }
  .progress b { color:#fff; }
  .wrap { position:relative; max-width:900px; margin-top:12px; touch-action:none; }
  canvas { width:100%; border-radius:10px; background:#000; cursor:crosshair;
           display:block; }
  .actions { width:100%; max-width:900px; display:flex; gap:10px; margin-top:12px; }
  .actions button { flex:1; border:0; border-radius:9px; padding:14px;
    cursor:pointer; font-size:15px; font-weight:600; color:#fff; }
  .save { background:#1f9d55; } .save:hover { background:#25b463; }
  .undo { background:#3a3d44; flex:0 0 130px; } .skip { background:#7a4a1e; flex:0 0 110px; }
  .hint { margin-top:12px; font-size:12px; color:#777; max-width:900px; text-align:center; }
  .empty { text-align:center; padding:60px; color:#7fd3a0; font-size:18px; }
</style></head>
<body>
<div class="bar">
  <h1>🎯 Dense label — box EVERY Pokémon</h1>
  <a href="/">live</a><a href="/review">grade</a>
  <span class="progress" id="progress"></span>
</div>
<div class="wrap"><canvas id="cv"></canvas></div>
<div class="actions">
  <button class="undo" onclick="undo()">Undo (U)</button>
  <button class="save" onclick="save()">Save &amp; next (Enter)</button>
  <button class="skip" onclick="skipFrame()">Skip (S)</button>
</div>
<div class="hint">Drag a box around <b>every</b> wild Pokémon — including the ones
  the model missed (yellow = model's guess, green = yours). Only wild spawns:
  skip gym/raid Pokémon and your buddy. <b>U</b> undo · <b>Enter</b> save · <b>S</b> skip.</div>
<script>
let frames = [], fi = 0, boxes = [], img = new Image(), drawing = null;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let seed = [];

async function loadFrames() {
  const r = await fetch('/label/frames'); frames = await r.json(); fi = 0;
  showFrame();
}
function showFrame() {
  const f = frames[fi];
  if (!f) { document.querySelector('.wrap').innerHTML =
      '<div class="empty">✓ Nothing left to hand-label right now.</div>';
    document.getElementById('progress').textContent = ''; return; }
  document.getElementById('progress').innerHTML =
    `frame <b>${fi+1}</b> / ${frames.length} · ${f.id}`;
  boxes = []; seed = f.seed || [];
  img = new Image();
  img.onload = () => { cv.width = f.frame[0]; cv.height = f.frame[1]; redraw(); };
  img.src = '/label/full/' + f.id + '.jpg';
}
function redraw() {
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  ctx.lineWidth = Math.max(3, cv.width/300);
  for (const b of seed) {   // model's boxes as faint yellow guides
    ctx.strokeStyle = 'rgba(255,210,0,0.7)';
    ctx.strokeRect((b[0]-b[2]/2)*cv.width, (b[1]-b[3]/2)*cv.height, b[2]*cv.width, b[3]*cv.height);
  }
  ctx.strokeStyle = '#22dd66';
  for (const b of boxes)
    ctx.strokeRect(b[0]*cv.width, b[1]*cv.height, b[2]*cv.width, b[3]*cv.height);
  if (drawing) { ctx.strokeStyle = '#88ffaa';
    ctx.strokeRect(drawing.x*cv.width, drawing.y*cv.height, drawing.w*cv.width, drawing.h*cv.height); }
}
function pos(e) {
  const r = cv.getBoundingClientRect();
  const t = e.touches ? e.touches[0] : e;
  return { x:(t.clientX-r.left)/r.width, y:(t.clientY-r.top)/r.height };
}
function down(e){ e.preventDefault(); const p=pos(e); drawing={x0:p.x,y0:p.y,x:p.x,y:p.y,w:0,h:0}; }
function move(e){ if(!drawing) return; e.preventDefault(); const p=pos(e);
  drawing.x=Math.min(p.x,drawing.x0); drawing.y=Math.min(p.y,drawing.y0);
  drawing.w=Math.abs(p.x-drawing.x0); drawing.h=Math.abs(p.y-drawing.y0); redraw(); }
function up(e){ if(!drawing) return; e.preventDefault();
  if(drawing.w>0.01 && drawing.h>0.01) boxes.push([drawing.x,drawing.y,drawing.w,drawing.h]);
  drawing=null; redraw(); }
cv.addEventListener('mousedown',down); cv.addEventListener('mousemove',move);
cv.addEventListener('mouseup',up);
cv.addEventListener('touchstart',down); cv.addEventListener('touchmove',move);
cv.addEventListener('touchend',up);
function undo(){ boxes.pop(); redraw(); }
function next(){ fi++; showFrame(); }
function skipFrame(){ next(); }
async function save(){
  const f = frames[fi]; if(!f) return;
  // convert [x,y,w,h] top-left-normalized -> [cx,cy,w,h] center-normalized
  const out = boxes.map(b => [b[0]+b[2]/2, b[1]+b[3]/2, b[2], b[3]]);
  await fetch('/label/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id:f.id, boxes:out})});
  next();
}
document.addEventListener('keydown', e => {
  if(e.key==='Enter') save();
  else if(e.key.toLowerCase()==='u') undo();
  else if(e.key.toLowerCase()==='s') skipFrame();
});
loadFrames();
</script>
</body></html>"""


class PhoneMonitor:
    """Shared state for one phone: the loop publishes, the server reads.

    The frame is JPEG-encoded ONCE per publish (not once per viewer per stream
    tick, as before) and handed to viewers through a condition variable, so the
    MJPEG feed pushes a new frame the instant the loop produces one -- no fixed
    fps cap, no redundant re-encodes. That is the bulk of the "5fps / laggy"
    fix: the old path re-encoded the same frame ~15x/s for every open tab."""

    _JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 72]

    def __init__(self, serial):
        self.serial = serial
        self.lock = threading.Lock()
        self._cond = threading.Condition(self.lock)
        self.pause_event = threading.Event()
        self._jpeg = None          # latest encoded frame (bytes)
        self._seq = 0              # bumps every publish -> wakes stream waiters
        self.state = "-"
        self.src = "-"
        self.note = "-"
        self.pan_speed = 0.0
        self.catches = 0
        self.wasted = 0
        self.panels = 0
        self.battery = None      # last polled battery %  (None until first poll)
        self.charging = False
        self.mirror = True       # False -> stream stopped + screen powered off

    def set_mirror(self, on):
        with self.lock:
            self.mirror = bool(on)

    def set_battery(self, level, charging):
        with self.lock:
            self.battery = level
            self.charging = charging

    def _store(self, small):
        ok, buf = cv2.imencode(".jpg", small, self._JPEG_PARAMS)
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def publish(self, img, state, target=None, fail_spots=(), pan_speed=0.0,
                note=None, tap=None):
        vis = img.copy()
        if target is not None:
            bx, by, bw, bh = target.bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 0, 255), 4)
            with self.lock:
                self.src = getattr(target, "src", "?")
        if tap is not None:
            cv2.circle(vis, (int(tap[0]), int(tap[1])), 26, (0, 255, 0), 5)
        for fx, fy, _exp in fail_spots:
            cv2.circle(vis, (int(fx), int(fy)), 40, (0, 200, 255), 3)
        cv2.putText(vis, str(state), (12, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (80, 240, 160), 3)
        small = cv2.resize(vis, (vis.shape[1] // 3, vis.shape[0] // 3))
        with self.lock:
            self.state = str(state)
            self.pan_speed = round(pan_speed, 1)
            if note:
                self.note = note
        self._store(small)

    def publish_raw(self, img):
        """Frame-only update (keeps the last state text) -- called from inside
        polling loops so the live feed stays smooth during encounters."""
        small = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
        self._store(small)

    def publish_off(self):
        """Show a 'mirroring off' card when the stream is stopped + screen slept."""
        img = np.full((260, 120, 3), 18, np.uint8)
        cv2.putText(img, "SCREEN", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (110, 110, 120), 2)
        cv2.putText(img, "OFF", (10, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (110, 110, 120), 2)
        with self.lock:
            self.state = "MIRROR OFF"
        self._store(img)

    def bump(self, outcome):
        with self.lock:
            if outcome == "encounter":
                self.catches += 1
            elif outcome == "panel":
                self.panels += 1
            else:
                self.wasted += 1
            self.note = outcome

    def jpeg(self):
        with self.lock:
            return self._jpeg

    def wait_frame(self, last_seq, timeout=2.0):
        """Block until a frame newer than `last_seq` is published (or timeout).
        Returns (bytes, seq). Event-driven: the stream sends exactly when the
        loop produces a frame, instead of polling on a timer."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            return self._jpeg, self._seq

    def status(self):
        with self.lock:
            return {
                "state": self.state, "src": self.src, "note": self.note,
                "pan_speed": self.pan_speed, "paused": self.pause_event.is_set(),
                "catches": self.catches, "wasted": self.wasted,
                "panels": self.panels,
                "battery": self.battery, "charging": self.charging,
                "mirror": self.mirror,
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
                elif path == "/review/queue" and review is not None:
                    graded, ungraded = review.counts()
                    body = {"items": review.queue(),
                            "graded": graded, "ungraded": ungraded}
                    self._send(200, "application/json", json.dumps(body).encode())
                elif path == "/label" and review is not None:
                    self._send(200, "text/html; charset=utf-8",
                               _LABEL_PAGE.encode())
                elif path == "/label/frames" and review is not None:
                    self._send(200, "application/json",
                               json.dumps(review.frames_for_labeling()).encode())
                elif (path.startswith("/label/full/") and path.endswith(".jpg")
                        and review is not None):
                    rid = path[len("/label/full/"):-len(".jpg")]
                    p = review.image_full_path(rid)
                    try:
                        with open(p, "rb") as f:
                            self._send(200, "image/jpeg", f.read())
                    except (OSError, TypeError):
                        self._send(404, "text/plain", b"gone")
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
                    seq = -1
                    try:
                        while True:
                            # Blocks until the loop publishes a NEW frame (or a
                            # 2s heartbeat), so the feed is as live as the loop
                            # and never resends a duplicate frame.
                            data, seq = pm.wait_frame(seq, timeout=2.0)
                            if data is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(
                                    f"Content-Length: {len(data)}\r\n\r\n".encode())
                                self.wfile.write(data)
                                self.wfile.write(b"\r\n")
                    except (ConnectionError, OSError):
                        return  # viewer closed the tab
                else:
                    self._send(404, "text/plain", b"not found")

            def do_POST(self):
                if self.path == "/label/save" and review is not None:
                    try:
                        n = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(n))
                        ok = review.save_boxes(body["id"], body.get("boxes", []))
                        self._send(200 if ok else 404, "text/plain",
                                   b"ok" if ok else b"unknown id")
                    except (ValueError, KeyError):
                        self._send(400, "text/plain", b"bad request")
                    return
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
                    action = parts[2]
                    if pm is not None and action in (
                            "pause", "resume", "mirror_off", "mirror_on"):
                        if action == "pause":
                            pm.pause_event.set()
                        elif action == "resume":
                            pm.pause_event.clear()
                        elif action == "mirror_off":
                            pm.set_mirror(False)
                        else:
                            pm.set_mirror(True)
                        self._send(200, "text/plain", b"ok")
                        return
                self._send(404, "text/plain", b"not found")

        server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
