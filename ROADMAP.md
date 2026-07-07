# Pincer — Roadmap

Where the project is and where it's going. Status legend: ✅ done · 🔨 in progress · ⏳ planned.

## Now (in progress)

- 🔨 **Retrain the detector on the full 5,445-frame, 2-class dataset**
  (`training/runs/pokemon_retrain/`, YOLO11s @ 1280px). Gate on held-out val metrics
  before it replaces the live model.
- ⏳ **Publish real benchmark stats** once that retrain is gated — a stats/accuracy table
  in the README and `docs/MODEL.md`:
  - **Model:** mAP@50, mAP@50-95, precision, recall (held-out val).
  - **Operational (live):** catch rate, catches/hour, empty-tap %, panel-tap %,
    mean map→tap and tap→catch latency.
  - *(Deliberately left blank until measured — see `HANDOFF.md` §0.)*

## Next

- ⏳ **Sustained two-phone live run.** Validate parallel worker threads, no dataset
  write collisions, and both devices catching in a real session.
- ⏳ **Runtime dependency slim-down.** Switch YOLO inference from Ultralytics to
  `onnxruntime` on the exported ONNX so `torch` leaves the runtime venv.
- ⏳ **Richer supervision.** Multi-box hand-labeling of hard event-VFX / stop-dense
  frames to lift recall in the worst scenes.

## Later

- ⏳ **UI: web → desktop app.** Move the live monitor + review UI from the stdlib
  web servers to a packaged desktop app (framework TBD). *(Design decision pending.)*
- ⏳ **Continuous evaluation.** A held-out "hard scenes" benchmark set that every new
  model is scored against automatically, so regressions can't ship.
- ⏳ **One-command setup / packaging** for a clean machine (deps, ADB, model download).

## Done

- ✅ Core perception→decision→actuation loop; wall-clock polling; verify-before-throw
  safety invariants (enforced + tested).
- ✅ Real-time I/O: continuous H.264 stream (600 ms → 1.3 ms capture); persistent input
  shell (200 ms → 20 ms taps); self-healing respawns + screen wake.
- ✅ Fine-tuned YOLO11 detector **live**; classical-CV fallback retained.
- ✅ Self-supervised data flywheel: auto-labeled positives + hard-negative `avoid` boxes.
- ✅ Human-in-the-loop review/voting UI feeding verified labels back into the dataset.
- ✅ Live MJPEG monitoring dashboard.
- ✅ Multi-device orchestration (thread-per-phone, connected-device filter).
- ✅ 127 automated tests.
- ✅ Private GitHub repo + resume-grade docs (README, model card, architecture diagram).
