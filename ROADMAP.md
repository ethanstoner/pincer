# Pincer — Roadmap

Where the project is and where it's going. Status legend: ✅ done · 🔨 in progress · ⏳ planned.

## Now (in progress)

- 🔨 **YOLO26 A/B.** Train `yolo26s` (and `yolo26m`, since inference isn't the bottleneck
  on the 4090) on the identical dataset and put them through the **same gate** as the live
  model. YOLO26's STAL + ProgLoss + NMS-free design targets exactly our small-object,
  dense-clutter recall problem; adopt it only if it beats the live model on *our* dense
  frames. See `docs/MODEL.md` "Alternatives".
- ⏳ **Richer supervision.** Keep hand-labeling hard event-VFX / stop-dense frames via the
  review UI — dense-frame recall is still ~20% absolute on the worst scenes, so this is
  the highest-leverage lever left.

## Next

- ⏳ **Runtime dependency slim-down.** Switch YOLO inference from Ultralytics to
  `onnxruntime` on the exported ONNX so `torch` leaves the runtime venv.
- ⏳ **Operational live stats.** Surface catch rate, catches/hour, empty-tap %, panel-tap %,
  and map→tap / tap→catch latency in the dashboard and README.

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
- ✅ **Retrain gated + deployed** on the full 5,417-frame / 2,238-vote dataset:
  **+13% val recall** (+27–38% on dense frames) over the model it replaced, on a hardened
  val split; prior weights kept as `best_prev.pt` for rollback.
- ✅ Self-supervised data flywheel: auto-labeled positives + hard-negative `avoid` boxes,
  with sibling-completion so dense frames aren't half-labeled.
- ✅ Human-in-the-loop review/voting UI (arrow-key grading; dedicated `player`/avatar
  avoid reason) feeding verified labels back into the dataset.
- ✅ Live MJPEG monitoring dashboard with battery meters + **stop/start mirroring**
  (screen power-off to charge paused phones faster).
- ✅ Multi-device orchestration (thread-per-phone, connected-device filter).
- ✅ Training speed hardening: batch/workers/disk-cache knobs; documented the 4090
  VRAM-spill cliff (batch ≤16 @ 1280px).
- ✅ 127 automated tests.
- ✅ Private GitHub repo + resume-grade docs (README, model card, architecture diagram).
