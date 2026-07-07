# Model Card — Pincer Target Detector

A fine-tuned **YOLO11-small** object detector that locates on-screen game targets in
live phone video, so the agent can decide where to tap. This card documents the data,
training setup, metrics, and the design decisions behind it.

## Task

Single-frame object detection on the game map screenshots (1080×2388, portrait).

| Class | ID | Meaning | Agent behavior |
|-------|----|---------|----------------|
| `pokemon` | 0 | A wild Pokémon sprite on the map | **Act on** — candidate tap target |
| `avoid`   | 1 | A UI element / landmark hitbox (gym, gym Pokémon, PokéStop, power spot, Rocket, raid icon, **player avatar / buddy**) | **Never tapped** — used to steer the tap point *away* |

The agent only ever acts on class-0 boxes. Class 1 exists so the model learns the
distractors that a naive detector taps by mistake, and so the tap-point selector can
place the tap on the target *farthest from* any nearby `avoid` box.

## Data

Collected entirely by the system operating on real devices — a **self-supervised data
flywheel** rather than a hand-annotated corpus.

- **5,417 labeled frames**, backed by **2,238 human review votes** (each Good/Bad+reason
  vote either corrects an auto-label or adds a hard case — including a dedicated
  `player`/avatar reason for a common mis-tap).
- **Positive labels** are written automatically on every confirmed catch (the tapped
  box is known-good ground truth).
- **Negative (`avoid`) labels** are written automatically whenever a tap opens a menu
  panel — a measured hard negative.
- **Human-in-the-loop review** (`src/review.py`): a browser UI surfaces every tap as a
  votable card; verified votes correct auto-labels and add hard cases back into the set.
- **Split:** 85 / 15 train / val, seeded and deterministic.
- **Dense-frame oversampling:** frames with ≥3 boxes (rare, high-value multi-object
  supervision) are duplicated ×12 **on the train split only**. The val split is never
  oversampled, so held-out metrics stay honest.

### Known limitations of the data

- Auto-labels are **sparse** — a confirmed-catch frame labels the one caught target,
  not every Pokémon visible. Volume averages this out; the review UI and hand-labeled
  dense frames add the multi-object signal.
- Distribution is biased toward the areas and times the bot has actually run.

## Training

Fine-tuned from a COCO-pretrained backbone with Ultralytics on an RTX 4090.

| Setting | Value | Why |
|---------|-------|-----|
| Base model | `yolo11s.pt` | Small beats nano on this dataset; still ~5–10 ms/frame |
| Image size | 1280 | Phone frames are tall; keeps small sprites ~40–60 px after resize |
| Epochs | 80 (live model) | Early-stop patience 40 |
| Batch | 16 | Fills the 4090 without spilling to system RAM at 1280 px |
| Augmentation | mosaic, h-flip, HSV jitter | Lighting/color variety |
| Augmentation **off** | rotation, shear, perspective | UI is axis-aligned — warping it hurts |

Output: a portable `best.pt` plus an ONNX export for lightweight inference. New models
train to a **separate run directory** and only replace the live model after clearing the
validation bar — the running agent never regresses silently.

## Metrics (held-out validation)

Both models below are scored on the **same, current** validation split. That split has
since been **hardened**: it now includes the human-verified dense/cluttered frames from
the review UI — a much tougher and more representative bar than the sparse, mostly
single-target auto-labeled val behind earlier figures. Absolute scores are therefore
lower than, and **not comparable to**, the pre-hardening numbers; what matters is the
head-to-head on the identical modern split.

Live model (`runs/pokemon`, YOLO11s, 80 epochs, 1280 px) vs. the model it replaced:

| Metric | Prior model | **Live model** | Δ |
|--------|-------------|----------------|-----|
| mAP@50 | 0.487 | **0.524** | +0.037 |
| mAP@50–95 | 0.400 | **0.401** | ~ |
| Precision | 0.681 | **0.629** | −0.052 |
| Recall | 0.478 | **0.541** | **+0.063 (+13%)** |

The live model trades a little precision for meaningfully higher **recall** — the right
call here: the agent's job is to *find* spawns, and the `avoid` class plus the tap-point
selector absorb the extra false positives. On **dense frames** (≥3 spawns) it detects
**+27–38% more** Pokémon than the prior model at equal confidence — the specific failure
mode ("clearly-visible spawns left un-tapped in crowded scenes") this retrain targeted.

The comparison is *conservative*: the prior model trained on a different split and had
seen some of these val frames, so its scores are if anything optimistic — the live model
wins anyway. New models are swapped in only after clearing this bar; the previous weights
are retained as `best_prev.pt` for instant rollback.

Inference latency ~5–10 ms on the RTX 4090 — negligible next to frame capture and the
game's own animation timing, which dominate the end-to-end loop.

## Alternatives considered

- **NVIDIA LocateAnything-3B (open-vocabulary localizer).** Evaluated directly and
  **rejected by measurement**: ~99 s/frame and it found zero game sprites
  (out-of-distribution for stylized game art). A purpose-trained small detector wins
  decisively when capture and animation latency dominate the loop anyway.
- **Classical CV (HSV + contour/shape filters).** The original detector; still present
  as an optional fallback. Works in moderate scenes but cannot segment sprites under
  heavy event VFX or in PokéStop-dense clutter — the failure mode YOLO fixes.

## Reproducing

```bash
# 1. Build the train/val split from the collected dataset
experiments/locate/venv/Scripts/python.exe training/prepare_split.py

# 2. Fine-tune (args: [epochs] [base_model] [run_name])
experiments/locate/venv/Scripts/python.exe training/train_yolo.py 120 yolo11s.pt pokemon_retrain

# 3. Gate on runs/<name>/results.csv, then point config.json at the new weights
```
