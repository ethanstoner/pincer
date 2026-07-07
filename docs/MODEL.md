# Model Card — Pincer Target Detector

A fine-tuned **YOLO11-small** object detector that locates on-screen game targets in
live phone video, so the agent can decide where to tap. This card documents the data,
training setup, metrics, and the design decisions behind it.

## Task

Single-frame object detection on the game map screenshots (1080×2388, portrait).

| Class | ID | Meaning | Agent behavior |
|-------|----|---------|----------------|
| `pokemon` | 0 | A wild Pokémon sprite on the map | **Act on** — candidate tap target |
| `avoid`   | 1 | A UI element / landmark hitbox (gym, PokéStop, power spot, Rocket, Route marker) | **Never tapped** — used to steer the tap point *away* |

The agent only ever acts on class-0 boxes. Class 1 exists so the model learns the
distractors that a naive detector taps by mistake, and so the tap-point selector can
place the tap on the target *farthest from* any nearby `avoid` box.

## Data

Collected entirely by the system operating on real devices — a **self-supervised data
flywheel** rather than a hand-annotated corpus.

- **5,445 labeled frames** — **4,589** `pokemon` boxes + **1,533** `avoid` boxes.
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

Live model (`runs/pokemon`, YOLO11s, 80 epochs, 1280 px):

| Metric | Value |
|--------|-------|
| mAP@50 | **0.84** |
| mAP@50–95 | **0.66** |
| Precision | **0.81** |
| Recall | **0.77** |

Inference latency ~5–10 ms on the RTX 4090 — negligible next to the ~1.3 ms frame
capture and the game's own animation timing, which dominate the end-to-end loop.

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
