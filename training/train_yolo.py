"""Fine-tune YOLO11-nano to detect wild Pokemon on the game map.

Run with the experiments venv (has ultralytics + CUDA):
    experiments/locate/venv/Scripts/python.exe training/train_yolo.py [epochs]

Data: training/yolo_data/{images,labels}/{train,val} (build it with
prepare_split.py). Phone frames are tall (1080x2388), so we train at a large
imgsz to keep the small Pokemon sprites resolvable. Output: a portable best.pt
under training/runs/pokemon/weights/ plus an ONNX export for lightweight live
inference.
"""
import os
import sys

import yaml
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "yolo_data")
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120

cfg = {"path": DATA, "train": "images/train", "val": "images/val", "names": {0: "pokemon"}}
yaml_path = os.path.join(HERE, "_dataset.yaml")
with open(yaml_path, "w") as f:
    yaml.safe_dump(cfg, f)

model = YOLO("yolo11n.pt")  # nano; COCO-pretrained backbone -> fast fine-tune
model.train(
    data=yaml_path,
    epochs=EPOCHS,
    imgsz=1280,          # tall phone frames -> keep sprites ~40-60px after resize
    batch=8,
    device=0,
    project=os.path.join(HERE, "runs"),
    name="pokemon",
    exist_ok=True,
    patience=40,
    degrees=0, shear=0, perspective=0,   # UI is axis-aligned; don't warp it
    mosaic=1.0, fliplr=0.5, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,  # color/lighting variety
)

best = os.path.join(HERE, "runs", "pokemon", "weights", "best.pt")
print("best model:", best)
try:
    YOLO(best).export(format="onnx", imgsz=1280, dynamic=False, simplify=True)
    print("exported ONNX next to best.pt")
except Exception as e:
    print("ONNX export skipped:", e)
