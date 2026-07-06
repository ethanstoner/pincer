"""Split the flat collected dataset (dataset/images + dataset/labels, written by
detector.save_label) into a YOLO train/val layout under training/yolo_data/.

Usage: python prepare_split.py [src_dataset_dir] [out_dir] [val_fraction]
"""
import os
import random
import shutil
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "dataset"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("training", "yolo_data")
VAL_FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

img_src = os.path.join(SRC, "images")
lbl_src = os.path.join(SRC, "labels")
imgs = sorted(f for f in os.listdir(img_src) if f.lower().endswith(".png"))
random.Random(42).shuffle(imgs)
n_val = max(1, int(len(imgs) * VAL_FRAC)) if len(imgs) > 1 else 0
val = set(imgs[:n_val])

for split in ("train", "val"):
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

kept = 0
for f in imgs:
    stem = os.path.splitext(f)[0]
    lbl = stem + ".txt"
    if not os.path.exists(os.path.join(lbl_src, lbl)):
        continue  # skip unlabeled frames
    split = "val" if f in val else "train"
    shutil.copy(os.path.join(img_src, f), os.path.join(OUT, "images", split, f))
    shutil.copy(os.path.join(lbl_src, lbl), os.path.join(OUT, "labels", split, lbl))
    kept += 1

print(f"{kept} labeled images -> {OUT}  (train {kept - len(val & set(imgs))}, val {len(val)})")
