"""Split the flat collected dataset (dataset/images + dataset/labels, written by
detector.save_label) into a YOLO train/val layout under training/yolo_data/.

Usage: python prepare_split.py [src_dataset_dir] [out_dir] [val_fraction]

DENSE OVERSAMPLING: frames with many pokemon boxes (hand-labelled crowded scenes
+ sibling-completed catches) are the rare, high-value supervision that teaches
"multiple objects per frame" -- but they're a tiny fraction of the data and get
drowned by ~3000 single-box auto-labels. So each dense TRAIN frame (>= DENSE_MIN
boxes) is duplicated OVERSAMPLE times. This happens on the TRAIN split ONLY --
the val split is never oversampled, so held-out metrics stay honest.
"""
import os
import random
import shutil
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "dataset"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("training", "yolo_data")
VAL_FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

DENSE_MIN = 3        # a label file with >= this many boxes is a "dense" frame
OVERSAMPLE = 12      # extra copies of each dense train frame

img_src = os.path.join(SRC, "images")
lbl_src = os.path.join(SRC, "labels")
imgs = sorted(f for f in os.listdir(img_src) if f.lower().endswith(".png"))
random.Random(42).shuffle(imgs)
n_val = max(1, int(len(imgs) * VAL_FRAC)) if len(imgs) > 1 else 0
val = set(imgs[:n_val])

for split in ("train", "val"):
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

def _box_count(path):
    with open(path) as fh:
        return sum(1 for line in fh if line.strip())

kept = 0
dense_dupes = 0
for f in imgs:
    stem = os.path.splitext(f)[0]
    lbl = stem + ".txt"
    lbl_path = os.path.join(lbl_src, lbl)
    if not os.path.exists(lbl_path):
        continue  # skip unlabeled frames
    split = "val" if f in val else "train"
    shutil.copy(os.path.join(img_src, f), os.path.join(OUT, "images", split, f))
    shutil.copy(lbl_path, os.path.join(OUT, "labels", split, lbl))
    kept += 1

    # Oversample dense TRAIN frames so their multi-object signal isn't drowned.
    if split == "train" and _box_count(lbl_path) >= DENSE_MIN:
        for k in range(OVERSAMPLE):
            dup = f"{stem}_ov{k:02d}"
            shutil.copy(os.path.join(img_src, f),
                        os.path.join(OUT, "images", "train", dup + ".png"))
            shutil.copy(lbl_path, os.path.join(OUT, "labels", "train", dup + ".txt"))
            dense_dupes += 1

print(f"{kept} labeled images -> {OUT}  (val {len(val)}); "
      f"+{dense_dupes} dense oversample copies (>= {DENSE_MIN} boxes x{OVERSAMPLE})")
