"""Click review store: every tap becomes a votable card in the live UI.

For each attempted tap we keep (in dataset/review/, rotated):
    <rid>.jpg       the audit composite (tapped crop + result screen)
    <rid>_full.jpg  the full pre-click frame (for turning votes into labels)
    <rid>.json      meta: bbox, outcome, detector source, vote status

Votes land in dataset/feedback/votes.jsonl AND become training labels:
    bad + an object reason (gym / dynamax / raid icon / pokestop / ...) ->
        class-1 "avoid" label at the tapped bbox (human-verified hard negative)
    good on a `nothing` outcome (a missed real Pokemon) ->
        class-0 "pokemon" label (human-verified recall example)
Blank-space bad votes are recorded but NOT auto-labelled: an arbitrary grass
box taught as "avoid" would poison the class semantics.
"""
import json
import os
import threading

import cv2
import numpy as np

from src.detector import _LABEL_LOCK, _next_label_index

# reasons that mark a real avoidable OBJECT at the tapped box
AVOID_REASONS = {"gym", "gym pokemon", "dynamax", "raid icon", "pokestop",
                 "rocket", "ui element"}
_PAD = 100


class ReviewStore:
    def __init__(self, dataset_dir, keep=400):
        self.dataset_dir = dataset_dir
        self.dir = os.path.join(dataset_dir, "review")
        self.feedback_dir = os.path.join(dataset_dir, "feedback")
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.keep = keep
        self._lock = threading.Lock()
        self._n = self._scan_next()

    def _scan_next(self):
        highest = -1
        for name in os.listdir(self.dir):
            if name.endswith(".json"):
                try:
                    highest = max(highest, int(name[1:-5]))
                except ValueError:
                    pass
        return highest + 1

    # --- recording (called by the catch loop on every tap) -----------------
    def record(self, img, target, outcome, result_img=None):
        h, w = img.shape[:2]
        bx, by, bw, bh = target.bbox
        x0, y0 = max(0, bx - _PAD), max(0, by - _PAD)
        x1, y1 = min(w, bx + bw + _PAD), min(h, by + bh + _PAD)
        crop = img[y0:y1, x0:x1].copy()
        cv2.rectangle(crop, (bx - x0, by - y0), (bx - x0 + bw, by - y0 + bh),
                      (0, 0, 255), 3)
        if result_img is not None:
            ch = crop.shape[0]
            rw = max(1, int(result_img.shape[1] * ch / result_img.shape[0]))
            crop = np.hstack([crop, np.full((ch, 6, 3), 255, np.uint8),
                              cv2.resize(result_img, (rw, ch))])

        with self._lock:
            rid = f"r{self._n:06d}"
            self._n += 1
            cv2.imwrite(os.path.join(self.dir, rid + ".jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 78])
            cv2.imwrite(os.path.join(self.dir, rid + "_full.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
            meta = {"id": rid, "outcome": outcome, "bbox": [bx, by, bw, bh],
                    "frame": [w, h], "src": getattr(target, "src", "?"),
                    "vote": None, "reason": None}
            with open(os.path.join(self.dir, rid + ".json"), "w") as f:
                json.dump(meta, f)
            self._rotate()
        return rid

    def _rotate(self):
        metas = sorted(n for n in os.listdir(self.dir) if n.endswith(".json"))
        excess = len(metas) - self.keep
        for name in metas[:max(0, excess)]:
            rid = name[:-5]
            for suffix in (".json", ".jpg", "_full.jpg"):
                try:
                    os.remove(os.path.join(self.dir, rid + suffix))
                except OSError:
                    pass

    # --- reading (the review page) -----------------------------------------
    def recent(self, n=60):
        metas = sorted((name for name in os.listdir(self.dir)
                        if name.endswith(".json")), reverse=True)[:n]
        out = []
        for name in metas:
            try:
                with open(os.path.join(self.dir, name)) as f:
                    out.append(json.load(f))
            except (OSError, ValueError):
                pass
        return out

    def image_path(self, rid):
        if not rid.startswith("r") or not rid[1:].isdigit():
            return None  # no path tricks
        return os.path.join(self.dir, rid + ".jpg")

    # --- voting -> labels ----------------------------------------------------
    def vote(self, rid, vote, reason=None):
        meta_path = os.path.join(self.dir, rid + ".json")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return False
        meta["vote"] = vote
        meta["reason"] = reason
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        with open(os.path.join(self.feedback_dir, "votes.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")

        full = cv2.imread(os.path.join(self.dir, rid + "_full.jpg"))
        if full is None:
            return True
        bx, by, bw, bh = meta["bbox"]
        h, w = full.shape[:2]
        cx, cy = (bx + bw / 2.0) / w, (by + bh / 2.0) / h

        if vote == "bad" and (reason or "").lower() in AVOID_REASONS:
            self._write_label(full, 1, "avoid_", cx, cy, bw / w, bh / h)
        elif vote == "good" and meta["outcome"] == "nothing":
            # a missed REAL Pokemon: human confirms the box -> recall example
            self._write_label(full, 0, "pokemon_", cx, cy, bw / w, bh / h)
        return True

    def _write_label(self, img, cls, prefix, cx, cy, nw, nh):
        images_dir = os.path.join(self.dataset_dir, "images")
        labels_dir = os.path.join(self.dataset_dir, "labels")
        with _LABEL_LOCK:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            index = _next_label_index(images_dir, prefix=prefix)
            stem = f"{prefix}{index:06d}"
            cv2.imwrite(os.path.join(images_dir, stem + ".png"), img)
            with open(os.path.join(labels_dir, stem + ".txt"), "w") as f:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
