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

# reasons that mark a real avoidable OBJECT at the tapped box -> class-1 label.
# "player" = the trainer avatar / buddy in the map center: a frequent mis-tap
# Ethan flagged, and teaching it as an explicit avoid box stops the model tapping
# it (it sits at a near-fixed spot, so it's a strong, learnable negative).
AVOID_REASONS = {"gym", "gym pokemon", "dynamax", "raid icon", "pokestop",
                 "rocket", "ui element", "player"}
_PAD = 100


class ReviewStore:
    # Cards that need HUMAN judgment. An 'encounter' outcome is self-evidently a
    # correct detection (the tap opened a catch) and is already auto-labelled a
    # pokemon on the catch path, so it never enters the grading queue -- grading
    # it would be busy-work. 'nothing'/'panel'/'timeout' are the ambiguous ones.
    GRADABLE_OUTCOMES = {"nothing", "panel", "timeout"}

    def __init__(self, dataset_dir, keep=5000):
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
                    "siblings": [list(s) for s in getattr(target, "siblings", ())],
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

    def queue(self, n=40):
        """Oldest UNGRADED, GRADABLE cards first -- the grading queue feeds one
        at a time from the front, so votes never re-show and Ethan grades without
        hunting or scrolling. Known-correct 'encounter' cards are skipped (see
        GRADABLE_OUTCOMES). Newest-first `recent()` still backs the gallery."""
        names = sorted(name for name in os.listdir(self.dir)
                       if name.endswith(".json"))
        out = []
        for name in names:
            try:
                with open(os.path.join(self.dir, name)) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            if (meta.get("vote") is None
                    and meta.get("outcome") in self.GRADABLE_OUTCOMES):
                out.append(meta)
                if len(out) >= n:
                    break
        return out

    def counts(self):
        """(graded, ungraded) where `ungraded` counts only GRADABLE cards still
        needing a vote -- the number the dashboard badge and progress show."""
        graded = ungraded = 0
        for name in os.listdir(self.dir):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.dir, name)) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            if meta.get("vote") is not None:
                graded += 1
            elif meta.get("outcome") in self.GRADABLE_OUTCOMES:
                ungraded += 1
        return graded, ungraded

    def image_path(self, rid):
        if not rid.startswith("r") or not rid[1:].isdigit():
            return None  # no path tricks
        return os.path.join(self.dir, rid + ".jpg")

    def image_full_path(self, rid):
        """Path to the FULL pre-click frame (for the dense hand-label canvas)."""
        if not rid.startswith("r") or not rid[1:].isdigit():
            return None
        return os.path.join(self.dir, rid + "_full.jpg")

    # --- dense hand-labelling (box EVERY spawn in a crowded frame) -----------
    def frames_for_labeling(self, n=60):
        """Newest full-frames not yet hand-labelled -- the source for the
        'box every Pokemon' pass that fixes zero-recall dense scenes. Each entry
        carries the model's own boxes as a starting point so Ethan only adds the
        ones it MISSED instead of drawing from scratch."""
        names = sorted((name for name in os.listdir(self.dir)
                        if name.endswith(".json")), reverse=True)
        out = []
        for name in names:
            try:
                with open(os.path.join(self.dir, name)) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            if meta.get("hand_labeled"):
                continue
            if not os.path.exists(os.path.join(self.dir, meta["id"] + "_full.jpg")):
                continue
            w, h = meta.get("frame", [1, 1])
            seed = []  # model boxes, normalized (cx,cy,nw,nh), as a starting set
            bx, by, bw, bh = meta["bbox"]
            seed.append([(bx + bw / 2.0) / w, (by + bh / 2.0) / h, bw / w, bh / h])
            for sx, sy, sw, sh in meta.get("siblings", []):
                seed.append([(sx + sw / 2.0) / w, (sy + sh / 2.0) / h, sw / w, sh / h])
            out.append({"id": meta["id"], "outcome": meta["outcome"],
                        "frame": [w, h], "seed": seed})
            if len(out) >= n:
                break
        return out

    def save_boxes(self, rid, boxes):
        """Persist a full hand-labelled frame: EVERY box (normalized [cx,cy,w,h])
        written as class 0 on the full frame, and the card marked hand_labeled so
        it leaves the label queue. Empty box list still marks it done (a frame
        with genuinely no spawns is a valid all-background example -- but we skip
        writing an empty label file, which YOLO reads as 'no objects')."""
        meta_path = os.path.join(self.dir, rid + ".json")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return False
        full = cv2.imread(os.path.join(self.dir, rid + "_full.jpg"))
        if full is None:
            return False
        rows = [(float(cx), float(cy), float(nw), float(nh))
                for cx, cy, nw, nh in boxes]
        if rows:
            self._write_label(full, 0, "pokemon_", rows)
        meta["hand_labeled"] = True
        meta["hand_boxes"] = len(rows)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        return True

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

        def norm(box):
            x, y, ww, hh = box
            return ((x + ww / 2.0) / w, (y + hh / 2.0) / h, ww / w, hh / h)

        if vote == "bad" and (reason or "").lower() in AVOID_REASONS:
            self._write_label(full, 1, "avoid_", [norm([bx, by, bw, bh])])
        elif vote == "good" and meta["outcome"] == "nothing":
            # a missed REAL Pokemon: human confirms the box -> recall example.
            # Complete the frame with the model's OTHER confident spawns so this
            # dense scene isn't half-labelled (same fix as the catch path).
            rows = [norm([bx, by, bw, bh])]
            rows += [norm(s) for s in meta.get("siblings", [])]
            self._write_label(full, 0, "pokemon_", rows)
        return True

    def _write_label(self, img, cls, prefix, rows):
        """Write one training image + a multi-row YOLO label file (each row a
        normalized (cx, cy, w, h) box of class `cls`)."""
        images_dir = os.path.join(self.dataset_dir, "images")
        labels_dir = os.path.join(self.dataset_dir, "labels")
        with _LABEL_LOCK:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            index = _next_label_index(images_dir, prefix=prefix)
            stem = f"{prefix}{index:06d}"
            cv2.imwrite(os.path.join(images_dir, stem + ".png"), img)
            with open(os.path.join(labels_dir, stem + ".txt"), "w") as f:
                for cx, cy, nw, nh in rows:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
