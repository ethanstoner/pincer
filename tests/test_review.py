import json
import os

import numpy as np

from src.detector import Target
from src.review import ReviewStore


def _img():
    return np.full((200, 100, 3), 60, np.uint8)


def test_record_and_recent_roundtrip(tmp_path):
    store = ReviewStore(str(tmp_path))
    rid = store.record(_img(), Target(x=50, y=100, bbox=(40, 90, 20, 20), src="yolo"),
                       "panel", result_img=_img())
    items = store.recent()
    assert items[0]["id"] == rid
    assert items[0]["outcome"] == "panel"
    assert items[0]["src"] == "yolo"
    assert items[0]["vote"] is None
    assert os.path.exists(store.image_path(rid))


def test_bad_vote_with_object_reason_writes_avoid_label(tmp_path):
    store = ReviewStore(str(tmp_path))
    rid = store.record(_img(), Target(x=50, y=100, bbox=(40, 90, 20, 20)), "panel")
    assert store.vote(rid, "bad", "gym") is True
    labels = os.listdir(tmp_path / "labels")
    assert any(n.startswith("avoid_") for n in labels)
    content = (tmp_path / "labels" / [n for n in labels if n.startswith("avoid_")][0]).read_text()
    assert content.startswith("1 ")          # class 1 = avoid
    votes = (tmp_path / "feedback" / "votes.jsonl").read_text().strip().splitlines()
    assert json.loads(votes[0])["reason"] == "gym"


def test_good_vote_on_missed_pokemon_writes_pokemon_label(tmp_path):
    store = ReviewStore(str(tmp_path))
    rid = store.record(_img(), Target(x=50, y=100, bbox=(40, 90, 20, 20)), "nothing")
    store.vote(rid, "good")
    labels = [n for n in os.listdir(tmp_path / "labels") if n.startswith("pokemon_")]
    assert len(labels) == 1
    assert (tmp_path / "labels" / labels[0]).read_text().startswith("0 ")


def test_bad_vote_blank_reason_writes_no_label(tmp_path):
    store = ReviewStore(str(tmp_path))
    rid = store.record(_img(), Target(x=50, y=100, bbox=(40, 90, 20, 20)), "nothing")
    store.vote(rid, "bad", "blank/nothing")
    assert not os.path.isdir(tmp_path / "labels")   # recorded, never labelled


def test_image_path_rejects_traversal(tmp_path):
    store = ReviewStore(str(tmp_path))
    assert store.image_path("../../etc/passwd") is None
    assert store.image_path("r000001") is not None
