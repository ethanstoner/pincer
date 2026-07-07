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


def test_player_bad_vote_writes_avoid_label(tmp_path):
    # 'player' (avatar/buddy mis-tap) is an object reason -> class-1 avoid label
    store = ReviewStore(str(tmp_path))
    rid = store.record(_img(), Target(x=50, y=100, bbox=(40, 90, 20, 20)), "nothing")
    assert store.vote(rid, "bad", "player") is True
    labels = [n for n in os.listdir(tmp_path / "labels") if n.startswith("avoid_")]
    assert len(labels) == 1
    assert (tmp_path / "labels" / labels[0]).read_text().startswith("1 ")


def test_good_vote_on_missed_pokemon_writes_pokemon_label(tmp_path):
    store = ReviewStore(str(tmp_path))
    rid = store.record(_img(), Target(x=50, y=100, bbox=(40, 90, 20, 20)), "nothing")
    store.vote(rid, "good")
    labels = [n for n in os.listdir(tmp_path / "labels") if n.startswith("pokemon_")]
    assert len(labels) == 1
    assert (tmp_path / "labels" / labels[0]).read_text().startswith("0 ")


def test_good_vote_completes_dense_frame_with_siblings(tmp_path):
    store = ReviewStore(str(tmp_path))
    target = Target(x=50, y=100, bbox=(40, 90, 20, 20),
                    siblings=((10, 10, 15, 15), (70, 150, 18, 18)))
    rid = store.record(_img(), target, "nothing")
    store.vote(rid, "good")
    labels = [n for n in os.listdir(tmp_path / "labels") if n.startswith("pokemon_")]
    content = (tmp_path / "labels" / labels[0]).read_text().splitlines()
    assert len(content) == 3                       # caught box + 2 siblings
    assert all(ln.startswith("0 ") for ln in content)


def test_bad_vote_blank_reason_writes_no_label(tmp_path):
    store = ReviewStore(str(tmp_path))
    rid = store.record(_img(), Target(x=50, y=100, bbox=(40, 90, 20, 20)), "nothing")
    store.vote(rid, "bad", "blank/nothing")
    assert not os.path.isdir(tmp_path / "labels")   # recorded, never labelled


def test_image_path_rejects_traversal(tmp_path):
    store = ReviewStore(str(tmp_path))
    assert store.image_path("../../etc/passwd") is None
    assert store.image_path("r000001") is not None


def test_queue_is_oldest_ungraded_first_and_hides_voted(tmp_path):
    store = ReviewStore(str(tmp_path))
    t = Target(x=50, y=100, bbox=(40, 90, 20, 20))
    r0 = store.record(_img(), t, "panel")
    r1 = store.record(_img(), t, "nothing")
    r2 = store.record(_img(), t, "panel")
    # grade the middle one -> it must drop out of the queue
    store.vote(r1, "good")
    q = store.queue()
    ids = [m["id"] for m in q]
    assert ids == [r0, r2]              # oldest-first, voted one gone
    assert store.counts() == (1, 2)    # (graded, ungraded)
