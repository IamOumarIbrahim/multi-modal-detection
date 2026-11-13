#   MMSAR - tests for label interpolation and the Option B split rules
#        Oumar Ibrahim U22200741

from scripts.split_dataset_episodes import (
    allocate_negative_parents_dynamically,
    allocate_positive_parents_dynamically,
    derive_positive_split_floor,
)
from mmsar.labels import box_to_yolo_line, interpolate_video_sequence


def test_interpolation():
    seq = [
        {"frame": 3, "x": 10, "y": 20, "width": 10, "height": 10, "enabled": True},
        {"frame": 5, "x": 20, "y": 20, "width": 10, "height": 10, "enabled": False},
    ]
    boxes = interpolate_video_sequence(seq, total_frames=7)
    assert boxes[1] is None and boxes[2] is None
    assert boxes[3]["x"] == 10.0
    assert boxes[4]["x"] == 15.0
    # the track is switched off at frame 5 and stays off
    assert boxes[5] is None and boxes[7] is None


def test_yolo_line():
    line = box_to_yolo_line({"x": 10.0, "y": 20.0, "width": 20.0, "height": 40.0})
    assert line == "0 0.200000 0.400000 0.200000 0.400000\n"


def test_positive_floor():
    assert derive_positive_split_floor(2) == 1
    assert derive_positive_split_floor(10) == 2
    assert derive_positive_split_floor(25) == 5


def test_positive_allocation():
    pres = {"p1": 0.99, "p2": 0.95, "p3": 0.52, "p4": 0.93, "p5": 0.10}
    a = allocate_positive_parents_dynamically(pres)
    assert len(a) == 5
    splits = list(a.values())
    assert "train" in splits and "val" in splits and "test" in splits


def test_negative_allocation():
    a = allocate_negative_parents_dynamically([f"n{i}" for i in range(10)])
    counts = {"train": 0, "val": 0, "test": 0}
    for s in a.values():
        counts[s] += 1
    assert counts == {"train": 6, "val": 2, "test": 2}


def test_committed_split_has_no_leakage():
    import json
    from pathlib import Path

    split = json.loads(Path("data/splits/dataset_episodes_split.json").read_text(encoding="utf-8"))
    where = {}
    for name in ["train", "val", "test"]:
        for ep in split["episodes"][name]:
            p = ep["canonical_parent_id"]
            if p in where:
                assert where[p] == name
            where[p] = name
    assert split["partitions"] == {"train_count": 88, "val_count": 44, "test_count": 48}
