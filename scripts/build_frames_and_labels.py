#   MMSAR - step 3: unpack every tile into frames and YOLO labels
#        Oumar Ibrahim U22200741

# positive clips: boxes come from the Label Studio video export (interpolated per frame)
# negative clips: every frame gets an empty label file
# thermal frames reuse the rgb box, the two streams are pixel aligned
# output  data/processed/images/{biome}/{scenario}/{rgb|thermal}/{stem}_frame_000000.png
#         data/processed/labels/... same layout with .txt
# use --labels-only to rebuild just the labels

import copy
import json
import re
import sys
from pathlib import Path

import cv2

from mmsar.labels import box_to_yolo_line, frame_number, interpolate_video_sequence, task_sequence

EXPORT_JSON = Path("data/annotations/raw_exports/project_2_rgb_annotations_adjusted.json")
OUT_ROOT = Path("data/processed")


# some clips were tracked with more than one box, merge them into one track
def merge_tracks(task):
    task = copy.deepcopy(task)
    anns = task.get("annotations", [])
    if len(anns) == 0:
        return task
    results = anns[0].get("result", [])
    tracks = []
    others = []
    for r in results:
        if r.get("type") == "videorectangle":
            tracks.append(r)
        else:
            others.append(r)
    if len(tracks) <= 1:
        return task

    merged = copy.deepcopy(tracks[0])
    seq = list(merged.get("value", {}).get("sequence", []))
    for t in tracks[1:]:
        seq += t.get("value", {}).get("sequence", [])
    seq.sort(key=frame_number)

    # keep the first keyframe for every frame number
    seen = set()
    dedup = []
    for kf in seq:
        if kf.get("frame") not in seen:
            seen.add(kf.get("frame"))
            dedup.append(kf)
    merged["value"]["sequence"] = dedup
    anns[0]["result"] = [merged] + others
    return task


def load_tasks():
    tasks = json.loads(EXPORT_JSON.read_text(encoding="utf-8"))
    by_stem = {}
    for t in tasks:
        name = Path(t.get("data", {}).get("video", "")).name
        # label studio puts an 8 char hash in front of uploaded files
        stem = re.sub(r"^[0-9a-fA-F]{8}-", "", name).replace(".mp4", "")
        by_stem[stem] = merge_tracks(t)
    return by_stem


def frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# one label line per frame (empty string = no person)
def clip_labels(rgb_path, task):
    n = frame_count(rgb_path)
    lines = []
    if task is None:
        for i in range(n):
            lines.append("")
        return lines

    seq = task_sequence(task)
    boxes = {}
    if len(seq) > 0:
        boxes = interpolate_video_sequence(seq, total_frames=n)
    for i in range(n):
        # label studio frames start at 1
        box = boxes.get(i + 1)
        if box is None:
            lines.append("")
        else:
            lines.append(box_to_yolo_line(box))
    return lines


def write_clip(rgb_path, lines, labels_only):
    biome = rgb_path.parent.parent.parent.name
    scenario = rgb_path.parent.parent.name
    th_path = rgb_path.parent.parent / "thermal" / rgb_path.name.replace("RGB", "Thermal")
    if not th_path.exists():
        raise FileNotFoundError(f"missing thermal twin {th_path}")

    for mod, path in [("rgb", rgb_path), ("thermal", th_path)]:
        img_dir = OUT_ROOT / "images" / biome / scenario / mod
        lbl_dir = OUT_ROOT / "labels" / biome / scenario / mod
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(lines)):
            (lbl_dir / f"{path.stem}_frame_{i:06d}.txt").write_text(lines[i], encoding="utf-8")

        if labels_only:
            continue
        cap = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(img_dir / f"{path.stem}_frame_{i:06d}.png"), frame)
            i += 1
        cap.release()
        if i != len(lines):
            raise RuntimeError(f"{path}: {i} frames but {len(lines)} labels")


def main():
    labels_only = "--labels-only" in sys.argv
    tasks = load_tasks()

    n_clips = 0
    n_frames = 0
    n_target = 0
    for rgb_path in sorted(Path("data/raw").glob("*/*/rgb/*.mp4")):
        scenario = rgb_path.parent.parent.name
        task = None
        if scenario == "positive":
            task = tasks.get(rgb_path.stem)
            if task is None:
                raise KeyError(f"no annotation task for {rgb_path.stem}")

        lines = clip_labels(rgb_path, task)
        write_clip(rgb_path, lines, labels_only)

        n_clips += 1
        n_frames += len(lines)
        for line in lines:
            if line != "":
                n_target += 1
        print(f"{rgb_path.stem}  {len(lines)} frames")

    print()
    print("----------------------")
    print("Clips= ", n_clips)
    print("Frames per stream= ", n_frames)
    print("Target frames= ", n_target)
    print("----------------------")


if __name__ == "__main__":
    main()
