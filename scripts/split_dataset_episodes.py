#   MMSAR - step 4: train / val / test split (Option B, grouped by parent video)
#        Oumar Ibrahim U22200741

# the left and right tile of one source video always land in the same split
# rgb and thermal tiles of the same video also stay together (canonical parent)
# positives are spread by difficulty (target presence rate), negatives 60/20/20 with seed 0
# output  data/splits/dataset_episodes_split.json
#         data/splits/{train,val,test}_{rgb,thermal}.txt   (frame lists for YOLO)
#         configs/data_rgb.yaml, configs/data_thermal.yaml

import json
import random
import re
from pathlib import Path

import cv2
import yaml


# shuffle until sibling tiles are at least min_distance apart in the feed order
def shuffle_with_anti_adjacency(episodes, seed=0, min_distance=2, max_attempts=1000):
    rng = random.Random(seed)
    if len(episodes) <= 2:
        out = list(episodes)
        rng.shuffle(out)
        return out

    for attempt in range(max_attempts):
        cand = list(episodes)
        rng.shuffle(cand)

        positions = {}
        for idx in range(len(cand)):
            positions.setdefault(cand[idx]["canonical_parent_id"], []).append(idx)

        ok = True
        for parent in positions:
            idxs = positions[parent]
            for i in range(len(idxs) - 1):
                if abs(idxs[i + 1] - idxs[i]) < min_distance:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return cand

    out = list(episodes)
    rng.shuffle(out)
    return out


# target presence and box size from the label files of one clip
def compute_episode_metrics(labels_root, biome, scenario, modality, episode_id, frame_count=240):
    if scenario != "positive":
        return {
            "target_presence_rate": 0.0,
            "target_frame_count": 0,
            "mean_bbox_area_px": 0.0,
            "mean_bbox_width_px": 0.0,
            "mean_bbox_height_px": 0.0,
            "frame_coverage_pct": 0.0,
            "difficulty_flag": "empty_negative",
        }

    search_dirs = [
        labels_root / biome / "positive" / modality / episode_id,
        labels_root / biome / "positive" / modality,
        labels_root / episode_id,
    ]
    files = []
    for d in search_dirs:
        if d.is_dir():
            found = sorted(d.glob(f"{episode_id}_frame_*.txt"))
            if len(found) == 0:
                found = sorted(d.glob("frame_*.txt"))
            if len(found) > 0:
                files = found
                break
    if len(files) == 0:
        files = sorted(labels_root.glob(f"**/{episode_id}_frame_*.txt"))

    hits = 0
    areas = []
    widths = []
    heights = []
    for f in files:
        if f.stat().st_size > 0:
            hits += 1
            for line in f.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    w = float(parts[3]) * 640.0
                    h = float(parts[4]) * 640.0
                    widths.append(w)
                    heights.append(h)
                    areas.append(w * h)

    n = frame_count
    if n <= 0:
        n = len(files)
    if n < 1:
        n = 1
    presence = hits / n

    mean_area = 0.0
    mean_w = 0.0
    mean_h = 0.0
    if len(areas) > 0:
        mean_area = float(sum(areas) / len(areas))
        mean_w = float(sum(widths) / len(widths))
        mean_h = float(sum(heights) / len(heights))

    flag = "open_high_contrast"
    if presence < 0.70 or (mean_area > 0 and mean_area < 5000):
        flag = "occluded_low_contrast"

    return {
        "target_presence_rate": round(presence, 4),
        "target_frame_count": hits,
        "mean_bbox_area_px": round(mean_area, 1),
        "mean_bbox_width_px": round(mean_w, 1),
        "mean_bbox_height_px": round(mean_h, 1),
        "frame_coverage_pct": round((mean_area / (640.0 * 640.0)) * 100.0, 4),
        "difficulty_flag": flag,
    }


# at least one positive parent in val and in test, more when the pool is bigger
def derive_positive_split_floor(total_positive_parents, val_ratio=0.20):
    return max(1, int(total_positive_parents * val_ratio))


# easiest parents first, handed out val, test, val, test ... the rest goes to train
def allocate_positive_parents_dynamically(positive_parent_difficulties, val_ratio=0.20, test_ratio=0.20):
    n = len(positive_parent_difficulties)
    if n == 0:
        return {}

    floor = derive_positive_split_floor(n, val_ratio)
    ranked = sorted(positive_parent_difficulties.items(), key=presence_of, reverse=True)
    pool = []
    for p in ranked:
        pool.append(p[0])

    val_target = floor
    test_target = floor
    left = n - (val_target + test_target)
    if left >= 2:
        test_target += 1
        left -= 1
    if left >= 3:
        val_target += 1
        left -= 1

    val = []
    test = []
    toggle = 0
    while (len(val) < val_target or len(test) < test_target) and len(pool) > 0:
        if len(val) < val_target and toggle == 0:
            val.append(pool.pop(0))
        elif len(test) < test_target and toggle == 1:
            test.append(pool.pop(0))
        elif len(test) < test_target:
            test.append(pool.pop(0))
        elif len(val) < val_target:
            val.append(pool.pop(0))
        toggle = 1 - toggle

    assignment = {}
    for p in val:
        assignment[p] = "val"
    for p in test:
        assignment[p] = "test"
    for p in pool:
        assignment[p] = "train"
    return assignment


def presence_of(item):
    return item[1]


def allocate_negative_parents_dynamically(negative_parents, train_ratio=0.60, val_ratio=0.20, seed=0):
    rng = random.Random(seed)
    parents = sorted(negative_parents)
    rng.shuffle(parents)

    n = len(parents)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = 0
    if n >= 3:
        n_val = max(1, int(round(n * val_ratio)))
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 2:
        n_test = 1
        if n_train > 1:
            n_train -= 1

    assignment = {}
    for p in parents[:n_train]:
        assignment[p] = "train"
    for p in parents[n_train : n_train + n_val]:
        assignment[p] = "val"
    for p in parents[n_train + n_val :]:
        assignment[p] = "test"
    return assignment


def split_episodes(
    raw_root=Path("data/raw"),
    labels_root=Path("data/processed/labels"),
    splits_out_dir=Path("data/splits"),
    train_ratio=0.60,
    val_ratio=0.20,
    test_ratio=0.20,
    seed=0,
):
    splits_out_dir.mkdir(parents=True, exist_ok=True)

    by_parent = {}
    ep_metrics = {}
    for vid in sorted(raw_root.glob("*/*/*/*.mp4")):
        stem = vid.stem
        if stem.endswith("_left"):
            parent = stem[:-5]
            tile = "left"
        elif stem.endswith("_right"):
            parent = stem[:-6]
            tile = "right"
        else:
            continue

        modality = vid.parent.name
        scenario = vid.parent.parent.name
        biome = vid.parent.parent.parent.name
        # desert_RGB_positive_1 and desert_Thermal_positive_1 -> desert_positive_1
        canonical = re.sub(r"_(RGB|Thermal)_", "_", parent)

        cap = cv2.VideoCapture(str(vid))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n_frames <= 0:
            n_frames = 240

        ep_id = f"{parent}_{tile}"
        ep = {
            "episode_id": ep_id,
            "parent_video_id": parent,
            "canonical_parent_id": canonical,
            "tile_position": tile,
            "biome": biome,
            "scenario": scenario,
            "modality": modality,
            "frame_count": n_frames,
            "start_frame": 0,
            "end_frame": n_frames - 1,
            "fps": 24,
            "video_path": str(vid.as_posix()),
        }
        by_parent.setdefault(canonical, []).append(ep)
        ep_metrics[ep_id] = compute_episode_metrics(labels_root, biome, scenario, modality, ep_id, n_frames)

    # positive parents are ranked by the mean rgb presence rate of their tiles
    pos_parents = {}
    neg_parents = {}
    for canonical in sorted(by_parent):
        eps = by_parent[canonical]
        biome = eps[0]["biome"]
        if eps[0]["scenario"] == "positive":
            rates = []
            for e in eps:
                if e["modality"] == "rgb":
                    rates.append(ep_metrics[e["episode_id"]]["target_presence_rate"])
            pos_parents.setdefault(biome, {})[canonical] = sum(rates) / max(1, len(rates))
        else:
            neg_parents.setdefault(biome, []).append(canonical)

    assignment = {}
    floors = {}
    for biome in pos_parents:
        floors[biome] = derive_positive_split_floor(len(pos_parents[biome]), val_ratio)
        assignment.update(allocate_positive_parents_dynamically(pos_parents[biome], val_ratio, test_ratio))
    for biome in neg_parents:
        assignment.update(allocate_negative_parents_dynamically(neg_parents[biome], train_ratio, val_ratio, seed))

    rgb_split = {"train": [], "val": [], "test": []}
    th_split = {"train": [], "val": [], "test": []}
    for canonical in sorted(by_parent):
        name = assignment.get(canonical, "train")
        for ep in by_parent[canonical]:
            if ep["modality"] == "rgb":
                rgb_split[name].append(ep)
            else:
                th_split[name].append(ep)

    all_splits = {"train": [], "val": [], "test": []}
    for name in ["train", "val", "test"]:
        for group in [rgb_split[name], th_split[name]]:
            ordered = shuffle_with_anti_adjacency(group, seed=seed, min_distance=2)
            for i in range(len(ordered)):
                ep = dict(ordered[i])
                ep["feed_order_index"] = i
                ep["metrics"] = ep_metrics.get(ep["episode_id"], {})
                all_splits[name].append(ep)

    balance = {}
    for name in all_splits:
        eps = all_splits[name]
        n_rgb = 0
        rates = []
        for e in eps:
            if e["modality"] == "rgb":
                n_rgb += 1
                if e["scenario"] == "positive":
                    rates.append(e["metrics"]["target_presence_rate"])
        mean_rate = 0.0
        if len(rates) > 0:
            mean_rate = sum(rates) / len(rates)
        balance[name] = {
            "total_episodes_rgb": n_rgb,
            "positive_episodes_rgb": len(rates),
            "mean_target_presence_rate": round(mean_rate, 4),
        }

    total = 0
    for name in all_splits:
        total += len(all_splits[name])

    summary = {
        "strategy": "Option B: Parent-Video Grouped Episodic Partitioning",
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "seed": seed,
        "anti_adjacency_min_distance": 2,
        "dynamic_floors_per_biome": floors,
        "total_canonical_parent_videos": len(by_parent),
        "total_episodes": total,
        "partitions": {
            "train_count": len(all_splits["train"]),
            "val_count": len(all_splits["val"]),
            "test_count": len(all_splits["test"]),
        },
        "split_difficulty_balance": balance,
        "canonical_parent_partition_assignment": assignment,
        "episodes": all_splits,
    }

    out_file = splits_out_dir / "dataset_episodes_split.json"
    out_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_yolo_lists(all_splits, splits_out_dir)
    return summary


# one repo-relative image path per line, in feed order, plus the two dataset yaml files
# (relative paths keep the lists portable; scripts are always run from the repo root)
def write_yolo_lists(all_splits, splits_out_dir):
    images = Path("data/processed/images")
    counts = {}
    for name in ["train", "val", "test"]:
        for mod in ["rgb", "thermal"]:
            paths = []
            for ep in all_splits[name]:
                if ep["modality"] != mod:
                    continue
                folder = images / ep["biome"] / ep["scenario"] / mod
                for f in sorted(folder.glob(f"{ep['episode_id']}_frame_*.png")):
                    paths.append(f.as_posix())
            out = splits_out_dir / f"{name}_{mod}.txt"
            out.write_text("\n".join(paths) + "\n", encoding="utf-8")
            counts[(name, mod)] = len(paths)

    Path("configs").mkdir(exist_ok=True)
    # "path" is filled in with the absolute repo root by train_full_detectors.py at run time
    for mod in ["rgb", "thermal"]:
        cfg = {
            "path": ".",
            "train": f"data/splits/train_{mod}.txt",
            "val": f"data/splits/val_{mod}.txt",
            "test": f"data/splits/test_{mod}.txt",
            "names": {0: "Person_Detected"},
        }
        with open(f"configs/data_{mod}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    print("split    rgb frames   thermal frames")
    for name in ["train", "val", "test"]:
        print(f"{name:<8} {counts[(name, 'rgb')]:<12} {counts[(name, 'thermal')]}")


def main():
    summary = split_episodes()
    print()
    print("----------------------")
    print("Parent videos= ", summary["total_canonical_parent_videos"])
    print("Train tiles= ", summary["partitions"]["train_count"])
    print("Val tiles= ", summary["partitions"]["val_count"])
    print("Test tiles= ", summary["partitions"]["test_count"])
    print("----------------------")


if __name__ == "__main__":
    main()
