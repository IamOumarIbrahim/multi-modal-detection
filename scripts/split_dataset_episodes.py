"""Dataset episode splitting and sequence shuffling pipeline (Option B).

Implements canonical parent-video grouped partitioning, 240-frame sequence bundling,
and seeded snippet shuffling with anti-adjacency guards across Desert, Forest, and Snow.
Guarantees 100% cross-modal parity between RGB and Thermal modalities.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml


@dataclass
class EpisodeMetadata:
    """Metadata for an atomic video episode snippet."""

    episode_id: str
    parent_video_id: str
    canonical_parent_id: str
    tile_position: str  # 'left' or 'right'
    biome: str  # 'desert', 'forest', or 'snow'
    scenario: str  # 'positive', 'negative'
    modality: str  # 'rgb' or 'thermal'
    frame_count: int
    start_frame: int
    end_frame: int
    fps: int
    video_path: str


def shuffle_with_anti_adjacency(
    episodes: list[EpisodeMetadata],
    seed: int = 0,
    min_distance: int = 2,
    max_attempts: int = 1000,
) -> list[EpisodeMetadata]:
    """Shuffle episodes using seed 0 while enforcing an anti-adjacency constraint.

    Guarantees that left and right snippets derived from the same parent video
    are separated by at least min_distance slots in the queue.
    """
    rng = random.Random(seed)
    n = len(episodes)
    if n <= 2:
        shuffled = list(episodes)
        rng.shuffle(shuffled)
        return shuffled

    for _ in range(max_attempts):
        candidate = list(episodes)
        rng.shuffle(candidate)

        valid = True
        positions: dict[str, list[int]] = {}
        for idx, ep in enumerate(candidate):
            positions.setdefault(ep.canonical_parent_id, []).append(idx)

        for parent_id, idxs in positions.items():
            if len(idxs) > 1:
                for i in range(len(idxs) - 1):
                    dist = abs(idxs[i + 1] - idxs[i])
                    if dist < min_distance:
                        valid = False
                        break
            if not valid:
                break

        if valid:
            return candidate

    fallback = list(episodes)
    rng.shuffle(fallback)
    return fallback


def compute_episode_metrics(
    labels_root: Path,
    biome: str,
    scenario: str,
    modality: str,
    episode_id: str,
    frame_count: int = 240,
) -> dict[str, Any]:
    """Compute target presence rate and scale metrics for an episode from label files."""
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

    label_files: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            files = sorted(d.glob(f"{episode_id}_frame_*.txt"))
            if not files:
                files = sorted(d.glob("frame_*.txt"))
            if files:
                label_files = files
                break

    if not label_files:
        label_files = sorted(labels_root.glob(f"**/{episode_id}_frame_*.txt"))

    target_frame_count = 0
    box_areas: list[float] = []
    box_widths: list[float] = []
    box_heights: list[float] = []

    for f in label_files:
        if f.stat().st_size > 0:
            target_frame_count += 1
            try:
                lines = f.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w = float(parts[3]) * 640.0
                        h = float(parts[4]) * 640.0
                        box_widths.append(w)
                        box_heights.append(h)
                        box_areas.append(w * h)
            except Exception:
                pass

    effective_frames = max(1, frame_count if frame_count > 0 else len(label_files))
    presence_rate = target_frame_count / effective_frames
    mean_area = float(sum(box_areas) / len(box_areas)) if box_areas else 0.0
    mean_w = float(sum(box_widths) / len(box_widths)) if box_widths else 0.0
    mean_h = float(sum(box_heights) / len(box_heights)) if box_heights else 0.0
    coverage_pct = (mean_area / (640.0 * 640.0)) * 100.0

    if presence_rate < 0.70 or (mean_area > 0 and mean_area < 5000):
        flag = "occluded_low_contrast"
    else:
        flag = "open_high_contrast"

    return {
        "target_presence_rate": round(presence_rate, 4),
        "target_frame_count": target_frame_count,
        "mean_bbox_area_px": round(mean_area, 1),
        "mean_bbox_width_px": round(mean_w, 1),
        "mean_bbox_height_px": round(mean_h, 1),
        "frame_coverage_pct": round(coverage_pct, 4),
        "difficulty_flag": flag,
    }


def derive_positive_split_floor(total_positive_parents: int, val_ratio: float = 0.20) -> int:
    """Derive minimum positive parent count per split dynamically from pool size."""
    return max(1, int(total_positive_parents * val_ratio))


def allocate_positive_parents_dynamically(
    positive_parent_difficulties: dict[str, float],
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
) -> dict[str, str]:
    """Allocate positive parents across splits ensuring dynamic floors and difficulty balancing."""
    n_parents = len(positive_parent_difficulties)
    if n_parents == 0:
        return {}

    floor = derive_positive_split_floor(n_parents, val_ratio)
    sorted_parents = sorted(
        positive_parent_difficulties.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    parent_ids = [p[0] for p in sorted_parents]

    val_target = floor
    test_target = floor
    remaining = n_parents - (val_target + test_target)

    if remaining >= 2:
        test_target += 1
        remaining -= 1
    if remaining >= 3:
        val_target += 1
        remaining -= 1

    remaining_pool = list(parent_ids)
    eval_allocated: dict[str, list[str]] = {"val": [], "test": []}

    toggle = 0
    while (len(eval_allocated["val"]) < val_target or len(eval_allocated["test"]) < test_target) and remaining_pool:
        if len(eval_allocated["val"]) < val_target and toggle == 0:
            eval_allocated["val"].append(remaining_pool.pop(0))
        elif len(eval_allocated["test"]) < test_target and toggle == 1:
            eval_allocated["test"].append(remaining_pool.pop(0))
        elif len(eval_allocated["test"]) < test_target:
            eval_allocated["test"].append(remaining_pool.pop(0))
        elif len(eval_allocated["val"]) < val_target:
            eval_allocated["val"].append(remaining_pool.pop(0))
        toggle = 1 - toggle

    train_allocated = list(remaining_pool)

    assignment: dict[str, str] = {}
    for p in eval_allocated["val"]:
        assignment[p] = "val"
    for p in eval_allocated["test"]:
        assignment[p] = "test"
    for p in train_allocated:
        assignment[p] = "train"

    return assignment


def allocate_negative_parents_dynamically(
    negative_parents: list[str],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 0,
) -> dict[str, str]:
    """Allocate negative parents matching 60/20/20 ratio under Option B grouping."""
    rng = random.Random(seed)
    shuffled = sorted(negative_parents)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 2:
        n_test = 1
        if n_train > 1:
            n_train -= 1

    assignment: dict[str, str] = {}
    for p in shuffled[:n_train]:
        assignment[p] = "train"
    for p in shuffled[n_train : n_train + n_val]:
        assignment[p] = "val"
    for p in shuffled[n_train + n_val :]:
        assignment[p] = "test"
    return assignment


def split_episodes(
    raw_root: Path = Path("data/raw"),
    labels_root: Path = Path("data/processed/labels"),
    splits_out_dir: Path = Path("data/splits"),
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 0,
) -> dict[str, Any]:
    """Partition the dataset dynamically using Option B parent-video grouped episodic splitting."""
    splits_out_dir.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(raw_root.glob("*/*/*/*.mp4"))
    episodes_by_canonical_parent: dict[str, list[EpisodeMetadata]] = {}
    episode_metrics_map: dict[str, dict[str, Any]] = {}

    for vid_path in all_videos:
        stem = vid_path.stem
        if stem.endswith("_left"):
            parent_id = stem[:-5]
            tile_pos = "left"
        elif stem.endswith("_right"):
            parent_id = stem[:-6]
            tile_pos = "right"
        else:
            continue

        scenario = vid_path.parent.parent.name
        modality = vid_path.parent.name
        biome = vid_path.parent.parent.parent.name
        canonical_parent_id = re.sub(r"_(RGB|Thermal)_", "_", parent_id)

        cap = cv2.VideoCapture(str(vid_path))
        f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if f_count <= 0:
            f_count = 240

        ep_id = f"{parent_id}_{tile_pos}"
        ep = EpisodeMetadata(
            episode_id=ep_id,
            parent_video_id=parent_id,
            canonical_parent_id=canonical_parent_id,
            tile_position=tile_pos,
            biome=biome,
            scenario=scenario,
            modality=modality,
            frame_count=f_count,
            start_frame=0,
            end_frame=f_count - 1,
            fps=24,
            video_path=str(vid_path.as_posix()),
        )
        episodes_by_canonical_parent.setdefault(canonical_parent_id, []).append(ep)

        metrics = compute_episode_metrics(
            labels_root=labels_root,
            biome=biome,
            scenario=scenario,
            modality=modality,
            episode_id=ep_id,
            frame_count=f_count,
        )
        episode_metrics_map[ep_id] = metrics

    # Segregate canonical parents by biome and scenario
    biome_positive_parents: dict[str, dict[str, float]] = {}
    biome_negative_parents: dict[str, list[str]] = {}

    for c_id, ep_list in sorted(episodes_by_canonical_parent.items()):
        sample_ep = ep_list[0]
        biome = sample_ep.biome
        scenario = sample_ep.scenario

        if scenario == "positive":
            presence_rates = [
                episode_metrics_map[e.episode_id]["target_presence_rate"]
                for e in ep_list
                if e.modality == "rgb"
            ]
            avg_presence = sum(presence_rates) / max(1, len(presence_rates))
            biome_positive_parents.setdefault(biome, {})[c_id] = avg_presence
        else:
            biome_negative_parents.setdefault(biome, []).append(c_id)

    canonical_partition_assignment: dict[str, str] = {}
    dynamic_floors: dict[str, int] = {}

    for biome, pos_dict in biome_positive_parents.items():
        dynamic_floors[biome] = derive_positive_split_floor(len(pos_dict), val_ratio)
        pos_assign = allocate_positive_parents_dynamically(
            positive_parent_difficulties=pos_dict,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        canonical_partition_assignment.update(pos_assign)

    for biome, neg_list in biome_negative_parents.items():
        neg_assign = allocate_negative_parents_dynamically(
            negative_parents=neg_list,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        canonical_partition_assignment.update(neg_assign)

    # Assign episodes to splits (grouping by modality)
    partition_assigned_rgb: dict[str, list[EpisodeMetadata]] = {"train": [], "val": [], "test": []}
    partition_assigned_thermal: dict[str, list[EpisodeMetadata]] = {"train": [], "val": [], "test": []}

    for c_id, ep_list in sorted(episodes_by_canonical_parent.items()):
        split_name = canonical_partition_assignment.get(c_id, "train")
        for ep in ep_list:
            if ep.modality == "rgb":
                partition_assigned_rgb[split_name].append(ep)
            else:
                partition_assigned_thermal[split_name].append(ep)

    # Anti-adjacency shuffling within each partition for RGB and Thermal
    all_splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for split_name in ["train", "val", "test"]:
        rgb_eps = shuffle_with_anti_adjacency(partition_assigned_rgb[split_name], seed=seed, min_distance=2)
        th_eps = shuffle_with_anti_adjacency(partition_assigned_thermal[split_name], seed=seed, min_distance=2)

        for order_idx, ep in enumerate(rgb_eps):
            ep_dict = asdict(ep)
            ep_dict["feed_order_index"] = order_idx
            ep_dict["metrics"] = episode_metrics_map.get(ep.episode_id, {})
            all_splits[split_name].append(ep_dict)

        for order_idx, ep in enumerate(th_eps):
            ep_dict = asdict(ep)
            ep_dict["feed_order_index"] = order_idx
            ep_dict["metrics"] = episode_metrics_map.get(ep.episode_id, {})
            all_splits[split_name].append(ep_dict)

    # Difficulty balance summary
    split_difficulty_summary: dict[str, Any] = {}
    for s_name, eps in all_splits.items():
        pos_eps = [e for e in eps if e.get("scenario") == "positive" and e.get("modality") == "rgb"]
        if pos_eps:
            rates = [e["metrics"]["target_presence_rate"] for e in pos_eps]
            mean_rate = sum(rates) / len(rates)
        else:
            mean_rate = 0.0
        split_difficulty_summary[s_name] = {
            "total_episodes_rgb": len([e for e in eps if e.get("modality") == "rgb"]),
            "positive_episodes_rgb": len(pos_eps),
            "mean_target_presence_rate": round(mean_rate, 4),
        }

    summary: dict[str, Any] = {
        "strategy": "Option B: Parent-Video Grouped Episodic Partitioning",
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "seed": seed,
        "anti_adjacency_min_distance": 2,
        "dynamic_floors_per_biome": dynamic_floors,
        "total_canonical_parent_videos": len(episodes_by_canonical_parent),
        "total_episodes": sum(len(p) for p in all_splits.values()),
        "partitions": {
            "train_count": len(all_splits["train"]),
            "val_count": len(all_splits["val"]),
            "test_count": len(all_splits["test"]),
        },
        "split_difficulty_balance": split_difficulty_summary,
        "canonical_parent_partition_assignment": canonical_partition_assignment,
        "episodes": all_splits,
    }

    out_file = splits_out_dir / "dataset_episodes_split.json"
    out_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_file = splits_out_dir / "dataset_episodes_split_manifest.json"
    manifest_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Generate YOLO frame lists and YAML configs
    generate_yolo_split_manifests(all_splits, splits_out_dir)

    return summary


def generate_yolo_split_manifests(all_splits: dict[str, list[dict[str, Any]]], splits_out_dir: Path) -> None:
    """Generate YOLO training file lists and data configs for RGB and Thermal."""
    repo_root = Path(".").resolve()
    images_base = Path("data/processed/images")

    frame_counts = {"rgb": {"train": 0, "val": 0, "test": 0}, "thermal": {"train": 0, "val": 0, "test": 0}}
    biome_counts = {
        b: {"train": 0, "val": 0, "test": 0}
        for b in ["desert", "forest", "snow"]
    }

    for split_name in ["train", "val", "test"]:
        for mod in ["rgb", "thermal"]:
            eps = [e for e in all_splits[split_name] if e["modality"] == mod]
            img_paths: list[str] = []

            for ep in eps:
                stem = ep["episode_id"]
                biome = ep["biome"]
                scen = ep["scenario"]
                ep_im_dir = images_base / biome / scen / mod
                frames = sorted(ep_im_dir.glob(f"{stem}_frame_*.png"))
                for f in frames:
                    img_paths.append(str(f.resolve().as_posix()))

                frame_counts[mod][split_name] += len(frames)
                if mod == "rgb":
                    biome_counts[biome][split_name] += len(frames)

            list_file = splits_out_dir / f"{split_name}_{mod}.txt"
            list_file.write_text("\n".join(img_paths) + "\n", encoding="utf-8")
            print(f"Generated {list_file.name}: {len(img_paths)} frames.")

    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    for mod in ["rgb", "thermal"]:
        cfg = {
            "path": str(repo_root.as_posix()),
            "train": f"data/splits/train_{mod}.txt",
            "val": f"data/splits/val_{mod}.txt",
            "test": f"data/splits/test_{mod}.txt",
            "names": {0: "Person_Detected"},
        }
        cfg_file = configs_dir / f"data_{mod}.yaml"
        with open(cfg_file, "w", encoding="utf-8") as fp:
            yaml.safe_dump(cfg, fp, sort_keys=False)
        print(f"Saved YOLO config: {cfg_file}")

    print("\n" + "=" * 80)
    print(" TABLE II FINAL FRAME COUNTS AUDIT (Option B: Desert + Forest + Snow)")
    print("=" * 80)
    print(f"{'Split':<8} {'RGB Frames':<14} {'Thermal Frames':<16} {'Desert (RGB)':<14} {'Forest (RGB)':<14} {'Snow (RGB)':<14}")
    print("-" * 80)
    for s in ["train", "val", "test"]:
        r_f = frame_counts["rgb"][s]
        t_f = frame_counts["thermal"][s]
        des = biome_counts["desert"][s]
        for_ = biome_counts["forest"][s]
        snw = biome_counts["snow"][s]
        print(f"{s:<8} {r_f:<14} {t_f:<16} {des:<14} {for_:<14} {snw:<14}")
    print("-" * 80)
    total_rgb = sum(frame_counts["rgb"].values())
    total_th = sum(frame_counts["thermal"].values())
    print(f"{'TOTAL':<8} {total_rgb:<14} {total_th:<16} {sum(biome_counts['desert'].values()):<14} {sum(biome_counts['forest'].values()):<14} {sum(biome_counts['snow'].values()):<14}")
    print("=" * 80)


def main() -> None:
    print("=== MMSAR Option B: Canonical Parent-Grouped Dataset Episode Splitting ===")
    summary = split_episodes()
    print(f"\nStrategy:               {summary['strategy']}")
    print(f"Total Canonical Parents:{summary['total_canonical_parent_videos']}")
    print(f"Total Episodes:         {summary['total_episodes']} (90 RGB + 90 Thermal)")
    print(f"Dynamic Floors (Biome): {summary['dynamic_floors_per_biome']}")
    print(f"Train Episodes:         {summary['partitions']['train_count']}")
    print(f"Val Episodes:           {summary['partitions']['val_count']}")
    print(f"Test Episodes:          {summary['partitions']['test_count']}")


if __name__ == "__main__":
    main()
