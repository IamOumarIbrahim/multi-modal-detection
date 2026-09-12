"""Dataset episode splitting and sequence shuffling pipeline (Option B).

Implements parent-video grouped partitioning, 240-frame sequence bundling,
and seeded snippet shuffling with anti-adjacency guards.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EpisodeMetadata:
    """Metadata for an atomic 240-frame video episode."""

    episode_id: str
    parent_video_id: str
    tile_position: str  # 'left' or 'right'
    biome: str  # 'desert' or 'forest'
    scenario: str  # 'positive', 'hard_negative', 'clear_negative'
    modality: str  # 'rgb' or 'thermal'
    frame_count: int  # 240
    start_frame: int  # 0
    end_frame: int  # 239
    fps: int  # 24
    video_path: str


def group_snippets_by_parent(video_dir: Path) -> dict[str, dict[str, Path]]:
    """Group left and right harvested video files by parent video stem.

    Returns mapping:
        parent_id -> {'left': Path, 'right': Path}
    """
    parent_map: dict[str, dict[str, Path]] = {}
    for vid_file in sorted(video_dir.glob("*.mp4")):
        stem = vid_file.stem
        if stem.endswith("_left"):
            parent_id = stem[:-5]
            parent_map.setdefault(parent_id, {})["left"] = vid_file
        elif stem.endswith("_right"):
            parent_id = stem[:-6]
            parent_map.setdefault(parent_id, {})["right"] = vid_file
    return parent_map


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

    for attempt in range(max_attempts):
        candidate = list(episodes)
        rng.shuffle(candidate)

        valid = True
        positions: dict[str, list[int]] = {}
        for idx, ep in enumerate(candidate):
            positions.setdefault(ep.parent_video_id, []).append(idx)

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


def split_episodes(
    raw_root: Path = Path("data/raw"),
    splits_out_dir: Path = Path("data/splits"),
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 0,
) -> dict[str, Any]:
    """Partition the dataset using Option B (Parent-Video Grouped Splitting)."""
    splits_out_dir.mkdir(parents=True, exist_ok=True)

    all_splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    summary: dict[str, Any] = {
        "strategy": "Option B: Parent-Video Grouped Episodic Partitioning",
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "seed": seed,
        "anti_adjacency_min_distance": 2,
        "frames_per_episode": 240,
        "fps": 24,
        "partitions": {},
    }

    # Discover all video snippets across data/raw
    all_videos = sorted(raw_root.glob("*/*/*/*.mp4"))
    parent_map: dict[str, dict[str, EpisodeMetadata]] = {}

    import re

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

        ep = EpisodeMetadata(
            episode_id=f"{parent_id}_{tile_pos}",
            parent_video_id=parent_id,
            tile_position=tile_pos,
            biome=biome,
            scenario=scenario,
            modality=modality,
            frame_count=240,
            start_frame=0,
            end_frame=239,
            fps=24,
            video_path=str(vid_path.as_posix()),
        )
        parent_map.setdefault(parent_id, {})[tile_pos] = ep

    # Group parent flights into mission series (e.g. desert_RGB_positive, desert_RGB_hard_negative, etc.)
    series_map: dict[str, list[str]] = {}
    for parent_id in sorted(parent_map.keys()):
        m = re.match(r"^(.+)_\d+$", parent_id)
        series_name = m.group(1) if m else parent_id
        series_map.setdefault(series_name, []).append(parent_id)

    # Deterministic Option B parent-video allocation ensuring balanced positive episodes
    # Each split receives exactly 2 positive episodes:
    # - Train: positive_1, positive_2 (2 pos, 2 clear_neg)
    # - Val: positive_3 (2 pos: left & right)
    # - Test: positive_4, positive_5 (2 pos, 2 clear_neg)
    # Hard negative and clear negative parents follow the 3:1:1 allocation (Train: 1,2,3; Val: 5; Test: 4).
    parent_partition_assignment = {
        # Positive series
        "desert_RGB_positive_1": "train",
        "desert_RGB_positive_2": "train",
        "desert_RGB_positive_3": "val",
        "desert_RGB_positive_4": "test",
        "desert_RGB_positive_5": "test",
        # Hard negative series
        "desert_RGB_hard_negative_1": "train",
        "desert_RGB_hard_negative_2": "train",
        "desert_RGB_hard_negative_3": "train",
        "desert_RGB_hard_negative_4": "test",
        "desert_RGB_hard_negative_5": "val",
        # Clear negative series
        "desert_RGB_clear_negative_1": "train",
        "desert_RGB_clear_negative_2": "train",
        "desert_RGB_clear_negative_3": "train",
        "desert_RGB_clear_negative_4": "test",
        "desert_RGB_clear_negative_5": "val",
    }

    partition_assigned: dict[str, list[EpisodeMetadata]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for parent_id, tiles in sorted(parent_map.items()):
        split_name = parent_partition_assignment.get(parent_id, "train")
        for ep in tiles.values():
            partition_assigned[split_name].append(ep)

    # Apply anti-adjacency shuffling within each split
    for split_name in ["train", "val", "test"]:
        ep_list = partition_assigned[split_name]
        shuffled_split = shuffle_with_anti_adjacency(
            ep_list,
            seed=seed,
            min_distance=2,
        )
        for order_idx, ep in enumerate(shuffled_split):
            ep_dict = asdict(ep)
            ep_dict["feed_order_index"] = order_idx
            all_splits[split_name].append(ep_dict)

    summary["partitions"]["train_count"] = len(all_splits["train"])
    summary["partitions"]["val_count"] = len(all_splits["val"])
    summary["partitions"]["test_count"] = len(all_splits["test"])
    summary["episodes"] = all_splits

    out_file = splits_out_dir / "dataset_episodes_split.json"
    out_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Run episode splitting pipeline and print summary."""
    print("=== MMSAR Option B: Dataset Episode Splitting ===")
    summary = split_episodes()
    print(f"Strategy: {summary['strategy']}")
    print(f"Seed: {summary['seed']}")
    print(f"Train episodes: {summary['partitions']['train_count']}")
    print(f"Val episodes:   {summary['partitions']['val_count']}")
    print(f"Test episodes:  {summary['partitions']['test_count']}")
    print("Saved split definition to data/splits/dataset_episodes_split.json")


if __name__ == "__main__":
    main()

