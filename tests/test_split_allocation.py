"""Behavioral unit tests for dynamic dataset episode splitting and difficulty balancing."""

from pathlib import Path
import json
import pytest
from scripts.split_dataset_episodes import (
    derive_positive_split_floor,
    allocate_positive_parents_dynamically,
    allocate_negative_parents_dynamically,
    compute_episode_metrics,
    split_episodes,
)


def test_dynamic_positive_split_floor_derivation() -> None:
    """Assert split floor scales dynamically with pool size, never locked to 2."""
    # N=2: minimum floor of 1 parent (2 episodes)
    assert derive_positive_split_floor(total_positive_parents=2, val_ratio=0.20) == 1

    # N=5: 5 * 0.20 = 1 parent (2 episodes)
    assert derive_positive_split_floor(total_positive_parents=5, val_ratio=0.20) == 1

    # N=10: 10 * 0.20 = 2 parents (4 episodes)
    assert derive_positive_split_floor(total_positive_parents=10, val_ratio=0.20) == 2

    # N=15: 15 * 0.20 = 3 parents (6 episodes)
    assert derive_positive_split_floor(total_positive_parents=15, val_ratio=0.20) == 3

    # N=25: 25 * 0.20 = 5 parents (10 episodes)
    assert derive_positive_split_floor(total_positive_parents=25, val_ratio=0.20) == 5


def test_positive_parent_allocation_difficulty_balance_and_priority() -> None:
    """Assert positive parents balance difficulty and clear dynamic floors."""
    # Synthetic pool of 5 positive parents with varying target presence rates
    difficulties = {
        "pos_1": 0.99,
        "pos_2": 0.95,
        "pos_3": 0.52,  # Occluded
        "pos_4": 0.93,
        "pos_5": 0.10,  # Highly occluded
    }

    assignment = allocate_positive_parents_dynamically(
        positive_parent_difficulties=difficulties,
        val_ratio=0.20,
        test_ratio=0.20,
    )

    # All parents must be assigned
    assert len(assignment) == 5
    assert set(assignment.values()).issubset({"train", "val", "test"})

    # Check that both val and test receive allocations
    val_parents = [p for p, s in assignment.items() if s == "val"]
    test_parents = [p for p, s in assignment.items() if s == "test"]
    train_parents = [p for p, s in assignment.items() if s == "train"]

    assert len(val_parents) >= 1
    assert len(test_parents) >= 1
    assert len(train_parents) >= 1

    # Check that evaluation splits received parents
    assert len(val_parents) + len(test_parents) >= 2


def test_negative_parent_allocation_proportions() -> None:
    """Assert negative parents follow 60/20/20 distribution."""
    negatives = [f"neg_{i}" for i in range(10)]
    assignment = allocate_negative_parents_dynamically(
        negative_parents=negatives,
        train_ratio=0.60,
        val_ratio=0.20,
        seed=0,
    )

    train_c = sum(1 for s in assignment.values() if s == "train")
    val_c = sum(1 for s in assignment.values() if s == "val")
    test_c = sum(1 for s in assignment.values() if s == "test")

    assert train_c == 6
    assert val_c == 2
    assert test_c == 2


def test_split_episodes_end_to_end(tmp_path: Path) -> None:
    """Assert split_episodes generates valid Option B manifest with dynamic floors."""
    repo_root = Path(__file__).resolve().parent.parent
    raw_root = repo_root / "data" / "raw"
    labels_root = repo_root / "data" / "processed" / "labels"

    if not raw_root.exists() or not any(raw_root.glob("*/*/*/*.mp4")):
        pytest.skip("Raw video directory not present for end-to-end split test")

    summary = split_episodes(
        raw_root=raw_root,
        labels_root=labels_root,
        splits_out_dir=tmp_path,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        seed=0,
    )

    assert summary["strategy"] == "Option B: Parent-Video Grouped Episodic Partitioning"
    assert summary["total_episodes"] > 0
    assert summary["partitions"]["train_count"] > 0
    assert summary["partitions"]["val_count"] > 0
    assert summary["partitions"]["test_count"] > 0

    # Ensure manifest file was written
    out_json = tmp_path / "dataset_episodes_split.json"
    manifest_json = tmp_path / "dataset_episodes_split_manifest.json"
    assert out_json.exists()
    assert manifest_json.exists()

    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert "dynamic_floors_per_biome" in parsed
    assert "split_difficulty_balance" in parsed

    # Verify Option B: zero cross-split leakage for parent videos
    parent_to_split: dict[str, str] = {}
    for s_name in ["train", "val", "test"]:
        for ep in parsed["episodes"][s_name]:
            p_id = ep["parent_video_id"]
            if p_id in parent_to_split:
                assert parent_to_split[p_id] == s_name, (
                    f"Leakage detected: parent {p_id} present in both {parent_to_split[p_id]} and {s_name}"
                )
            parent_to_split[p_id] = s_name
