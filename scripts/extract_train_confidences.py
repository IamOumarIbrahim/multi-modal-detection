"""Extract detector confidences on training split and create unified cache."""

import json
from pathlib import Path
import sys
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_and_fill_manuscript import parse_split_episodes, extract_confidences

def main():
    rgb_weights = "C:/Dev/repos/Public repos/DMS-Eval/runs/detect/runs/detect/yolo11n_rgb_60e/weights/best.pt"
    thermal_weights = "C:/Dev/repos/Public repos/DMS-Eval/runs/detect/runs/detect/yolo11n_thermal_60e/weights/best.pt"

    print("Loading YOLO models...")
    model_rgb = YOLO(rgb_weights)
    model_th = YOLO(thermal_weights)

    print("Parsing train episodes...")
    train_eps = parse_split_episodes("data/splits/train_rgb.txt", "data/splits/train_thermal.txt")
    print(f"Loaded {len(train_eps)} train episodes.")

    train_rgb_seqs = []
    train_th_seqs = []
    train_fused_seqs = []
    train_gt_seqs = []
    train_biomes = []

    for idx, ep in enumerate(train_eps):
        print(f"  Extracting [{idx+1}/{len(train_eps)}] {ep['episode_id']} ({len(ep['rgb_frames'])} frames)...")
        c_rgb = extract_confidences(model_rgb, ep["rgb_frames"], batch_size=32)
        c_th = extract_confidences(model_th, ep["thermal_frames"], batch_size=32)
        fused = [max(r, t) for r, t in zip(c_rgb, c_th)]
        train_rgb_seqs.append(c_rgb)
        train_th_seqs.append(c_th)
        train_fused_seqs.append(fused)
        train_gt_seqs.append(ep["labels"])
        train_biomes.append(ep["biome"])

    print("Loading existing validation and test cache...")
    with open("runs/detect/extracted_confidences_cache.json") as f:
        existing = json.load(f)

    unified = {
        "train_rgb_seqs": train_rgb_seqs,
        "train_th_seqs": train_th_seqs,
        "train_fused_seqs": train_fused_seqs,
        "train_gt_seqs": train_gt_seqs,
        "train_biomes": train_biomes,
        "val_fused_seqs": existing["val_fused_seqs"],
        "val_gt_seqs": existing["val_gt_seqs"],
        "test_rgb_seqs": existing["test_rgb_seqs"],
        "test_th_seqs": existing["test_th_seqs"],
        "test_fused_seqs": existing["test_fused_seqs"],
        "test_gt_seqs": existing["test_gt_seqs"],
        "test_biomes": existing["test_biomes"],
    }

    out_path = Path("runs/detect/unified_confidences_cache.json")
    with open(out_path, "w") as f:
        json.dump(unified, f)
    print(f"SUCCESS: Unified confidences cache saved to {out_path} ({out_path.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
