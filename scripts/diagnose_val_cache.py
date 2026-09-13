"""Diagnose why validation-derived tau* diverges from the paper's calibrated values."""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmsar.postprocessing.baseline import BaselinePostProcessor
from mmsar.postprocessing.optimizer import find_optimal_threshold

CACHE = Path("runs/detect/unified_confidences_cache.json")
cache = json.loads(CACHE.read_text())

for split in ["val", "test"]:
    fused = cache[f"{split}_fused_seqs"]
    gt = cache[f"{split}_gt_seqs"]
    n_tiles = len(fused)
    n_frames = sum(len(s) for s in fused)
    n_pos_frames = int(sum(sum(s) for s in gt))
    flat_conf = np.array([v for s in fused for v in s])
    flat_gt = np.array([v for s in gt for v in s])
    pos_conf = flat_conf[flat_gt == 1]
    neg_conf = flat_conf[flat_gt == 0]

    print(f"--- {split} split ---")
    print(f"  tiles={n_tiles}  frames={n_frames}  positive_frames={n_pos_frames} ({100*n_pos_frames/n_frames:.1f}%)")
    if len(pos_conf):
        print(f"  confidence at gt=1: mean={pos_conf.mean():.4f} median={np.median(pos_conf):.4f} max={pos_conf.max():.4f}")
    else:
        print("  NO POSITIVE FRAMES FOUND")
    if len(neg_conf):
        print(f"  confidence at gt=0: mean={neg_conf.mean():.4f} median={np.neg_conf.max():.4f}" if False else f"  confidence at gt=0: mean={neg_conf.mean():.4f} median={np.median(neg_conf):.4f} max={neg_conf.max():.4f}")
    else:
        print("  NO NEGATIVE FRAMES FOUND")

print("\n--- Re-deriving M1 tau* on val split with plain F1 (paper's original protocol) ---")
val_fused, val_gt = cache["val_fused_seqs"], cache["val_gt_seqs"]
tau, f1 = find_optimal_threshold(BaselinePostProcessor(), val_fused, val_gt, metric="f1")
print(f"  M1 tau* = {tau:.2f} (val F1 = {f1:.4f})  -- paper reports tau*=0.43")
