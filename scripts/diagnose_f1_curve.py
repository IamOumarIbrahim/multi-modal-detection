"""Print the full F1(tau) sweep curve and positive-frame confidence histogram on val."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mmsar.metrics.frame_metrics import calculate_frame_metrics

cache = json.loads(Path("runs/detect/unified_confidences_cache.json").read_text())
val_fused, val_gt = cache["val_fused_seqs"], cache["val_gt_seqs"]
y_true = np.array([v for s in val_gt for v in s])
y_score = np.array([v for s in val_fused for v in s])

print("--- F1(tau) sweep on val, M1 (raw thresholding) ---")
print(f"{'tau':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
tau = 0.05
while tau <= 0.95 + 1e-6:
    y_pred = (y_score >= tau).astype(int)
    m = calculate_frame_metrics(y_true.tolist(), y_pred.tolist())
    if round(tau, 2) in (0.05, 0.07, 0.09, 0.15, 0.25, 0.35, 0.43, 0.49, 0.57, 0.65, 0.75, 0.85):
        print(f"{tau:5.2f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {int(m['tp']):5d} {int(m['fp']):5d} {int(m['fn']):5d}")
    tau += 0.02

print("\n--- Confidence histogram for gt=1 (positive) val frames ---")
pos_conf = y_score[y_true == 1]
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.43, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(pos_conf, bins=bins)
for i in range(len(hist)):
    print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:5d} frames ({100*hist[i]/len(pos_conf):.1f}%)")

print("\n--- Confidence histogram for gt=0 (negative) val frames, zoomed to low end ---")
neg_conf = y_score[y_true == 0]
bins2 = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.43, 0.5, 1.0]
hist2, _ = np.histogram(neg_conf, bins=bins2)
for i in range(len(hist2)):
    print(f"  [{bins2[i]:.2f}, {bins2[i+1]:.2f}): {hist2[i]:5d} frames ({100*hist2[i]/len(neg_conf):.1f}%)")
