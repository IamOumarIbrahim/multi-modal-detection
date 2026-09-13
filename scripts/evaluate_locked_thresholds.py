"""Compute verified metrics across all methods and biomes under locked thresholds."""

import json
import numpy as np
from scipy import stats
import random

from mmsar.postprocessing.baseline import BaselinePostProcessor
from mmsar.postprocessing.moving_average import MovingAveragePostProcessor
from mmsar.postprocessing.median_filter import MedianFilterPostProcessor
from mmsar.postprocessing.history_tracking import HistoryTrackingPostProcessor
from mmsar.metrics.frame_metrics import calculate_frame_metrics
from mmsar.metrics.alarm_metrics import count_alarm_false_triggers, calculate_time_to_alarm, _extract_events

def compute_bootstrap_ci(data: list[float], b_iterations: int = 1000, alpha: float = 0.05, seed: int = 42):
    rng = random.Random(seed)
    n = len(data)
    boot_means = []
    for _ in range(b_iterations):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lower = boot_means[int((alpha / 2.0) * b_iterations)]
    upper = boot_means[int((1.0 - alpha / 2.0) * b_iterations)]
    return round(float(np.mean(data)), 4), round(lower, 4), round(upper, 4)

def main():
    with open("runs/detect/unified_confidences_cache.json") as f:
        d = json.load(f)

    test_fused = d["test_fused_seqs"]
    test_gt = d["test_gt_seqs"]
    test_biomes = d["test_biomes"]

    total_frames = sum(len(s) for s in test_fused)
    total_minutes = total_frames / (24 * 60)
    pos_idxs = [i for i, g in enumerate(test_gt) if sum(g) > 0]
    total_pos_tiles = len(pos_idxs)

    biomes = ["desert", "forest", "snow"]
    biome_names = {"desert": "Arid Desert", "forest": "Temperate Forest", "snow": "Snow/Alpine"}

    methods = {
        "M1": ("Baseline Raw Thresholding", BaselinePostProcessor(), 0.43, "0.02 $\\mu$s", "0 B ($O(1)$)"),
        "M2": ("Five-Frame Moving Average", MovingAveragePostProcessor(5), 0.57, "0.30 $\\mu$s", "20 B ($O(1)$)"),
        "M3": ("Five-Frame Median Filter", MedianFilterPostProcessor(5), 0.49, "0.42 $\\mu$s", "20 B ($O(W)$)"),
        "M4": ("Five-Frame History Consensus", HistoryTrackingPostProcessor(5, required_hits=3), 0.49, "0.63 $\\mu$s", "5 B ($O(W)$)"),
    }

    print("=" * 80)
    print(" VERIFIED LOCKED BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total test frames: {total_frames} ({total_minutes:.3f} min), Positive tiles: {total_pos_tiles}/24\n")

    m1_fp_total = None
    results = {}

    for key, (label, proc, tau, runtime_str, state_str) in methods.items():
        decisions = [proc.decide(s, tau) for s in test_fused]
        y_true = [v for s in test_gt for v in s]
        y_pred = [v for s in decisions for v in s]
        fm = calculate_frame_metrics(y_true, y_pred)

        # Tile recall
        det_pos = sum(1 for i in pos_idxs if sum(p and g for p, g in zip(decisions[i], test_gt[i])) > 0)
        tile_recall_pct = (det_pos / total_pos_tiles) * 100.0

        # FA events and FP frames
        fa_events = sum(len(_extract_events([pv if gv == 0 else 0 for pv, gv in zip(decisions[i], test_gt[i])])) for i in range(len(test_gt)))
        fa_rate = fa_events / total_minutes
        fp_frames = int(fm["fp"])

        if key == "M1":
            m1_fp_total = fp_frames
            fasr = 0.0
        else:
            fasr = ((m1_fp_total - fp_frames) / m1_fp_total) * 100.0

        # Latency
        delays = []
        for i in pos_idxs:
            res_lat = calculate_time_to_alarm(test_gt[i], decisions[i], fps=24.0)
            delays.extend(res_lat["delays_frames"])
        mean_delay_frames = float(np.mean(delays)) if delays else 0.0
        mean_delay_ms = (mean_delay_frames / 24.0) * 1000.0

        # Bootstrap CI on episode F1
        ep_f1s = [calculate_frame_metrics(test_gt[i], decisions[i])["f1"] for i in range(len(test_gt))]
        _, ci_low, ci_high = compute_bootstrap_ci(ep_f1s)

        # Per-biome breakdown
        per_biome = {}
        for b in biomes:
            b_idxs = [i for i, bm in enumerate(test_biomes) if bm == b]
            b_dur_min = sum(len(test_fused[i]) for i in b_idxs) / (24 * 60)
            b_pos_idxs = [i for i in b_idxs if sum(test_gt[i]) > 0]
            b_det_pos = sum(1 for i in b_pos_idxs if sum(p and g for p, g in zip(decisions[i], test_gt[i])) > 0)
            b_rec_pct = (b_det_pos / len(b_pos_idxs)) * 100.0 if b_pos_idxs else 0.0
            b_fa_events = sum(len(_extract_events([pv if gv == 0 else 0 for pv, gv in zip(decisions[i], test_gt[i])])) for i in b_idxs)
            b_fa_rate = b_fa_events / b_dur_min
            b_fp_frames = sum(sum(p == 1 and g == 0 for p, g in zip(decisions[i], test_gt[i])) for i in b_idxs)
            per_biome[b] = {
                "recall_pct": round(b_rec_pct, 1),
                "det_pos": b_det_pos,
                "total_pos": len(b_pos_idxs),
                "fa_events": b_fa_events,
                "fa_rate": round(b_fa_rate, 2),
                "fp_frames": b_fp_frames,
            }

        results[key] = {
            "label": label,
            "tau": tau,
            "tile_recall_pct": round(tile_recall_pct, 1),
            "det_pos": det_pos,
            "total_pos": total_pos_tiles,
            "f1": round(fm["f1"], 3),
            "ci_low": round(ci_low, 3),
            "ci_high": round(ci_high, 3),
            "fa_events": fa_events,
            "fa_rate": round(fa_rate, 1),
            "fp_frames": fp_frames,
            "fasr": round(fasr, 1),
            "mean_delay_frames": round(mean_delay_frames, 1),
            "mean_delay_ms": round(mean_delay_ms, 1),
            "runtime_str": runtime_str,
            "state_str": state_str,
            "ep_f1s": ep_f1s,
            "per_biome": per_biome,
        }

        print(f"[{key}] {label}")
        print(f"  Val tau*: {tau:.2f}")
        print(f"  Alert Recall: {tile_recall_pct:.1f}% ({det_pos}/{total_pos_tiles})")
        print(f"  Frame F1: {fm['f1']:.3f} [{ci_low:.3f}, {ci_high:.3f}] (P={fm['precision']:.3f}, R={fm['recall']:.3f})")
        print(f"  FA/min: {fa_rate:.1f} ({fa_events} events across {total_minutes:.2f} min)")
        print(f"  FP Frames: {fp_frames} (FASR: {fasr:.1f}%)")
        print(f"  Latency: {mean_delay_ms:.1f} ms ({mean_delay_frames:.1f} frames)")
        print(f"  Per-biome:")
        for b in biomes:
            pb = per_biome[b]
            print(f"    {b:7s}: Recall={pb['recall_pct']:5.1f}% ({pb['det_pos']}/{pb['total_pos']}) | FA/min={pb['fa_rate']:5.2f} | FP={pb['fp_frames']:2d}")
        print()

    # Episode-level paired tests
    m1_f1s = results["M1"]["ep_f1s"]
    m2_f1s = results["M2"]["ep_f1s"]
    diffs = [b - a for a, b in zip(m1_f1s, m2_f1s)]
    w_res = stats.wilcoxon(diffs, alternative="two-sided")
    print(f"Episode-Level Paired Wilcoxon Test (M2 vs M1):")
    print(f"  Mean delta F1: {np.mean(diffs):+.4f}")
    print(f"  Statistic W = {w_res.statistic:.2f}, p-value = {w_res.pvalue:.5f}")

    # Export to JSON
    with open("runs/detect/locked_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved locked results to runs/detect/locked_benchmark_results.json")

if __name__ == "__main__":
    main()
