#   MMSAR - paper Tables III and IV (fixed tau* per method)
#        Oumar Ibrahim U22200741

# the tau* values below are the ones reported in the paper, they are inputs here
# (they are not re-derived, the val F1 sweep in evaluate_mmsar_unified.py gives others)
# M5 / M6 are trained on the train split, 40 epochs, 5 seeds
# frame F1 CI = bootstrap over test clips of the pooled frame F1 (B = 1000)
# output  runs/detect/locked_benchmark_results.json

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

from mmsar.evaluation import (
    bootstrap_ci,
    load_cache,
    minutes,
    pooled_f1_ci,
    score_decisions,
    target_alerted,
)
from mmsar.filters import BaselineFilter, HistoryConsensusFilter, MedianFilter, MovingAverageFilter
from mmsar.metrics import count_false_alerts
from mmsar.sequence_models import SequenceFilter

SEEDS = [42, 101, 2024, 777, 999]
BIOMES = ["desert", "forest", "snow"]

METHODS = {
    "M1": ("Baseline Raw Thresholding", BaselineFilter(), 0.43, "0.02 $\\mu$s", "0 B ($O(1)$)"),
    "M2": ("Five-Frame Moving Average", MovingAverageFilter(5), 0.57, "0.30 $\\mu$s", "20 B ($O(1)$)"),
    "M3": ("Five-Frame Median Filter", MedianFilter(5), 0.49, "0.42 $\\mu$s", "20 B ($O(W)$)"),
    "M4": ("Five-Frame History Consensus", HistoryConsensusFilter(5, hits=3), 0.49, "0.63 $\\mu$s", "5 B ($O(W)$)"),
}
LEARNED = {
    "M5": ("Learned Mamba-SSSM", "mamba", 0.73, "150.59 $\\mu$s", "512 B"),
    "M6": ("Learned GRU Baseline", "gru", 0.50, "21.80 $\\mu$s", "64 B"),
}


# clip ids in the same order as the cache (first time each clip shows up in the list)
def test_parents():
    ids = []
    for line in Path("data/splits/test_rgb.txt").read_text(encoding="utf-8").splitlines():
        m = re.search(r"/([^/]+)_frame_\d+\.png", line.strip())
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    parents = []
    for e in ids:
        parents.append(re.sub(r"_(left|right)$", "", e))
    return parents


def fp_frames(pred, gt):
    n = 0
    for i in range(len(gt)):
        if pred[i] == 1 and gt[i] == 0:
            n += 1
    return n


# one method, one set of decisions -> one row of Table III plus its Table IV rows
def table_row(decisions, test_gt, test_biomes, parents, test_min):
    s = score_decisions(decisions, test_gt, test_biomes, BIOMES)
    pos = s["pos_idx"]

    hits = []
    for i in pos:
        if target_alerted(decisions[i], test_gt[i]):
            hits.append(100.0)
        else:
            hits.append(0.0)
    _, rec_low, rec_high = bootstrap_ci(hits)

    # a parent video counts as found if either of its two tiles raised an alert
    pos_parents = []
    for i in pos:
        if parents[i] not in pos_parents:
            pos_parents.append(parents[i])
    found = 0
    for p in pos_parents:
        for i in pos:
            if parents[i] == p and target_alerted(decisions[i], test_gt[i]):
                found += 1
                break

    f1_low, f1_high = pooled_f1_ci(decisions, test_gt)

    per_biome = {}
    for b in BIOMES:
        rec = 0.0
        if s["pos_by_biome"][b] > 0:
            rec = (s["det_by_biome"][b] / s["pos_by_biome"][b]) * 100.0
        per_biome[b] = {
            "recall_pct": round(rec, 1),
            "det_pos": s["det_by_biome"][b],
            "total_pos": s["pos_by_biome"][b],
            "fa_events": s["fa_by_biome"][b],
            "fa_rate": round(s["fa_by_biome"][b] / s["minutes_by_biome"][b], 2),
            "fp_frames": s["fp_by_biome"][b],
        }

    return {
        "tile_recall_pct": round((s["detected"] / len(pos)) * 100.0, 1),
        "tile_recall_ci": [round(rec_low, 1), round(rec_high, 1)],
        "det_pos": s["detected"],
        "total_pos": len(pos),
        "parent_recall": f"{found}/{len(pos_parents)}",
        "precision": round(s["frame"]["precision"], 3),
        "recall": round(s["frame"]["recall"], 3),
        "f1": round(s["frame"]["f1"], 3),
        "f1_ci": [round(f1_low, 3), round(f1_high, 3)],
        "fa_events": s["fa_events"],
        "fa_rate": round(s["fa_events"] / test_min, 1),
        "fp_frames": int(s["frame"]["fp"]),
        "mean_delay_frames": round(s["mean_delay_frames"], 1),
        "mean_delay_ms": round(s["mean_delay_ms"], 1),
        "ep_f1s": s["episode_f1s"],
        "per_biome": per_biome,
    }


def mean_std(values):
    return round(float(np.mean(values)), 3), round(float(np.std(values, ddof=1)), 3)


def main():
    cache = load_cache()
    test_fused = cache["test_fused_seqs"]
    test_gt = cache["test_gt_seqs"]
    test_biomes = cache["test_biomes"]
    parents = test_parents()
    test_min = minutes(test_fused)
    if len(parents) != len(test_fused):
        raise ValueError("data/splits/test_rgb.txt does not match the confidence cache")

    results = {}
    decisions = {}
    for key in METHODS:
        label, filt, tau, runtime, state = METHODS[key]
        dec = []
        for s in test_fused:
            dec.append(filt.decide(s, tau))
        decisions[key] = dec

        row = table_row(dec, test_gt, test_biomes, parents, test_min)
        row["label"] = label
        row["tau"] = tau
        row["runtime_str"] = runtime
        row["state_str"] = state
        results[key] = row

    # false alarm suppression ratio = FP frames removed relative to raw thresholding
    m1_fp = results["M1"]["fp_frames"]
    for key in results:
        results[key]["fasr"] = round(((m1_fp - results[key]["fp_frames"]) / m1_fp) * 100.0, 1)

    train_fused = cache["train_fused_seqs"]
    train_gt = cache["train_gt_seqs"]
    for key in LEARNED:
        label, kind, tau, runtime, state = LEARNED[key]
        rows = []
        for seed in SEEDS:
            model = SequenceFilter(kind=kind, seed=seed)
            model.fit(train_fused, train_gt, epochs=40, lr=0.01)
            dec = []
            for s in test_fused:
                dec.append(model.decide(s, tau))
            rows.append(table_row(dec, test_gt, test_biomes, parents, test_min))

        rec = []
        f1 = []
        fa = []
        fp = []
        delay = []
        lows = []
        highs = []
        for r in rows:
            # use the unrounded recall so the mean is not off by rounding
            rec.append(r["det_pos"] / r["total_pos"] * 100.0)
            f1.append(r["f1"])
            fa.append(r["fa_events"] / test_min)
            fp.append(r["fp_frames"])
            delay.append(r["mean_delay_ms"])
            lows.append(r["f1_ci"][0])
            highs.append(r["f1_ci"][1])
        mean_fp = float(np.mean(fp))

        results[key] = {
            "label": label,
            "tau": tau,
            "seeds": SEEDS,
            "tile_recall_pct": mean_std(rec),
            "f1": mean_std(f1),
            "f1_ci": [round(float(np.mean(lows)), 3), round(float(np.mean(highs)), 3)],
            "fa_rate": mean_std(fa),
            "fp_frames": mean_std(fp),
            "fasr": round(((m1_fp - mean_fp) / m1_fp) * 100.0, 1),
            "mean_delay_ms": round(float(np.mean(delay)), 1),
            "parent_recall": [r["parent_recall"] for r in rows],
            "runtime_str": runtime,
            "state_str": state,
            "seed_results": rows,
        }

    # mission savings vs raw thresholding, every avoided false alert is one hover we skip
    # hover 280 W for 20 s = 5.6 kJ, telemetry 1.2 kB per alert (45 kB with a thumbnail)
    savings = {}
    for key in ["M2", "M3", "M4"]:
        savings[key] = {}
        for b in BIOMES:
            m1b = results["M1"]["per_biome"][b]
            mb = results[key]["per_biome"][b]
            avoided = m1b["fa_events"] - mb["fa_events"]
            fp_avoided = m1b["fp_frames"] - mb["fp_frames"]
            savings[key][b] = {
                "fa_events_avoided": avoided,
                "fp_frames_avoided": fp_avoided,
                "fp_frames_avoided_pct": round(fp_avoided / m1b["fp_frames"] * 100.0, 1),
                "bandwidth_kb": round(avoided * 1.2, 1),
                "bandwidth_thumb_kb": round(avoided * 45.0, 1),
                "battery_kj": round(avoided * 5.6, 1),
                "flight_time_min": round(avoided * 5.6 * 1000.0 / 280.0 / 60.0, 1),
            }
    results["savings_vs_m1"] = savings

    # paired wilcoxon M2 vs M1 over the 24 test clips
    tests = {}
    for name in ["fp_frames", "fa_events", "f1"]:
        a = []
        b = []
        for i in range(len(test_gt)):
            if name == "fp_frames":
                a.append(fp_frames(decisions["M1"][i], test_gt[i]))
                b.append(fp_frames(decisions["M2"][i], test_gt[i]))
            elif name == "fa_events":
                a.append(count_false_alerts(test_gt[i], decisions["M1"][i]))
                b.append(count_false_alerts(test_gt[i], decisions["M2"][i]))
            else:
                a.append(results["M1"]["ep_f1s"][i])
                b.append(results["M2"]["ep_f1s"][i])
        diffs = []
        for i in range(len(a)):
            diffs.append(b[i] - a[i])
        res = stats.wilcoxon(diffs, alternative="two-sided")
        tests[f"m2_vs_m1_{name}"] = {"W": float(res.statistic), "p_value": round(float(res.pvalue), 4)}
    results["wilcoxon"] = tests

    print(f"test: {sum(len(s) for s in test_fused)} frames, {test_min:.3f} min")
    print()
    print("method  tau   recall           parents  F1 [95% CI]              FA/min  FP   latency")
    for key in ["M1", "M2", "M3", "M4"]:
        r = results[key]
        print(f"{key}      {r['tau']:.2f}  {r['tile_recall_pct']:5.1f}% [{r['tile_recall_ci'][0]}, {r['tile_recall_ci'][1]}]"
              f"  {r['parent_recall']}     {r['f1']:.3f} [{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}]"
              f"     {r['fa_rate']:4.1f}   {r['fp_frames']:3d}  {r['mean_delay_ms']} ms")
    for key in ["M5", "M6"]:
        r = results[key]
        print(f"{key}      {r['tau']:.2f}  {r['tile_recall_pct'][0]:.1f} +- {r['tile_recall_pct'][1]:.1f}%"
              f"           {r['f1'][0]:.3f} +- {r['f1'][1]:.3f} [{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}]"
              f"  {r['fa_rate'][0]:.1f} +- {r['fa_rate'][1]:.1f}  {r['fp_frames'][0]:.0f}  {r['mean_delay_ms']} ms")
    print()
    print("per biome (Table IV)")
    for b in BIOMES:
        for key in ["M1", "M2", "M3", "M4"]:
            pb = results[key]["per_biome"][b]
            print(f"  {b:<7} {key}  {pb['det_pos']}/{pb['total_pos']}  FA/min {pb['fa_rate']:5.2f}  FP {pb['fp_frames']}")
    print()
    for name in tests:
        print(f"wilcoxon {name}: W = {tests[name]['W']}, p = {tests[name]['p_value']}")

    out = Path("runs/detect/locked_benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("----------------------")
    print("Saved= ", out)
    print("----------------------")


if __name__ == "__main__":
    main()
