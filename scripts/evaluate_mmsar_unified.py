#   MMSAR - validation calibrated benchmark (tau* = best val F1)
#        Oumar Ibrahim U22200741

# this is the protocol where every tau* is picked by sweeping validation F1
# the paper tables use fixed tau* instead, see evaluate_locked_thresholds.py
# 1. upstream rgb / thermal / fused detection at tau = 0.50 on test
# 2. window length ablation W in {1, 3, 5, 7} on validation
# 3. M5 and M6 trained on train (5 seeds), every tau* from validation
# 4. all methods M1..M6 on test
# 5. paired wilcoxon tests on clip F1
# 6. tau sensitivity of M1 vs M2 on test
# output  runs/detect/unified_evaluation_results.json

import json
from pathlib import Path

import numpy as np
from scipy import stats

from mmsar.evaluation import (
    best_threshold,
    best_threshold_by_decide,
    bootstrap_ci,
    continuous_scores,
    flatten,
    load_cache,
    minutes,
    score_decisions,
)
from mmsar.filters import BaselineFilter, HistoryConsensusFilter, MedianFilter, MovingAverageFilter
from mmsar.metrics import calculate_frame_metrics, count_false_alerts
from mmsar.sequence_models import SequenceFilter

SEEDS = [42, 101, 2024, 777, 999]
BIOMES = ["desert", "forest", "snow"]
BIOME_NAMES = {"desert": "Arid Desert", "forest": "Temperate Forest", "snow": "Snow/Alpine"}


def threshold_all(seqs, tau):
    out = []
    for s in seqs:
        row = []
        for v in s:
            if v >= tau:
                row.append(1)
            else:
                row.append(0)
        out.append(row)
    return out


# paired wilcoxon of clip F1 (b minus a), only when at least 5 clips differ
def paired_test(scores_a, scores_b):
    diffs = []
    for i in range(len(scores_a)):
        diffs.append(scores_b[i] - scores_a[i])

    nonzero = 0
    for d in diffs:
        if abs(d) > 1e-7:
            nonzero += 1
    stat = 0.0
    p = 1.0
    if nonzero >= 5:
        res = stats.wilcoxon(diffs, alternative="two-sided")
        stat = float(res.statistic)
        p = float(res.pvalue)

    return {
        "mean_diff": round(float(np.mean(diffs)), 4),
        "statistic": round(stat, 4),
        "p_value": round(p, 5),
        "n_episodes": len(scores_a),
    }


def test_row(decisions, test_gt, test_biomes, test_min):
    s = score_decisions(decisions, test_gt, test_biomes, BIOMES)
    fm = s["frame"]
    n_pos = len(s["pos_idx"])
    _, ci_low, ci_high = bootstrap_ci(s["episode_f1s"])

    recall_by_biome = {}
    fa_rate_by_biome = {}
    for b in BIOMES:
        recall_by_biome[b] = 0.0
        if s["pos_by_biome"][b] > 0:
            recall_by_biome[b] = (s["det_by_biome"][b] / s["pos_by_biome"][b]) * 100.0
        fa_rate_by_biome[b] = round(s["fa_by_biome"][b] / s["minutes_by_biome"][b], 2)

    return {
        "precision": round(fm["precision"], 4),
        "recall": round(fm["recall"], 4),
        "f1": round(fm["f1"], 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "episode_f1s": s["episode_f1s"],
        "tile_recall_overall": round((s["detected"] / n_pos) * 100.0, 1),
        "detected_pos_overall": s["detected"],
        "total_pos_tiles": n_pos,
        "tile_recall_by_biome": recall_by_biome,
        "detected_pos_by_biome": s["det_by_biome"],
        "total_pos_by_biome": s["pos_by_biome"],
        "fa_events_overall": s["fa_events"],
        "fa_rate_overall": round(s["fa_events"] / test_min, 2),
        "fa_events_by_biome": s["fa_by_biome"],
        "fa_rate_by_biome": fa_rate_by_biome,
        "fp_frames_overall": int(fm["fp"]),
        "fp_frames_by_biome": s["fp_by_biome"],
        "mean_delay_frames": round(s["mean_delay_frames"], 1),
        "mean_delay_ms": round(s["mean_delay_ms"], 1),
    }


# mean / std over the 5 seeds, the median F1 seed is kept as the representative one
def aggregate_seeds(rows, m1_fp, runtime_us, state_desc):
    f1s = []
    fa_rates = []
    fps = []
    recalls = []
    lows = []
    highs = []
    delays = []
    taus = []
    for r in rows:
        f1s.append(r["f1"])
        fa_rates.append(r["fa_rate_overall"])
        fps.append(r["fp_frames_overall"])
        recalls.append(r["tile_recall_overall"])
        lows.append(r["ci_low"])
        highs.append(r["ci_high"])
        delays.append(r["mean_delay_ms"])
        taus.append(r["tau_star"])

    mean_fp = float(np.mean(fps))
    fasr = 0.0
    if m1_fp > 0:
        fasr = ((m1_fp - mean_fp) / m1_fp) * 100.0
    rep = rows[int(np.argsort(f1s)[len(f1s) // 2])]

    return {
        "tau_star": round(float(np.mean(taus)), 2),
        "tile_recall_overall": round(float(np.mean(recalls)), 1),
        "f1_mean": round(float(np.mean(f1s)), 3),
        "f1_std": round(float(np.std(f1s, ddof=1)), 3),
        "ci_low": round(float(np.mean(lows)), 3),
        "ci_high": round(float(np.mean(highs)), 3),
        "fa_rate_mean": round(float(np.mean(fa_rates)), 1),
        "fa_rate_std": round(float(np.std(fa_rates, ddof=1)), 1),
        "mean_delay_ms": round(float(np.mean(delays)), 1),
        "fp_frames_overall": int(round(mean_fp)),
        "fasr": round(fasr, 1),
        "runtime_us": runtime_us,
        "state_size": state_desc,
        "rep_seed_metrics": rep,
        "seed_results": rows,
    }


def main():
    cache = load_cache()
    train_fused = cache["train_fused_seqs"]
    train_gt = cache["train_gt_seqs"]
    val_fused = cache["val_fused_seqs"]
    val_gt = cache["val_gt_seqs"]
    test_rgb = cache["test_rgb_seqs"]
    test_th = cache["test_th_seqs"]
    test_fused = cache["test_fused_seqs"]
    test_gt = cache["test_gt_seqs"]
    test_biomes = cache["test_biomes"]
    test_min = minutes(test_fused)
    print(f"train {len(train_fused)} clips, val {len(val_fused)} clips, test {len(test_fused)} clips ({test_min:.3f} min)")

    # 1. upstream detection at tau = 0.50
    upstream = {}
    for b in BIOMES:
        idx = []
        for i in range(len(test_biomes)):
            if test_biomes[i] == b:
                idx.append(i)
        y_true = []
        for i in idx:
            y_true += test_gt[i]

        for label, seqs in [("YOLO11n-RGB", test_rgb), ("YOLO11n-Thermal", test_th), ("Late Fusion Max Gate", test_fused)]:
            picked = []
            for i in idx:
                picked.append(seqs[i])
            m = calculate_frame_metrics(y_true, flatten(threshold_all(picked, 0.50)))
            upstream[f"{label} ({BIOME_NAMES[b]})"] = m
            print(f"  {b:<7} {label:<21} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    # 2. window ablation, validation only
    val_min = minutes(val_fused)
    ablation = {}
    for w in [1, 3, 5, 7]:
        filt = MovingAverageFilter(window_size=w)
        scores = []
        for s in val_fused:
            scores.append(continuous_scores(filt, s))
        tau_w, _ = best_threshold(scores, val_gt)
        dec = threshold_all(scores, tau_w)
        m = calculate_frame_metrics(flatten(val_gt), flatten(dec))

        fa = 0
        for i in range(len(dec)):
            fa += count_false_alerts(val_gt[i], dec[i])

        ablation[w] = {
            "window_size": w,
            "latency_ms": round((w / 24.0) * 1000.0, 1),
            "tau_star": tau_w,
            "val_f1": round(m["f1"], 4),
            "val_precision": round(m["precision"], 4),
            "val_recall": round(m["recall"], 4),
            "val_fa_events": fa,
            "val_fa_rate": round(fa / val_min, 2),
        }
        print(f"  W={w} tau*={tau_w:.2f} F1={m['f1']:.4f} FA/min={fa / val_min:.2f}")

    # 3. tau* for every method from validation
    filters = {
        "M1": BaselineFilter(),
        "M2": MovingAverageFilter(window_size=5),
        "M3": MedianFilter(window_size=5),
        "M4": HistoryConsensusFilter(window_size=5, hits=3),
    }
    taus = {}
    for key in ["M1", "M2", "M3"]:
        scores = []
        for s in val_fused:
            scores.append(continuous_scores(filters[key], s))
        taus[key], f1 = best_threshold(scores, val_gt)
        print(f"  {key} val tau*={taus[key]:.2f} (val F1 {f1:.4f})")
    taus["M4"], f1 = best_threshold_by_decide(filters["M4"], val_fused, val_gt)
    print(f"  M4 val tau*={taus['M4']:.2f} (val F1 {f1:.4f})")

    learned = {"M5": [], "M6": []}
    for key, kind in [("M5", "mamba"), ("M6", "gru")]:
        for seed in SEEDS:
            model = SequenceFilter(kind=kind, seed=seed)
            model.fit(train_fused, train_gt, epochs=40, lr=0.01)
            probs = []
            for s in val_fused:
                probs.append(model.predict_probs(s))
            tau_s, f1 = best_threshold(probs, val_gt)
            learned[key].append((model, tau_s))
            print(f"  {key} seed {seed:4d} val tau*={tau_s:.2f} (val F1 {f1:.4f})")

    # 4. test split
    runtimes = {"M1": 0.02, "M2": 0.30, "M3": 0.42, "M4": 0.63}
    states = {"M1": "0 B ($O(1)$)", "M2": "20 B ($O(1)$)", "M3": "20 B ($O(W)$)", "M4": "5 B ($O(W)$)"}
    results = {}
    for key in ["M1", "M2", "M3", "M4"]:
        dec = []
        for s in test_fused:
            dec.append(filters[key].decide(s, taus[key]))
        row = test_row(dec, test_gt, test_biomes, test_min)
        row["tau_star"] = taus[key]
        row["runtime_us"] = runtimes[key]
        row["state_size"] = states[key]
        results[key] = row

    m1_fp = results["M1"]["fp_frames_overall"]
    for key in ["M1", "M2", "M3", "M4"]:
        fasr = 0.0
        if m1_fp > 0:
            fasr = ((m1_fp - results[key]["fp_frames_overall"]) / m1_fp) * 100.0
        results[key]["fasr"] = round(fasr, 1)

    for key in ["M5", "M6"]:
        rows = []
        for model, tau in learned[key]:
            dec = []
            for s in test_fused:
                dec.append(model.decide(s, tau))
            row = test_row(dec, test_gt, test_biomes, test_min)
            row["tau_star"] = tau
            rows.append(row)
        if key == "M5":
            results[key] = aggregate_seeds(rows, m1_fp, 150.59, "512 B ($O(d_{\\text{model}} d_{\\text{state}})$)")
        else:
            results[key] = aggregate_seeds(rows, m1_fp, 21.80, "64 B ($O(d_{\\text{hidden}}^2)$)")

    print()
    for key in ["M1", "M2", "M3", "M4"]:
        r = results[key]
        print(f"  {key} tau*={r['tau_star']:.2f} recall={r['tile_recall_overall']:.1f}% F1={r['f1']:.3f} FA/min={r['fa_rate_overall']:.1f} FP={r['fp_frames_overall']}")
    for key in ["M5", "M6"]:
        r = results[key]
        print(f"  {key} tau*={r['tau_star']:.2f} recall={r['tile_recall_overall']:.1f}% F1={r['f1_mean']:.3f} +- {r['f1_std']:.3f} FA/min={r['fa_rate_mean']:.1f}")

    # 5. paired tests on clip F1
    m1_f1s = results["M1"]["episode_f1s"]
    m2_f1s = results["M2"]["episode_f1s"]
    sig = {
        "m2_vs_m1": paired_test(m1_f1s, m2_f1s),
        "m2_vs_m5": paired_test(results["M5"]["rep_seed_metrics"]["episode_f1s"], m2_f1s),
        "m2_vs_m6": paired_test(results["M6"]["rep_seed_metrics"]["episode_f1s"], m2_f1s),
    }

    # 6. tau sensitivity on test, M1 vs M2
    y_true = flatten(test_gt)
    m2_scores = []
    for s in test_fused:
        m2_scores.append(continuous_scores(filters["M2"], s))

    sensitivity = []
    for t in np.arange(0.20, 0.85, 0.05):
        t = round(t, 2)
        dec1 = threshold_all(test_fused, t)
        dec2 = threshold_all(m2_scores, t)
        m1 = calculate_frame_metrics(y_true, flatten(dec1))
        m2 = calculate_frame_metrics(y_true, flatten(dec2))
        ev1 = 0
        ev2 = 0
        for i in range(len(test_gt)):
            ev1 += count_false_alerts(test_gt[i], dec1[i])
            ev2 += count_false_alerts(test_gt[i], dec2[i])
        fa1 = ev1 / test_min
        fa2 = ev2 / test_min
        sensitivity.append({
            "tau": t,
            "m1_f1": round(m1["f1"], 4),
            "m1_fa_rate": round(fa1, 2),
            "m2_f1": round(m2["f1"], 4),
            "m2_fa_rate": round(fa2, 2),
            "m2_dominates": bool(m2["f1"] >= m1["f1"] and fa2 <= fa1),
        })

    ledger = {
        "protocol": "Strict no-leakage: detectors fixed; M5/M6 trained on train split; tau* and W calibrated on val split; test split evaluated once.",
        "upstream_detection": upstream,
        "window_ablation_val": ablation,
        "test_results": results,
        "significance_tests": sig,
        "threshold_sensitivity": sensitivity,
    }
    out = Path("runs/detect/unified_evaluation_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(ledger, f, indent=2)

    print()
    print("----------------------")
    print("Saved= ", out)
    print("----------------------")


if __name__ == "__main__":
    main()
