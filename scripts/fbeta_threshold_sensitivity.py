#   MMSAR - recall weighted threshold sensitivity (paper Table V)
#        Oumar Ibrahim U22200741

# tau* is re-picked on validation with a recall heavy objective, then M1..M4 run on test
#   --objective f2     maximise F2 (beta = 2)
#   --objective cost   minimise 5 * FN + 1 * FP
#   --min_precision    only accept tau where val precision >= this (paper uses 0.95)
#   --max_fa_rate      only accept tau where val FA/min <= this
# output  runs/detect/threshold_sensitivity_{objective}[_p95][_faN].json

import argparse
import json
from pathlib import Path

import numpy as np

from mmsar.evaluation import continuous_scores, flatten, load_cache, minutes, target_alerted
from mmsar.filters import BaselineFilter, HistoryConsensusFilter, MedianFilter, MovingAverageFilter
from mmsar.metrics import calculate_frame_metrics, count_false_alerts, time_to_alarm


def fbeta(precision, recall, beta):
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom > 0:
        return (1 + b2) * precision * recall / denom
    return 0.0


def fa_rate(decisions, gt_seqs, total_min):
    events = 0
    for i in range(len(gt_seqs)):
        events += count_false_alerts(gt_seqs[i], decisions[i])
    if total_min > 0:
        return events / total_min
    return 0.0


# val decisions for one tau: threshold the continuous scores, or run the filter (M4)
def val_decisions(inputs, tau, filt):
    out = []
    for x in inputs:
        if filt is None:
            out.append((x >= tau).astype(int).tolist())
        else:
            out.append(filt.decide(x, tau))
    return out


def sweep(inputs, gt_seqs, args, filt=None, tau_min=0.05, tau_max=0.95, tau_step=0.02):
    y_true = np.array(flatten(gt_seqs), dtype=np.int32)
    val_min = len(y_true) / (24.0 * 60.0)

    best_tau = 0.50
    best_score = -np.inf
    best_p = 0.0
    fallback_tau = 0.50
    fallback_score = -np.inf

    tau = tau_min
    while tau <= tau_max + 1e-6:
        decisions = val_decisions(inputs, tau, filt)
        y_pred = np.array(flatten(decisions), dtype=np.int32)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        # no alerts at all counts as precision 1 here
        precision = 1.0
        if tp + fp > 0:
            precision = tp / (tp + fp)
        recall = 0.0
        if tp + fn > 0:
            recall = tp / (tp + fn)

        if args.objective == "f1":
            score = fbeta(precision, recall, 1.0)
        elif args.objective in ("f2", "fbeta"):
            score = fbeta(precision, recall, args.beta)
        elif args.objective == "cost":
            score = -(args.c_fn * fn + args.c_fp * fp)
        else:
            raise ValueError("unknown objective " + args.objective)

        if score > fallback_score:
            fallback_score = score
            fallback_tau = round(tau, 4)

        ok = args.min_precision is None or precision >= args.min_precision
        if ok and args.max_fa_rate is not None:
            ok = fa_rate(decisions, gt_seqs, val_min) <= args.max_fa_rate

        if ok:
            if score > best_score or (abs(score - best_score) < 1e-9 and abs(tau - 0.50) < abs(best_tau - 0.50)):
                best_score = score
                best_tau = round(tau, 4)
                best_p = precision
        tau += tau_step

    if best_score == -np.inf:
        print(f"  constraints never met, using unconstrained tau*={fallback_tau:.2f}")
        return fallback_tau, 0.0
    return best_tau, best_p


def test_metrics(decisions, test_gt, test_min):
    fm = calculate_frame_metrics(flatten(test_gt), flatten(decisions))

    pos = []
    for i in range(len(test_gt)):
        if sum(test_gt[i]) > 0:
            pos.append(i)
    detected = 0
    delays = []
    for i in pos:
        if target_alerted(decisions[i], test_gt[i]):
            detected += 1
        delays += time_to_alarm(test_gt[i], decisions[i])

    delay_ms = 0.0
    if len(delays) > 0:
        delay_ms = float(np.mean(delays)) / 24.0 * 1000.0
    recall_pct = 0.0
    if len(pos) > 0:
        recall_pct = (detected / len(pos)) * 100.0

    return {
        "precision": round(fm["precision"], 3),
        "recall_frame": round(fm["recall"], 3),
        "f1_frame": round(fm["f1"], 3),
        "tile_recall_pct": round(recall_pct, 1),
        "detected_pos": detected,
        "total_pos_tiles": len(pos),
        "fp_frames": int(fm["fp"]),
        "fa_per_min": round(fa_rate(decisions, test_gt, test_min), 2),
        "mean_delay_ms": round(delay_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["f1", "f2", "fbeta", "cost"], default="f2")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--c_fn", type=float, default=5.0)
    ap.add_argument("--c_fp", type=float, default=1.0)
    ap.add_argument("--min_precision", type=float, default=None)
    ap.add_argument("--max_fa_rate", type=float, default=None)
    args = ap.parse_args()

    cache = load_cache()
    val_fused = cache["val_fused_seqs"]
    val_gt = cache["val_gt_seqs"]
    test_fused = cache["test_fused_seqs"]
    test_gt = cache["test_gt_seqs"]
    test_min = minutes(test_fused)

    print(f"objective {args.objective}  min precision {args.min_precision}  max FA/min {args.max_fa_rate}")
    print("method  tau*  val prec  recall   F1     FA/min  FP   delay")

    results = {}
    for key, filt in [("M1", BaselineFilter()), ("M2", MovingAverageFilter(5)), ("M3", MedianFilter(5))]:
        # threshold the continuous score directly, same as the decide() of these filters
        scores = []
        for s in val_fused:
            scores.append(np.array(continuous_scores(filt, s), dtype=np.float32))
        tau, val_p = sweep(scores, val_gt, args)
        dec = []
        for s in test_fused:
            dec.append(filt.decide(s, tau))
        res = test_metrics(dec, test_gt, test_min)
        res["tau_star"] = tau
        results[key] = res
        print(f"{key}      {tau:.2f}  {val_p:.3f}     {res['tile_recall_pct']:5.1f}%  {res['f1_frame']:.3f}  {res['fa_per_min']:5.2f}  {res['fp_frames']:3d}  {res['mean_delay_ms']} ms")

    m4 = HistoryConsensusFilter(window_size=5, hits=3)
    tau4, val_p4 = sweep(val_fused, val_gt, args, filt=m4)
    dec4 = []
    for s in test_fused:
        dec4.append(m4.decide(s, tau4))
    res4 = test_metrics(dec4, test_gt, test_min)
    res4["tau_star"] = tau4
    results["M4"] = res4
    print(f"M4      {tau4:.2f}  {val_p4:.3f}     {res4['tile_recall_pct']:5.1f}%  {res4['f1_frame']:.3f}  {res4['fa_per_min']:5.2f}  {res4['fp_frames']:3d}  {res4['mean_delay_ms']} ms")

    suffix = args.objective
    if args.min_precision is not None:
        suffix += f"_p{int(args.min_precision * 100)}"
    if args.max_fa_rate is not None:
        suffix += f"_fa{int(args.max_fa_rate)}"
    out = Path(f"runs/detect/threshold_sensitivity_{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print()
    print("----------------------")
    print("Saved= ", out)
    print("----------------------")


if __name__ == "__main__":
    main()
