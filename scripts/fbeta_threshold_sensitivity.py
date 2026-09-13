"""Recall-weighted threshold sensitivity analysis with operational constraints."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmsar.metrics.frame_metrics import calculate_frame_metrics
from mmsar.metrics.alarm_metrics import _extract_events, calculate_time_to_alarm
from mmsar.postprocessing.baseline import BaselinePostProcessor
from mmsar.postprocessing.moving_average import MovingAveragePostProcessor
from mmsar.postprocessing.median_filter import MedianFilterPostProcessor
from mmsar.postprocessing.history_tracking import HistoryTrackingPostProcessor

CACHE_CANDIDATES = [
    Path("runs/detect/unified_confidences_cache.json"),
    Path("runs/detect/extracted_confidences_cache.json"),
]


def load_cache() -> dict:
    for p in CACHE_CANDIDATES:
        if p.exists():
            print(f"Loading cached confidences from: {p}")
            return json.loads(p.read_text())
    raise FileNotFoundError("No confidence cache found in runs/detect/.")


def get_continuous_predictions(processor, seq: list[float]) -> list[float]:
    if isinstance(processor, BaselinePostProcessor):
        return list(seq)
    if isinstance(processor, MovingAveragePostProcessor):
        w = processor.window_size
        return [float(np.mean(seq[max(0, i - w + 1) : i + 1])) for i in range(len(seq))]
    if isinstance(processor, MedianFilterPostProcessor):
        w = processor.window_size
        return [float(statistics.median(seq[max(0, i - w + 1) : i + 1])) for i in range(len(seq))]
    return list(seq)


def fbeta_score(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def compute_fa_rate(decisions: list[list[int]], ground_truth: list[list[int]], total_minutes: float) -> float:
    fa_events = 0
    for pred, gt in zip(decisions, ground_truth):
        masked = [pv if gv == 0 else 0 for pv, gv in zip(pred, gt)]
        fa_events += len(_extract_events(masked))
    return fa_events / total_minutes if total_minutes > 0 else 0.0


def sweep_threshold(
    continuous_seqs: list[list[float]],
    gt_seqs: list[list[int]],
    objective: str,
    beta: float,
    c_fn: float,
    c_fp: float,
    min_precision: Optional[float] = None,
    max_fa_rate: Optional[float] = None,
    tau_min: float = 0.05,
    tau_max: float = 0.95,
    tau_step: float = 0.02,
) -> tuple[float, float, float, float]:
    y_true = np.array([v for s in gt_seqs for v in s], dtype=np.int32)
    y_score = np.array([v for s in continuous_seqs for v in s], dtype=np.float32)
    total_val_minutes = len(y_true) / (24.0 * 60.0)

    best_tau, best_score = 0.50, -np.inf
    best_p, best_fa = 0.0, 0.0
    fallback_tau, fallback_score = 0.50, -np.inf

    tau = tau_min
    while tau <= tau_max + 1e-6:
        y_pred = (y_score >= tau).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        if objective in ("f2", "fbeta"):
            score = fbeta_score(precision, recall, beta)
        elif objective == "cost":
            score = -(c_fn * fn + c_fp * fp)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if score > fallback_score:
            fallback_score, fallback_tau = score, round(tau, 4)

        satisfies_prec = min_precision is None or precision >= min_precision
        satisfies_fa = True
        if satisfies_prec and max_fa_rate is not None:
            decisions = [[int(v >= tau) for v in s] for s in continuous_seqs]
            fa_rate = compute_fa_rate(decisions, gt_seqs, total_val_minutes)
            satisfies_fa = fa_rate <= max_fa_rate
        else:
            fa_rate = 0.0

        if satisfies_prec and satisfies_fa:
            if score > best_score or (abs(score - best_score) < 1e-9 and abs(tau - 0.50) < abs(best_tau - 0.50)):
                best_score = score
                best_tau = round(tau, 4)
                best_p = precision
                best_fa = fa_rate

        tau += tau_step

    if best_score == -np.inf:
        print(f"  [WARN] Constraints not met. Fallback to tau*={fallback_tau:.2f}")
        return fallback_tau, fallback_score, 0.0, 0.0

    return best_tau, best_score, best_p, best_fa


def sweep_history_consensus(
    seqs: list[list[float]],
    gt_seqs: list[list[int]],
    objective: str,
    beta: float,
    c_fn: float,
    c_fp: float,
    min_precision: Optional[float] = None,
    max_fa_rate: Optional[float] = None,
    w: int = 5,
    hits: int = 3,
    tau_min: float = 0.05,
    tau_max: float = 0.95,
    tau_step: float = 0.02,
) -> tuple[float, float, float, float]:
    proc = HistoryTrackingPostProcessor(window_size=w, required_hits=hits)
    y_true = np.array([v for s in gt_seqs for v in s], dtype=np.int32)
    total_val_minutes = len(y_true) / (24.0 * 60.0)

    best_tau, best_score = 0.50, -np.inf
    best_p, best_fa = 0.0, 0.0
    fallback_tau, fallback_score = 0.50, -np.inf

    tau = tau_min
    while tau <= tau_max + 1e-6:
        decisions = [proc.decide(s, tau=tau) for s in seqs]
        y_pred = np.array([p for s in decisions for p in s], dtype=np.int32)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        if objective in ("f2", "fbeta"):
            score = fbeta_score(precision, recall, beta)
        else:
            score = -(c_fn * fn + c_fp * fp)

        if score > fallback_score:
            fallback_score, fallback_tau = score, round(tau, 4)

        satisfies_prec = min_precision is None or precision >= min_precision
        satisfies_fa = True
        if satisfies_prec and max_fa_rate is not None:
            fa_rate = compute_fa_rate(decisions, gt_seqs, total_val_minutes)
            satisfies_fa = fa_rate <= max_fa_rate
        else:
            fa_rate = 0.0

        if satisfies_prec and satisfies_fa:
            if score > best_score or (abs(score - best_score) < 1e-9 and abs(tau - 0.50) < abs(best_tau - 0.50)):
                best_score = score
                best_tau = round(tau, 4)
                best_p = precision
                best_fa = fa_rate

        tau += tau_step

    if best_score == -np.inf:
        print(f"  [WARN] Constraints not met for M4. Fallback to tau*={fallback_tau:.2f}")
        return fallback_tau, fallback_score, 0.0, 0.0

    return best_tau, best_score, best_p, best_fa


def evaluate_on_test(
    decisions: list[list[int]], test_gt: list[list[int]], total_test_minutes: float, pos_test_indices: list[int], total_pos_tiles: int
) -> dict:
    y_true_flat = [v for s in test_gt for v in s]
    y_pred_flat = [v for s in decisions for v in s]
    fm = calculate_frame_metrics(y_true_flat, y_pred_flat)

    detected_pos = 0
    for idx in pos_test_indices:
        gt, pred = test_gt[idx], decisions[idx]
        if sum(p and g for p, g in zip(pred, gt)) > 0:
            detected_pos += 1
    tile_recall = (detected_pos / total_pos_tiles) * 100.0 if total_pos_tiles else 0.0

    fa_events = 0
    for pred, gt in zip(decisions, test_gt):
        masked = [pv if gv == 0 else 0 for pv, gv in zip(pred, gt)]
        fa_events += len(_extract_events(masked))
    fa_rate = fa_events / total_test_minutes if total_test_minutes else 0.0

    delays = []
    for idx in pos_test_indices:
        res = calculate_time_to_alarm(test_gt[idx], decisions[idx], fps=24.0)
        delays.extend(res["delays_frames"])
    mean_delay_ms = (float(np.mean(delays)) / 24.0 * 1000.0) if delays else 0.0

    return {
        "precision": round(fm["precision"], 3),
        "recall_frame": round(fm["recall"], 3),
        "f1_frame": round(fm["f1"], 3),
        "tile_recall_pct": round(tile_recall, 1),
        "detected_pos": detected_pos,
        "total_pos_tiles": total_pos_tiles,
        "fp_frames": int(fm["fp"]),
        "fa_per_min": round(fa_rate, 2),
        "mean_delay_ms": round(mean_delay_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", choices=["f1", "f2", "fbeta", "cost"], default="f2")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--c_fn", type=float, default=5.0)
    ap.add_argument("--c_fp", type=float, default=1.0)
    ap.add_argument("--min_precision", type=float, default=None)
    ap.add_argument("--max_fa_rate", type=float, default=None)
    args = ap.parse_args()

    cache = load_cache()
    val_fused, val_gt = cache["val_fused_seqs"], cache["val_gt_seqs"]
    test_fused, test_gt = cache["test_fused_seqs"], cache["test_gt_seqs"]
    total_test_minutes = sum(len(s) for s in test_fused) / (24.0 * 60.0)
    pos_test_indices = [i for i, gt in enumerate(test_gt) if sum(gt) > 0]
    total_pos_tiles = len(pos_test_indices)

    methods = {
        "M1": BaselinePostProcessor(),
        "M2": MovingAveragePostProcessor(window_size=5),
        "M3": MedianFilterPostProcessor(window_size=5),
    }

    constr_info = []
    if args.min_precision is not None:
        constr_info.append(f"Val Prec >= {args.min_precision:.2f}")
    if args.max_fa_rate is not None:
        constr_info.append(f"Val FA/min <= {args.max_fa_rate:.1f}")
    constr_str = f" [Constraints: {', '.join(constr_info)}]" if constr_info else " [Unconstrained]"

    tag = f"beta={args.beta}" if args.objective in ("f2", "fbeta") else f"c_fn={args.c_fn}, c_fp={args.c_fp}"
    print(f"\nOptimization: {args.objective} ({tag}){constr_str}")
    print(f"{'Method':10s} {'tau*':>6s} {'ValPrec':>8s} {'TileRecall':>11s} {'FrameF1':>8s} {'FA/min':>8s} {'FPframes':>9s} {'Delay(ms)':>10s}")

    results = {}
    for key, proc in methods.items():
        c_val = [get_continuous_predictions(proc, s) for s in val_fused]
        tau, _, val_p, _ = sweep_threshold(
            c_val, val_gt, args.objective, args.beta, args.c_fn, args.c_fp,
            min_precision=args.min_precision, max_fa_rate=args.max_fa_rate
        )
        decisions = [proc.decide(s, tau) for s in test_fused]
        res = evaluate_on_test(decisions, test_gt, total_test_minutes, pos_test_indices, total_pos_tiles)
        res["tau_star"] = tau
        results[key] = res
        print(f"{key:10s} {tau:6.2f} {val_p:8.3f} {res['tile_recall_pct']:9.1f}% {res['f1_frame']:8.3f} {res['fa_per_min']:8.2f} {res['fp_frames']:9d} {res['mean_delay_ms']:10.1f}")

    tau4, _, val_p4, _ = sweep_history_consensus(
        val_fused, val_gt, args.objective, args.beta, args.c_fn, args.c_fp,
        min_precision=args.min_precision, max_fa_rate=args.max_fa_rate
    )
    proc4 = HistoryTrackingPostProcessor(window_size=5, required_hits=3)
    decisions4 = [proc4.decide(s, tau4) for s in test_fused]
    res4 = evaluate_on_test(decisions4, test_gt, total_test_minutes, pos_test_indices, total_pos_tiles)
    res4["tau_star"] = tau4
    results["M4"] = res4
    print(f"{'M4':10s} {tau4:6.2f} {val_p4:8.3f} {res4['tile_recall_pct']:9.1f}% {res4['f1_frame']:8.3f} {res4['fa_per_min']:8.2f} {res4['fp_frames']:9d} {res4['mean_delay_ms']:10.1f}")

    out_suffix = f"{args.objective}"
    if args.min_precision is not None:
        out_suffix += f"_p{int(args.min_precision*100)}"
    if args.max_fa_rate is not None:
        out_suffix += f"_fa{int(args.max_fa_rate)}"

    out_path = Path(f"runs/detect/threshold_sensitivity_{out_suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
