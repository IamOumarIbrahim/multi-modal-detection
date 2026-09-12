"""Threshold optimization utilities for post-processing methods."""

from typing import Sequence, Any, Union
import numpy as np
from mmsar.postprocessing.base import PostProcessor
from mmsar.metrics.frame_metrics import calculate_frame_metrics


def find_optimal_threshold(
    processor: PostProcessor,
    confidence_sequences: Sequence[Sequence[float]],
    ground_truth_sequences: Sequence[Sequence[int]],
    tau_min: float = 0.05,
    tau_max: float = 0.95,
    tau_step: float = 0.02,
    metric: str = "f1",
) -> tuple[float, float]:
    """Find the optimal decision threshold tau* on validation sequences.

    Sweeps tau across [tau_min, tau_max] with step tau_step to maximize the
    specified metric (default: F1-score) across the concatenated validation sequences.

    Args:
        processor: PostProcessor instance with decide(confidence, tau) method.
        confidence_sequences: Sequence of per-frame confidence lists.
        ground_truth_sequences: Sequence of per-frame binary ground truth lists.
        tau_min: Lower bound for threshold sweep.
        tau_max: Upper bound for threshold sweep.
        tau_step: Step size for threshold sweep.
        metric: Optimization target metric ("f1", "precision", "recall").

    Returns:
        tuple (tau_star, best_score): Optimal threshold and corresponding score on validation set.
    """
    if len(confidence_sequences) != len(ground_truth_sequences):
        raise ValueError("Mismatched sequence counts between confidences and ground truth.")

    # Flatten ground truth for evaluation
    y_true_flat: list[int] = []
    for gt in ground_truth_sequences:
        y_true_flat.extend(gt)

    best_tau = 0.50
    best_score = -1.0

    current_tau = tau_min
    while current_tau <= tau_max + 1e-6:
        y_pred_flat: list[int] = []
        for seq in confidence_sequences:
            decisions = processor.decide(seq, tau=current_tau)
            y_pred_flat.extend(decisions)

        metrics = calculate_frame_metrics(y_true_flat, y_pred_flat)
        score = metrics.get(metric, 0.0)

        # Tie-breaker: prefer threshold closest to 0.50
        if score > best_score or (abs(score - best_score) < 1e-6 and abs(current_tau - 0.50) < abs(best_tau - 0.50)):
            best_score = score
            best_tau = round(current_tau, 4)

        current_tau += tau_step

    return best_tau, best_score


def optimize_all_thresholds(
    processors: dict[str, PostProcessor],
    confidence_sequences: Sequence[Sequence[float]],
    ground_truth_sequences: Sequence[Sequence[int]],
    tau_min: float = 0.05,
    tau_max: float = 0.95,
    tau_step: float = 0.02,
    metric: str = "f1",
) -> dict[str, dict[str, float]]:
    """Tune optimal threshold tau* for a dictionary of post-processing methods.

    Args:
        processors: Dict mapping method name to PostProcessor instance.
        confidence_sequences: Sequence of per-frame confidence lists.
        ground_truth_sequences: Sequence of per-frame binary ground truth lists.
        tau_min: Lower bound for sweep.
        tau_max: Upper bound for sweep.
        tau_step: Step size for sweep.
        metric: Optimization metric.

    Returns:
        Dict mapping method name to dict containing "tau_star" and "val_score".
    """
    results: dict[str, dict[str, float]] = {}
    for name, proc in processors.items():
        tau_star, val_score = find_optimal_threshold(
            processor=proc,
            confidence_sequences=confidence_sequences,
            ground_truth_sequences=ground_truth_sequences,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_step=tau_step,
            metric=metric,
        )
        results[name] = {
            "tau_star": tau_star,
            "val_score": val_score,
        }
    return results
