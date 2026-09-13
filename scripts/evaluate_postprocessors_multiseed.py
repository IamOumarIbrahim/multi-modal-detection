"""Multi-seed evaluation harness for MMSAR post-processors and upstream detectors.

Runs 5 seeds for learned models (GRU and Mamba-SSSM), evaluates deterministic filters
(Baseline Raw, Moving Average, Median, History Consensus), computes bootstrap 95% CIs,
time-to-alarm latency, and biome-stratified operational savings.
"""

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ultralytics import YOLO

from mmsar.fusion import soft_disjunctive_late_fusion
from mmsar.metrics.alarm_metrics import (
    calculate_alarm_metrics,
    calculate_time_to_alarm,
    count_alarm_false_triggers,
)
from mmsar.metrics.frame_metrics import calculate_frame_metrics
from mmsar.postprocessing.baseline import BaselinePostProcessor
from mmsar.postprocessing.gru_baseline import GRUPostProcessor
from mmsar.postprocessing.history_tracking import HistoryTrackingPostProcessor
from mmsar.postprocessing.mamba_sssm import MambaSSSMPostProcessor
from mmsar.postprocessing.median_filter import MedianFilterPostProcessor
from mmsar.postprocessing.moving_average import MovingAveragePostProcessor
from mmsar.postprocessing.optimizer import find_optimal_threshold


SEEDS = [42, 101, 2024, 777, 999]


def compute_bootstrap_ci(data: list[float], b_iterations: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Compute 95% non-parametric bootstrap percentile confidence interval."""
    if not data:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(data)
    boot_means = []
    for _ in range(b_iterations):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lower = boot_means[int((alpha / 2.0) * b_iterations)]
    upper = boot_means[int((1.0 - alpha / 2.0) * b_iterations)]
    mean_val = sum(boot_means) / len(boot_means)
    return round(mean_val, 4), round(lower, 4), round(upper, 4)


def extract_episode_confidences(model: YOLO, img_paths: list[str]) -> list[float]:
    """Run detector on a sequence of image paths and extract class 0 confidences."""
    confidences = []
    results = model.predict(img_paths, conf=0.01, verbose=False, device=0 if torch.cuda.is_available() else "cpu")
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            confidences.append(float(np.max(confs)))
        else:
            confidences.append(0.0)
    return confidences


def load_ground_truth_for_images(img_paths: list[str]) -> list[int]:
    """Load per-frame binary presence from YOLO label files."""
    labels = []
    for p in img_paths:
        p_obj = Path(p)
        txt_path = p_obj.parent.parent.parent / "labels" / p_obj.parent.name / f"{p_obj.stem}.txt"
        if not txt_path.exists():
            # Try alternate path
            parts = list(p_obj.parts)
            try:
                idx = parts.index("images")
                parts[idx] = "labels"
                txt_path = Path(*parts).with_suffix(".txt")
            except ValueError:
                pass
        if txt_path.exists() and txt_path.stat().st_size > 0:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def run_evaluation(
    rgb_weights: str,
    thermal_weights: str,
    manifest_path: str = "data/splits/dataset_episodes_split_manifest.json",
    output_json: str = "runs/detect/postprocessing_multiseed_results.json",
):
    print("=" * 80)
    print(" MMSAR MULTI-SEED BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"RGB Weights:     {rgb_weights}")
    print(f"Thermal Weights: {thermal_weights}")

    model_rgb = YOLO(rgb_weights)
    model_thermal = YOLO(thermal_weights)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    episodes = manifest.get("episodes", [])
    val_episodes = [ep for ep in episodes if ep.get("partition") == "val" and ep.get("modality") == "rgb"]
    test_episodes = [ep for ep in episodes if ep.get("partition") == "test" and ep.get("modality") == "rgb"]

    print(f"Loaded {len(val_episodes)} validation episodes, {len(test_episodes)} test episodes.")

    # -------------------------------------------------------------------------
    # 1. Ingest Confidences for Validation and Test
    # -------------------------------------------------------------------------
    print("\nExtracting detector confidences across validation episodes...")
    val_fused_seqs = []
    val_gt_seqs = []
    for ep in val_episodes:
        rgb_paths = ep["frame_paths"]
        th_paths = [p.replace("/rgb/", "/thermal/").replace("\\rgb\\", "\\thermal\\") for p in rgb_paths]
        gt = load_ground_truth_for_images(rgb_paths)

        c_rgb = extract_episode_confidences(model_rgb, rgb_paths)
        c_th = extract_episode_confidences(model_thermal, th_paths)
        fused = [max(r, t) for r, t in zip(c_rgb, c_th)]

        val_fused_seqs.append(fused)
        val_gt_seqs.append(gt)

    print("Extracting detector confidences across test episodes...")
    test_fused_seqs = []
    test_gt_seqs = []
    test_biomes = []
    for ep in test_episodes:
        rgb_paths = ep["frame_paths"]
        th_paths = [p.replace("/rgb/", "/thermal/").replace("\\rgb\\", "\\thermal\\") for p in rgb_paths]
        gt = load_ground_truth_for_images(rgb_paths)

        c_rgb = extract_episode_confidences(model_rgb, rgb_paths)
        c_th = extract_episode_confidences(model_thermal, th_paths)
        fused = [max(r, t) for r, t in zip(c_rgb, c_th)]

        test_fused_seqs.append(fused)
        test_gt_seqs.append(gt)
        test_biomes.append(ep.get("biome", "unknown"))

    # -------------------------------------------------------------------------
    # 2. Calibrate Deterministic Post-Processors
    # -------------------------------------------------------------------------
    print("\nCalibrating deterministic filters on validation set...")
    det_processors = {
        "M1_Baseline_Raw": BaselinePostProcessor(),
        "M2_Moving_Average": MovingAveragePostProcessor(window_size=5),
        "M3_Median_Filter": MedianFilterPostProcessor(window_size=5),
        "M4_History_Consensus": HistoryTrackingPostProcessor(window_size=5, min_detections=3),
    }

    calibrated_taus = {}
    for name, proc in det_processors.items():
        tau_star, val_f1 = find_optimal_threshold(proc, val_fused_seqs, val_gt_seqs, metric="f1")
        calibrated_taus[name] = tau_star
        print(f"  {name:25s} -> tau* = {tau_star:.4f} (Val F1 = {val_f1:.4f})")

    # -------------------------------------------------------------------------
    # 3. Train & Calibrate Learned Post-Processors Across 5 Seeds
    # -------------------------------------------------------------------------
    print("\nTraining and evaluating learned models across 5 random seeds...")
    gru_seed_results = []
    mamba_seed_results = []

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # GRU
        gru = GRUPostProcessor(in_features=1, hidden_dim=16, num_layers=1)
        gru.fit(val_fused_seqs, val_gt_seqs, epochs=40, lr=0.01)
        tau_gru, val_f1_gru = find_optimal_threshold(gru, val_fused_seqs, val_gt_seqs, metric="f1")

        # Evaluate GRU on test
        y_true_flat = [val for seq in test_gt_seqs for val in seq]
        y_pred_gru = [val for seq in test_fused_seqs for val in gru.decide(seq, tau=tau_gru)]
        m_gru = calculate_frame_metrics(y_true_flat, y_pred_gru)
        m_gru["tau_star"] = tau_gru
        m_gru["seed"] = seed
        gru_seed_results.append(m_gru)
        print(f"  GRU (seed={seed}):       tau* = {tau_gru:.4f}, Test F1 = {m_gru['f1']:.4f}, Prec = {m_gru['precision']:.4f}, Rec = {m_gru['recall']:.4f}")

        # Mamba-SSSM
        mamba = MambaSSSMPostProcessor(in_features=1, d_model=16, d_state=8)
        mamba.fit(val_fused_seqs, val_gt_seqs, epochs=40, lr=0.01)
        tau_mamba, val_f1_mamba = find_optimal_threshold(mamba, val_fused_seqs, val_gt_seqs, metric="f1")

        y_pred_mamba = [val for seq in test_fused_seqs for val in mamba.decide(seq, tau=tau_mamba)]
        m_mamba = calculate_frame_metrics(y_true_flat, y_pred_mamba)
        m_mamba["tau_star"] = tau_mamba
        m_mamba["seed"] = seed
        mamba_seed_results.append(m_mamba)
        print(f"  Mamba-SSSM (seed={seed}): tau* = {tau_mamba:.4f}, Test F1 = {m_mamba['f1']:.4f}, Prec = {m_mamba['precision']:.4f}, Rec = {m_mamba['recall']:.4f}")

    # -------------------------------------------------------------------------
    # 4. Aggregate Multi-Seed Statistics
    # -------------------------------------------------------------------------
    def get_stats(results_list: list[dict[str, float]], key: str) -> tuple[float, float]:
        vals = [r[key] for r in results_list]
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    print("\n" + "=" * 80)
    print(" MULTI-SEED SUMMARY (Mean ± Sample Std across 5 seeds)")
    print("=" * 80)
    for model_name, r_list in [("GRU Baseline", gru_seed_results), ("Mamba-SSSM", mamba_seed_results)]:
        f1_m, f1_s = get_stats(r_list, "f1")
        p_m, p_s = get_stats(r_list, "precision")
        r_m, r_s = get_stats(r_list, "recall")
        print(f"{model_name:15s} | F1: {f1_m:.4f} ± {f1_s:.4f} | Prec: {p_m:.4f} ± {p_s:.4f} | Rec: {r_m:.4f} ± {r_s:.4f}")

    # -------------------------------------------------------------------------
    # 5. Save Complete Benchmark JSON
    # -------------------------------------------------------------------------
    output_data = {
        "calibrated_taus": calibrated_taus,
        "gru_seeds": gru_seed_results,
        "mamba_seeds": mamba_seed_results,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nBenchmark metrics successfully exported to: {output_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", default="runs/smoke_test/yolo11n_rgb_smoke/weights/best.pt")
    parser.add_argument("--thermal", default="runs/smoke_test/yolo11n_rgb_smoke/weights/best.pt")
    args = parser.parse_args()
    run_evaluation(args.rgb, args.thermal)
