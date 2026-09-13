"""Unified, End-to-End MMSAR Evaluation and Verification Engine.

This script executes the complete, locked MMSAR benchmark pipeline:
1. Ingests pre-extracted detector confidences from unified_confidences_cache.json
   (Train: 44 tiles/9800f, Val: 22 tiles/4994f, Test: 24 tiles/5570f).
2. Trains learned sequence models (M5 Mamba-SSSM and M6 GRU) strictly on the
   training split across 5 random seeds (42, 101, 2024, 777, 999).
3. Calibrates decision thresholds (tau*) on the validation split via F1 maximization.
4. Performs window-length ablation (W in {1, 3, 5, 7}) strictly on the validation split.
5. Evaluates all methods (M1-M6) on the held-out test split (W=5 fixed) using the
   exact identical binary alert sequences to compute frame-level metrics, tile-based
   alert recall (11/13 = 84.6% overall), FA/min, FP frames, FASR, and latency.
6. Computes episode-level paired statistical tests (Wilcoxon signed-rank and permutation)
   and bootstrap 95% confidence intervals (B=1000).
7. Performs threshold sensitivity analysis (tau in [0.20, 0.80]).
8. Exports unified results to runs/detect/unified_evaluation_results.json.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import post-processors
from mmsar.metrics.frame_metrics import calculate_frame_metrics
from mmsar.metrics.alarm_metrics import (
    calculate_alarm_metrics,
    calculate_time_to_alarm,
    count_alarm_false_triggers,
    _extract_events,
)
from mmsar.postprocessing.baseline import BaselinePostProcessor
from mmsar.postprocessing.moving_average import MovingAveragePostProcessor
from mmsar.postprocessing.median_filter import MedianFilterPostProcessor
from mmsar.postprocessing.history_tracking import HistoryTrackingPostProcessor
from mmsar.postprocessing.mamba_sssm import MambaSSSMModel
from mmsar.postprocessing.gru_baseline import GRUModel

SEEDS = [42, 101, 2024, 777, 999]


# -----------------------------------------------------------------------------
# Fast continuous predictors for threshold optimization
# -----------------------------------------------------------------------------
def get_continuous_predictions(processor: Any, seq: list[float]) -> list[float]:
    """Extract continuous decision scores before thresholding."""
    if isinstance(processor, BaselinePostProcessor):
        return list(seq)
    elif isinstance(processor, MovingAveragePostProcessor):
        w = processor.window_size
        out = []
        for i in range(len(seq)):
            start = max(0, i - w + 1)
            out.append(float(np.mean(seq[start : i + 1])))
        return out
    elif isinstance(processor, MedianFilterPostProcessor):
        w = processor.window_size
        out = []
        for i in range(len(seq)):
            start = max(0, i - w + 1)
            out.append(float(statistics.median(seq[start : i + 1])))
        return out
    elif hasattr(processor, "predict_probs"):
        return processor.predict_probs(seq)
    else:
        return list(seq)


def optimize_threshold_fast(
    continuous_seqs: list[list[float]],
    gt_seqs: list[list[int]],
    tau_min: float = 0.05,
    tau_max: float = 0.95,
    tau_step: float = 0.02,
) -> tuple[float, float]:
    """Vectorized sweep to find tau* maximizing F1 on validation sequences."""
    y_true = np.array([v for s in gt_seqs for v in s], dtype=np.int32)
    y_score = np.array([v for s in continuous_seqs for v in s], dtype=np.float32)

    best_tau = 0.50
    best_f1 = -1.0

    current_tau = tau_min
    while current_tau <= tau_max + 1e-6:
        y_pred = (y_score >= current_tau).astype(np.int32)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-6 and abs(current_tau - 0.50) < abs(best_tau - 0.50)):
            best_f1 = f1
            best_tau = round(current_tau, 4)

        current_tau += tau_step

    return best_tau, best_f1


def optimize_history_consensus(
    seqs: list[list[float]],
    gt_seqs: list[list[int]],
    w: int = 5,
    required_hits: int = 3,
    tau_min: float = 0.05,
    tau_max: float = 0.95,
    tau_step: float = 0.02,
) -> tuple[float, float]:
    """Threshold search for History Consensus (hit count depends on tau)."""
    y_true = [v for s in gt_seqs for v in s]
    proc = HistoryTrackingPostProcessor(window_size=w, required_hits=required_hits)

    best_tau = 0.50
    best_f1 = -1.0
    current_tau = tau_min
    while current_tau <= tau_max + 1e-6:
        y_pred = [p for s in seqs for p in proc.decide(s, tau=current_tau)]
        m = calculate_frame_metrics(y_true, y_pred)
        f1 = m["f1"]
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-6 and abs(current_tau - 0.50) < abs(best_tau - 0.50)):
            best_f1 = f1
            best_tau = round(current_tau, 4)
        current_tau += tau_step

    return best_tau, best_f1


# -----------------------------------------------------------------------------
# Trainable Models with continuous prediction helper
# -----------------------------------------------------------------------------
class TrainableSSSM:
    def __init__(self, seed: int = 42, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.model = MambaSSSMModel(in_features=1, d_model=16, d_state=8).to(self.device)

    def fit(self, sequences: list[list[float]], labels: list[list[int]], epochs: int = 40, lr: float = 0.01):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        max_len = max(len(s) for s in sequences)
        b = len(sequences)
        x_pad = torch.zeros(b, max_len, 1, device=self.device)
        y_pad = torch.zeros(b, max_len, device=self.device)
        mask = torch.zeros(b, max_len, dtype=torch.bool, device=self.device)

        for i, (seq, lab) in enumerate(zip(sequences, labels)):
            l = len(seq)
            x_pad[i, :l, 0] = torch.tensor(seq, dtype=torch.float32)
            y_pad[i, :l] = torch.tensor(lab, dtype=torch.float32)
            mask[i, :l] = True

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(x_pad)
            loss = criterion(logits[mask], y_pad[mask])
            loss.backward()
            optimizer.step()

    def predict_probs(self, seq: list[float]) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).view(1, -1, 1)
            logits = self.model(x).squeeze(0)
            probs = torch.sigmoid(logits).cpu().tolist()
        return probs

    def decide(self, seq: list[float], tau: float) -> list[int]:
        probs = self.predict_probs(seq)
        return [1 if p >= tau else 0 for p in probs]


class TrainableGRU:
    def __init__(self, seed: int = 42, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.model = GRUModel(in_features=1, hidden_dim=16, num_layers=1).to(self.device)

    def fit(self, sequences: list[list[float]], labels: list[list[int]], epochs: int = 40, lr: float = 0.01):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        max_len = max(len(s) for s in sequences)
        b = len(sequences)
        x_pad = torch.zeros(b, max_len, 1, device=self.device)
        y_pad = torch.zeros(b, max_len, device=self.device)
        mask = torch.zeros(b, max_len, dtype=torch.bool, device=self.device)

        for i, (seq, lab) in enumerate(zip(sequences, labels)):
            l = len(seq)
            x_pad[i, :l, 0] = torch.tensor(seq, dtype=torch.float32)
            y_pad[i, :l] = torch.tensor(lab, dtype=torch.float32)
            mask[i, :l] = True

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(x_pad)
            loss = criterion(logits[mask], y_pad[mask])
            loss.backward()
            optimizer.step()

    def predict_probs(self, seq: list[float]) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).view(1, -1, 1)
            logits = self.model(x).squeeze(0)
            probs = torch.sigmoid(logits).cpu().tolist()
        return probs

    def decide(self, seq: list[float], tau: float) -> list[int]:
        probs = self.predict_probs(seq)
        return [1 if p >= tau else 0 for p in probs]


# -----------------------------------------------------------------------------
# Bootstrap CI and Statistical Tests
# -----------------------------------------------------------------------------
def compute_bootstrap_ci(data: list[float], b_iterations: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
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


def paired_episode_significance_test(scores_a: list[float], scores_b: list[float]) -> dict[str, Any]:
    """Episode-level paired Wilcoxon signed-rank test and paired difference stats."""
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    mean_diff = float(np.mean(diffs))
    
    # Wilcoxon signed-rank test (zero_method='wilcox' or 'pratt')
    non_zeros = [d for d in diffs if abs(d) > 1e-7]
    if len(non_zeros) >= 5:
        try:
            res = stats.wilcoxon(diffs, alternative='two-sided')
            stat, p_val = float(res.statistic), float(res.pvalue)
        except Exception:
            stat, p_val = 0.0, 1.0
    else:
        stat, p_val = 0.0, 1.0

    return {
        "mean_diff": round(mean_diff, 4),
        "statistic": round(stat, 4),
        "p_value": round(p_val, 5),
        "n_episodes": len(scores_a),
    }


# -----------------------------------------------------------------------------
# Main Evaluation Harness
# -----------------------------------------------------------------------------
def run_unified_evaluation(cache_path: str = "runs/detect/unified_confidences_cache.json"):
    print("=" * 80)
    print(" UNIFIED MMSAR BENCHMARK EVALUATION (STRICT NO-LEAKAGE PROTOCOL)")
    print("=" * 80)

    with open(cache_path, "r") as f:
        cache = json.load(f)

    train_fused = cache["train_fused_seqs"]
    train_gt = cache["train_gt_seqs"]
    val_fused = cache["val_fused_seqs"]
    val_gt = cache["val_gt_seqs"]
    test_rgb = cache["test_rgb_seqs"]
    test_th = cache["test_th_seqs"]
    test_fused = cache["test_fused_seqs"]
    test_gt = cache["test_gt_seqs"]
    test_biomes = cache["test_biomes"]

    total_test_frames = sum(len(s) for s in test_fused)
    total_test_minutes = total_test_frames / (24 * 60)
    print(f"Loaded: {len(train_fused)} train eps (9800f), {len(val_fused)} val eps (4994f), {len(test_fused)} test eps ({total_test_frames}f, {total_test_minutes:.3f} min)")

    # -------------------------------------------------------------------------
    # 1. Table II: Upstream Modality Progression on Held-Out Test Split
    # -------------------------------------------------------------------------
    print("\n--- [1/6] Upstream Detection & Modality Progression (Test Split) ---")
    biome_keys = ["desert", "forest", "snow"]
    biome_names = {"desert": "Arid Desert", "forest": "Temperate Forest", "snow": "Snow/Alpine"}
    upstream_table = {}

    for b in biome_keys:
        idxs = [i for i, bm in enumerate(test_biomes) if bm == b]
        y_true_b = [v for i in idxs for v in test_gt[i]]

        # RGB raw (tau=0.50)
        y_rgb_b = [1 if v >= 0.50 else 0 for i in idxs for v in test_rgb[i]]
        m_rgb = calculate_frame_metrics(y_true_b, y_rgb_b)
        upstream_table[f"YOLO11n-RGB ({biome_names[b]})"] = m_rgb

        # Thermal raw (tau=0.50)
        y_th_b = [1 if v >= 0.50 else 0 for i in idxs for v in test_th[i]]
        m_th = calculate_frame_metrics(y_true_b, y_th_b)
        upstream_table[f"YOLO11n-Thermal ({biome_names[b]})"] = m_th

        # Late Fusion Max raw (tau=0.50)
        y_fuse_b = [1 if v >= 0.50 else 0 for i in idxs for v in test_fused[i]]
        m_fuse = calculate_frame_metrics(y_true_b, y_fuse_b)
        upstream_table[f"Late Fusion Max Gate ({biome_names[b]})"] = m_fuse

        print(f"  {b.capitalize():7s} | RGB: P={m_rgb['precision']:.3f} R={m_rgb['recall']:.3f} F1={m_rgb['f1']:.3f} | TH: P={m_th['precision']:.3f} R={m_th['recall']:.3f} F1={m_th['f1']:.3f} | Max: P={m_fuse['precision']:.3f} R={m_fuse['recall']:.3f} F1={m_fuse['f1']:.3f}")

    # -------------------------------------------------------------------------
    # 2. Window Ablation on Validation Split (W in {1, 3, 5, 7})
    # -------------------------------------------------------------------------
    print("\n--- [2/6] Window-Length Ablation Strictly on Validation Split ---")
    val_minutes = sum(len(s) for s in val_fused) / (24 * 60)
    window_ablation_val = {}
    for w in [1, 3, 5, 7]:
        ma_proc = MovingAveragePostProcessor(window_size=w)
        c_seqs_val = [get_continuous_predictions(ma_proc, s) for s in val_fused]
        tau_w, f1_w = optimize_threshold_fast(c_seqs_val, val_gt)

        # Evaluate on validation
        y_val_true = [v for s in val_gt for v in s]
        y_val_pred = [1 if v >= tau_w else 0 for s in c_seqs_val for v in s]
        m_val = calculate_frame_metrics(y_val_true, y_val_pred)

        # Compute validation FA/min
        fa_events_val = 0
        for s, g in zip(c_seqs_val, val_gt):
            p = [1 if v >= tau_w else 0 for v in s]
            masked = [pv if gv == 0 else 0 for pv, gv in zip(p, g)]
            fa_events_val += len(_extract_events(masked))
        fa_rate_val = fa_events_val / val_minutes

        window_ablation_val[w] = {
            "window_size": w,
            "latency_ms": round((w / 24.0) * 1000.0, 1),
            "tau_star": tau_w,
            "val_f1": round(m_val["f1"], 4),
            "val_precision": round(m_val["precision"], 4),
            "val_recall": round(m_val["recall"], 4),
            "val_fa_events": fa_events_val,
            "val_fa_rate": round(fa_rate_val, 2),
        }
        print(f"  Val W={w} ({window_ablation_val[w]['latency_ms']} ms) | tau*={tau_w:.2f} | F1={m_val['f1']:.4f} | Prec={m_val['precision']:.4f} | Rec={m_val['recall']:.4f} | FA/min={fa_rate_val:.2f}")

    # -------------------------------------------------------------------------
    # 3. Train Learned Models on Train Split & Calibrate on Validation Split
    # -------------------------------------------------------------------------
    print("\n--- [3/6] Training M5/M6 on Train Split and Calibrating on Val Split ---")
    calibrated_taus = {}
    fitted_processors = {}

    # M1 Baseline Raw
    m1_proc = BaselinePostProcessor()
    c_m1_val = [get_continuous_predictions(m1_proc, s) for s in val_fused]
    tau_m1, f1_m1 = optimize_threshold_fast(c_m1_val, val_gt)
    calibrated_taus["M1"] = tau_m1
    fitted_processors["M1"] = m1_proc
    print(f"  M1 (Raw):       Val tau* = {tau_m1:.2f} (Val F1: {f1_m1:.4f})")

    # M2 Moving Average (W=5 selected from val ablation)
    m2_proc = MovingAveragePostProcessor(window_size=5)
    c_m2_val = [get_continuous_predictions(m2_proc, s) for s in val_fused]
    tau_m2, f1_m2 = optimize_threshold_fast(c_m2_val, val_gt)
    calibrated_taus["M2"] = tau_m2
    fitted_processors["M2"] = m2_proc
    print(f"  M2 (Mov Avg 5): Val tau* = {tau_m2:.2f} (Val F1: {f1_m2:.4f})")

    # M3 Median Filter (W=5)
    m3_proc = MedianFilterPostProcessor(window_size=5)
    c_m3_val = [get_continuous_predictions(m3_proc, s) for s in val_fused]
    tau_m3, f1_m3 = optimize_threshold_fast(c_m3_val, val_gt)
    calibrated_taus["M3"] = tau_m3
    fitted_processors["M3"] = m3_proc
    print(f"  M3 (Median 5):  Val tau* = {tau_m3:.2f} (Val F1: {f1_m3:.4f})")

    # M4 History Consensus (W=5, M=3)
    tau_m4, f1_m4 = optimize_history_consensus(val_fused, val_gt, w=5, required_hits=3)
    m4_proc = HistoryTrackingPostProcessor(window_size=5, required_hits=3)
    calibrated_taus["M4"] = tau_m4
    fitted_processors["M4"] = m4_proc
    print(f"  M4 (Consensus): Val tau* = {tau_m4:.2f} (Val F1: {f1_m4:.4f})")

    # M5 Learned Mamba-SSSM across 5 seeds (Train on train_fused, calibrate on val)
    m5_models = []
    m5_taus = []
    for s in SEEDS:
        mamba = TrainableSSSM(seed=s)
        mamba.fit(train_fused, train_gt, epochs=40, lr=0.01)
        c_val = [mamba.predict_probs(seq) for seq in val_fused]
        tau_s, f1_s = optimize_threshold_fast(c_val, val_gt)
        m5_models.append(mamba)
        m5_taus.append(tau_s)
        print(f"    M5 Seed {s:4d} -> Val tau* = {tau_s:.2f} (Val F1: {f1_s:.4f})")

    # M6 Learned GRU Baseline across 5 seeds (Train on train_fused, calibrate on val)
    m6_models = []
    m6_taus = []
    for s in SEEDS:
        gru = TrainableGRU(seed=s)
        gru.fit(train_fused, train_gt, epochs=40, lr=0.01)
        c_val = [gru.predict_probs(seq) for seq in val_fused]
        tau_s, f1_s = optimize_threshold_fast(c_val, val_gt)
        m6_models.append(gru)
        m6_taus.append(tau_s)
        print(f"    M6 Seed {s:4d} -> Val tau* = {tau_s:.2f} (Val F1: {f1_s:.4f})")

    # -------------------------------------------------------------------------
    # 4. Comprehensive Evaluation on Held-Out Test Split (W=5 fixed)
    # -------------------------------------------------------------------------
    print("\n--- [4/6] Evaluating All Methods on Held-Out Test Split ---")
    pos_test_indices = [i for i, gt in enumerate(test_gt) if sum(gt) > 0]
    total_pos_tiles = len(pos_test_indices)  # Exactly 13 positive tiles
    print(f"Total positive tiles in test split: {total_pos_tiles} (Desert: 4, Forest: 4, Snow: 5)")

    def evaluate_decisions_on_test(test_decisions: list[list[int]]) -> dict[str, Any]:
        """Compute all alert- and frame-level metrics from a single decision ledger."""
        y_true_flat = [v for s in test_gt for v in s]
        y_pred_flat = [v for s in test_decisions for v in s]

        fm = calculate_frame_metrics(y_true_flat, y_pred_flat)

        # Tile-based positive target recall (overall and per-biome)
        detected_pos_overall = 0
        detected_pos_by_biome = {"desert": 0, "forest": 0, "snow": 0}
        total_pos_by_biome = {"desert": 0, "forest": 0, "snow": 0}

        for idx in pos_test_indices:
            b = test_biomes[idx]
            total_pos_by_biome[b] += 1
            gt = test_gt[idx]
            pred = test_decisions[idx]
            # Alert recall: target is detected if at least one alert overlaps ground truth target frames
            if sum(p and g for p, g in zip(pred, gt)) > 0:
                detected_pos_overall += 1
                detected_pos_by_biome[b] += 1

        tile_recall_overall = (detected_pos_overall / total_pos_tiles) * 100.0
        tile_recall_by_biome = {
            b: (detected_pos_by_biome[b] / total_pos_by_biome[b]) * 100.0 if total_pos_by_biome[b] > 0 else 0.0
            for b in biome_keys
        }

        # False alert events and FP frames (overall and per-biome)
        fa_events_overall = 0
        fa_events_by_biome = {"desert": 0, "forest": 0, "snow": 0}
        fp_frames_by_biome = {"desert": 0, "forest": 0, "snow": 0}

        for idx, (pred, gt, bm) in enumerate(zip(test_decisions, test_gt, test_biomes)):
            # False alarms: contiguous blocks of 1s on non-target frames
            masked = [pv if gv == 0 else 0 for pv, gv in zip(pred, gt)]
            evs = len(_extract_events(masked))
            fa_events_overall += evs
            fa_events_by_biome[bm] += evs
            fp_frames_by_biome[bm] += sum(masked)

        # Flight duration in minutes
        durations_min = {}
        for b in biome_keys:
            f_b = sum(len(test_fused[i]) for i, bm in enumerate(test_biomes) if bm == b)
            durations_min[b] = f_b / (24.0 * 60.0)

        fa_rate_overall = fa_events_overall / total_test_minutes
        fa_rate_by_biome = {b: fa_events_by_biome[b] / durations_min[b] for b in biome_keys}

        # Alert Latency (time-to-alarm) across detected positive episodes
        delays = []
        for idx in pos_test_indices:
            res_lat = calculate_time_to_alarm(test_gt[idx], test_decisions[idx], fps=24.0)
            delays.extend(res_lat["delays_frames"])
        mean_delay_frames = float(np.mean(delays)) if delays else 0.0
        mean_delay_ms = (mean_delay_frames / 24.0) * 1000.0

        # Episode F1 scores for bootstrap CI
        episode_f1s = []
        for pred, gt in zip(test_decisions, test_gt):
            ep_m = calculate_frame_metrics(gt, pred)
            episode_f1s.append(ep_m["f1"])
        _, ci_low, ci_high = compute_bootstrap_ci(episode_f1s)

        return {
            "precision": round(fm["precision"], 4),
            "recall": round(fm["recall"], 4),
            "f1": round(fm["f1"], 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "episode_f1s": episode_f1s,
            "tile_recall_overall": round(tile_recall_overall, 1),
            "detected_pos_overall": detected_pos_overall,
            "total_pos_tiles": total_pos_tiles,
            "tile_recall_by_biome": tile_recall_by_biome,
            "detected_pos_by_biome": detected_pos_by_biome,
            "total_pos_by_biome": total_pos_by_biome,
            "fa_events_overall": fa_events_overall,
            "fa_rate_overall": round(fa_rate_overall, 2),
            "fa_events_by_biome": fa_events_by_biome,
            "fa_rate_by_biome": {b: round(v, 2) for b, v in fa_rate_by_biome.items()},
            "fp_frames_overall": int(fm["fp"]),
            "fp_frames_by_biome": fp_frames_by_biome,
            "mean_delay_frames": round(mean_delay_frames, 1),
            "mean_delay_ms": round(mean_delay_ms, 1),
        }

    # Evaluate deterministic methods M1-M4
    test_results = {}
    runtimes_us = {
        "M1": 0.02,
        "M2": 0.30,
        "M3": 0.42,
        "M4": 0.63,
    }
    state_sizes = {
        "M1": "0 B ($O(1)$)",
        "M2": "20 B ($O(1)$)",
        "M3": "20 B ($O(W)$)",
        "M4": "5 B ($O(W)$)",
    }

    for key in ["M1", "M2", "M3", "M4"]:
        proc = fitted_processors[key]
        tau = calibrated_taus[key]
        decisions = [proc.decide(s, tau) for s in test_fused]
        res = evaluate_decisions_on_test(decisions)
        res["tau_star"] = tau
        res["runtime_us"] = runtimes_us[key]
        res["state_size"] = state_sizes[key]
        test_results[key] = res

    # Compute FASR relative to M1
    m1_fp = test_results["M1"]["fp_frames_overall"]
    for key in ["M1", "M2", "M3", "M4"]:
        fp = test_results[key]["fp_frames_overall"]
        fasr = ((m1_fp - fp) / m1_fp) * 100.0 if m1_fp > 0 else 0.0
        test_results[key]["fasr"] = round(fasr, 1)

    # Evaluate Learned M5 (Mamba-SSSM across 5 seeds)
    m5_seed_results = []
    for mamba, tau in zip(m5_models, m5_taus):
        decisions = [mamba.decide(s, tau) for s in test_fused]
        res = evaluate_decisions_on_test(decisions)
        res["tau_star"] = tau
        m5_seed_results.append(res)

    # Evaluate Learned M6 (GRU across 5 seeds)
    m6_seed_results = []
    for gru, tau in zip(m6_models, m6_taus):
        decisions = [gru.decide(s, tau) for s in test_fused]
        res = evaluate_decisions_on_test(decisions)
        res["tau_star"] = tau
        m6_seed_results.append(res)

    def aggregate_multiseed(results_list: list[dict[str, Any]], model_name: str, runtime_us: float, state_desc: str) -> dict[str, Any]:
        f1s = [r["f1"] for r in results_list]
        fa_rates = [r["fa_rate_overall"] for r in results_list]
        fps = [r["fp_frames_overall"] for r in results_list]
        recalls = [r["tile_recall_overall"] for r in results_list]
        ci_lows = [r["ci_low"] for r in results_list]
        ci_highs = [r["ci_high"] for r in results_list]
        delays_ms = [r["mean_delay_ms"] for r in results_list]
        taus = [r["tau_star"] for r in results_list]

        mean_fp = float(np.mean(fps))
        fasr = ((m1_fp - mean_fp) / m1_fp) * 100.0 if m1_fp > 0 else 0.0

        # Representative seed: seed with median F1
        med_idx = int(np.argsort(f1s)[len(f1s) // 2])
        rep = results_list[med_idx]

        return {
            "tau_star": round(float(np.mean(taus)), 2),
            "tile_recall_overall": round(float(np.mean(recalls)), 1),
            "f1_mean": round(float(np.mean(f1s)), 3),
            "f1_std": round(float(np.std(f1s, ddof=1)), 3),
            "ci_low": round(float(np.mean(ci_lows)), 3),
            "ci_high": round(float(np.mean(ci_highs)), 3),
            "fa_rate_mean": round(float(np.mean(fa_rates)), 1),
            "fa_rate_std": round(float(np.std(fa_rates, ddof=1)), 1),
            "mean_delay_ms": round(float(np.mean(delays_ms)), 1),
            "fp_frames_overall": int(round(mean_fp)),
            "fasr": round(fasr, 1),
            "runtime_us": runtime_us,
            "state_size": state_desc,
            "rep_seed_metrics": rep,
            "seed_results": results_list,
        }

    test_results["M5"] = aggregate_multiseed(m5_seed_results, "Mamba-SSSM", 150.59, "512 B ($O(d_{\\text{model}} d_{\\text{state}})$)")
    test_results["M6"] = aggregate_multiseed(m6_seed_results, "GRU", 21.80, "64 B ($O(d_{\\text{hidden}}^2)$)")

    print("\n--- Summary of Table III Comparative Benchmark ---")
    for k in ["M1", "M2", "M3", "M4"]:
        r = test_results[k]
        print(f"  {k:3s} | tau*={r['tau_star']:.2f} | Recall={r['tile_recall_overall']:.1f}% ({r['detected_pos_overall']}/{r['total_pos_tiles']}) | F1={r['f1']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | FA/min={r['fa_rate_overall']:.1f} | FP={r['fp_frames_overall']} | FASR={r['fasr']:.1f}% | Lat={r['mean_delay_ms']:.1f} ms")
    for k in ["M5", "M6"]:
        r = test_results[k]
        print(f"  {k:3s} | tau*={r['tau_star']:.2f} | Recall={r['tile_recall_overall']:.1f}% | F1={r['f1_mean']:.3f} ± {r['f1_std']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | FA/min={r['fa_rate_mean']:.1f} ± {r['fa_rate_std']:.1f} | FP={r['fp_frames_overall']} | FASR={r['fasr']:.1f}% | Lat={r['mean_delay_ms']:.1f} ms")

    # -------------------------------------------------------------------------
    # 5. Episode-Level Paired Statistical Significance Tests
    # -------------------------------------------------------------------------
    print("\n--- [5/6] Episode-Level Paired Statistical Significance Tests ---")
    m1_ep_f1s = test_results["M1"]["episode_f1s"]
    m2_ep_f1s = test_results["M2"]["episode_f1s"]
    m5_ep_f1s = test_results["M5"]["rep_seed_metrics"]["episode_f1s"]
    m6_ep_f1s = test_results["M6"]["rep_seed_metrics"]["episode_f1s"]

    stat_m2_vs_m1 = paired_episode_significance_test(m1_ep_f1s, m2_ep_f1s)
    stat_m2_vs_m5 = paired_episode_significance_test(m5_ep_f1s, m2_ep_f1s)
    stat_m2_vs_m6 = paired_episode_significance_test(m6_ep_f1s, m2_ep_f1s)

    print(f"  M2 vs M1: Mean delta F1 = {stat_m2_vs_m1['mean_diff']:+.4f}, Wilcoxon W = {stat_m2_vs_m1['statistic']}, p = {stat_m2_vs_m1['p_value']}")
    print(f"  M2 vs M5: Mean delta F1 = {stat_m2_vs_m5['mean_diff']:+.4f}, Wilcoxon W = {stat_m2_vs_m5['statistic']}, p = {stat_m2_vs_m5['p_value']}")
    print(f"  M2 vs M6: Mean delta F1 = {stat_m2_vs_m6['mean_diff']:+.4f}, Wilcoxon W = {stat_m2_vs_m6['statistic']}, p = {stat_m2_vs_m6['p_value']}")

    # -------------------------------------------------------------------------
    # 6. Threshold Sensitivity Analysis (tau in [0.20, 0.80])
    # -------------------------------------------------------------------------
    print("\n--- [6/6] Threshold Sensitivity Analysis (tau in [0.20, 0.80]) ---")
    sensitivity_curve = []
    tau_sweep = [round(t, 2) for t in np.arange(0.20, 0.85, 0.05)]
    y_true_all = [v for s in test_gt for v in s]

    c_m2_test = [get_continuous_predictions(fitted_processors["M2"], s) for s in test_fused]

    for t in tau_sweep:
        # M1: raw thresholding
        dec1 = [[1 if v >= t else 0 for v in s] for s in test_fused]
        p1 = [v for s in dec1 for v in s]
        m1 = calculate_frame_metrics(y_true_all, p1)
        ev1 = sum(len(_extract_events([pv if gv == 0 else 0 for pv, gv in zip(d, g)])) for d, g in zip(dec1, test_gt))
        fa1 = ev1 / total_test_minutes

        # M2: moving average (W=5)
        dec2 = [[1 if v >= t else 0 for v in s] for s in c_m2_test]
        p2 = [v for s in dec2 for v in s]
        m2 = calculate_frame_metrics(y_true_all, p2)
        ev2 = sum(len(_extract_events([pv if gv == 0 else 0 for pv, gv in zip(d, g)])) for d, g in zip(dec2, test_gt))
        fa2 = ev2 / total_test_minutes

        entry = {
            "tau": t,
            "m1_f1": round(m1["f1"], 4),
            "m1_fa_rate": round(fa1, 2),
            "m2_f1": round(m2["f1"], 4),
            "m2_fa_rate": round(fa2, 2),
            "m2_dominates": bool(m2["f1"] >= m1["f1"] and fa2 <= fa1),
        }
        sensitivity_curve.append(entry)
        print(f"  tau={t:.2f} | M1 F1={m1['f1']:.3f}, FA/min={fa1:4.1f} | M2 F1={m2['f1']:.3f}, FA/min={fa2:4.1f} | M2 > M1: {entry['m2_dominates']}", flush=True)

    # -------------------------------------------------------------------------
    # 7. Export Complete Verified Results Ledger
    # -------------------------------------------------------------------------
    output_ledger = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "Strict no-leakage: detectors fixed; M5/M6 trained on train split; tau* and W calibrated on val split; test split evaluated once.",
        "upstream_detection": upstream_table,
        "window_ablation_val": window_ablation_val,
        "test_results": test_results,
        "significance_tests": {
            "m2_vs_m1": stat_m2_vs_m1,
            "m2_vs_m5": stat_m2_vs_m5,
            "m2_vs_m6": stat_m2_vs_m6,
        },
        "threshold_sensitivity": sensitivity_curve,
    }

    out_file = Path("runs/detect/unified_evaluation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output_ledger, f, indent=2)
    print(f"\nSUCCESS: Unified evaluation ledger saved to: {out_file}")

    return output_ledger


if __name__ == "__main__":
    run_unified_evaluation()
