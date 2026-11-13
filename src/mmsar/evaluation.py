#   MMSAR - shared evaluation helpers used by the benchmark scripts
#        Oumar Ibrahim U22200741

import json
import random
import statistics

import numpy as np

from mmsar.filters import BaselineFilter, MedianFilter, MovingAverageFilter
from mmsar.metrics import calculate_frame_metrics, count_false_alerts, time_to_alarm

FPS = 24.0
CACHE_PATH = "runs/detect/unified_confidences_cache.json"


def load_cache(path=CACHE_PATH):
    with open(path) as f:
        return json.load(f)


def flatten(seqs):
    out = []
    for s in seqs:
        for v in s:
            out.append(v)
    return out


def minutes(seqs):
    total = 0
    for s in seqs:
        total += len(s)
    return total / (FPS * 60)


# continuous score before the threshold, used for the fast threshold sweep
def continuous_scores(filt, seq):
    out = []
    if isinstance(filt, MovingAverageFilter):
        w = filt.window_size
        for i in range(len(seq)):
            start = max(0, i - w + 1)
            out.append(float(np.mean(seq[start : i + 1])))
    elif isinstance(filt, MedianFilter):
        w = filt.window_size
        for i in range(len(seq)):
            start = max(0, i - w + 1)
            out.append(float(statistics.median(seq[start : i + 1])))
    elif isinstance(filt, BaselineFilter):
        out = list(seq)
    else:
        out = filt.predict_probs(seq)
    return out


# sweep tau in [0.05, 0.95] step 0.02 and keep the best f1
# ties go to the tau closest to 0.50
def best_threshold(score_seqs, gt_seqs, tau_min=0.05, tau_max=0.95, tau_step=0.02):
    y_true = np.array(flatten(gt_seqs), dtype=np.int32)
    y_score = np.array(flatten(score_seqs), dtype=np.float32)

    best_tau = 0.50
    best_f1 = -1.0
    tau = tau_min
    while tau <= tau_max + 1e-6:
        y_pred = (y_score >= tau).astype(np.int32)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        prec = 0.0
        if tp + fp > 0:
            prec = tp / (tp + fp)
        rec = 0.0
        if tp + fn > 0:
            rec = tp / (tp + fn)
        f1 = 0.0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)

        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-6 and abs(tau - 0.50) < abs(best_tau - 0.50)):
            best_f1 = f1
            best_tau = round(tau, 4)
        tau += tau_step
    return best_tau, best_f1


# same sweep but runs the filter every time, needed for the consensus filter
def best_threshold_by_decide(filt, seqs, gt_seqs, tau_min=0.05, tau_max=0.95, tau_step=0.02):
    y_true = flatten(gt_seqs)
    best_tau = 0.50
    best_f1 = -1.0
    tau = tau_min
    while tau <= tau_max + 1e-6:
        y_pred = []
        for s in seqs:
            y_pred += filt.decide(s, tau)
        f1 = calculate_frame_metrics(y_true, y_pred)["f1"]
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-6 and abs(tau - 0.50) < abs(best_tau - 0.50)):
            best_f1 = f1
            best_tau = round(tau, 4)
        tau += tau_step
    return best_tau, best_f1


def target_alerted(pred, gt):
    for i in range(len(gt)):
        if pred[i] == 1 and gt[i] == 1:
            return True
    return False


# percentile bootstrap over clips, returns (mean of boot means, low, high)
def bootstrap_ci(values, b_iterations=1000, alpha=0.05, seed=42):
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(b_iterations):
        total = 0
        for _ in range(n):
            total += values[rng.randint(0, n - 1)]
        boot.append(total / n)
    boot.sort()
    low = boot[int((alpha / 2.0) * b_iterations)]
    high = boot[int((1.0 - alpha / 2.0) * b_iterations)]
    return round(sum(boot) / len(boot), 4), round(low, 4), round(high, 4)


# bootstrap of the pooled frame f1: resample whole clips, then pool their frames
def pooled_f1_ci(decisions, gt_seqs, b_iterations=1000, alpha=0.05, seed=42):
    rng = random.Random(seed)
    n = len(gt_seqs)
    boot = []
    for _ in range(b_iterations):
        y_true = []
        y_pred = []
        for _ in range(n):
            k = rng.randint(0, n - 1)
            y_true += gt_seqs[k]
            y_pred += decisions[k]
        boot.append(calculate_frame_metrics(y_true, y_pred)["f1"])
    boot.sort()
    low = boot[int((alpha / 2.0) * b_iterations)]
    high = boot[int((1.0 - alpha / 2.0) * b_iterations)]
    return low, high


# every test metric we report, computed from one set of alert decisions
def score_decisions(decisions, gt_seqs, biomes, biome_keys=("desert", "forest", "snow")):
    fm = calculate_frame_metrics(flatten(gt_seqs), flatten(decisions))

    pos_idx = []
    for i in range(len(gt_seqs)):
        if sum(gt_seqs[i]) > 0:
            pos_idx.append(i)

    # alert recall: a positive clip counts if any alert lands on a target frame
    detected = 0
    det_by_biome = {}
    pos_by_biome = {}
    for b in biome_keys:
        det_by_biome[b] = 0
        pos_by_biome[b] = 0
    for i in pos_idx:
        pos_by_biome[biomes[i]] += 1
        if target_alerted(decisions[i], gt_seqs[i]):
            detected += 1
            det_by_biome[biomes[i]] += 1

    fa_events = 0
    fa_by_biome = {}
    fp_by_biome = {}
    frames_by_biome = {}
    for b in biome_keys:
        fa_by_biome[b] = 0
        fp_by_biome[b] = 0
        frames_by_biome[b] = 0
    for i in range(len(gt_seqs)):
        b = biomes[i]
        evs = count_false_alerts(gt_seqs[i], decisions[i])
        fa_events += evs
        fa_by_biome[b] += evs
        frames_by_biome[b] += len(gt_seqs[i])
        for k in range(len(gt_seqs[i])):
            if decisions[i][k] == 1 and gt_seqs[i][k] == 0:
                fp_by_biome[b] += 1

    delays = []
    for i in pos_idx:
        delays += time_to_alarm(gt_seqs[i], decisions[i])
    mean_delay = 0.0
    if len(delays) > 0:
        mean_delay = float(np.mean(delays))

    episode_f1s = []
    for i in range(len(gt_seqs)):
        episode_f1s.append(calculate_frame_metrics(gt_seqs[i], decisions[i])["f1"])

    return {
        "frame": fm,
        "pos_idx": pos_idx,
        "detected": detected,
        "det_by_biome": det_by_biome,
        "pos_by_biome": pos_by_biome,
        "fa_events": fa_events,
        "fa_by_biome": fa_by_biome,
        "fp_by_biome": fp_by_biome,
        "minutes_by_biome": {b: frames_by_biome[b] / (FPS * 60.0) for b in biome_keys},
        "mean_delay_frames": mean_delay,
        "mean_delay_ms": (mean_delay / FPS) * 1000.0,
        "episode_f1s": episode_f1s,
    }
