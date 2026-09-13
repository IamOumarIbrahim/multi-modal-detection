"""MMSAR Benchmark Evaluation & Automated LaTeX Manuscript Filler.

Ingests trained YOLO11n-RGB and YOLO11n-Thermal weights, executes multi-biome evaluation,
calibrates causal post-processing filters (M1-M5), calculates operational resource savings,
and directly fills every [TBD] placeholder in docs/manuscript/main.tex.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ultralytics import YOLO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmsar.metrics.frame_metrics import calculate_frame_metrics
from mmsar.postprocessing.baseline import BaselinePostProcessor
from mmsar.postprocessing.moving_average import MovingAveragePostProcessor
from mmsar.postprocessing.median_filter import MedianFilterPostProcessor
from mmsar.postprocessing.history_tracking import HistoryTrackingPostProcessor
from mmsar.postprocessing.mamba_sssm import MambaSSSMPostProcessor
from mmsar.postprocessing.optimizer import find_optimal_threshold


SEEDS = [42, 101, 2024, 777, 999]


def find_weights(model_name: str) -> Path | None:
    """Locate best.pt weights from local or Ultralytics runs directory."""
    candidates = [
        Path(f"C:/Dev/repos/Public repos/DMS-Eval/runs/detect/runs/detect/{model_name}/weights/best.pt"),
        Path(f"runs/detect/{model_name}/weights/best.pt"),
        Path(f"C:/Dev/repos/Public repos/DMS-Eval/runs/detect/{model_name}/weights/best.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def parse_split_episodes(rgb_txt_path: str | Path, thermal_txt_path: str | Path) -> list[dict[str, Any]]:
    """Parse aligned RGB and Thermal split text files into ordered episodic sequences."""
    p_rgb = Path(rgb_txt_path)
    p_th = Path(thermal_txt_path)
    if not p_rgb.exists():
        raise FileNotFoundError(f"RGB split file not found: {p_rgb}")
    if not p_th.exists():
        raise FileNotFoundError(f"Thermal split file not found: {p_th}")

    lines_rgb = [l.strip() for l in p_rgb.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines_th = [l.strip() for l in p_th.read_text(encoding="utf-8").splitlines() if l.strip()]

    if len(lines_rgb) != len(lines_th):
        raise ValueError(f"Line count mismatch: {len(lines_rgb)} RGB vs {len(lines_th)} Thermal")

    episodes_dict: dict[str, dict[str, Any]] = {}

    for rgb_path, th_path in zip(lines_rgb, lines_th):
        norm = rgb_path.replace("\\", "/")
        match = re.search(r"images/([^/]+)/([^/]+)/rgb/([^/]+)_frame_\d+\.png", norm)
        if match:
            biome = match.group(1)
            scenario = match.group(2)
            ep_id = match.group(3)
        else:
            ep_id = Path(norm).stem.split("_frame_")[0]
            biome = "unknown"
            scenario = "unknown"

        if ep_id not in episodes_dict:
            episodes_dict[ep_id] = {
                "episode_id": ep_id,
                "biome": biome,
                "scenario": scenario,
                "rgb_frames": [],
                "thermal_frames": [],
                "labels": [],
            }

        episodes_dict[ep_id]["rgb_frames"].append(rgb_path)
        episodes_dict[ep_id]["thermal_frames"].append(th_path)
        episodes_dict[ep_id]["labels"].append(load_binary_label(rgb_path))

    return list(episodes_dict.values())


def load_binary_label(image_path: str) -> int:
    """Load binary presence label (1 if target bounding box exists, 0 otherwise)."""
    norm = image_path.replace("\\", "/")
    lbl_path = norm.replace("/images/", "/labels/").replace(".png", ".txt")
    p = Path(lbl_path)
    if p.exists() and p.stat().st_size > 0:
        return 1
    return 0


def extract_confidences(model: YOLO, frame_paths: list[str], batch_size: int = 32) -> list[float]:
    """Run batched inference to extract class 0 person detection confidences."""
    confidences = []
    device = 0 if torch.cuda.is_available() else "cpu"

    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i : i + batch_size]
        results = model.predict(batch, conf=0.01, verbose=False, device=device)
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                confidences.append(float(np.max(confs)))
            else:
                confidences.append(0.0)

    return confidences


def compute_bootstrap_ci(
    data: list[float], b_iterations: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """Compute 95% non-parametric bootstrap percentile confidence interval across episodes."""
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


def run_benchmark_and_fill(
    rgb_weights_path: Path,
    thermal_weights_path: Path,
    main_tex_path: Path = Path("docs/manuscript/main.tex"),
    output_json: Path = Path("runs/detect/benchmark_evaluation_results.json"),
) -> dict[str, Any]:
    print("=" * 80)
    print(" MMSAR END-TO-END BENCHMARK EVALUATION & LATEX AUTO-FILLER")
    print("=" * 80)
    print(f"RGB Weights:     {rgb_weights_path}")
    print(f"Thermal Weights: {thermal_weights_path}")

    model_rgb = YOLO(str(rgb_weights_path))
    model_thermal = YOLO(str(thermal_weights_path))

    val_episodes = parse_split_episodes("data/splits/val_rgb.txt", "data/splits/val_thermal.txt")
    test_episodes = parse_split_episodes("data/splits/test_rgb.txt", "data/splits/test_thermal.txt")
    print(f"Loaded {len(val_episodes)} validation episodes, {len(test_episodes)} test episodes.")

    # -------------------------------------------------------------------------
    # 1. Ingest Confidences (with disk cache)
    # -------------------------------------------------------------------------
    cache_file = Path("runs/detect/extracted_confidences_cache.json")
    if cache_file.exists():
        print(f"\n[1/5] Loading pre-extracted confidences from cache: {cache_file}", flush=True)
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        val_fused_seqs = cache_data["val_fused_seqs"]
        val_gt_seqs = cache_data["val_gt_seqs"]
        test_rgb_seqs = cache_data["test_rgb_seqs"]
        test_th_seqs = cache_data["test_th_seqs"]
        test_fused_seqs = cache_data["test_fused_seqs"]
        test_gt_seqs = cache_data["test_gt_seqs"]
        test_biomes = cache_data["test_biomes"]
    else:
        print("\n[1/5] Extracting validation confidences...", flush=True)
        val_fused_seqs = []
        val_gt_seqs = []
        for ep in val_episodes:
            c_rgb = extract_confidences(model_rgb, ep["rgb_frames"])
            c_th = extract_confidences(model_thermal, ep["thermal_frames"])
            fused = [max(r, t) for r, t in zip(c_rgb, c_th)]
            val_fused_seqs.append(fused)
            val_gt_seqs.append(ep["labels"])

        print("[2/5] Extracting test confidences across all biomes...", flush=True)
        test_rgb_seqs = []
        test_th_seqs = []
        test_fused_seqs = []
        test_gt_seqs = []
        test_biomes = []
        for ep in test_episodes:
            c_rgb = extract_confidences(model_rgb, ep["rgb_frames"])
            c_th = extract_confidences(model_thermal, ep["thermal_frames"])
            fused = [max(r, t) for r, t in zip(c_rgb, c_th)]
            test_rgb_seqs.append(c_rgb)
            test_th_seqs.append(c_th)
            test_fused_seqs.append(fused)
            test_gt_seqs.append(ep["labels"])
            test_biomes.append(ep["biome"])

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({
                "val_fused_seqs": val_fused_seqs,
                "val_gt_seqs": val_gt_seqs,
                "test_rgb_seqs": test_rgb_seqs,
                "test_th_seqs": test_th_seqs,
                "test_fused_seqs": test_fused_seqs,
                "test_gt_seqs": test_gt_seqs,
                "test_biomes": test_biomes,
            }, f)
        print(f"Saved extracted confidences cache to: {cache_file}", flush=True)

    # -------------------------------------------------------------------------
    # 2. Table III: Frame-Level Upstream Detection (tau = 0.50 on test split)
    # -------------------------------------------------------------------------
    print("\n[3/5] Computing Table III Upstream Detection Metrics (tau = 0.50)...", flush=True)
    upstream_results: dict[str, dict[str, float]] = {}

    biome_keys = ["desert", "forest", "snow"]
    biome_display = {
        "desert": "Arid Desert",
        "forest": "Temperate Forest",
        "snow": "Snow/Alpine",
    }

    for b in biome_keys:
        idxs = [i for i, bm in enumerate(test_biomes) if bm == b]
        y_true_b = [v for i in idxs for v in test_gt_seqs[i]]
        
        y_rgb_b = [1 if v >= 0.50 else 0 for i in idxs for v in test_rgb_seqs[i]]
        m_rgb = calculate_frame_metrics(y_true_b, y_rgb_b)
        upstream_results[f"YOLO11n-RGB ({biome_display[b]})"] = m_rgb

        y_th_b = [1 if v >= 0.50 else 0 for i in idxs for v in test_th_seqs[i]]
        m_th = calculate_frame_metrics(y_true_b, y_th_b)
        upstream_results[f"YOLO11n-Thermal ({biome_display[b]})"] = m_th

        y_fuse_b = [1 if v >= 0.50 else 0 for i in idxs for v in test_fused_seqs[i]]
        m_fuse = calculate_frame_metrics(y_true_b, y_fuse_b)
        upstream_results[f"Late Fusion Gate ({biome_display[b]})"] = m_fuse

        print(f"  {b.capitalize():7s} | RGB: P={m_rgb['precision']:.3f} R={m_rgb['recall']:.3f} F1={m_rgb['f1']:.3f} | TH: P={m_th['precision']:.3f} R={m_th['recall']:.3f} F1={m_th['f1']:.3f} | Fuse: P={m_fuse['precision']:.3f} R={m_fuse['recall']:.3f} F1={m_fuse['f1']:.3f}", flush=True)

    # -------------------------------------------------------------------------
    # 3. Table I: Comparative Benchmark of Causal Temporal Post-Processing
    # -------------------------------------------------------------------------
    print("\n[4/5] Calibrating & Benchmarking Post-Processing Methods (M1-M5)...", flush=True)
    methods: dict[str, Any] = {
        "M1": BaselinePostProcessor(),
        "M2": MovingAveragePostProcessor(window_size=5),
        "M3": MedianFilterPostProcessor(window_size=5),
        "M4": HistoryTrackingPostProcessor(window_size=5, required_hits=3),
    }

    calibrated_taus: dict[str, float] = {}
    for key, proc in methods.items():
        tau_star, val_f1 = find_optimal_threshold(proc, val_fused_seqs, val_gt_seqs, metric="f1")
        calibrated_taus[key] = tau_star
        print(f"  {key:3s} optimal tau* = {tau_star:.2f} (Val F1: {val_f1:.4f})", flush=True)

    print("  Training Learned Mamba-SSSM across seeds...")
    best_mamba_val_f1 = -1.0
    best_mamba_proc = None
    best_mamba_tau = 0.50

    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        random.seed(s)
        mamba = MambaSSSMPostProcessor(in_features=1, d_model=16, d_state=8)
        mamba.fit(val_fused_seqs, val_gt_seqs, epochs=40, lr=0.01)
        tau_m, f1_m = find_optimal_threshold(mamba, val_fused_seqs, val_gt_seqs, metric="f1")
        if f1_m > best_mamba_val_f1:
            best_mamba_val_f1 = f1_m
            best_mamba_proc = mamba
            best_mamba_tau = tau_m

    methods["M5"] = best_mamba_proc
    calibrated_taus["M5"] = best_mamba_tau
    print(f"  M5  optimal tau* = {best_mamba_tau:.2f} (Val F1: {best_mamba_val_f1:.4f})")

    y_test_true_flat = [v for seq in test_gt_seqs for v in seq]
    total_pos_frames = sum(y_test_true_flat)

    postprocessing_results: dict[str, dict[str, Any]] = {}
    m1_fp_total = 0

    for key in ["M1", "M2", "M3", "M4", "M5"]:
        proc = methods[key]
        tau = calibrated_taus[key]

        t0 = time.perf_counter()
        test_preds = []
        episode_f1s = []
        for seq, gt in zip(test_fused_seqs, test_gt_seqs):
            pred = proc.decide(seq, tau=tau)
            test_preds.extend(pred)
            ep_m = calculate_frame_metrics(gt, pred)
            episode_f1s.append(ep_m["f1"])
        lat_ms = ((time.perf_counter() - t0) / len(y_test_true_flat)) * 1000.0

        m = calculate_frame_metrics(y_test_true_flat, test_preds)
        _, ci_low, ci_high = compute_bootstrap_ci(episode_f1s)

        neg_fp = int(m["fp"])
        pos_recall_pct = round((m["tp"] / total_pos_frames) * 100.0, 1)

        if key == "M1":
            m1_fp_total = neg_fp
            fasr_pct = 0.0
        else:
            fasr_pct = round((1.0 - (neg_fp / m1_fp_total)) * 100.0, 1) if m1_fp_total > 0 else 0.0

        postprocessing_results[key] = {
            "val_tau": tau,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "pos_recall_pct": pos_recall_pct,
            "negative_fp": neg_fp,
            "fasr_pct": fasr_pct,
            "latency_ms": lat_ms,
        }
        print(f"  {key} -> Prec: {m['precision']:.3f}, Rec: {m['recall']:.3f}, F1: {m['f1']:.3f} [{ci_low:.3f}, {ci_high:.3f}], FASR: {fasr_pct}%, Lat: {lat_ms:.3f}ms")

    # -------------------------------------------------------------------------
    # 4. Table IV: Operational Impact Across Biomes
    # -------------------------------------------------------------------------
    print("\n[5/5] Computing Table IV Operational Resource Savings Across Biomes...")
    operational_results: dict[str, dict[str, dict[str, Any]]] = {}

    m1_proc = methods["M1"]
    m1_tau = calibrated_taus["M1"]
    m1_fp_per_biome: dict[str, int] = {}
    for b in biome_keys:
        idxs = [i for i, bm in enumerate(test_biomes) if bm == b]
        gt_b = [v for i in idxs for v in test_gt_seqs[i]]
        pred_b = [v for i in idxs for v in m1_proc.decide(test_fused_seqs[i], tau=m1_tau)]
        m_b = calculate_frame_metrics(gt_b, pred_b)
        m1_fp_per_biome[b] = int(m_b["fp"])

    for key in ["M2", "M3", "M4", "M5"]:
        proc = methods[key]
        tau = calibrated_taus[key]
        operational_results[key] = {}

        for b in biome_keys:
            idxs = [i for i, bm in enumerate(test_biomes) if bm == b]
            gt_b = [v for i in idxs for v in test_gt_seqs[i]]
            pred_b = [v for i in idxs for v in proc.decide(test_fused_seqs[i], tau=tau)]
            m_b = calculate_frame_metrics(gt_b, pred_b)
            fp_b = int(m_b["fp"])
            base_fp = m1_fp_per_biome[b]

            delta_fp = max(0, base_fp - fp_b)
            fasr_val = round((1.0 - (fp_b / base_fp)) * 100.0, 1) if base_fp > 0 else 0.0
            bandwidth_kb = round(delta_fp * 1.2, 1)
            battery_kj = round(delta_fp * 5.6, 1)
            flight_time_min = round(delta_fp / 3.0, 1)

            operational_results[key][b] = {
                "fasr": fasr_val,
                "delta_fp": delta_fp,
                "bandwidth_kb": bandwidth_kb,
                "battery_kj": battery_kj,
                "flight_time_min": flight_time_min,
            }
            print(f"  {key} ({b:7s}) | FASR: {fasr_val:5.1f}% | ΔFP: {delta_fp:4d} | ΔΩ: {bandwidth_kb:6.1f} kB | ΔE: {battery_kj:6.1f} kJ | Δt: {flight_time_min:5.1f} min")

    # -------------------------------------------------------------------------
    # 5. Fill main.tex Placeholders
    # -------------------------------------------------------------------------
    print("\nFilling [TBD] placeholders in docs/manuscript/main.tex...")
    main_tex_content = main_tex_path.read_text(encoding="utf-8")

    # Table I Replacement
    t1_lines = {
        "M1": f"M1: Baseline Raw Thresholding   & $O(1)$             & 0 B   & {postprocessing_results['M1']['val_tau']:.2f} & {postprocessing_results['M1']['precision']:.3f} & {postprocessing_results['M1']['recall']:.3f} & {postprocessing_results['M1']['f1']:.3f} & [{postprocessing_results['M1']['ci_low']:.3f}, {postprocessing_results['M1']['ci_high']:.3f}] & {postprocessing_results['M1']['pos_recall_pct']:.1f} & {postprocessing_results['M1']['negative_fp']} & --- & {postprocessing_results['M1']['latency_ms']:.3f} ms \\\\",
        "M2": f"M2: Five-Frame Moving Average   & $O(1)$ amortized   & 20 B  & {postprocessing_results['M2']['val_tau']:.2f} & {postprocessing_results['M2']['precision']:.3f} & {postprocessing_results['M2']['recall']:.3f} & {postprocessing_results['M2']['f1']:.3f} & [{postprocessing_results['M2']['ci_low']:.3f}, {postprocessing_results['M2']['ci_high']:.3f}] & {postprocessing_results['M2']['pos_recall_pct']:.1f} & {postprocessing_results['M2']['negative_fp']} & {postprocessing_results['M2']['fasr_pct']:.1f} & {postprocessing_results['M2']['latency_ms']:.3f} ms \\\\",
        "M3": f"M3: Five-Frame Median Filter    & $O(W)$             & 20 B  & {postprocessing_results['M3']['val_tau']:.2f} & {postprocessing_results['M3']['precision']:.3f} & {postprocessing_results['M3']['recall']:.3f} & {postprocessing_results['M3']['f1']:.3f} & [{postprocessing_results['M3']['ci_low']:.3f}, {postprocessing_results['M3']['ci_high']:.3f}] & {postprocessing_results['M3']['pos_recall_pct']:.1f} & {postprocessing_results['M3']['negative_fp']} & {postprocessing_results['M3']['fasr_pct']:.1f} & {postprocessing_results['M3']['latency_ms']:.3f} ms \\\\",
        "M4": f"M4: Five-Frame History Consensus& $O(W)$             & 5 B   & {postprocessing_results['M4']['val_tau']:.2f} & {postprocessing_results['M4']['precision']:.3f} & {postprocessing_results['M4']['recall']:.3f} & {postprocessing_results['M4']['f1']:.3f} & [{postprocessing_results['M4']['ci_low']:.3f}, {postprocessing_results['M4']['ci_high']:.3f}] & {postprocessing_results['M4']['pos_recall_pct']:.1f} & {postprocessing_results['M4']['negative_fp']} & {postprocessing_results['M4']['fasr_pct']:.1f} & {postprocessing_results['M4']['latency_ms']:.3f} ms \\\\",
        "M5": f"\\textbf{{M5: Learned Mamba-SSSM}} & \\textbf{{$O(d_{{\\text{{model}}}} d_{{\\text{{state}}}})$}} & \\textbf{{512 B}} & \\textbf{{{postprocessing_results['M5']['val_tau']:.2f}}} & \\textbf{{{postprocessing_results['M5']['precision']:.3f}}} & \\textbf{{{postprocessing_results['M5']['recall']:.3f}}} & \\textbf{{{postprocessing_results['M5']['f1']:.3f}}} & \\textbf{{[{postprocessing_results['M5']['ci_low']:.3f}, {postprocessing_results['M5']['ci_high']:.3f}]}} & \\textbf{{{postprocessing_results['M5']['pos_recall_pct']:.1f}}} & \\textbf{{{postprocessing_results['M5']['negative_fp']}}} & \\textbf{{{postprocessing_results['M5']['fasr_pct']:.1f}}} & \\textbf{{{postprocessing_results['M5']['latency_ms']:.3f} ms}} \\\\",
    }

    lines_tex = main_tex_content.splitlines()
    for key, repl in t1_lines.items():
        for idx, l in enumerate(lines_tex):
            if key == "M5" and "M5: Learned Mamba-SSSM" in l and "[TBD]" in l:
                lines_tex[idx] = repl
                break
            elif l.strip().startswith(key) and "[TBD]" in l:
                lines_tex[idx] = repl
                break
    main_tex_content = "\n".join(lines_tex)

    # Table III Replacement
    t3_map = {
        "YOLO11n-RGB (Arid Desert)": upstream_results["YOLO11n-RGB (Arid Desert)"],
        "YOLO11n-RGB (Temperate Forest)": upstream_results["YOLO11n-RGB (Temperate Forest)"],
        "YOLO11n-RGB (Snow/Alpine)": upstream_results["YOLO11n-RGB (Snow/Alpine)"],
        "YOLO11n-Thermal (Arid Desert)": upstream_results["YOLO11n-Thermal (Arid Desert)"],
        "YOLO11n-Thermal (Temperate Forest)": upstream_results["YOLO11n-Thermal (Temperate Forest)"],
        "YOLO11n-Thermal (Snow/Alpine)": upstream_results["YOLO11n-Thermal (Snow/Alpine)"],
        "Late Fusion Gate (Desert)": upstream_results["Late Fusion Gate (Arid Desert)"],
        "Late Fusion Gate (Forest)": upstream_results["Late Fusion Gate (Temperate Forest)"],
        "Late Fusion Gate (Snow/Alpine)": upstream_results["Late Fusion Gate (Snow/Alpine)"],
    }

    lines_tex = main_tex_content.splitlines()
    for row_name, metrics in t3_map.items():
        is_bold = "Late Fusion" in row_name
        p, r, f = metrics["precision"], metrics["recall"], metrics["f1"]
        if is_bold:
            new_line = f"\\textbf{{{row_name}}}  & \\textbf{{{p:.3f}}} & \\textbf{{{r:.3f}}} & \\textbf{{{f:.3f}}} \\\\"
        else:
            new_line = f"{row_name:38s} & {p:.3f} & {r:.3f} & {f:.3f} \\\\"

        for idx, l in enumerate(lines_tex):
            if row_name in l and "[TBD]" in l:
                lines_tex[idx] = new_line
                break
    main_tex_content = "\n".join(lines_tex)

    # Table IV Replacement
    t4_name_map = {
        "M2": "M2: Moving Avg.",
        "M3": "M3: Median",
        "M4": "M4: Consensus",
        "M5": "M5: Mamba-SSSM",
    }
    biome_label_map = {
        "desert": "Desert",
        "forest": "Forest",
        "snow": "Snow/Alpine",
    }

    lines_tex = main_tex_content.splitlines()
    for m_key, m_label in t4_name_map.items():
        for b_key, b_label in biome_label_map.items():
            op = operational_results[m_key][b_key]
            new_line = f"{m_label:16s} & {b_label:11s} & {op['fasr']:.1f} & {op['delta_fp']} & {op['bandwidth_kb']:.1f} & {op['battery_kj']:.1f} & {op['flight_time_min']:.1f} \\\\"
            for idx, l in enumerate(lines_tex):
                if m_label in l and b_label in l and "[TBD]" in l:
                    lines_tex[idx] = new_line
                    break
    main_tex_content = "\n".join(lines_tex)

    main_tex_path.write_text(main_tex_content, encoding="utf-8")
    
    remaining = re.findall(r"\[(.*?(?:TBD|TODO|PLACEHOLDER).*?)\]", main_tex_content, re.IGNORECASE)
    print(f"\nRemaining bracketed placeholders in {main_tex_path}: {len(remaining)}")
    if remaining:
        print(f"  Leftover items: {remaining}")
    else:
        print("  SUCCESS: Exactly 0 [TBD] placeholders remaining in manuscript!")

    summary_data = {
        "rgb_weights": str(rgb_weights_path),
        "thermal_weights": str(thermal_weights_path),
        "upstream_detection": upstream_results,
        "postprocessing_benchmark": postprocessing_results,
        "operational_impact": operational_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved complete benchmark metrics to: {output_json}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMSAR models and fill LaTeX manuscript tables.")
    parser.add_argument("--rgb", default=None, help="Path to best.pt for YOLO11n-RGB")
    parser.add_argument("--thermal", default=None, help="Path to best.pt for YOLO11n-Thermal")
    args = parser.parse_args()

    rgb_path = Path(args.rgb) if args.rgb else find_weights("yolo11n_rgb_60e")
    thermal_path = Path(args.thermal) if args.thermal else find_weights("yolo11n_thermal_60e")

    if not rgb_path or not rgb_path.exists():
        print("Error: YOLO11n-RGB weights not found! (looked for yolo11n_rgb_60e)")
        sys.exit(1)

    if not thermal_path or not thermal_path.exists():
        print("Error: YOLO11n-Thermal weights not found! (looked for yolo11n_thermal_60e)")
        sys.exit(1)

    run_benchmark_and_fill(rgb_path, thermal_path)


if __name__ == "__main__":
    main()
