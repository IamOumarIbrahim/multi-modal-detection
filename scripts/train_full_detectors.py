"""Full Training Pipeline for YOLO11n-RGB and YOLO11n-Thermal on Combined MMSAR Dataset."""

import os
import sys
import time
import json
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    print("=" * 80)
    print(" STARTING FULL DUAL-DETECTOR TRAINING (YOLO11n-RGB & YOLO11n-Thermal)")
    print("=" * 80)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name:     {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    start_time_all = time.time()
    summary = {}

    # -------------------------------------------------------------------------
    # 1. Train YOLO11n-RGB (60 Epochs)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [PHASE 1/2] TRAINING YOLO11n-RGB (60 EPOCHS)")
    print("=" * 80)
    t0_rgb = time.time()
    model_rgb = YOLO("yolo11n.pt")
    results_rgb = model_rgb.train(
        data="configs/data_rgb.yaml",
        epochs=60,
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project="runs/detect",
        name="yolo11n_rgb_60e",
        exist_ok=True,
        verbose=True,
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
    )
    time_rgb = time.time() - t0_rgb
    summary["rgb"] = {
        "training_time_seconds": round(time_rgb, 2),
        "training_time_minutes": round(time_rgb / 60.0, 2),
        "weights_path": str(Path("runs/detect/yolo11n_rgb_60e/weights/best.pt").resolve()),
    }
    print(f"YOLO11n-RGB training completed in {time_rgb / 60.0:.2f} minutes.")

    # -------------------------------------------------------------------------
    # 2. Train YOLO11n-Thermal (60 Epochs)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [PHASE 2/2] TRAINING YOLO11n-Thermal (60 EPOCHS)")
    print("=" * 80)
    t0_thermal = time.time()
    model_thermal = YOLO("yolo11n.pt")
    results_thermal = model_thermal.train(
        data="configs/data_thermal.yaml",
        epochs=60,
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project="runs/detect",
        name="yolo11n_thermal_60e",
        exist_ok=True,
        verbose=True,
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
    )
    time_thermal = time.time() - t0_thermal
    summary["thermal"] = {
        "training_time_seconds": round(time_thermal, 2),
        "training_time_minutes": round(time_thermal / 60.0, 2),
        "weights_path": str(Path("runs/detect/yolo11n_thermal_60e/weights/best.pt").resolve()),
    }
    print(f"YOLO11n-Thermal training completed in {time_thermal / 60.0:.2f} minutes.")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_time = time.time() - start_time_all
    summary["total_training_time_minutes"] = round(total_time / 60.0, 2)
    summary["total_training_time_hours"] = round(total_time / 3600.0, 2)

    os.makedirs("runs/detect", exist_ok=True)
    summary_path = "runs/detect/training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" ALL DETECTOR TRAINING FINISHED SUCCESSFULLY!")
    print(f" Total wall clock: {total_time / 60.0:.2f} min ({total_time / 3600.0:.2f} hours)")
    print(f" Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
