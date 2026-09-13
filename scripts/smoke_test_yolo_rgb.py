"""2-Epoch Smoke Test on YOLO11n-RGB with real final dataset to measure wall-clock per epoch."""

import time
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    print("=" * 70)
    print(" STARTING 2-EPOCH SMOKE TEST (YOLO11n-RGB)")
    print("=" * 70)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name:    {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:     {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    model = YOLO("yolo11n.pt")

    t0 = time.time()
    results = model.train(
        data="configs/data_rgb.yaml",
        epochs=2,
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project="runs/smoke_test",
        name="yolo11n_rgb_smoke",
        exist_ok=True,
        verbose=True,
    )
    total_time = time.time() - t0
    sec_per_epoch = total_time / 2.0

    print("\n" + "=" * 70)
    print(" SMOKE TEST BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Wall-Clock for 2 Epochs: {total_time:.2f} s ({total_time / 60.0:.2f} min)")
    print(f"Wall-Clock per Epoch:          {sec_per_epoch:.2f} s ({sec_per_epoch / 60.0:.2f} min)")
    print(f"Projected 60 Epochs:           {(sec_per_epoch * 60) / 3600.0:.2f} hours")
    print(f"Projected 80 Epochs:           {(sec_per_epoch * 80) / 3600.0:.2f} hours")
    print(f"Projected 100 Epochs:          {(sec_per_epoch * 100) / 3600.0:.2f} hours")
    print("=" * 70)


if __name__ == "__main__":
    main()

