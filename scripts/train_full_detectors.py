#   MMSAR - step 5: train the two YOLO11n detectors (rgb and pseudo-thermal)
#        Oumar Ibrahim U22200741

# settings come from configs/hyperparams.yaml, data from configs/data_{rgb,thermal}.yaml
# weights end up in runs/detect/yolo11n_{rgb,thermal}_60e/weights/best.pt
# --smoke runs 2 epochs on rgb only to check the setup and time one epoch

import json
import sys
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


# ultralytics resolves a relative dataset "path" against its own datasets folder,
# so write a copy of the dataset yaml with the absolute repo root filled in
def dataset_yaml(mod):
    with open(ROOT / f"configs/data_{mod}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = ROOT.as_posix()
    out = ROOT / "runs" / "detect" / f"_resolved_data_{mod}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(out)


def train_one(mod, hp, epochs, project, name):
    device = "cpu"
    if torch.cuda.is_available():
        device = 0

    model = YOLO(hp["weights"])
    t0 = time.time()
    model.train(
        data=dataset_yaml(mod),
        epochs=epochs,
        batch=hp["batch"],
        imgsz=hp["imgsz"],
        device=device,
        workers=hp["workers"],
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        cos_lr=hp["cos_lr"],
        lr0=hp["lr0"],
        lrf=hp["lrf"],
        seed=hp["seed"],
    )
    return time.time() - t0


def main():
    with open("configs/hyperparams.yaml", encoding="utf-8") as f:
        hp = yaml.safe_load(f)

    print("torch", torch.__version__, " cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    if "--smoke" in sys.argv:
        secs = train_one("rgb", hp, 2, "runs/smoke_test", "yolo11n_rgb_smoke")
        print()
        print("----------------------")
        print("Seconds per epoch= ", round(secs / 2, 1))
        print("Projected full run (h)= ", round(secs / 2 * hp["epochs"] / 3600, 2))
        print("----------------------")
        return

    summary = {}
    total = 0
    for mod in ["rgb", "thermal"]:
        name = f"yolo11n_{mod}_{hp['epochs']}e"
        secs = train_one(mod, hp, hp["epochs"], "runs/detect", name)
        total += secs
        summary[mod] = {
            "training_time_seconds": round(secs, 2),
            "training_time_minutes": round(secs / 60.0, 2),
            "weights_path": str(Path(f"runs/detect/{name}/weights/best.pt").resolve()),
        }
        print(f"{mod} done in {secs / 60.0:.1f} min")

    summary["total_training_time_minutes"] = round(total / 60.0, 2)
    summary["total_training_time_hours"] = round(total / 3600.0, 2)
    Path("runs/detect").mkdir(parents=True, exist_ok=True)
    with open("runs/detect/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("----------------------")
    print("Total training (h)= ", summary["total_training_time_hours"])
    print("----------------------")


if __name__ == "__main__":
    main()
