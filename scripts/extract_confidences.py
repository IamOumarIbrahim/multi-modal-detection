#   MMSAR - step 6: run both detectors on every split and cache the confidences
#        Oumar Ibrahim U22200741

# per frame we keep the highest person confidence (0.0 when nothing is found)
# the two streams are fused with max(), labels come from the YOLO label files
# output  runs/detect/unified_confidences_cache.json, used by all evaluate_* scripts
# usage   python scripts/extract_confidences.py [rgb_best.pt] [thermal_best.pt]

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from mmsar.fusion import late_fusion


# the txt lists are in feed order, group the frames back into clips
def parse_split(split):
    rgb_lines = Path(f"data/splits/{split}_rgb.txt").read_text(encoding="utf-8").splitlines()
    th_lines = Path(f"data/splits/{split}_thermal.txt").read_text(encoding="utf-8").splitlines()
    rgb = []
    th = []
    for line in rgb_lines:
        if line.strip() != "":
            rgb.append(line.strip())
    for line in th_lines:
        if line.strip() != "":
            th.append(line.strip())
    if len(rgb) != len(th):
        raise ValueError(f"{split}: {len(rgb)} rgb frames vs {len(th)} thermal frames")

    episodes = {}
    for i in range(len(rgb)):
        path = rgb[i].replace("\\", "/")
        m = re.search(r"images/([^/]+)/([^/]+)/rgb/([^/]+)_frame_\d+\.png", path)
        ep_id = m.group(3)
        if ep_id not in episodes:
            episodes[ep_id] = {"biome": m.group(1), "rgb": [], "thermal": [], "labels": []}
        episodes[ep_id]["rgb"].append(rgb[i])
        episodes[ep_id]["thermal"].append(th[i])

        # a non empty label file means a person is in the frame
        lbl = Path(path.replace("/images/", "/labels/").replace(".png", ".txt"))
        if lbl.exists() and lbl.stat().st_size > 0:
            episodes[ep_id]["labels"].append(1)
        else:
            episodes[ep_id]["labels"].append(0)
    return list(episodes.values())


def confidences(model, frames, batch_size=32):
    device = "cpu"
    if torch.cuda.is_available():
        device = 0
    out = []
    for i in range(0, len(frames), batch_size):
        results = model.predict(frames[i : i + batch_size], conf=0.01, verbose=False, device=device)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                out.append(float(np.max(r.boxes.conf.cpu().numpy())))
            else:
                out.append(0.0)
    return out


def main():
    rgb_w = "runs/detect/yolo11n_rgb_60e/weights/best.pt"
    th_w = "runs/detect/yolo11n_thermal_60e/weights/best.pt"
    if len(sys.argv) > 2:
        rgb_w = sys.argv[1]
        th_w = sys.argv[2]
    model_rgb = YOLO(rgb_w)
    model_th = YOLO(th_w)

    cache = {}
    for split in ["train", "val", "test"]:
        eps = parse_split(split)
        rgb_seqs = []
        th_seqs = []
        fused_seqs = []
        gt_seqs = []
        biomes = []
        for k in range(len(eps)):
            ep = eps[k]
            print(f"{split} [{k + 1}/{len(eps)}] {len(ep['rgb'])} frames")
            c_rgb = confidences(model_rgb, ep["rgb"])
            c_th = confidences(model_th, ep["thermal"])
            rgb_seqs.append(c_rgb)
            th_seqs.append(c_th)
            fused_seqs.append(late_fusion(c_rgb, c_th))
            gt_seqs.append(ep["labels"])
            biomes.append(ep["biome"])

        cache[f"{split}_rgb_seqs"] = rgb_seqs
        cache[f"{split}_th_seqs"] = th_seqs
        cache[f"{split}_fused_seqs"] = fused_seqs
        cache[f"{split}_gt_seqs"] = gt_seqs
        cache[f"{split}_biomes"] = biomes

    out = Path("runs/detect/unified_confidences_cache.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(cache, f)
    print("saved", out)


if __name__ == "__main__":
    main()
