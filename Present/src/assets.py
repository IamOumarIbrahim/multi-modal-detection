"""Harvest the paper's own material for the slide templates.

  assets/frames/*.jpg   640x640 RGB + pseudo-thermal tiles (same frame), desert / forest / snow
  assets/index.json     one entry per frame: biome, positive/negative, ground-truth person box (px)
  assets/traces.json    real per-frame confidences from the test split (fused, ground truth, raw / filtered alerts)
  assets/render_uav.png the drone-footprint render from the README

Run once:  python -I assets.py
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE.parent / "assets"
RAW = REPO / "data" / "raw"
ANN = REPO / "data" / "annotations" / "raw_exports" / "project_2_rgb_annotations_adjusted.json"
CACHE = REPO / "runs" / "detect" / "unified_confidences_cache.json"


def keyframes():
    out = {}
    for t in json.loads(ANN.read_text(encoding="utf-8")):
        name = t["data"]["video"].split("/")[-1].split("-", 1)[1]
        for a in t["annotations"]:
            for r in a["result"]:
                if r.get("type") == "videorectangle":
                    out.setdefault(name, []).extend(r["value"]["sequence"])
    for k in out:
        out[k].sort(key=lambda s: s["frame"])
    return out


def box_at(seq, f):
    """Label Studio interpolation: linear between keyframes while the earlier one is enabled."""
    prev = [s for s in seq if s["frame"] <= f]
    if not prev:
        return None
    a = prev[-1]
    if not a["enabled"] and a["frame"] != f:
        return None
    nxt = [s for s in seq if s["frame"] > f]
    if nxt and a["enabled"]:
        b = nxt[0]
        t = (f - a["frame"]) / (b["frame"] - a["frame"])
        return [a[k] + (b[k] - a[k]) * t for k in ("x", "y", "width", "height")]
    return [a[k] for k in ("x", "y", "width", "height")]


def grab(path, idx):
    c = cv2.VideoCapture(str(path))
    c.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = c.read()
    c.release()
    return fr if ok else None


def main():
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    kf = keyframes()
    index = []
    for vid, seq in sorted(kf.items()):
        biome = vid.split("_")[0]
        rgb = RAW / biome / "positive" / "rgb" / vid
        th = RAW / biome / "positive" / "thermal" / vid.replace("_RGB_", "_Thermal_")
        if not (rgb.exists() and th.exists()):
            continue
        # two frames per clip where the person box is present and reasonably big
        cand = [f for f in range(1, 240, 6) if (b := box_at(seq, f)) and b[3] > 9]
        if not cand:
            continue
        for f in (cand[len(cand) // 3], cand[(2 * len(cand)) // 3]):
            b = box_at(seq, f)
            a, t = grab(rgb, f - 1), grab(th, f - 1)
            if a is None or t is None:
                continue
            h, w = a.shape[:2]
            px = [b[0] / 100 * w, b[1] / 100 * h, (b[0] + b[2]) / 100 * w, (b[1] + b[3]) / 100 * h]
            fid = f"{vid[:-4].replace('_RGB', '')}_f{f:03d}"
            cv2.imwrite(str(OUT / "frames" / f"{fid}_rgb.jpg"), a, [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(str(OUT / "frames" / f"{fid}_th.jpg"), t, [cv2.IMWRITE_JPEG_QUALITY, 93])
            index.append(dict(id=fid, biome=biome, positive=True, box=[round(v, 1) for v in px]))
    for biome in ("desert", "forest", "snow"):            # empty scenes, the source of false alerts
        for p in sorted((RAW / biome / "negative" / "rgb").glob("*.mp4"))[:4]:
            th = p.parents[1] / "thermal" / p.name.replace("_RGB_", "_Thermal_")
            a, t = grab(p, 100), grab(th, 100)
            if a is None or t is None:
                continue
            fid = p.stem.replace("_RGB", "") + "_f101"
            cv2.imwrite(str(OUT / "frames" / f"{fid}_rgb.jpg"), a, [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(str(OUT / "frames" / f"{fid}_th.jpg"), t, [cv2.IMWRITE_JPEG_QUALITY, 93])
            index.append(dict(id=fid, biome=biome, positive=False, box=None))
    (OUT / "index.json").write_text(json.dumps(index, indent=1))

    # filmstrips: 8 consecutive moments of one clip per biome (person present the whole time)
    strips = []
    for biome in ("desert", "forest", "snow"):
        best = None
        for vid, seq in sorted(kf.items()):
            if not vid.startswith(biome):
                continue
            fr = [f for f in range(1, 240, 4) if (b := box_at(seq, f)) and b[3] > 8]
            if best is None or len(fr) > len(best[1]):
                best = (vid, fr)
        vid, fr = best
        seq = kf[vid]
        rgb = RAW / biome / "positive" / "rgb" / vid
        th = RAW / biome / "positive" / "thermal" / vid.replace("_RGB_", "_Thermal_")
        pick = [fr[int(i * (len(fr) - 1) / 7)] for i in range(8)]
        items = []
        for f in pick:
            b = box_at(seq, f)
            a_, t_ = grab(rgb, f - 1), grab(th, f - 1)
            fid = f"strip_{vid[:-4].replace('_RGB', '')}_f{f:03d}"
            cv2.imwrite(str(OUT / "frames" / f"{fid}_rgb.jpg"), a_, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(OUT / "frames" / f"{fid}_th.jpg"), t_, [cv2.IMWRITE_JPEG_QUALITY, 90])
            items.append(dict(id=fid, frame=f, box=[round(b[0] * 6.4, 1), round(b[1] * 6.4, 1),
                                                   round((b[0] + b[2]) * 6.4, 1), round((b[1] + b[3]) * 6.4, 1)]))
        strips.append(dict(biome=biome, video=vid, frames=items))
    (OUT / "strips.json").write_text(json.dumps(strips, indent=1))

    # real confidence traces: raw threshold (M1, tau 0.43) vs moving average (M2, W 5, tau 0.57)
    d = json.loads(CACHE.read_text())
    tr = []
    for i, (fu, gt, bi) in enumerate(zip(d["test_fused_seqs"], d["test_gt_seqs"], d["test_biomes"])):
        fu = np.asarray(fu, float)
        ma = np.array([fu[max(0, k - 4):k + 1].mean() for k in range(len(fu))])
        raw_alert, ma_alert = fu >= 0.43, ma >= 0.57
        gt = np.asarray(gt, int)
        tr.append(dict(i=i, biome=bi, fused=np.round(fu, 3).tolist(), ma=np.round(ma, 3).tolist(), gt=gt.tolist(),
                       raw_fa=int((raw_alert & (gt == 0)).sum()), ma_fa=int((ma_alert & (gt == 0)).sum()),
                       flicker=float(np.abs(np.diff(fu)).mean())))
    (OUT / "traces.json").write_text(json.dumps(tr))
    shutil.copy(REPO / "figures" / "render_uav.png", OUT / "render_uav.png")
    print(len(index), "frames,", sum(e["positive"] for e in index), "positive;", len(tr), "traces")


if __name__ == "__main__":
    main()
