#   MMSAR - README and presentation figures
#        Oumar Ibrahim U22200741

# charts come only from committed results (runs/detect/*.json + the confidence cache)
# frame figures and the demo clips also need the local videos (data/raw, videos/), they
# are skipped with a note when those are missing
# output  figures/fig_*.png, figures/demo_*.mp4, figures/demo_forest.gif
# usage   python scripts/make_figures.py [--only name1,name2] [--no-video]

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from mmsar.filters import BaselineFilter, HistoryConsensusFilter, MedianFilter, MovingAverageFilter  # noqa: E402
from mmsar.metrics import calculate_frame_metrics, count_false_alerts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "figures"
FPS = 24.0

# one colour per role, the same on every figure (validated, CVD safe as a 3 set)
RAW = "#eb6834"       # M1 raw thresholding, the problem
FILT = "#2a78d6"      # classical causal filters M2 to M4
LEARN = "#1baf7a"     # learned models M5, M6
GT = "#d9d8d3"        # ground truth band
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
SURF = "#ffffff"
NAVY = "#14285a"

TAU = {"M1": 0.43, "M2": 0.57, "M3": 0.49, "M4": 0.49}
FILTERS = {"M1": BaselineFilter(), "M2": MovingAverageFilter(5), "M3": MedianFilter(5),
           "M4": HistoryConsensusFilter(5, hits=3)}
NAMES = {"M1": "Raw threshold", "M2": "Moving average", "M3": "Median", "M4": "Consensus 3 of 5",
         "M5": "Mamba-SSSM", "M6": "GRU"}
COLOR = {"M1": RAW, "M2": FILT, "M3": FILT, "M4": FILT, "M5": LEARN, "M6": LEARN}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 13,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
})


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15, metadata={"Software": None})
    plt.close(fig)
    print("saved", path.relative_to(ROOT))


def load(name):
    return json.loads((ROOT / "runs/detect" / name).read_text(encoding="utf-8"))


def test_ids():
    ids = []
    for line in (ROOT / "data/splits/test_rgb.txt").read_text(encoding="utf-8").splitlines():
        m = re.search(r"/([^/]+)_frame_\d+\.png", line)
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def raw_video(stem, modality="rgb"):
    biome, _, scen = stem.split("_")[:3]
    if modality == "thermal":
        stem = stem.replace("RGB", "Thermal")
    return ROOT / "data/raw" / biome / scen / modality / f"{stem}.mp4"


def read_frames(path, idx=None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx is None or i in idx:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def gt_boxes(stem):
    from build_frames_and_labels import clip_labels, load_tasks
    lines = clip_labels(raw_video(stem), load_tasks().get(stem))
    out = []
    for line in lines:
        out.append([float(v) for v in line.split()[1:]] if line else None)
    return out


def draw_box(ax, box, w, h, color="white", lw=2.0, ls="-"):
    xc, yc, bw, bh = box
    # pad tiny boxes so the person stays visible at slide size
    bw2 = max(bw, 0.06)
    bh2 = max(bh, 0.06)
    ax.add_patch(Rectangle(((xc - bw2 / 2) * w, (yc - bh2 / 2) * h), bw2 * w, bh2 * h,
                           fill=False, ec=color, lw=lw, ls=ls))


def have_videos(stems):
    missing = [s for s in stems if not raw_video(s).exists()]
    if missing:
        print(f"  skipped: raw videos not found ({missing[0]} ...)")
        return False
    return True


# ---------------------------------------------------------------- frame figures

def fig_biomes():
    picks = [("Desert", "desert_RGB_positive_4_left", 120), ("Forest", "forest_RGB_positive_5_left", 209),
             ("Snow", "snow_RGB_positive_10_left", 150)]
    if not have_videos([p[1] for p in picks]):
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.6), gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    for c, (name, stem, k) in enumerate(picks):
        box = gt_boxes(stem)[k]
        for r, mod in enumerate(["rgb", "thermal"]):
            img = read_frames(raw_video(stem, mod), {k})[0]
            ax = axes[r, c]
            ax.imshow(img)
            if box:
                draw_box(ax, box, img.shape[1], img.shape[0], color="#ffffff", lw=2.2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(name, fontsize=18, loc="center", pad=8)
            if c == 0:
                ax.set_ylabel(["Visible RGB", "Pseudo-thermal (LUT)"][r], fontsize=15, color=INK)
    fig.text(0.5, 0.06, "Same 640x640 tile, two streams. White box: the person (ground truth). "
             "Pseudo-thermal is a colour map of RGB brightness, not real heat.",
             ha="center", color=INK2, fontsize=12)
    save(fig, "fig_biomes")


def fig_dual_crop():
    src = ROOT / "videos/desert/positive/rgb/desert_RGB_positive_4.mp4"
    stem = "desert_RGB_positive_4_left"
    if not src.exists() or not have_videos([stem]):
        return
    k = 120
    full = read_frames(src, {k})[0]
    tile = read_frames(raw_video(stem), {k})[0]
    tile_th = read_frames(raw_video(stem, "thermal"), {k})[0]
    box = gt_boxes(stem)[k]

    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.95, 1, 1], wspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax.imshow(full)
    for x, col, lab in [(0, FILT, "left tile"), (640, RAW, "right tile")]:
        ax.add_patch(Rectangle((x + 3, 43), 634, 634, fill=False, ec=col, lw=3))
        ax.text(x + 14, 70, lab, color="white", fontsize=13, weight="bold",
                bbox=dict(fc=col, ec="none", pad=3))
    ax.set_title("1280x720 UAV frame (generated video)", fontsize=14)
    for i, (img, t) in enumerate([(tile, "Left tile, RGB"), (tile_th, "Left tile, pseudo-thermal")]):
        a = fig.add_subplot(gs[i + 1])
        a.imshow(img)
        if box:
            draw_box(a, box, 640, 640, color="white", lw=2.2)
        a.set_title(t, fontsize=14)
    for a in fig.axes:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    save(fig, "fig_dual_crop")


def fig_flicker():
    ids = test_ids()
    k = ids.index("desert_RGB_negative_4_right")
    cache = load("unified_confidences_cache.json")
    s = np.array(cache["test_fused_seqs"][k])
    gt = cache["test_gt_seqs"][k]
    d1 = FILTERS["M1"].decide(list(s), TAU["M1"])
    d2 = FILTERS["M2"].decide(list(s), TAU["M2"])
    ma = np.array([s[max(0, i - 4): i + 1].mean() for i in range(len(s))])
    t = np.arange(len(s)) / FPS
    n1 = count_false_alerts(gt, d1)
    n2 = count_false_alerts(gt, d2)

    with_frames = have_videos([ids[k]])
    fig = plt.figure(figsize=(16, 8.4 if with_frames else 5.8))
    gs = fig.add_gridspec(3 if with_frames else 2, 4, height_ratios=([1.25] if with_frames else []) + [2.2, 0.75],
                          hspace=0.35, wspace=0.05)
    row = 0
    if with_frames:
        # one thumbnail at the start of each of the first four false alerts
        starts = [i for i in range(len(d1)) if d1[i] == 1 and (i == 0 or d1[i - 1] == 0)][:4]
        imgs = read_frames(raw_video(ids[k]), set(starts))
        for j, (i, img) in enumerate(zip(starts, imgs)):
            a = fig.add_subplot(gs[0, j])
            a.imshow(img)
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(RAW)
                sp.set_linewidth(3)
            a.set_title(f"t = {i / FPS:.1f} s | confidence {s[i]:.2f}", fontsize=12, color=INK2, loc="center", weight="normal")
        row = 1

    ax = fig.add_subplot(gs[row, :])
    ax.plot(t, s, color=RAW, lw=1.6, label="Raw fused confidence")
    ax.plot(t, ma, color=FILT, lw=2.6, label="5-frame moving average")
    ax.axhline(TAU["M1"], color=RAW, lw=1.2, ls=(0, (4, 3)))
    ax.axhline(TAU["M2"], color=FILT, lw=1.2, ls=(0, (4, 3)))
    ax.text(t[-1] + 0.1, TAU["M1"], "τ = 0.43", color=INK2, va="center", fontsize=11)
    ax.text(t[-1] + 0.1, TAU["M2"], "τ = 0.57", color=INK2, va="center", fontsize=11)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Detector confidence")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title(f"No person in this clip (dark distractor, left): raw thresholding raises {n1} false alerts, "
                 f"the moving average {n2}", fontsize=16)

    rast = fig.add_subplot(gs[row + 1, :], sharex=ax)
    for y, d, col, lab in [(1, d1, RAW, f"Raw alerts ({n1})"), (0, d2, FILT, f"Filtered alerts ({n2})")]:
        on = np.array(d) == 1
        rast.bar(t[on], 0.7, width=1 / FPS, bottom=y - 0.35, color=col, align="edge", lw=0)
        rast.text(-0.1, y, lab, ha="right", va="center", color=INK, fontsize=12)
    rast.set_yticks([])
    rast.set_ylim(-0.6, 1.6)
    rast.set_xlabel("Time in clip (s)")
    rast.grid(False)
    rast.spines["left"].set_visible(False)
    save(fig, "fig_flicker")


# ---------------------------------------------------------------- explainer figures

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16, 4.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tc=INK, size=12.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=size,
                weight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6, shrinkA=0, shrinkB=0))

    box(0.1, 1.55, 1.9, 1.5, "UAV video\n1280x720\n24 fps", "#eef3fa", FILT)
    box(2.5, 1.55, 1.7, 1.5, "Two\n640x640\ntiles", "#eef3fa", FILT)
    box(4.7, 2.75, 2.0, 1.1, "RGB tile", "#ffffff", MUTED)
    box(4.7, 0.75, 2.0, 1.1, "Pseudo-thermal\n(256-level LUT)", "#ffffff", MUTED, size=11.5)
    box(7.2, 2.75, 1.9, 1.1, "YOLO11n\nRGB", "#dbe5f4", FILT)
    box(7.2, 0.75, 1.9, 1.1, "YOLO11n\nthermal", "#dbe5f4", FILT)
    box(9.6, 1.6, 1.6, 1.4, "Late fusion\ns[n] = max", "#ffe8d9", RAW)
    box(11.7, 1.6, 2.2, 1.4, "Causal filter\nlast W = 5 frames\n(166.7 ms)", "#d7ecd9", LEARN)
    box(14.35, 1.75, 1.5, 1.1, "Alert\ny[n]", NAVY, NAVY, tc="white", bold=True)
    arrow(2.0, 2.3, 2.5, 2.3)
    arrow(4.2, 2.6, 4.7, 3.3)
    arrow(4.2, 2.0, 4.7, 1.3)
    arrow(6.7, 3.3, 7.2, 3.3)
    arrow(6.7, 1.3, 7.2, 1.3)
    arrow(9.1, 3.3, 9.6, 2.6)
    arrow(9.1, 1.3, 9.6, 2.0)
    arrow(11.2, 2.3, 11.7, 2.3)
    arrow(13.9, 2.3, 14.35, 2.3)
    ax.text(12.8, 1.25, "M1 raw | M2 mean | M3 median\nM4 consensus | M5 Mamba | M6 GRU",
            ha="center", va="top", color=INK2, fontsize=11)
    ax.text(8.15, 4.25, "frozen detectors, one confidence per frame each", ha="center", color=INK2, fontsize=11)
    ax.text(12.8, 3.35, "the part we compare", ha="center", color=INK, fontsize=12, weight="bold")
    save(fig, "fig_pipeline")


TOY = [0.1, 0.2, 0.8, 0.1, 0.2, 0.6, 0.7, 0.8, 0.2, 0.1]


def fig_filters_toy():
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1.5, 1.6], "hspace": 0.3})
    n = np.arange(len(TOY))
    a0.bar(n, TOY, width=0.55, color=[RAW if v >= 0.5 else MUTED for v in TOY], lw=0)
    a0.axhline(0.5, color=INK2, ls=(0, (4, 3)), lw=1.2)
    a0.text(9.45, 0.52, "τ = 0.5", color=INK2, fontsize=11, ha="right", va="bottom")
    a0.annotate("one-frame spike", xy=(2, 0.82), xytext=(3.2, 0.95), fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2))
    a0.annotate("person really there", xy=(6, 0.72), xytext=(6.9, 0.95), fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2))
    a0.set_ylim(0, 1.05)
    a0.grid(axis="x", visible=False)
    a0.set_xticks(n)
    a0.set_xlim(-0.6, 9.6)
    a0.set_ylabel("Confidence")
    a0.set_title("Ten frames, one spike: which rules alert?")
    rows = [("M1", BaselineFilter()), ("M2", MovingAverageFilter(5)), ("M3", MedianFilter(5)),
            ("M4", HistoryConsensusFilter(5, hits=3))]
    for r, (key, f) in enumerate(rows):
        d = f.decide(TOY, 0.5)
        y = len(rows) - 1 - r
        for i, v in enumerate(d):
            a1.add_patch(Rectangle((i - 0.3, y - 0.32), 0.6, 0.64, fc=COLOR[key] if v else "#f3f2ee",
                                   ec="none"))
        a1.text(-0.75, y, f"{key} {NAMES[key]}", ha="right", va="center", fontsize=12.5, color=INK)
    a1.set_xlim(-0.6, 9.6)
    a1.set_ylim(-0.6, len(rows) - 0.4)
    a1.set_yticks([])
    a1.set_xticks(n)
    a1.set_xlabel("Frame")
    a1.grid(False)
    a1.spines["left"].set_visible(False)
    a1.set_title("Alert decisions (filled = alert). Every rule only sees the current and past frames.",
                 fontsize=13, weight="normal", color=INK2)
    save(fig, "fig_filters_toy")


# ---------------------------------------------------------------- result charts

def fig_fusion():
    up = load("unified_evaluation_results.json")["upstream_detection"]
    cache = load("unified_confidences_cache.json")
    gt = sum(cache["test_gt_seqs"], [])
    overall = {}
    for lab, key in [("RGB", "test_rgb_seqs"), ("Pseudo-thermal", "test_th_seqs"), ("Fused (max)", "test_fused_seqs")]:
        pred = [1 if v >= 0.5 else 0 for v in sum(cache[key], [])]
        overall[lab] = calculate_frame_metrics(gt, pred)["f1"]
    groups = ["All test tiles", "Desert", "Forest", "Snow"]
    bnames = {"Desert": "Arid Desert", "Forest": "Temperate Forest", "Snow": "Snow/Alpine"}
    series = [("RGB", "YOLO11n-RGB", "#b9b8b2"), ("Pseudo-thermal", "YOLO11n-Thermal", "#7d7c77"),
              ("Fused (max)", "Late Fusion Max Gate", FILT)]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(groups))
    w = 0.26
    for j, (lab, key, col) in enumerate(series):
        vals = [overall[lab]] + [up[f"{key} ({bnames[g]})"]["f1"] for g in groups[1:]]
        bars = ax.bar(x + (j - 1) * w, vals, width=w - 0.02, color=col, label=lab, lw=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=10.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Frame F1 at τ = 0.50")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    ax.set_title(f"Taking the higher of the two confidences beats either stream alone "
                 f"(F1 {overall['Fused (max)']:.3f} vs {overall['RGB']:.3f} and {overall['Pseudo-thermal']:.3f})",
                 fontsize=14, pad=14)
    save(fig, "fig_fusion")
    return overall


def method_points():
    r = load("locked_benchmark_results.json")
    pts = []
    for k in ["M1", "M2", "M3", "M4"]:
        pts.append((k, r[k]["fa_rate"], 0.0, r[k]["tile_recall_pct"], 0.0, r[k]["f1"], r[k]["mean_delay_ms"]))
    for k in ["M5", "M6"]:
        pts.append((k, r[k]["fa_rate"][0], r[k]["fa_rate"][1], r[k]["tile_recall_pct"][0], r[k]["tile_recall_pct"][1],
                    r[k]["f1"][0], r[k]["mean_delay_ms"]))
    return r, pts


def fig_tradeoff():
    _, pts = method_points()
    mem = {"M1": "0 B", "M2": "20 B", "M3": "20 B", "M4": "5 B", "M5": "512 B, trained", "M6": "64 B, trained"}
    offs = {"M1": (0, 14, "center"), "M2": (0, -34, "center"), "M3": (14, 8, "left"),
            "M5": (-14, -40, "right"), "M6": (30, -4, "left")}
    fig, ax = plt.subplots(figsize=(12, 6.6))
    for k, fa, fa_sd, rec, rec_sd, f1, _ in pts:
        if fa_sd:
            ax.errorbar(fa, rec, xerr=fa_sd, yerr=rec_sd, fmt="none", ecolor=COLOR[k], elinewidth=1.4, capsize=4)
        ax.scatter(fa, rec, s=170, color=COLOR[k], edgecolor="white", linewidth=2, zorder=3)
        if k == "M4":
            continue  # same point as M3, labelled together
        dx, dy, ha = offs[k]
        label = f"{k} {NAMES[k]}\n({mem[k]})" if k == "M5" else f"{k} {NAMES[k]}  ({mem[k]})"
        if k == "M3":
            label = "M3 Median (20 B) and\nM4 Consensus (5 B): same point"
        ax.annotate(label, (fa, rec), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=12, color=INK)
    ax.annotate("better", xy=(0.6, 95.2), xytext=(2.0, 93.6), fontsize=12, color=INK2,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(76, 96)
    ax.set_yticks(range(76, 97, 4))
    ax.set_xlabel("False alerts per minute (test, lower is better)")
    ax.set_ylabel("Tile alert recall, % (higher is better)")
    ax.set_title("Filtering cuts false alerts 2.5 to 5x; recall drops by at most one or two tiles")
    ax.text(0.0, -0.16, "13 positive test tiles, so one tile = 7.7 points of recall. M5, M6: mean ± sd over 5 seeds.",
            transform=ax.transAxes, color=INK2, fontsize=11)
    save(fig, "fig_tradeoff")


def fig_raster():
    ids = test_ids()
    cache = load("unified_confidences_cache.json")
    order = sorted(range(len(ids)), key=lambda i: (cache["test_biomes"][i], -sum(cache["test_gt_seqs"][i]), ids[i]))
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2), sharey=True, gridspec_kw={"wspace": 0.04})
    for ax, key in zip(axes, ["M1", "M2"]):
        total = 0
        for row, i in enumerate(order):
            s = cache["test_fused_seqs"][i]
            g = np.array(cache["test_gt_seqs"][i])
            d = np.array(FILTERS[key].decide(s, TAU[key]))
            total += count_false_alerts(list(g), list(d))
            y = len(order) - 1 - row
            t = np.arange(len(s)) / FPS
            ax.bar(t[g == 1], 0.8, width=1 / FPS, bottom=y - 0.4, color=GT, align="edge", lw=0)
            on = d == 1
            ax.bar(t[on & (g == 1)], 0.36, width=1 / FPS, bottom=y - 0.18, color=COLOR[key], align="edge", lw=0)
            off = on & (g == 0)
            ax.bar(t[off], 0.8, width=1 / FPS, bottom=y - 0.4, color=COLOR[key], align="edge", lw=0)
        ax.set_title(f"{key} {NAMES[key]}: {total} false alerts", color=INK)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Time in clip (s)")
        ax.grid(False)
        ax.spines["left"].set_visible(False)
    labels = []
    for i in order:
        b = cache["test_biomes"][i]
        labels.append(f"{b} | {'person' if sum(cache['test_gt_seqs'][i]) else 'no person'}")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(labels[::-1], fontsize=10)
    axes[0].tick_params(axis="y", length=0)
    fig.suptitle("All 24 test tiles. Grey: person in view. Thin bar: alert on the person. "
                 "Tall bar outside grey: false alert.", x=0.12, ha="left", fontsize=13, color=INK2, y=0.955)
    save(fig, "fig_raster")


def fig_biome_fa():
    r = load("locked_benchmark_results.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, gridspec_kw={"wspace": 0.12})
    keys = ["M1", "M2", "M3", "M4"]
    for ax, b in zip(axes, ["desert", "forest", "snow"]):
        vals = [r[k]["per_biome"][b]["fa_rate"] for k in keys]
        recs = [f"{r[k]['per_biome'][b]['det_pos']}/{r[k]['per_biome'][b]['total_pos']}" for k in keys]
        y = np.arange(len(keys))[::-1]
        ax.barh(y, vals, color=[COLOR[k] for k in keys], height=0.62, lw=0)
        for yy, v, rc in zip(y, vals, recs):
            ax.text(v + 0.25, yy, f"{v:.1f}   ({rc} found)", va="center", fontsize=11, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{k} {NAMES[k]}" for k in keys] if b == "desert" else [""] * 4)
        ax.tick_params(axis="y", length=0)
        ax.set_title(b.capitalize(), loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 17)
        ax.set_xlabel("False alerts / min")
    fig.suptitle("The moving average gives the fewest false alerts in every biome", x=0.08, ha="left",
                 fontsize=16, weight="bold", y=1.04)
    save(fig, "fig_biome_fa")


def fig_budget():
    r = load("locked_benchmark_results.json")
    runtime = {"M1": 0.02, "M2": 0.30, "M3": 0.42, "M4": 0.63, "M5": 150.59, "M6": 21.80}
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(13, 6.6), gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.55})
    budget = 1000 / FPS
    a0.barh(0, 10.2, color=FILT, height=0.5, lw=0)
    a0.barh(0, budget - 10.2, left=10.2, color="#eceae4", height=0.5, lw=0)
    a0.text(5.1, 0, "2 x YOLO11n\n10.2 ms (GPU)", ha="center", va="center", color="white", fontsize=11, weight="bold")
    a0.text(10.2 + (budget - 10.2) / 2, 0, "free time in the frame", ha="center", va="center", color=INK2, fontsize=12)
    a0.annotate("slowest filter (Mamba): 0.15 ms", xy=(10.35, 0.25), xytext=(16, 0.62), fontsize=11.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2), annotation_clip=False)
    a0.set_xlim(0, budget)
    a0.set_yticks([])
    a0.set_xlabel("ms")
    a0.grid(False)
    a0.spines["left"].set_visible(False)
    a0.set_title(f"One frame at 24 fps = {budget:.1f} ms", fontsize=14)
    keys = ["M1", "M2", "M3", "M4", "M6", "M5"]
    y = np.arange(len(keys))[::-1]
    a1.barh(y, [runtime[k] for k in keys], color=[COLOR[k] for k in keys], height=0.6, lw=0)
    state = {k: r[k]["state_str"].split(" ")[0] + " B" for k in keys}
    for yy, k in zip(y, keys):
        a1.text(runtime[k] * 1.25, yy, f"{runtime[k]:g} µs | {state[k]} state", va="center", fontsize=11, color=INK2)
    a1.set_xscale("log")
    a1.set_xlim(0.01, 3000)
    a1.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
    a1.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1000"])
    a1.set_yticks(y)
    a1.set_yticklabels([f"{k} {NAMES[k]}" for k in keys])
    a1.tick_params(axis="y", length=0)
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("Filter time per frame, µs (log scale)")
    a1.set_title("Every filter costs < 0.4 % of the frame; the real cost is memory and training", fontsize=14)
    save(fig, "fig_budget")


def fig_tau_sweep():
    sens = load("unified_evaluation_results.json")["threshold_sensitivity"]
    t = [s["tau"] for s in sens]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"wspace": 0.22})
    a0.plot(t, [s["m1_fa_rate"] for s in sens], color=RAW, lw=2.4, marker="o", ms=6, label="M1 raw")
    a0.plot(t, [s["m2_fa_rate"] for s in sens], color=FILT, lw=2.4, marker="o", ms=6, label="M2 moving average")
    a0.set_xlabel("Threshold τ")
    a0.set_ylabel("False alerts / min (test)")
    a0.set_title("False alerts")
    a0.legend()
    a1.plot(t, [s["m1_f1"] for s in sens], color=RAW, lw=2.4, marker="o", ms=6, label="M1 raw")
    a1.plot(t, [s["m2_f1"] for s in sens], color=FILT, lw=2.4, marker="o", ms=6, label="M2 moving average")
    a1.set_xlabel("Threshold τ")
    a1.set_ylabel("Frame F1 (test)")
    a1.set_title("Frame F1")
    a1.set_ylim(0, 1)
    ratio = [s["m1_fa_rate"] / s["m2_fa_rate"] for s in sens if s["m2_fa_rate"] > 0]
    fig.suptitle(f"At every threshold the moving average has {min(ratio):.1f} to {max(ratio):.1f}x fewer false "
                 "alerts, at similar F1", x=0.08, ha="left", fontsize=16, weight="bold", y=1.04)
    save(fig, "fig_tau_sweep")


def fig_window():
    ab = load("unified_evaluation_results.json")["window_ablation_val"]
    ws = sorted(ab, key=lambda k: int(k))
    W = [int(w) for w in ws]
    fa = [ab[w]["val_fa_rate"] for w in ws]
    f1 = [ab[w]["val_f1"] for w in ws]
    lat = [ab[w]["latency_ms"] for w in ws]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.25})
    a0.bar(W, fa, color=[FILT if w == 5 else "#9ec5f4" for w in W], width=1.2, lw=0)
    for w, v, l in zip(W, fa, lat):
        a0.text(w, v + 0.4, f"{v:.1f}", ha="center", fontsize=11, color=INK2)
        a0.text(w, -2.6, f"{l:.0f} ms", ha="center", fontsize=10, color=MUTED)
    a0.set_xticks(W)
    a0.set_xticklabels([f"W={w}" for w in W])
    a0.set_ylabel("False alerts / min (validation)")
    a0.set_title("Longer window, fewer false alerts")
    a0.grid(axis="x", visible=False)
    a0.set_ylim(0, max(fa) * 1.15)
    a1.plot(W, f1, color=FILT, lw=2.4, marker="o", ms=7)
    for w, v in zip(W, f1):
        a1.text(w, v + 0.0008, f"{v:.3f}", ha="center", fontsize=11, color=INK2)
    a1.set_xticks(W)
    a1.set_ylim(0.84, 0.865)
    a1.set_ylabel("Frame F1 (validation)")
    a1.set_title("F1 hardly changes")
    fig.suptitle("Longer windows cut false alerts; W = 5 (208 ms of video) already gets most of the gain", x=0.08, ha="left",
                 fontsize=16, weight="bold", y=1.05)
    save(fig, "fig_window")


def fig_recall_mode():
    base = load("locked_benchmark_results.json")
    sens = load("threshold_sensitivity_f2_p95.json")
    keys = ["M1", "M2", "M3", "M4"]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    y = np.arange(len(keys))[::-1]
    for yy, k in zip(y, keys):
        b = base[k]["fa_rate"]
        s = sens[k]["fa_per_min"]
        ax.plot([b, s], [yy, yy], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([b], [yy], s=130, color="white", edgecolor=COLOR[k], linewidth=2.4, zorder=3)
        ax.scatter([s], [yy], s=130, color=COLOR[k], zorder=3)
        ax.text(s + 0.5, yy, f"{s:.1f}/min, recall {sens[k]['tile_recall_pct']:.0f} %  (τ = {sens[k]['tau_star']:.2f})",
                va="center", fontsize=11.5, color=INK)
        ax.text(b - 0.5, yy, f"{b:.1f}", va="center", ha="right", fontsize=11.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k} {NAMES[k]}" for k in keys])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 34)
    ax.set_xlabel("False alerts / min (test). Hollow: paper τ*. Filled: recall-weighted τ (F2, val precision ≥ 0.95)")
    ax.set_title("Lowering τ to catch faint targets: raw alerts jump to 22/min, filters stay near 8 to 10")
    save(fig, "fig_recall_mode")


def fig_dataset():
    split = json.loads((ROOT / "data/splits/dataset_episodes_split.json").read_text(encoding="utf-8"))
    tiles = []
    for name in ["train", "val", "test"]:
        for ep in split["episodes"][name]:
            if ep["modality"] == "rgb":
                tiles.append((ep["biome"], name, ep["metrics"]["target_frame_count"] > 0, ep["episode_id"]))
    col = {"train": "#9ec5f4", "val": "#3987e5", "test": NAVY}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"wspace": 0.08})
    for ax, b in zip(axes, ["desert", "forest", "snow"]):
        ts = sorted([t for t in tiles if t[0] == b], key=lambda t: (["train", "val", "test"].index(t[1]), not t[2]))
        for i, (_, sp, pos, _) in enumerate(ts):
            r, c = divmod(i, 6)
            ax.add_patch(Rectangle((c, -r), 0.86, 0.86, fc=col[sp], ec="none"))
            if pos:
                ax.add_patch(Rectangle((c + 0.3, -r + 0.3), 0.26, 0.26, fc="white", ec="none"))
        ax.set_xlim(-0.2, 6)
        ax.set_ylim(-5.3, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        npos = sum(1 for t in ts if t[2])
        ax.set_title(f"{b.capitalize()}: {len(ts)} tiles, {npos} with a person", loc="left", fontsize=14)
    handles = [Rectangle((0, 0), 1, 1, fc=col[s]) for s in col] + [Rectangle((0, 0), 1, 1, fc="white", ec=MUTED)]
    n = {s: sum(1 for t in tiles if t[1] == s) for s in col}
    fig.legend(handles, [f"train ({n['train']})", f"validation ({n['val']})", f"test ({n['test']})",
                         "white dot: person in tile"], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("45 generated videos, 90 tiles; both tiles of a video always go to the same split",
                 x=0.12, ha="left", fontsize=16, weight="bold", y=1.0)
    save(fig, "fig_dataset")


# ---------------------------------------------------------------- demo clip

def demo_clip(stem, key="M2", out_name="demo_forest", gif=False):
    ids = test_ids()
    if stem not in ids or not have_videos([stem]):
        return
    k = ids.index(stem)
    cache = load("unified_confidences_cache.json")
    s = np.array(cache["test_fused_seqs"][k])
    gt = cache["test_gt_seqs"][k]
    d1 = FILTERS["M1"].decide(list(s), TAU["M1"])
    d2 = FILTERS[key].decide(list(s), TAU[key])
    ma = np.array([s[max(0, i - 4): i + 1].mean() for i in range(len(s))])
    frames = read_frames(raw_video(stem))
    boxes = gt_boxes(stem) if "positive" in stem else [None] * len(frames)
    n = min(len(frames), len(s))
    t = np.arange(n) / FPS

    W, H = 1600, 900
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = OUT / f"{out_name}.mp4"
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "20", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1], wspace=0.12, hspace=0.35,
                          left=0.03, right=0.97, top=0.9, bottom=0.08)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_lamp = fig.add_subplot(gs[1, 1])
    fa1 = 0
    fa2 = 0
    for i in range(n):
        if d1[i] and gt[i] == 0 and (i == 0 or not (d1[i - 1] and gt[i - 1] == 0)):
            fa1 += 1
        if d2[i] and gt[i] == 0 and (i == 0 or not (d2[i - 1] and gt[i - 1] == 0)):
            fa2 += 1
        ax_img.clear()
        ax_img.imshow(frames[i])
        if boxes[i]:
            draw_box(ax_img, boxes[i], 640, 640, color="white", lw=2.5, ls=(0, (4, 2)))
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.grid(False)
        ax_img.set_title(f"{stem.split('_')[0]} test tile, t = {t[i]:4.1f} s"
                         + ("   (dashed: person)" if boxes[i] else ""), fontsize=14, loc="left")

        ax_tr.clear()
        ax_tr.fill_between(t[: i + 1], 0, np.array(gt[: i + 1]) * 0.9, step="post", color=GT, lw=0,
                           label="person in view")
        ax_tr.plot(t[: i + 1], s[: i + 1], color=RAW, lw=1.5, label="raw confidence")
        ax_tr.plot(t[: i + 1], ma[: i + 1], color=FILT, lw=2.5, label="5-frame average")
        ax_tr.axhline(TAU["M1"], color=RAW, lw=1, ls=(0, (4, 3)))
        ax_tr.axhline(TAU["M2"], color=FILT, lw=1, ls=(0, (4, 3)))
        ax_tr.set_xlim(0, t[n - 1])
        ax_tr.set_ylim(0, 1)
        ax_tr.set_xlabel("s")
        ax_tr.legend(loc="upper left", ncol=3, fontsize=11)
        ax_tr.set_title("Fused detector confidence", fontsize=14)

        ax_lamp.clear()
        ax_lamp.axis("off")
        ax_lamp.set_xlim(0, 2)
        ax_lamp.set_ylim(0, 1)
        for x0, on, fa, col, lab in [(0.02, d1[i], fa1, RAW, "RAW THRESHOLD"), (1.02, d2[i], fa2, FILT, "MOVING AVERAGE")]:
            wrong = on and gt[i] == 0
            ax_lamp.add_patch(FancyBboxPatch((x0, 0.25), 0.94, 0.6, boxstyle="round,pad=0.0,rounding_size=0.08",
                                             fc=col if on else "#f3f2ee", ec=col, lw=2.5))
            ax_lamp.text(x0 + 0.47, 0.66, lab, ha="center", fontsize=15, weight="bold",
                         color="white" if on else col)
            ax_lamp.text(x0 + 0.47, 0.42, ("ALERT (false)" if wrong else "ALERT") if on else "quiet",
                         ha="center", fontsize=14, color="white" if on else INK2)
            ax_lamp.text(x0 + 0.47, 0.08, f"false alerts so far: {fa}", ha="center", fontsize=14, color=INK)
        fig.suptitle("Same detector, same video: only the decision rule differs", x=0.03, ha="left",
                     fontsize=18, weight="bold")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        proc.stdin.write(np.ascontiguousarray(buf).tobytes())
    plt.close(fig)
    proc.stdin.close()
    proc.wait()
    print("saved", out.relative_to(ROOT))
    if gif:
        g = OUT / f"{out_name}.gif"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(out), "-vf",
                        "fps=8,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        str(g)], check=True)
        print("saved", g.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    jobs = {
        "pipeline": fig_pipeline, "filters_toy": fig_filters_toy, "fusion": fig_fusion, "tradeoff": fig_tradeoff,
        "raster": fig_raster, "biome_fa": fig_biome_fa, "budget": fig_budget, "tau_sweep": fig_tau_sweep,
        "window": fig_window, "recall_mode": fig_recall_mode, "dataset": fig_dataset, "biomes": fig_biomes,
        "dual_crop": fig_dual_crop, "flicker": fig_flicker,
    }
    only = [x for x in a.only.split(",") if x]
    for name, fn in jobs.items():
        if not only or name in only:
            fn()
    if not a.no_video and (not only or "demo" in only):
        demo_clip("forest_RGB_positive_3_right", out_name="demo_forest", gif=True)
        demo_clip("desert_RGB_negative_4_right", out_name="demo_desert_empty")


if __name__ == "__main__":
    main()
