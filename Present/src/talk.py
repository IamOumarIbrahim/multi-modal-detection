"""The finished talk ("Causal Temporal Filtering.pptx", 15 slides) rebuilt in the surviving looks.

Images from the talk are treated by what they are, so nothing loses its meaning:
  chart / diagram   kept exact; printed onto the style's paper (light looks) or shown in night mode on the
                    style's card (dark looks: neutrals inverted, colours kept and lifted so they stay readable)
  photo             the style's medium, kept recognisable; the annotated person is re-marked in the style
  modality figure   RGB / pseudo-thermal comparisons keep their colours (only grain + the style's print frame)
  logo / QR code    untouched (QR stays black on white so it still scans)

Run:  python -I talk.py                 all looks
      python -I talk.py graffiti 1 4    one look, some slides (JPEG only)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import art                                                     # noqa: E402
from engine import Slide, W, H, M, ROOT, fnt, tlen               # noqa: E402
from styles import STYLES, Ember                               # noqa: E402

SRC = ROOT / "talk" / "source"
OUT = ROOT / "talk"
KEYS = ["ember", "graffiti", "collage", "pburn", "signal"]      # the looks still in Present/
N = 15
CT, CB = 560, 1900
BOXES = json.loads((SRC / "boxes.json").read_text())           # person boxes (px in the 640x640 tile)


# ====================================================================== helpers
def load(name):
    im = Image.open(SRC / name).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def new(T, n, section):
    s = Slide(T, n, chrome=False)
    s.mono(M, 84, "Causal Temporal Filtering  /  Dual-stream UAV SAR", 30)
    s.mono(W - M, 84, f"{n:02d} / {N}", 30, anchor="ra")
    yb = H - 128
    T.footer(s, yb)
    s.mono(M, yb + 40, section, 28)
    s.mono(W - M, yb + 40, "University of Sharjah", 28, anchor="ra")
    return s


def fit_size(T, f, txt, size, maxw):
    sz = int(size * T.scale.get(f, 1))
    w = tlen(txt, fnt(T, f, sz), T.track.get(f, 0), sz)
    return int(size * min(1.0, maxw / w))


def night(a):
    """Dark-mode a chart: neutrals (background, text, axes, grey bars) flip lightness, colours keep their
    hue and are only lifted so dark ones stay visible on a dark card."""
    mx, mn = a.max(-1), a.min(-1)
    chroma = np.clip((mx - mn - 40) / 50.0, 0, 1)[..., None]     # pastels flip too
    flip = (255 - mx - mn)[..., None]
    lift = np.maximum(0, 140 - (mx + mn) / 2)[..., None]
    return np.clip(a + flip * (1 - chroma) + lift * chroma, 0, 255)


def chart(T, s, box, name, label, crop=None, pad=70, valign="top"):
    """A chart/diagram on the style's card. box = the space it may use; the card shrinks to the chart."""
    x, y, wmax, hmax = box
    im = load(name)
    if crop:
        im = im.crop(crop)
    k = min((wmax - 2 * pad) / im.width, (hmax - 2 * pad) / im.height)
    iw, ih = int(im.width * k), int(im.height * k)
    im = im.resize((iw, ih), Image.LANCZOS)
    if k > 1.6:
        im = im.filter(ImageFilter.UnsharpMask(2, 60, 2))
    w, h = iw + 2 * pad, ih + 2 * pad
    y = y + (hmax - h) // 2 if valign == "mid" else y
    s.card((x, y, x + w, y + h))
    cx, cy = x + pad, y + pad
    under = np.asarray(s.g.crop((cx, cy, cx + iw, cy + ih)), np.float32)
    a = np.asarray(im, np.float32)
    if under.mean() > 128:                       # light paper: print the chart (multiply keeps the paper texture)
        out = under * a / 255
    else:                                        # dark card: night mode, screened onto the card
        out = 255 - (255 - under) * (255 - night(a)) / 255
    s.pic(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), cx, cy, 0, f"Image - {label}")
    return x, y, w, h


def _grain(im, seed, amt=6):
    a = art.grain(np.asarray(im, np.float32), np.random.default_rng(seed), amt)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def soft(T, im, box, seed):
    """The style's medium for a content photo, kept readable."""
    k = T.key
    if k == "graffiti":                          # photocopy, but gentler than the poster art so detail survives
        from PIL import ImageEnhance, ImageOps
        g = ImageEnhance.Contrast(ImageOps.autocontrast(im.convert("L"), cutoff=1)).enhance(1.12)
        return _grain(g.convert("RGB"), seed, 9)
    if k == "collage":
        return T.treat(im, box, seed, "rgb")
    if k == "signal":
        return T._burn.treat(im, box, seed, "rgb")
    a = np.asarray(im, np.float32)
    L = art.stretch(art.lum(im), 1, 99.5)
    if k == "ember":
        out = art.apply_lut(L * 0.9, "fire") * 0.6 + a * 0.4 * np.array([1.0, 0.82, 0.68])
        out = art.grain(out, np.random.default_rng(seed), 7)
        if box:                                  # the person glows as embers
            glow = np.asarray(Ember.treat(T, im, box, seed, "th"), np.float32)
            hh, ww = L.shape
            yy, xx = np.mgrid[0:hh, 0:ww]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            sg = 2.2 * max(box[2] - box[0], box[3] - box[1])
            m = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg ** 2))[..., None]
            out = 255 - (255 - out) * (255 - glow * m) / 255
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    # pixel burn: fire-graded LED cells, one edge dissolving
    w, h = im.size
    cell = max(8, min(w, h) // 120)
    small = art.pixelate(im, cell, 0.03, seed)
    Ls = (small @ np.array([0.299, 0.587, 0.114], np.float32)) / 255
    Ls = (Ls - np.percentile(Ls, 1)) / max(1e-3, np.percentile(Ls, 99.5) - np.percentile(Ls, 1))
    small = art.apply_lut(np.clip(Ls, 0, 1) * 0.92, "fire") * 0.55 + small * 0.45
    gh, gw = small.shape[:2]
    keep = art.edge_keep(gw, gh, ["left", "right", "top", "bottom"][seed % 4], 0.0, 0.16)
    return art.cells_to_image(small, cell, w, h, T.bg, 2, keep, seed)


def photo(T, s, box, name, label, seed, keep_color=False):
    """A photo from the talk in a slot. keep_color: modality figures, only grain + the frame."""
    x, y, w, h = [int(v) for v in box]
    cw, ch = T.inner(w, h)
    im = load(name)
    pb = BOXES.get(name)
    SW, SH = im.size
    a = cw / ch
    if keep_color:                                  # contain: never crop a figure
        k = min(cw / SW, ch / SH)
        im = im.resize((int(SW * k), int(SH * k)), Image.LANCZOS)
        cw, ch = im.size
        w, h = cw + (w - T.inner(w, h)[0]), ch + (h - T.inner(w, h)[1])
        im = _grain(im, seed, 5)
    else:                                           # cover: crop to the slot, keep the person in frame
        cwid, chei = (SW, SW / a) if SW / a <= SH else (SH * a, SH)
        cx, cy = ((pb[0] + pb[2]) / 2 + 0.12 * cwid, (pb[1] + pb[3]) / 2) if pb else (SW / 2, SH / 2)   # room for the label
        x0 = min(max(0, cx - cwid / 2), SW - cwid)
        y0 = min(max(0, cy - chei / 2), SH - chei)
        im = im.crop((int(x0), int(y0), int(x0 + cwid), int(y0 + chei))).resize((cw, ch), Image.LANCZOS)
        if pb:
            k = cw / cwid
            pb = [(pb[0] - x0) * k, (pb[1] - y0) * k, (pb[2] - x0) * k, (pb[3] - y0) * k]
        im = soft(T, im, pb, seed)
        if pb:
            im = T.mark(im, pb, seed)
    out = T.frame(im, w, h, seed, label)
    s.pic(out, x, y, T.radius, f"Image - {label}")
    return x, y, out.width, out.height


def light(T):
    return T.bg[0] > 128


def logos(T, s, x, y, hgt=150):
    """University logo (white + green; white turns to ink on light looks) and the ICSPIS logo on a plate."""
    u = Image.open(SRC / "s01_23.png").convert("RGBA")
    if light(T):
        a = np.asarray(u).copy()
        wht = a[..., :3].min(-1) > 200
        a[wht, :3] = T.fg
        u = Image.fromarray(a, "RGBA")
    u = u.resize((int(u.width * hgt / u.height), hgt), Image.LANCZOS)
    s.pic(u, x, y, 0, "Logo - University of Sharjah")
    ic = load("s01_17.png")
    ih = int(hgt * 0.62)
    ic = ic.resize((int(ic.width * ih / ic.height), ih), Image.LANCZOS)
    plate = Image.new("RGB", (ic.width + 40, ic.height + 30), (255, 255, 255))
    plate.paste(ic, (20, 15))
    s.pic(plate, x + u.width + 60, y + (hgt - plate.height) // 2, 0, "Logo - ICSPIS")


def num(T):
    return getattr(T, "num_col", "acc")


def step_side(T, s, x, w, n, title, sub):
    s.text(x - 10, CT - 40, n, 330, "D", num(T))
    yy = CT + int(330 * T.scale.get("D", 1) * 1.25)
    yy = s.block(x, yy, w, title, 70, "BB", lh=1.15)
    s.block(x, yy + 30, w, sub, 44, fill="mute")


# ====================================================================== the 15 slides
def t01(T):
    s = Slide(T, 1, chrome=False)
    T.cover(s)
    mw = 1420
    s.mono(M, 150, "Department of Computer Engineering", 32, "fg")
    s.mono(M, 198, "University of Sharjah, Sharjah, UAE", 32)
    sz = min(fit_size(T, "H", "Causal Temporal", 220, mw), fit_size(T, "HI", "Filtering", 220, mw))
    y2 = 470 + int(sz * T.scale.get("H", 1) * 1.12)
    s.head(M, 470, "Causal Temporal", sz)
    s.head(M, y2, "Filtering", sz, f="HI")
    yy = y2 + int(sz * T.scale.get("HI", 1) * 1.32)
    yy = s.block(M, yy, mw, "for Reliable Alerts in Dual-Stream UAV Search and Rescue Detection", 62, "BB", lh=1.22)
    s.block(M, yy + 70, mw, "Oumar Mamoun Ibrahim\nMohamad Khairi bin Ishak", 44, lh=1.35)
    logos(T, s, M, 1720, 200)
    return s


def t02(T):
    s = new(T, 2, "Introduction & Objectives")
    s.title("The problem", "Introduction & Objectives")
    s.bullets(M, CT + 30, 1720, [
        "Autonomous unmanned aerial vehicles in wilderness search and rescue require reliable alerts under strict "
        "size, weight, power, and cost constraints.",
        "In these missions, false detections trigger costly hover maneuvers that waste battery life, and distract "
        "operators from search areas.",
        "A false alarm may cascade downstream edge mechanisms such as enabling other types of detection sensors to "
        "verify targets, upload current drone location via telemetry, or in extreme cases land the drone in an area "
        "with no person to rescue."], 52)
    chart(T, s, (2060, CT - 60, W - M - 2060, 1360), "s02_5.png", "False alarm vs real target", valign="mid")
    return s


def t03(T):
    s = new(T, 3, "Introduction & Objectives")
    s.title("The motivation", "Introduction & Objectives")
    yy = s.block(M, CT + 30, 2250, "Dealing with false alarms in the domain of Search & Rescue is a tricky endeavor. "
                                   "The primary reasons are:", 70, "BB", lh=1.22)
    s.bullets(M, yy + 70, 2250, [
        "The lack of diverse Search & Rescue UAV Point-of-View (POV) temporal datasets.",
        "Search & Rescue UAVs often contain multi-modal video streams such as Thermal and Near Infra-Red paired with "
        "regular RGB streams. No reliable training data exists."], 60)
    photo(T, s, (2780, 250, 880, 1640), "s03_8.png", "One tile, three views", 3, keep_color=True)
    return s


def t04(T):
    s = new(T, 4, "Introduction & Objectives")
    s.title("The objectives", "Introduction & Objectives")
    s.block(M, CT + 20, 1800, "Hence, our work aims to:", 56, "BB")
    items = ["Curate a diverse RGB and Pseudo-Thermal synthetic video dataset from the POV of a Search & Rescue UAV "
             "across different environments.",
             "Evaluate temporal causal filters on the number of false alerts generated: classic vs models that learn.",
             "Provide a recommendation based off the benchmark."]
    yy = CT + 150
    for i, t in enumerate(items):
        s.text(M, yy - 8, f"{i + 1:02d}", 84, "D", num(T))
        e = s.block(M + 230, yy, 1580, t, 50)
        yy = max(e, yy + 120) + 70
    photo(T, s, (2150, 330, 1510, 1540), "s04_12.png", "Desert, person annotated", 4)
    return s


def t05(T):
    s = new(T, 5, "Literature Review / Background")
    s.title("Existing literature", "Literature Review / Background")
    groups = [("Datasets & sensing", ["KAIST [1] and LLVIP [2]: ground datasets that handle challenging outdoor "
                                       "environments.", "WiSARD [3]: multi-view benchmark.",
                                       "Rudol and Doherty [4]: dual-band geolocation from UAVs."]),
              ("Detection on the edge", ["Large Neural Nets drain battery quickly [5].",
                                         "Plain YOLO11n achieves high FPS but detections fluctuate due to motion "
                                         "blur and terrain [6]."]),
              ("Temporal decisions", ["Classical sequential detection handles fluctuations using statistical "
                                      "methods [7], [8].",
                                      "More recently, Mamba enabled sequence modelling with linear complexity [9]."])]
    gap = 70
    cw = (W - 2 * M - 2 * gap) // 3
    for i, (hd, it) in enumerate(groups):
        x = M + i * (cw + gap)
        s.card((x, CT, x + cw, 1380))
        s.mono(x + 60, CT + 64, hd, 32, "acc")
        s.bullets(x + 60, CT + 150, cw - 120, it, 50, gap=0.6)
    s.card((M, 1470, W - M, 1860), hi=True)
    s.mono(M + 60, 1520, "The gap", 32, "on_acc")
    s.block(M + 60, 1590, W - 2 * M - 120,
            "Most SAR literature covers bounding box accuracy; no previous work examines whether learned models are "
            "needed to provide reliable alerts within edge budgets.", 56, "BB", lh=1.25, fill="on_acc")
    return s


def t06(T):
    s = new(T, 6, "Methodology")
    s.title("Our proposed approach", "Methodology")
    chart(T, s, (M, CT - 20, 2560, 1360), "s06_7.png", "Dual-stream late fusion", valign="mid")
    step_side(T, s, 2900, W - M - 2900, "01", "Dual-stream detection",
              "RGB and pseudo-thermal YOLO11n detectors, combined by late decision fusion.")
    return s


def t07(T):
    s = new(T, 7, "Methodology")
    s.title("Our proposed approach", "Methodology")
    chart(T, s, (M, CT - 20, 2560, 1360), "s07_18.png", "Sliding window mean", valign="mid")
    step_side(T, s, 2900, W - M - 2900, "02", "Moving average",
              "A causal sliding window smooths the volatile per-frame confidence.")
    return s


def t08(T):
    s = new(T, 8, "Methodology")
    s.title("Our proposed approach", "Methodology")
    chart(T, s, (M, CT - 20, 2560, 1360), "s08_14.png", "Median filter", valign="mid")
    step_side(T, s, 2900, W - M - 2900, "03", "Median filter",
              "An order-statistic filter rejects single-frame impulse spikes.")
    return s


def t09(T):
    s = new(T, 9, "Methodology")
    s.title("Six decision strategies", "Methodology")
    cards = [("M1", "Raw Baseline", "Direct score threshold", "Instantaneous confidence", "classic"),
             ("M2", "Moving Average", "Sliding window mean", "5-frame window, 166.7 ms span", "classic"),
             ("M3", "Median Filter", "Order-statistic median", "Spike rejection, 5 frames", "classic"),
             ("M4", "History Consensus", "Majority rule persistence", "At least 3 of 5 detections", "classic"),
             ("M5", "Mamba-SSSM", "Selective state-space filter", "Learned recurrent dynamics", "learned"),
             ("M6", "GRU Baseline", "Single-layer GRU, hidden size 16",
              "1 681 params, 64 B state, edge-compatible; 44 tiles / 5 seeds", "learned")]
    gap = 60
    cw, chh = (W - 2 * M - 2 * gap) // 3, (CB - 40 - CT - gap) // 2
    for i, (m, t, sub, note, kind) in enumerate(cards):
        x, y = M + (i % 3) * (cw + gap), CT + (i // 3) * (chh + gap)
        hi = kind == "learned"
        s.card((x, y, x + cw, y + chh), hi=hi)
        fc = "on_acc" if hi else "fg"
        s.text(x + 60, y + 50, m, 96, "D", "on_acc" if hi else num(T))
        s.mono(x + cw - 60, y + 70, kind, 28, "on_acc" if hi else "mute", anchor="ra")
        s.text(x + 60, y + 220, t, 76, "BB", fc)
        s.text(x + 60, y + 340, sub, 50, "B", fc)
        s.block(x + 60, y + 430, cw - 120, note, 40, fill="on_acc" if hi else "mute", lh=1.3)
    return s


def t10(T):
    s = new(T, 10, "Experimental Setup")
    s.title("How we ran the experiment", "Experimental Setup")
    s.bullets(M, CT + 30, 2250, [
        "3 Environments: Desert, Forest, Snow",
        "1280x720 Video at 24 Hz, 10 s",
        "90 video tiles corpus (42 target containing, 48 non-target containing)",
        "Evaluation on YOLO11n",
        "Train/Val/Test split follows 49% / 24% / 27%",
        "Videos are generated in RGB via Grok/Gemini simulating UAV flights at 30–50 m above ground level (AGL) "
        "with 30–45° downward camera angles.",
        "LUT-based thermal representation is used instead of real thermal video."], 50, gap=0.5)
    photo(T, s, (2620, 250, 1040, 1640), "s10_9.png", "RGB and pseudo-thermal tiles", 5, keep_color=True)
    return s


def t11(T):
    s = new(T, 11, "Results & Discussion")
    s.title("Our results and what they mean", "Results & Discussion")
    chart(T, s, (M, CT - 20, 2620, 1360), "s11_7.png", "F1 vs false alerts", valign="mid")
    x = 3000
    s.mono(x, CT + 40, "How to read it", 32, "acc")
    s.block(x, CT + 120, W - M - x, "Up and to the left is better: higher frame F1, fewer false alerts per minute.",
            56, "BB", lh=1.25)
    return s


def t12(T):
    s = new(T, 12, "Results & Discussion")
    s.title("Results across environments", "Results & Discussion")
    yy = s.block(M, CT + 30, 1480, "Causal filters performance across the three different environments. Forest "
                                   "achieves the highest false alerts generated, caused by the high terrain clutter.",
                 58)
    s.block(M, yy + 50, 1480, "Alpine environments achieve the lowest false alerts due to its visual difference in "
                              "color.", 58)
    x0 = 1860
    _, cy, _, chh = chart(T, s, (x0, CT - 20, W - M - x0, 900), "s12_8.png", "False alerts per biome")
    y = cy + chh + 90
    cw = (W - M - x0) // 3
    for i, (b, raw, best) in enumerate((("Desert", "10.50", "1.50"), ("Forest", "12.49", "3.33"),
                                        ("Snow / Alpine", "4.50", "1.50"))):
        x = x0 + i * cw
        s.mono(x, y, b, 30, "acc")
        s.text(x - 4, y + 60, best, 150, "D", "fg")
        s.mono(x, y + 60 + int(150 * T.scale.get("D", 1) * 1.25), f"M2 / min, raw {raw}", 26)
    return s


def t13(T):
    s = new(T, 13, "Results & Discussion")
    s.title("Examples", "Results & Discussion")
    w, h, g = 1700, 1340, W - 2 * M - 2 * 1700
    photo(T, s, (M, CT - 30, w, h), "s13_11.png", "Snow / Alpine", 6)
    photo(T, s, (M + w + g, CT - 30, w, h), "s13_7.png", "Temperate forest", 7)
    return s


def t14(T):
    s = new(T, 14, "Results & Discussion")
    s.title("Base vs recall-weighted", "Results & Discussion")
    chart(T, s, (M, CT - 20, 2400, 1360), "s14_7.png", "False alerts per minute", valign="mid")
    x, w = 2780, W - M - 2780
    for i, (v, a, b) in enumerate((("2.10", "false alerts / min with M2", "down from 9.00 raw, at the F1 threshold"),
                                   ("9.57", "with recall weighting", "F2, 5x cost: down from 21.97 raw"))):
        y = CT + i * 660
        s.text(x - 6, y, v, 230, "D", num(T) if i == 0 else "fg")
        yy = y + int(230 * T.scale.get("D", 1) * 1.34)
        s.text(x, yy, a, 48, "BB")
        s.mono(x, yy + 80, b, 28)
    return s


def t15(T):
    s = new(T, 15, "Conclusion")
    s.title("Final notes", "Conclusion")
    yy = s.block(M, CT + 30, 2150, "This paper investigated causal temporal decision strategies to convert noisy "
                                   "dual-stream visible and pseudo-thermal detections into reliable search and rescue "
                                   "alerts.", 56, "BB", lh=1.28)
    s.block(M, yy + 70, 2150, "Simple causal filtering offers an efficient, training-free baseline, while learned "
                              "sequence models provide maximum false-alarm suppression when memory allows. Future "
                              "work will test these methods on real RGB-thermal datasets and embedded UAV hardware.",
            50)
    q = 900
    qx, qy = W - M - q - 40, CT
    s.card((qx - 40, qy - 40, qx + q + 40, qy + q + 40))
    qr = load("s15_8.png").resize((q, q), Image.LANCZOS)
    s.pic(qr, qx, qy, 0, "Image - QR code")
    s.mono(qx + q / 2, qy + q + 90, "Code at", 30, "acc", anchor="ma")
    s.mono(qx + q / 2, qy + q + 140, "github.com/IamOumarIbrahim", 28, "fg", anchor="ma")
    s.mono(qx + q / 2, qy + q + 184, "/multi-modal-detection", 28, "fg", anchor="ma")
    s.hw(M, 1640, "thank you!", 110, "acc")
    return s


SLIDES = [t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15]
NAMES = ["cover", "problem", "motivation", "objectives", "literature", "method_fusion", "method_moving_average",
         "method_median", "method_strategies", "setup", "results_tradeoff", "results_biomes", "examples",
         "results_recall", "conclusion"]


def build(key, only=None, pptx=True):
    from pptx import Presentation
    from pptx.util import Emu
    T = STYLES[key]()
    folder = OUT / T.folder.replace("_40", "")
    folder.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
    for i, (fn, nm) in enumerate(zip(SLIDES, NAMES), 1):
        if only and i not in only:
            continue
        s = fn(T)
        s.render().save(folder / f"{i:02d}_{nm}.jpg", quality=92, subsampling=0)
        if pptx:
            s.to_pptx(prs)
        print(f"{key} {i}", end="  ", flush=True)
    print()
    if pptx and not only:
        out = OUT / f"Causal Temporal Filtering - {T.name}.pptx"
        prs.save(out)
        print("saved", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    keys = [a for a in args if a in STYLES] or KEYS
    only = set(int(a) for a in args if a.isdigit()) or None
    for k in keys:
        build(k, only, pptx=not only)
