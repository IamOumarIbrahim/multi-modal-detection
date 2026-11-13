"""Image treatments. Every style turns the paper's own frames (UAV tiles, pseudo-thermal, person boxes,
confidence traces) into its medium: film burn, pixel mosaic, spray paint, stipple, halftone, collage."""
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# ====================================================================== colour ramps
RAMPS = {
    "fire":   [(0, (13, 10, 9)), (.22, (70, 8, 5)), (.42, (190, 34, 14)), (.6, (250, 96, 24)),
               (.76, (255, 160, 60)), (.9, (255, 222, 168)), (1, (255, 247, 236))],
    "rose":   [(0, (13, 10, 9)), (.25, (80, 10, 30)), (.45, (196, 36, 70)), (.65, (240, 96, 96)),
               (.82, (250, 170, 150)), (1, (255, 240, 230))],
    "amber":  [(0, (13, 10, 9)), (.3, (110, 30, 6)), (.55, (240, 120, 20)), (.75, (255, 190, 60)),
               (.9, (255, 232, 150)), (1, (255, 250, 230))],
    "rainbow": [(0, (255, 40, 110)), (.2, (255, 120, 20)), (.4, (255, 210, 0)), (.6, (30, 200, 120)),
                (.8, (20, 120, 255)), (1, (120, 50, 220))],
    "blue":    [(0, (175, 230, 255)), (.4, (40, 170, 245)), (.75, (20, 90, 230)), (1, (12, 30, 120))],
    "pink":    [(0, (255, 215, 232)), (.45, (255, 110, 170)), (.8, (240, 20, 120)), (1, (170, 0, 90))],
    "violet":  [(0, (255, 30, 70)), (.5, (200, 20, 140)), (1, (70, 30, 200))],
    "teal":    [(0, (190, 250, 235)), (.5, (20, 200, 190)), (1, (10, 80, 140))],
    "sunset":  [(0, (255, 225, 90)), (.4, (255, 120, 40)), (.75, (230, 40, 90)), (1, (90, 30, 140))],
}


@lru_cache(None)
def lut(name):
    st = RAMPS[name]
    xs = np.array([p for p, _ in st])
    out = np.zeros((256, 3), np.float32)
    for c in range(3):
        out[:, c] = np.interp(np.linspace(0, 1, 256), xs, [col[c] for _, col in st])
    return out


def apply_lut(I, name):
    return lut(name)[(np.clip(I, 0, 1) * 255).astype(np.uint8)]


# ====================================================================== the paper's material
@lru_cache(None)
def index():
    return json.loads((ASSETS / "index.json").read_text())


@lru_cache(None)
def strips():
    return json.loads((ASSETS / "strips.json").read_text())


@lru_cache(None)
def traces():
    return json.loads((ASSETS / "traces.json").read_text())


@lru_cache(None)
def _pool(kind):
    ix = index()
    if kind == "neg":
        return [e for e in ix if not e["positive"]]
    # "hero" frames: person clearly visible and well inside the tile
    good = [e for e in ix if e["positive"] and 50 < e["box"][3] - e["box"][1] < 330
            and e["box"][0] > 20 and e["box"][2] < 620 and e["box"][1] > 20 and e["box"][3] < 620]
    order = {"desert": 0, "snow": 1, "forest": 2}
    good.sort(key=lambda e: (order[e["biome"]], e["id"]))
    # interleave biomes so consecutive seeds change landscape
    by = [[e for e in good if e["biome"] == b] for b in ("desert", "snow", "forest")]
    out = []
    for i in range(max(map(len, by))):
        for b in by:
            if i < len(b):
                out.append(b[i])
    return out


def pick(seed, kind="pos", biome=None):
    p = _pool("neg" if kind == "neg" else "pos")
    if biome:
        p = [e for e in p if e["biome"] == biome] or p
    return p[(seed * 7 + 3) % len(p)]


@lru_cache(64)
def load(fid, kind="rgb"):
    return Image.open(ASSETS / "frames" / f"{fid}_{kind}.jpg").convert("RGB")


def fit(entry, w, h, zoom=1.0, kind="rgb", src=None):
    """Crop a 640 tile to w:h around the person (zoom > 1 = tighter); returns image and mapped box."""
    im = src or load(entry["id"], kind)
    SW, SH = im.size
    a = w / h
    cw, ch = (SW, SW / a) if SW / a <= SH else (SH * a, SH)       # largest w:h window inside the source
    cw, ch = cw / zoom, ch / zoom
    b = entry.get("box")
    cx, cy = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if b else (SW / 2, SH / 2)
    x0 = min(max(0, cx - cw / 2), SW - cw)
    y0 = min(max(0, cy - ch / 2), SH - ch)
    out = im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((int(w), int(h)), Image.LANCZOS)
    if not b:
        return out, None
    k = w / cw
    return out, [(b[0] - x0) * k, (b[1] - y0) * k, (b[2] - x0) * k, (b[3] - y0) * k]


def lum(im):
    return np.asarray(im.convert("L"), np.float32) / 255


def stretch(a, lo=2, hi=98, gamma=1.0):
    l, h = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - l) / (h - l + 1e-6), 0, 1) ** gamma


def heat(im, box=None, boost=0.55):
    """Our own 'pseudo-thermal': darkness of the RGB tile, plus a warm bump where the person is."""
    L = lum(im.filter(ImageFilter.GaussianBlur(1.2)))
    H = stretch(1 - L, 3, 99.5, 1.35)
    if box:
        h, w = H.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        sx, sy = max(8, (box[2] - box[0]) * 0.7), max(8, (box[3] - box[1]) * 0.6)
        H = np.clip(H * 0.75 + boost * np.exp(-((xx - cx) / sx) ** 2 - ((yy - cy) / sy) ** 2), 0, 1)
    return H


# ====================================================================== film burn (55)
def _noise(rng, gh, gw, cells=(5, 8)):
    n = rng.random(cells).astype(np.float32)
    return np.asarray(Image.fromarray(n, "F").resize((gw, gh), Image.BICUBIC))


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def fade_w(fade, X, Y, A):
    if not fade:
        return 1.0
    side, a, b = fade
    xn = X / A
    return {"left": smooth(a, b, xn), "right": smooth(a, b, 1 - xn),
            "top": smooth(a, b, Y), "bottom": smooth(a, b, 1 - Y)}[side]


def intensity(gw, gh, seed, style=None, fade=None):
    rng = np.random.default_rng(seed)
    A = gw / gh
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    X, Y = xx / max(1, gw - 1) * A, yy / max(1, gh - 1)
    I = np.zeros_like(X)
    style = style or rng.choice(["leak", "streak", "flare", "wave", "ramp"], p=[.34, .28, .2, .08, .1])
    if style == "ramp":
        ang = rng.uniform(-0.5, 0.5)
        I += (X / A * math.cos(ang) + Y * math.sin(ang)) * 0.9
    for _ in range(rng.integers(2, 5)):
        cx, cy = rng.uniform(-0.1, 1.1) * A, rng.uniform(-0.15, 1.15)
        sx, sy = rng.uniform(0.18, 0.6) * A, rng.uniform(0.2, 0.7)
        th = rng.uniform(0, math.pi)
        dx, dy = X - cx, Y - cy
        u = dx * math.cos(th) + dy * math.sin(th)
        v = -dx * math.sin(th) + dy * math.cos(th)
        I += rng.uniform(0.35, 0.9) * np.exp(-(u / sx) ** 2 - (v / sy) ** 2)
    if style in ("streak", "leak"):
        ang = rng.uniform(-0.3, 0.3)
        for _ in range(rng.integers(3, 9)):
            xc, wd = rng.uniform(0, A), rng.uniform(0.006, 0.07) * A
            d = X - xc - (Y - 0.5) * ang
            I += rng.uniform(0.25, 0.9) * np.exp(-(d / wd) ** 2) * \
                (0.4 + 0.6 * np.exp(-((Y - rng.uniform(0, 1)) / 0.45) ** 2))
    if style == "flare":
        cx, cy = rng.uniform(0.3, 0.8) * A, rng.uniform(0.3, 0.7)
        I += 1.1 * np.exp(-((X - cx) / (0.5 * A)) ** 2 - ((Y - cy) / 0.09) ** 2)
        I += 0.9 * np.exp(-((X - cx) / 0.12) ** 2 - ((Y - cy) / 0.16) ** 2)
    if style == "wave":
        f, ph, amp = rng.uniform(0.6, 1.4), rng.uniform(0, 6.28), rng.uniform(0.15, 0.3)
        yc = 0.5 + amp * np.sin(2 * math.pi * f * X / A + ph)
        I += 1.0 * np.exp(-((Y - yc) / 0.035) ** 2) + 0.5 * np.exp(-((Y - yc) / 0.14) ** 2)
    I *= 0.7 + 0.6 * _noise(rng, gh, gw)
    lo, hi = np.percentile(I, 4), np.percentile(I, 99.7)
    I = np.clip((I - lo) / (hi - lo + 1e-6), 0, 1) ** rng.uniform(1.15, 1.6)
    I *= fade_w(fade, X, Y, A)
    return np.clip(I, 0, 1)


def leak_field(w, h, seed, style=None, fade=None):
    I = intensity(max(4, w // 8), max(4, h // 8), seed, style, fade)
    return np.asarray(Image.fromarray(I.astype(np.float32), "F").resize((w, h), Image.BICUBIC))


def grain(rgb, rng, amt=8.0, I=None):
    h, w = rgb.shape[:2]
    g = rng.normal(0, 1, (h, w)).astype(np.float32) * (amt if I is None else amt * (0.6 + 0.9 * np.clip(I, 0, 1)))
    return rgb + g[..., None] + rng.normal(0, amt * 0.3, (h, w, 3)).astype(np.float32)


def dust(im, rng, n=None, light=(255, 244, 230), dark=(8, 6, 5), scratches=True):
    w, h = im.size
    d = ImageDraw.Draw(im)
    for _ in range(n if n is not None else int(w * h / 90000)):
        x, y, r = rng.integers(0, w), rng.integers(0, h), rng.uniform(0.8, 3.2)
        d.ellipse((x - r, y - r, x + r, y + r), fill=light if rng.random() < 0.5 else dark)
    if scratches:
        for _ in range(rng.integers(1, 3)):
            x, y0 = rng.integers(0, w), rng.integers(0, max(1, h // 2))
            d.line((x, y0, x + rng.integers(-12, 12), y0 + rng.integers(h // 8, h // 3)), fill=(120, 96, 84), width=1)
    return im


def tex_burn(w, h, seed, style=None, fade=None, ramp="fire", dusty=True):
    rng = np.random.default_rng(seed + 991)
    I = leak_field(w, h, seed, style, fade)
    yy, xx = np.mgrid[0:h, 0:w]
    I = I * (1 - 0.28 * ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2)
    rgb = grain(apply_lut(I, ramp), rng, 9, I)
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    return dust(im, rng) if dusty else im


def burn_photo(entry, w, h, seed, zoom=1.0, kind="rgb", ramp="fire"):
    """A drone tile shot on burnt film: warm duotone + light leak screened over it + grain + dust."""
    im, box = fit(entry, w, h, zoom, "rgb")
    rng = np.random.default_rng(seed)
    if kind == "th":
        T = heat(im, box)
        base = apply_lut(T * 0.95, ramp)
    else:
        L = stretch(lum(im), 1, 99.5, 1.15)
        base = apply_lut(0.18 + 0.8 * L, ramp) * 0.75 + np.asarray(im, np.float32) * np.array([1.0, 0.72, 0.55]) * 0.3
    leak = leak_field(w, h, seed + 5, rng.choice(["leak", "streak", "flare"]))
    lk = apply_lut(leak, ramp) / 255
    out = 255 - (255 - base) * (1 - lk * 0.85)                       # screen
    yy, xx = np.mgrid[0:h, 0:w]
    out *= (1 - 0.35 * ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2)[..., None]
    out = grain(out, rng, 9)
    return dust(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), rng, int(w * h / 150000)), box


# ====================================================================== pixel mosaic (56)
def pixelate(im, cell, jitter=0.05, seed=0, levels=None):
    w, h = im.size
    gw, gh = max(1, math.ceil(w / cell)), max(1, math.ceil(h / cell))
    small = np.asarray(im.resize((gw, gh), Image.BOX), np.float32)
    if levels:
        small = np.round(small / 255 * (levels - 1)) / (levels - 1) * 255
    rng = np.random.default_rng(seed)
    small *= (1 + rng.normal(0, jitter, (gh, gw)))[..., None]
    return small


def cells_to_image(small, cell, w, h, bg, gap=0, keep=None, seed=0, half=0.18):
    rng = np.random.default_rng(seed + 17)
    gh, gw = small.shape[:2]
    rgb = small.copy()
    bgc = np.array(bg, np.float32)
    if keep is not None:
        r = rng.random((gh, gw))
        gone = r > keep
        hf = (~gone) & (r > keep * (1 - half))
        rgb[hf] = rgb[hf] * 0.55 + bgc * 0.45
        rgb[gone] = bgc
    big = np.repeat(np.repeat(np.clip(rgb, 0, 255), cell, 0), cell, 1)[:h, :w]
    if gap:
        big[(np.arange(h) % cell) < gap, :] = bgc
        big[:, (np.arange(w) % cell) < gap] = bgc
    return Image.fromarray(big.astype(np.uint8), "RGB")


def edge_keep(gw, gh, side, a=0.0, b=0.35, seed=0):
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    xn, yn = xx / max(1, gw - 1), yy / max(1, gh - 1)
    t = {"left": xn, "right": 1 - xn, "top": yn, "bottom": 1 - yn}[side]
    return smooth(a, b, t)


def mosaic_photo(entry, w, h, seed, bg, cell=40, zoom=1.0, kind="rgb", side=None, gap=0, ramp=None, sat=1.25):
    im, box = fit(entry, w, h, zoom, "th" if kind == "th" else "rgb")
    if ramp:                                   # palette-mapped mosaic (heat through a ramp)
        H = heat(im, box) if kind != "th" else stretch(1 - lum(im))
        im = Image.fromarray(apply_lut(H, ramp).astype(np.uint8), "RGB")
    elif sat != 1:
        from PIL import ImageEnhance
        im = ImageEnhance.Contrast(ImageEnhance.Color(im).enhance(sat)).enhance(1.12)
    small = pixelate(im, cell, 0.05, seed)
    gh, gw = small.shape[:2]
    side = side or ["top", "left", "right", "bottom"][seed % 4]
    keep = edge_keep(gw, gh, side, 0.0, 0.3) if side != "none" else None
    return cells_to_image(small, cell, w, h, bg, gap, keep, seed), box


# ====================================================================== stipple / halftone / dot grid (58)
def stipple(D, ink, paper, density=1.0, rmin=1.2, rmax=3.4, seed=0, color_fn=None, gamma=1.0, alpha=False):
    """Weighted random dots: darkness map D (0..1, h x w) -> dot image."""
    h, w = D.shape
    rng = np.random.default_rng(seed)
    n = int(w * h * density / 22)
    xs, ys = rng.random(n) * w, rng.random(n) * h
    v = D[ys.astype(int).clip(0, h - 1), xs.astype(int).clip(0, w - 1)] ** gamma
    keep = rng.random(n) < v
    xs, ys, v = xs[keep], ys[keep], v[keep]
    im = Image.new("RGBA" if alpha else "RGB", (w, h), (0, 0, 0, 0) if alpha else paper)
    d = ImageDraw.Draw(im)
    rr = rmin + (rmax - rmin) * v * rng.uniform(0.7, 1.15, len(v))
    for x, y, r, vv in zip(xs, ys, rr, v):
        c = color_fn(vv) if color_fn else ink
        d.ellipse((x - r, y - r, x + r, y + r), fill=c)
    return im


def halftone(D, cell=14, angle=45, ink=(20, 20, 20), paper=(240, 236, 228), alpha=False, scale=1.18):
    """Classic AM halftone of a darkness map, fully vectorised."""
    h, w = D.shape
    t = math.radians(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx * math.cos(t) + yy * math.sin(t)
    v = -xx * math.sin(t) + yy * math.cos(t)
    cu, cv = (np.floor(u / cell) + 0.5) * cell, (np.floor(v / cell) + 0.5) * cell
    cx = (cu * math.cos(t) - cv * math.sin(t)).clip(0, w - 1).astype(int)
    cy = (cu * math.sin(t) + cv * math.cos(t)).clip(0, h - 1).astype(int)
    Ds = np.asarray(Image.fromarray((D * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(cell / 2)), np.float32) / 255
    r = cell * 0.5 * np.sqrt(Ds[cy, cx]) * scale
    dot = (np.hypot(u - cu, v - cv) < r)
    if alpha:
        a = np.zeros((h, w, 4), np.uint8)
        a[dot] = (*ink, 255)
        return Image.fromarray(a, "RGBA")
    out = np.empty((h, w, 3), np.uint8)
    out[:] = paper
    out[dot] = ink
    return Image.fromarray(out, "RGB")


def dotgrid(im, cell, bg, keep=None, seed=0, minr=0.12, sat=1.3):
    """Coloured dots on a strict grid; radius follows darkness/saturation (LED / dot-matrix look)."""
    from PIL import ImageEnhance
    im = ImageEnhance.Color(im).enhance(sat)
    w, h = im.size
    gw, gh = math.ceil(w / cell), math.ceil(h / cell)
    small = np.asarray(im.resize((gw, gh), Image.BOX), np.float32)
    L = small.mean(2) / 255
    r = cell * 0.5 * (minr + (1 - minr) * np.clip(1.15 - L * 0.9, 0, 1))
    out = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(out)
    rng = np.random.default_rng(seed)
    for j in range(gh):
        for i in range(gw):
            if keep is not None and rng.random() > keep[j, i]:
                continue
            cx, cy, rr = i * cell + cell / 2, j * cell + cell / 2, r[j, i]
            d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=tuple(int(c) for c in small[j, i]))
    return out


def sparkle(d, x, y, r, fill, thin=0.18):
    """Four-point star (the 'detection' sparkle)."""
    t = r * thin
    d.polygon([(x, y - r), (x + t, y - t), (x + r, y), (x + t, y + t), (x, y + r), (x - t, y + t), (x - r, y),
               (x - t, y - t)], fill=fill)


# ====================================================================== spray paint (57)
def _spray_mask(size, draw_fn, width, seed, drips=True, speck=1.0):
    """Paint a shape with draw_fn on a mask, then turn it into a spray: soft core, overspray, drips."""
    w, h = size
    rng = np.random.default_rng(seed)
    m = Image.new("L", size, 0)
    draw_fn(ImageDraw.Draw(m))
    core = np.asarray(m.filter(ImageFilter.GaussianBlur(max(1, width * 0.18))), np.float32) / 255
    halo = np.asarray(m.filter(ImageFilter.GaussianBlur(max(2, width * 0.9))), np.float32) / 255
    noise = rng.random((h, w)).astype(np.float32)
    a = np.clip(core * 1.15, 0, 1)
    a = np.maximum(a, (noise < halo * 0.33 * speck) * rng.uniform(0.5, 1.0, (h, w)))
    out = Image.fromarray((a * 255).astype(np.uint8), "L")
    if drips:
        ys, xs = np.nonzero(np.asarray(m) > 128)
        if len(xs):
            d = ImageDraw.Draw(out)
            # drips start from the lowest painted pixel of random columns
            for _ in range(int(rng.integers(2, 6) * max(1, len(xs) / 40000))):
                k = rng.integers(0, len(xs))
                x = xs[k]
                col = ys[xs == x]
                y0 = col.max()
                L = rng.uniform(0.6, 4.5) * width
                ww = max(2, width * rng.uniform(0.18, 0.32))
                d.line((x, y0, x, y0 + L), fill=235, width=int(ww))
                d.ellipse((x - ww * 0.9, y0 + L - ww * 0.6, x + ww * 0.9, y0 + L + ww * 1.2), fill=235)
    return out


def spray(layer, draw_fn, color, width, seed=0, drips=True, speck=1.0):
    """Composite a sprayed shape onto an RGBA layer."""
    a = _spray_mask(layer.size, draw_fn, width, seed, drips, speck)
    col = Image.new("RGBA", layer.size, (*color, 255))
    col.putalpha(a)
    layer.alpha_composite(col)
    return layer


def spray_rect(layer, box, color, width=18, seed=0, drips=True):
    x0, y0, x1, y1 = box
    rng = np.random.default_rng(seed)
    j = lambda: rng.uniform(-width * 0.6, width * 0.6)
    pts = [(x0 + j(), y0 + j()), (x1 + j(), y0 + j()), (x1 + j(), y1 + j()), (x0 + j(), y1 + j()),
           (x0 + j(), y0 - width * 0.8)]
    return spray(layer, lambda d: d.line(pts, fill=255, width=int(width), joint="curve"), color, width, seed, drips)


def spray_line(layer, pts, color, width=18, seed=0, drips=False):
    return spray(layer, lambda d: d.line(pts, fill=255, width=int(width), joint="curve"), color, width, seed, drips)


def spray_text(layer, xy, text, font, color, seed=0, drips=True, width=None):
    width = width or font.size * 0.12
    return spray(layer, lambda d: d.text(xy, text, font=font, fill=255), color, width, seed, drips, 0.8)


def bezier(p0, p1, bend=0.18, n=30):
    """Quadratic curve from p0 to p1, bowed sideways by `bend` of its length."""
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2 + bend * (y1 - y0), (y0 + y1) / 2 - bend * (x1 - x0)
    return [((1 - t) ** 2 * x0 + 2 * (1 - t) * t * mx + t * t * x1,
             (1 - t) ** 2 * y0 + 2 * (1 - t) * t * my + t * t * y1) for t in np.linspace(0, 1, n)]


def arrow_head(curve, L):
    (xa, ya), (x1, y1) = curve[-3], curve[-1]
    ang = math.atan2(y1 - ya, x1 - xa)
    return [(x1 - L * math.cos(ang - 0.5), y1 - L * math.sin(ang - 0.5)), (x1, y1),
            (x1 - L * math.cos(ang + 0.5), y1 - L * math.sin(ang + 0.5))]


def spray_arrow(layer, p0, p1, color, width=18, seed=0, bend=0.18):
    curve = bezier(p0, p1, bend)
    head = arrow_head(curve, width * 3.2)

    def fn(d):
        d.line(curve, fill=255, width=int(width), joint="curve")
        d.line(head, fill=255, width=int(width), joint="curve")
    return spray(layer, fn, color, width, seed, False)


# ====================================================================== surfaces
def concrete(w, h, base=(196, 192, 186), seed=0, cracks=True):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for k, amp in ((6, 0.5), (24, 0.28), (96, 0.14), (380, 0.08)):
        n = rng.random((max(2, h * k // w), k)).astype(np.float32)
        acc += amp * np.asarray(Image.fromarray(n, "F").resize((w, h), Image.BICUBIC))
    acc = (acc - acc.mean()) / (acc.std() + 1e-6)
    fine = rng.normal(0, 1, (h, w)).astype(np.float32)
    v = acc * 4.5 + fine * 6
    rgb = np.array(base, np.float32)[None, None] + v[..., None]
    # stains
    st = np.asarray(Image.fromarray(rng.random((6, 10)).astype(np.float32), "F").resize((w, h), Image.BICUBIC))
    rgb *= (1 - 0.1 * np.clip(st - 0.55, 0, 1) * 3)[..., None]
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    for _ in range(int(w * h / 25000)):                    # pores
        x, y, r = rng.integers(0, w), rng.integers(0, h), rng.uniform(0.6, 2.4)
        d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(c * 0.72) for c in base))
    if cracks:
        for _ in range(rng.integers(1, 3)):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            ang = rng.uniform(0, 6.28)
            pts = [(x, y)]
            for _ in range(rng.integers(12, 30)):
                ang += rng.normal(0, 0.35)
                x, y = x + 14 * math.cos(ang), y + 14 * math.sin(ang)
                pts.append((x, y))
            d.line(pts, fill=tuple(int(c * 0.62) for c in base), width=2)
    return im


def paper(w, h, base=(239, 233, 221), seed=0, grid=None, fiber=5.0, mottle=6.0):
    rng = np.random.default_rng(seed)
    n = np.asarray(Image.fromarray(rng.random((max(2, h // 160), max(2, w // 160))).astype(np.float32), "F")
                   .resize((w, h), Image.BICUBIC))
    rgb = np.array(base, np.float32)[None, None] + ((n - 0.5) * mottle * 2)[..., None]
    rgb += rng.normal(0, fiber, (h, w))[..., None]
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    if grid:
        step, col = grid
        d = ImageDraw.Draw(im)
        for x in range(0, w, step):
            d.line((x, 0, x, h), fill=col, width=2)
        for y in range(0, h, step):
            d.line((0, y, w, y), fill=col, width=2)
    return im


def torn_mask(w, h, seed=0, amp=10, step=14):
    rng = np.random.default_rng(seed)
    pts = []

    def edge(p0, p1):
        n = max(2, int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step))
        nx, ny = (p1[1] - p0[1]), -(p1[0] - p0[0])
        L = math.hypot(nx, ny) or 1
        nx, ny = nx / L, ny / L
        off = 0
        for i in range(n):
            t = i / n
            off = np.clip(off + rng.normal(0, amp * 0.45), -amp, amp)
            pts.append((p0[0] + (p1[0] - p0[0]) * t + nx * (off - amp), p0[1] + (p1[1] - p0[1]) * t + ny * (off - amp)))
    edge((0, 0), (w, 0))
    edge((w, 0), (w, h))
    edge((w, h), (0, h))
    edge((0, h), (0, 0))
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return m


def print_card(im, seed=0, border=0, border_col=(250, 248, 242), torn=True, angle=0.0, shadow=True,
               shadow_col=(0, 0, 0), shadow_a=90):
    """Turn a picture into a pasted print: optional white border, torn edges, tilt and soft shadow (RGBA)."""
    if border:
        b = Image.new("RGB", (im.width + 2 * border, im.height + 2 * border), border_col)
        b.paste(im, (border, border))
        im = b
    w, h = im.size
    m = torn_mask(w, h, seed, amp=max(4, w / 160)) if torn else Image.new("L", (w, h), 255)
    rgba = im.convert("RGBA")
    rgba.putalpha(m)
    if angle:
        rgba = rgba.rotate(angle, Image.BICUBIC, expand=True)
    if not shadow:
        return rgba
    pad = int(min(w, h) * 0.05) + 6
    out = Image.new("RGBA", (rgba.width + 2 * pad, rgba.height + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("RGBA", rgba.size, (*shadow_col, 0))
    sh.putalpha(rgba.getchannel("A").point(lambda v: v * shadow_a // 255))
    out.alpha_composite(sh.filter(ImageFilter.GaussianBlur(pad * 0.45)), (int(pad * 1.25), int(pad * 1.5)))
    out.alpha_composite(rgba, (pad, pad))
    return out


def tape(w, h, seed=0, col=(232, 220, 190), a=170):
    """A strip of masking tape with jagged ends (RGBA)."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pts = [(0, 0), (w, 0)]
    pts += [(w - rng.uniform(0, h * 0.15), y) for y in np.linspace(0, h, 7)]
    pts += [(w, h), (0, h)]
    pts += [(rng.uniform(0, h * 0.15), y) for y in np.linspace(h, 0, 7)]
    ImageDraw.Draw(im).polygon(pts, fill=(*col, a))
    n = rng.normal(0, 8, (h, w))
    arr = np.asarray(im).astype(np.float32)
    arr[..., :3] += n[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


# ====================================================================== hand-drawn (59)
def scribble_ellipse(d, box, color, width=8, seed=0, loops=1.35):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    t0 = rng.uniform(0, 6.28)
    ts = np.linspace(t0, t0 + 6.28 * loops, 120)
    wob = 1 + 0.06 * np.sin(ts * 3 + rng.uniform(0, 6)) + np.linspace(0, 0.12, 120)
    pts = [(cx + rx * k * math.cos(t), cy + ry * k * math.sin(t)) for t, k in zip(ts, wob)]
    d.line(pts, fill=color, width=int(width), joint="curve")


def marker_arrow(d, p0, p1, color, width=8, bend=0.25):
    curve = bezier(p0, p1, bend, 40)
    d.line(curve, fill=color, width=int(width), joint="curve")
    d.line(arrow_head(curve, width * 4), fill=color, width=int(width), joint="curve")


def brush_stroke(w, h, color, seed=0, a=235):
    """Dry-brush stroke (RGBA): a wavy band with bristle streaks that thin out towards both ends."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / w
    ys = h / 2 + (h * 0.12) * np.sin(xn * rng.uniform(2, 5) + rng.uniform(0, 6))
    band = np.abs(yy - ys) < h * 0.38 * (0.85 + 0.15 * np.sin(xn * 9))
    bristle = np.asarray(Image.fromarray(rng.random((h, 1)).astype(np.float32), "F").resize((w, h)), np.float32)
    ends = np.minimum(smooth(0.0, 0.1, xn), smooth(0.0, 0.25, 1 - xn))       # dry at the start, drags out at the end
    al = band & (bristle < 0.15 + 0.85 * ends)
    al &= rng.random((h, w)) < 0.55 + 0.45 * ends
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = color
    out[..., 3] = (al * a).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def concentric(d, cx, cy, r, color, n=7, width=10):
    for i in range(n):
        rr = r * (1 - i / n)
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=color, width=width)


# ====================================================================== drone + footprint motif
def drone(d, cx, cy, s, fill, rotor=None, width=None):
    """Top-down quadcopter: X arms, four rotors, body, camera."""
    width = width or max(2, int(s * 0.07))
    rotor = rotor or fill
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        d.line((cx, cy, cx + dx * s * 0.62, cy + dy * s * 0.62), fill=fill, width=int(s * 0.1))
        r = s * 0.36
        x, y = cx + dx * s * 0.62, cy + dy * s * 0.62
        d.ellipse((x - r, y - r, x + r, y + r), outline=rotor, width=width)
        d.ellipse((x - s * 0.06, y - s * 0.06, x + s * 0.06, y + s * 0.06), fill=fill)
    d.rounded_rectangle((cx - s * 0.24, cy - s * 0.3, cx + s * 0.24, cy + s * 0.3), int(s * 0.1), fill=fill)


# ====================================================================== confidence traces
def trace(seed):
    """A real test sequence with visible flicker (fused confidence, its moving average, ground truth)."""
    t = [x for x in traces() if sum(x["gt"]) > 40 and x["raw_fa"] > 0] or traces()
    t.sort(key=lambda x: -x["flicker"])
    return t[seed % len(t)]
