"""Ten looks, each built from the paper's own material.

The paper (MMSAR): a drone's person detector flickers; RGB + pseudo-thermal are fused; causal filters decide
when a flicker becomes an alert. Every style below shows that story in its own medium:
  * image slots are real 640x640 UAV tiles (desert / forest / snow), RGB or pseudo-thermal
  * the person is always marked - box, sprayed tag, marker circle, red stipple, sparkle
  * empty tiles carry a crossed-out FALSE ALERT (the thing the filters remove)
  * wide strips are filmstrips of consecutive frames, or the real per-frame confidence trace
"""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

import art
from engine import W, H, M, fnt, tlen

ASSETS = art.ASSETS


def rgba(w, h):
    return Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))


def fit_into(im, w, h):
    """Shrink an image (keeping aspect) so it fits inside w x h."""
    if im.width <= w and im.height <= h:
        return im
    k = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)


def fade_alpha(im, side="left", a=0.0, b=0.4):
    """Alpha ramp so a picture melts into the surface on one side."""
    im = im.convert("RGBA")
    w, h = im.size
    t = np.linspace(0, 1, w if side in ("left", "right") else h)
    if side in ("right", "bottom"):
        t = t[::-1]
    ramp = (art.smooth(a, b, t) * 255).astype(np.uint8)
    al = np.broadcast_to(ramp[None, :] if side in ("left", "right") else ramp[:, None], (h, w))
    im.putalpha(Image.fromarray(np.minimum(np.asarray(im.getchannel("A")), al).astype(np.uint8)))
    return im


@lru_cache(None)
def render_uav():
    return Image.open(ASSETS / "render_uav.png").convert("RGB")


def trace_xy(tr, w, h, pad=0.06, top=0.08, bottom=0.30):
    """Map a confidence trace to pixel coords; returns xs, raw ys, filtered ys, gt, alert masks, plot box."""
    f = np.asarray(tr["fused"])
    ma = np.asarray(tr["ma"])
    gt = np.asarray(tr["gt"])
    n = len(f)
    x0, x1 = w * pad, w * (1 - pad)
    y0, y1 = h * top, h * (1 - bottom)
    xs = x0 + np.arange(n) / (n - 1) * (x1 - x0)
    Y = lambda v: y1 - v * (y1 - y0)
    return dict(xs=xs, raw=Y(f), ma=Y(ma), gt=gt, raw_alert=f >= 0.43, ma_alert=ma >= 0.57, tau=Y(0.43),
                box=(x0, y0, x1, y1), n=n, Y=Y)


# ====================================================================== base style
class Style:
    key = name = folder = pptx = ""
    source = ""
    bg = (13, 10, 9)
    fg = (246, 236, 222)
    mute = (150, 132, 120)
    line = (66, 50, 44)
    card_c = (26, 20, 18)
    acc = (255, 106, 30)
    acc2 = (214, 40, 22)
    acc3 = (255, 200, 80)
    on_acc = (22, 10, 6)
    fonts = {}
    track = {}
    scale = {}
    radius = 0
    bullet = "■"
    mono_upper = True
    png = False
    zoom = 1.25
    shape = "square"
    label_neg = "False alert"
    label_pos = "Person"

    # surfaces -------------------------------------------------------
    @property
    def card(self):
        return self._card

    def base(self, n):
        return self._base()

    @lru_cache(None)
    def _base(self):
        return Image.new("RGB", (W, H), self.bg)

    # chrome ---------------------------------------------------------
    def chrome(self, s, section):
        s.mono(M, 84, "Project name  /  Team", 30)
        s.mono(W - M, 84, f"{s.n:02d} / 40", 30, anchor="ra")
        yb = H - 128
        self.footer(s, yb)
        s.mono(M, yb + 40, section or "Section title", 28)
        s.mono(W - M, yb + 40, "Organisation  /  Term", 28, anchor="ra")

    def footer(self, s, yb):
        s.rule(M, yb, W - M, "line", 3)

    def title_deco(self, s, x, y, w, size):
        pass

    def divider_deco(self, s, x, y):
        pass

    # cards / tags / bars --------------------------------------------
    def _card(self, s, box, hi):
        x0, y0, x1, y1 = box
        f = self.acc if hi else self.card_c
        if self.shape == "round":
            s.d.rounded_rectangle(box, self.radius, fill=f, outline=self.acc if hi else self.line, width=3)
        elif self.shape == "shadow":
            s.d.rectangle((x0 + 16, y0 + 16, x1 + 16, y1 + 16), fill=self.fg)
            s.d.rectangle(box, fill=f, outline=self.fg, width=5)
        else:
            s.d.rectangle(box, fill=f, outline=self.acc if hi else self.line, width=3)
            s.d.rectangle((x0, y0, x0 + 24, y0 + 24), fill=self.on_acc if hi else self.acc)

    def tag(self, s, x, y, txt, size=40, hi=True):
        fo = fnt(self, "M", size)
        txt = txt.upper() if self.mono_upper else txt
        tw = tlen(txt, fo, self.track.get("M", 0), size)
        px, ph = int(size * 0.55), int(size * 1.7)
        box = (x, y, x + tw + 2 * px, y + ph)
        fill = self.acc if hi else self.fg
        if self.shape == "round":
            s.d.rounded_rectangle(box, ph // 2, fill=fill)
        else:
            s.d.rectangle(box, fill=fill)
        bb = fo.getbbox("H")
        s.text(x + px, y + (ph - (bb[3] - bb[1])) / 2 - bb[1], txt, size, "M", self.on_acc if hi else self.bg)
        return box[2]

    def bar(self, s, x, y, w, h, frac, seed, label):
        bw = max(8, int(w * frac))
        s.d.rectangle((x, y, x + w, y + h), fill=self.card_c, outline=self.line, width=2)
        s.d.rectangle((x, y, x + bw, y + h), fill=self.acc)
        self.bar_label(s, x, y, w, h, bw, frac, label)

    def bar_label(self, s, x, y, w, h, bw, frac, label):
        if not label:
            return
        if frac > 0.72:
            s.mono(x + bw - 30, y + (h - 28) // 2, label, 28, self.on_acc, anchor="ra")
        else:
            s.mono(x + bw + 30, y + (h - 28) // 2, label, 28)

    # pictures -------------------------------------------------------
    def inner(self, w, h):
        return w, h

    def photo(self, s, w, h, seed, kind, label, biome=None, zoom=None):
        cw, ch = self.inner(w, h)
        if kind == "strip":
            im = self.strip(cw, ch, seed)
            if im.mode == "RGBA":                   # separate prints: keep the gaps transparent, no outer frame
                out = rgba(w, h)
                out.alpha_composite(im, ((w - cw) // 2, (h - ch) // 2))
                return out
        elif kind == "trace":
            im = self.trace_art(cw, ch, seed)
        elif kind == "scene":
            e = dict(id="scene", box=[520, 330, 1180, 900])      # drone + footprint region of the render
            im, box = art.fit(e, cw, ch, zoom or 1.0, src=render_uav())
            im = self.treat(im, None, seed, "scene")
        else:
            e = art.pick(seed, "neg" if kind == "neg" else "pos", biome)
            src = "th" if kind == "th" and self.real_thermal else "rgb"
            im, box = art.fit(e, cw, ch, zoom or self.zoom, src)
            im = self.treat(im, box, seed, kind)
            if kind == "neg":
                im = self.mark_neg(im, seed)
            elif box:
                im = self.mark(im, box, seed)
        return self.frame(im, w, h, seed, label)

    real_thermal = False

    def treat(self, im, box, seed, kind):
        return im

    def mark(self, im, box, seed):
        d = ImageDraw.Draw(im)
        k = max(3, int(min(im.size) * 0.004))
        x0, y0, x1, y1 = self._pad(box, im)
        d.rectangle((x0, y0, x1, y1), outline=self.fg, width=k)
        self._chip(im, x0, y0 - 8, self.label_pos, self.fg, self.bg, above=True)
        return im

    def mark_neg(self, im, seed):
        rng = np.random.default_rng(seed)
        w, h = im.size
        bw, bh = w * rng.uniform(0.12, 0.2), h * rng.uniform(0.2, 0.32)
        x0, y0 = w * rng.uniform(0.35, 0.6), h * rng.uniform(0.25, 0.5)
        d = ImageDraw.Draw(im)
        k = max(3, int(min(w, h) * 0.004))
        for i in range(0, int(bw), 24):                                   # dashed box
            d.line((x0 + i, y0, x0 + min(bw, i + 12), y0), fill=self.acc2, width=k)
            d.line((x0 + i, y0 + bh, x0 + min(bw, i + 12), y0 + bh), fill=self.acc2, width=k)
        for i in range(0, int(bh), 24):
            d.line((x0, y0 + i, x0, y0 + min(bh, i + 12)), fill=self.acc2, width=k)
            d.line((x0 + bw, y0 + i, x0 + bw, y0 + min(bh, i + 12)), fill=self.acc2, width=k)
        d.line((x0, y0, x0 + bw, y0 + bh), fill=self.acc2, width=k)
        d.line((x0 + bw, y0, x0, y0 + bh), fill=self.acc2, width=k)
        self._chip(im, x0, y0 - 8, self.label_neg, self.acc2, self.bg, above=True)
        return im

    def _pad(self, box, im, k=0.18):
        x0, y0, x1, y1 = box
        px, py = (x1 - x0) * k + 6, (y1 - y0) * k * 0.6 + 6
        return max(4, x0 - px), max(4, y0 - py), min(im.width - 4, x1 + px), min(im.height - 4, y1 + py)

    def _chip(self, im, x, y, txt, bgc, fgc, above=False, size=None):
        size = size or max(22, int(min(im.size) * 0.028))
        size = min(size, 34)
        fo = fnt(self, "M", size)
        txt = txt.upper() if self.mono_upper else txt
        tw = fo.getlength(txt)
        ph = int(size * 1.6)
        if above:
            y = y - ph
        x = min(max(4, x), im.width - tw - size - 4)
        y = min(max(4, y), im.height - ph - 4)
        d = ImageDraw.Draw(im)
        box = (x, y, x + tw + size, y + ph)
        if self.shape == "round":
            d.rounded_rectangle(box, ph // 2, fill=bgc)
        else:
            d.rectangle(box, fill=bgc)
        bb = fo.getbbox("H")
        d.text((x + size / 2, y + (ph - (bb[3] - bb[1])) / 2 - bb[1]), txt, font=fo, fill=fgc)

    def frame(self, im, w, h, seed, label):
        """Final touch on an image slot: the small slot label in the corner."""
        im = im.convert("RGB")
        fo = fnt(self, "M", 26)
        s = f"[ {label.upper()} ]"
        tw = fo.getlength(s)
        d = ImageDraw.Draw(im)
        bx, by = 28, im.height - 28 - 54
        box = (bx, by, bx + tw + 36, by + 54)
        if self.shape == "round":
            d.rounded_rectangle(box, 27, fill=(12, 9, 8))
            d.text((bx + 18, by + 13), s, font=fo, fill=self.fg)
        elif self.bg[0] > 128:
            d.rectangle(box, fill=self.fg)
            d.text((bx + 18, by + 13), s, font=fo, fill=self.bg)
        else:
            d.rectangle(box, fill=(10, 8, 8))
            d.text((bx + 18, by + 13), s, font=fo, fill=self.fg)
        return im

    # filmstrip + trace ------------------------------------------------
    def strip(self, w, h, seed):
        st = art.strips()[seed % 3]
        n = int(min(8, max(3, round(w / h))))
        gap = max(10, int(h * 0.04))
        tw = (w - gap * (n - 1)) / n
        out = Image.new("RGB", (w, h), self.bg)
        frames = st["frames"][:: max(1, 8 // n)][:n]
        for i, fr in enumerate(frames):
            e = dict(id=fr["id"], box=fr["box"])
            im, box = art.fit(e, int(tw), h, 1.15, "rgb")
            im = self.treat(im, box, seed + i, "rgb")
            im = self.mark(im, box, seed + i) if box else im
            out.paste(im.convert("RGB"), (int(i * (tw + gap)), 0))
        return out

    def trace_art(self, w, h, seed):
        """The paper's core picture: raw confidence flickers across the threshold, the filter holds steady."""
        tr = art.trace(seed)
        P = trace_xy(tr, w, h)
        im = Image.new("RGB", (w, h), self.card_c if self.card_c != self.bg else self.bg)
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = P["box"]
        xs = P["xs"]
        step = xs[1] - xs[0]
        for i, g in enumerate(P["gt"]):                                    # person in view
            if g:
                d.rectangle((xs[i] - step / 2, y0, xs[i] + step / 2, y1), fill=self._mix(0.12))
        d.line([(x0, P["tau"]), (x1, P["tau"])], fill=self.mute, width=2)
        d.line(list(zip(xs, P["raw"])), fill=self.acc, width=max(3, int(h * 0.008)))
        d.line(list(zip(xs, P["ma"])), fill=self.fg, width=max(4, int(h * 0.012)))
        self._ticks(d, P, h)
        return im

    def _mix(self, t, c=None):
        c = c or self.fg
        base = self.card_c if self.card_c != self.bg else self.bg
        return tuple(int(b + (cc - b) * t) for b, cc in zip(base, c))

    def _ticks(self, d, P, h):
        """Two rows of alert ticks under the plot: raw (top) and filtered (bottom); false ones in acc2."""
        xs, step = P["xs"], P["xs"][1] - P["xs"][0]
        r0, r1 = h * 0.76, h * 0.86
        th = h * 0.065
        for row, al in ((r0, P["raw_alert"]), (r1, P["ma_alert"])):
            for i in np.nonzero(al)[0]:
                col = self.fg if P["gt"][i] else self.acc2
                d.rectangle((xs[i] - step * 0.4, row, xs[i] + step * 0.4, row + th), fill=col)

    # cover / thanks ----------------------------------------------------
    cover_biome = "snow"

    def cover(self, s):
        """Default: a full-height hero tile on the right that melts into the surface."""
        w = int(W * 0.6)
        im = self.photo(s, w, H, 2, "rgb", "Cover image", self.cover_biome, 1.05)
        s.pic(fade_alpha(im, "left", 0.0, 0.45), W - w, 0, 0, "Image - Cover")

    def thanks(self, s):
        w = int(W * 0.55)
        im = self.photo(s, w, H, 5, "th" if self.real_thermal else "rgb", "Closing image", "desert", 1.05)
        s.pic(fade_alpha(im, "left", 0.0, 0.45), W - w, 0, 0, "Image - Closing")


# ====================================================================== 55  film burn
class Burn(Style):
    key, name, folder, pptx, source = "burn", "Film Burn", "film_burn_40", "FilmBurn_Template_40.pptx", "55"
    fonts = dict(H="InstrumentSerif-Regular.ttf", HI="InstrumentSerif-Italic.ttf", D="InstrumentSerif-Regular.ttf",
                 B="Inter-Regular.ttf", BB="Inter-SemiBold.ttf", M="JetBrainsMono-Medium.ttf")
    track = dict(H=-0.01, HI=-0.01, D=-0.02, BB=-0.01, M=0.02)
    scale = dict(H=1.18, HI=1.18)
    shape, radius, bullet = "round", 34, "●"

    def footer(self, s, yb):
        I = np.linspace(0.42, 1, W - 2 * M)[None, :].repeat(6, 0)
        s.g.paste(Image.fromarray(art.apply_lut(I, "fire").astype(np.uint8), "RGB"), (M, yb))

    def treat(self, im, box, seed, kind):
        rng = np.random.default_rng(seed)
        w, h = im.size
        if kind == "th":
            base = art.apply_lut(art.heat(im, box) * 0.97, "fire")
        else:
            L = art.stretch(art.lum(im), 1, 99.5, 1.15)
            ramp = ["fire", "fire", "rose", "amber"][seed % 4]
            base = art.apply_lut(0.16 + 0.8 * L, ramp) * 0.72 + np.asarray(im, np.float32) * \
                np.array([1.0, 0.7, 0.52]) * 0.32
        leak = art.leak_field(w, h, seed + 5, ["leak", "streak", "flare"][seed % 3])
        lk = art.apply_lut(leak, "fire") / 255
        out = 255 - (255 - base) * (1 - lk * 0.8)
        yy, xx = np.mgrid[0:h, 0:w]
        out *= (1 - 0.35 * ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2)[..., None]
        out = art.grain(out, rng, 9)
        return art.dust(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), rng, int(w * h / 150000))

    def mark(self, im, box, seed):
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = self._pad(box, im)
        k = max(3, int(min(im.size) * 0.0035))
        d.rounded_rectangle((x0, y0, x1, y1), 10, outline=(255, 244, 228), width=k)
        self._chip(im, x0, y0 - 10, self.label_pos, (255, 244, 228), (30, 12, 6), above=True)
        return im

    def bar(self, s, x, y, w, h, frac, seed, label):
        bw = max(8, int(w * frac))
        s.d.rectangle((x, y, x + w, y + h), fill=self.card_c, outline=self.line, width=2)
        I = np.linspace(0.42, 0.42 + 0.58 * frac, bw)[None, :].repeat(h, 0)
        s.g.paste(Image.fromarray(art.apply_lut(I, "fire").astype(np.uint8), "RGB"), (x, y))
        self.bar_label(s, x, y, w, h, bw, frac, label)

    def trace_art(self, w, h, seed):
        tr = art.trace(seed)
        P = trace_xy(tr, w, h)
        glow = Image.new("RGB", (w, h), (0, 0, 0))
        d = ImageDraw.Draw(glow)
        d.line(list(zip(P["xs"], P["raw"])), fill=(255, 110, 30), width=max(4, int(h * 0.012)))
        g = np.asarray(glow.filter(ImageFilter.GaussianBlur(h * 0.03)), np.float32) * 1.8 + np.asarray(glow, np.float32)
        base = np.asarray(art.tex_burn(w, h, seed + 40, "leak", None, "fire", False), np.float32) * 0.35
        im = Image.fromarray(np.clip(base + g, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = P["box"]
        for i in range(int(x0), int(x1), 26):
            d.line((i, P["tau"], i + 12, P["tau"]), fill=(200, 170, 150), width=2)
        d.line(list(zip(P["xs"], P["ma"])), fill=(255, 244, 228), width=max(4, int(h * 0.012)))
        self._ticks(d, P, h)
        return art.dust(im, np.random.default_rng(seed), int(w * h / 120000))


# ====================================================================== 56  pixel mosaic
class Pixel(Style):
    key, name, folder, pptx, source = "pixel", "Pixel Mosaic", "pixel_mosaic_40", "PixelMosaic_Template_40.pptx", "56"
    bg, fg, mute, line = (243, 241, 236), (16, 16, 22), (112, 110, 106), (206, 202, 194)
    card_c, acc, acc2, on_acc = (255, 255, 255), (36, 80, 255), (255, 40, 130), (255, 255, 255)
    fonts = dict(H="SpaceGrotesk-Bold.ttf", HI="SpaceGrotesk-Regular.ttf", D="Silkscreen-Regular.ttf",
                 B="SpaceGrotesk-Regular.ttf", BB="SpaceGrotesk-Bold.ttf", M="SpaceMono-Regular.ttf")
    track = dict(H=-0.03, HI=-0.03, D=-0.12, BB=-0.01)
    shape, png, real_thermal = "shadow", True, True
    cell = 40
    divider_full = True

    def footer(self, s, yb):
        c, n = 16, (W - 2 * M) // 16
        rng = np.random.default_rng(s.n)
        cols = art.apply_lut(np.linspace(0.0, 1, n), ["rainbow", "blue", "pink", "sunset"][s.n % 4])
        for i in range(n):
            if rng.random() < (i / n) ** 0.6 + 0.08:
                x = M + i * c
                s.d.rectangle((x + 1, yb - 4, x + c - 2, yb + c - 7), fill=tuple(int(v) for v in cols[i]))

    def treat(self, im, box, seed, kind):
        w, h = im.size
        side = ["top", "left", "right", "bottom"][seed % 4]
        from PIL import ImageEnhance
        im = ImageEnhance.Contrast(ImageEnhance.Color(im).enhance(1.3)).enhance(1.12)
        small = art.pixelate(im, self.cell, 0.05, seed)
        gh, gw = small.shape[:2]
        keep = art.edge_keep(gw, gh, side, 0.0, 0.28)
        return art.cells_to_image(small, self.cell, w, h, self.bg, 0, keep, seed)

    def mark(self, im, box, seed):
        c = self.cell
        x0, y0, x1, y1 = self._pad(box, im, 0.1)
        x0, y0 = int(x0 // c) * c, int(y0 // c) * c
        x1, y1 = int(math.ceil(x1 / c)) * c, int(math.ceil(y1 / c)) * c
        d = ImageDraw.Draw(im)
        d.rectangle((x0, y0, x1, y1), outline=self.acc, width=10)
        self._chip(im, x0, y0, self.label_pos, self.acc, (255, 255, 255), above=True)
        return im

    def bar(self, s, x, y, w, h, frac, seed, label):
        bw = max(20, int(w * frac))
        s.d.rectangle((x, y, x + w, y + h), fill=self.card_c, outline=self.line, width=2)
        cell = 20
        gw, rows = max(1, bw // cell), max(1, h // cell)
        I = np.linspace(0.25, 0.25 + 0.75 * frac, gw)[None, :].repeat(rows, 0)
        arr = art.apply_lut(I, ["rainbow", "blue", "pink", "violet", "teal", "sunset"][seed % 6])
        arr *= (1 + np.random.default_rng(seed).normal(0, 0.06, (rows, gw)))[..., None]
        big = np.repeat(np.repeat(np.clip(arr, 0, 255), cell, 0), cell, 1)
        s.g.paste(Image.fromarray(big.astype(np.uint8), "RGB").resize((gw * cell, h), Image.NEAREST), (x, y))
        self.bar_label(s, x, y, w, h, gw * cell, frac, label)

    def trace_art(self, w, h, seed):
        """Confidence as columns of cells - one column per few frames, coloured by height."""
        tr = art.trace(seed)
        P = trace_xy(tr, w, h, top=0.06, bottom=0.3)
        im = Image.new("RGB", (w, h), self.bg)
        d = ImageDraw.Draw(im)
        c = max(12, int(h * 0.05))
        x0, y0, x1, y1 = P["box"]
        ncol = int((x1 - x0) // c)
        f = np.asarray(tr["fused"])
        lutc = art.lut("rainbow")
        for j in range(ncol):
            a, b = int(j / ncol * len(f)), int((j + 1) / ncol * len(f))
            v = f[a:b].max() if b > a else 0
            k = int(round(v * (y1 - y0) / c))
            for r in range(k):
                col = tuple(int(q) for q in lutc[int(min(255, (r * c) / (y1 - y0) * 255))])
                d.rectangle((x0 + j * c + 1, y1 - (r + 1) * c + 1, x0 + (j + 1) * c - 2, y1 - r * c - 2), fill=col)
        d.line([(x0, P["tau"]), (x1, P["tau"])], fill=self.fg, width=3)
        self._ticks(d, P, h)
        return im

    def cover(self, s):
        w = int(W * 0.62)
        e = art.pick(2, "pos", "snow")
        im, box = art.fit(e, w, H, 1.05, "rgb")
        small = art.pixelate(ImageEnhance.Contrast(ImageEnhance.Color(im).enhance(1.3)).enhance(1.12), self.cell, .05, 3)
        gh, gw = small.shape[:2]
        keep = art.edge_keep(gw, gh, "left", 0.0, 0.55)
        im = self.mark(art.cells_to_image(small, self.cell, w, H, self.bg, 0, keep, 3), box, 3)
        s.pic(self.frame(im, w, H, 3, "Cover image"), W - w, 0, 0, "Image - Cover")

    def thanks(self, s):
        w = int(W * 0.55)
        e = art.pick(5, "pos", "forest")
        im, box = art.fit(e, w, H, 1.05, "th")
        small = art.pixelate(im, self.cell, .05, 9)
        gh, gw = small.shape[:2]
        im = art.cells_to_image(small, self.cell, w, H, self.bg, 0, art.edge_keep(gw, gh, "left", 0.0, 0.55), 9)
        s.pic(self.frame(self.mark(im, box, 9), w, H, 9, "Closing image"), W - w, 0, 0, "Image - Closing")


# ====================================================================== 55 x 56  pixel burn
class PixelBurn(Pixel):
    key, name, folder, pptx, source = "pburn", "Pixel Burn", "pixel_burn_40", "PixelBurn_Template_40.pptx", "55 + 56"
    bg, fg, mute, line = (11, 9, 9), (250, 238, 224), (150, 130, 118), (64, 46, 40)
    card_c, acc, acc2, on_acc = (24, 17, 15), (255, 98, 26), (230, 40, 70), (20, 8, 4)
    fonts = dict(H="InstrumentSerif-Regular.ttf", HI="InstrumentSerif-Italic.ttf", D="Silkscreen-Regular.ttf",
                 B="Inter-Regular.ttf", BB="Inter-SemiBold.ttf", M="Silkscreen-Regular.ttf")
    track = dict(H=-0.01, HI=-0.01, D=-0.12, BB=-0.01, M=0.02)
    scale = dict(H=1.18, HI=1.18)
    shape, png, real_thermal = "square", False, False
    divider_full = False

    def footer(self, s, yb):
        c, n = 16, (W - 2 * M) // 16
        rng = np.random.default_rng(s.n)
        cols = art.apply_lut(np.linspace(0.35, 1, n), "fire")
        for i in range(n):
            if rng.random() < (i / n) ** 0.6 + 0.08:
                s.d.rectangle((M + i * c + 1, yb - 4, M + i * c + c - 2, yb + c - 7), fill=tuple(int(v) for v in cols[i]))

    def _heat_cells(self, im, box, seed, side, a=0.0, b=0.3):
        w, h = im.size
        Hm = art.heat(im, box)
        hot = Image.fromarray(art.apply_lut(Hm, "fire").astype(np.uint8), "RGB")
        small = art.pixelate(hot, self.cell, 0.06, seed)
        gh, gw = small.shape[:2]
        Hs = np.asarray(Image.fromarray((Hm * 255).astype(np.uint8)).resize((gw, gh), Image.BOX), np.float32) / 255
        keep = np.clip(Hs * 2.2 + 0.05, 0, 1)
        if side:
            keep = keep * art.edge_keep(gw, gh, side, a, b)
        return art.cells_to_image(small, self.cell, w, h, self.bg, 4, keep, seed)

    def treat(self, im, box, seed, kind):
        return self._heat_cells(im, box, seed, None)

    def mark(self, im, box, seed):
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = self._pad(box, im, 0.12)
        L, k = max(24, (x1 - x0) * 0.3), 8
        col = (255, 244, 228)
        for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            # an L-shaped corner bracket: one horizontal and one vertical bar
            d.rectangle((min(cx, cx + sx * L), min(cy, cy + sy * k), max(cx, cx + sx * L), max(cy, cy + sy * k)), fill=col)
            d.rectangle((min(cx, cx + sx * k), min(cy, cy + sy * L), max(cx, cx + sx * k), max(cy, cy + sy * L)), fill=col)
        self._chip(im, x0, y0 - 6, self.label_pos, col, (20, 8, 4), above=True)
        return im

    def bar(self, s, x, y, w, h, frac, seed, label):
        bw = max(20, int(w * frac))
        s.d.rectangle((x, y, x + w, y + h), fill=self.card_c, outline=self.line, width=2)
        cell = 20
        gw, rows = max(1, bw // cell), max(1, h // cell)
        I = np.linspace(0.42, 0.42 + 0.58 * frac, gw)[None, :].repeat(rows, 0)
        arr = art.apply_lut(I, "fire") * (1 + np.random.default_rng(seed).normal(0, 0.06, (rows, gw)))[..., None]
        big = np.repeat(np.repeat(np.clip(arr, 0, 255), cell, 0), cell, 1)
        s.g.paste(Image.fromarray(big.astype(np.uint8), "RGB").resize((gw * cell, h), Image.NEAREST), (x, y))
        self.bar_label(s, x, y, w, h, gw * cell, frac, label)

    def trace_art(self, w, h, seed):
        tr = art.trace(seed)
        P = trace_xy(tr, w, h, top=0.06, bottom=0.3)
        im = Image.new("RGB", (w, h), self.bg)
        d = ImageDraw.Draw(im)
        c = max(12, int(h * 0.05))
        x0, y0, x1, y1 = P["box"]
        ncol = int((x1 - x0) // c)
        f = np.asarray(tr["fused"])
        for j in range(ncol):
            a, b = int(j / ncol * len(f)), int((j + 1) / ncol * len(f))
            v = f[a:b].max() if b > a else 0
            for r in range(int(round(v * (y1 - y0) / c))):
                col = tuple(int(q) for q in art.lut("fire")[int(min(255, 90 + r * c / (y1 - y0) * 165))])
                d.rectangle((x0 + j * c + 2, y1 - (r + 1) * c + 2, x0 + (j + 1) * c - 2, y1 - r * c - 2), fill=col)
        d.line([(x0, P["tau"]), (x1, P["tau"])], fill=self.fg, width=3)
        self._ticks(d, P, h)
        return im

    def cover(self, s):
        w = int(W * 0.62)
        e = art.pick(2, "pos", "snow")
        im, box = art.fit(e, w, H, 1.05, "rgb")
        im = self.mark(self._heat_cells(im, box, 3, "left", 0.0, 0.5), box, 3)
        s.pic(self.frame(im, w, H, 3, "Cover image"), W - w, 0, 0, "Image - Cover")

    def thanks(self, s):
        w = int(W * 0.55)
        e = art.pick(5, "pos", "desert")
        im, box = art.fit(e, w, H, 1.05, "rgb")
        im = self.mark(self._heat_cells(im, box, 9, "left", 0.0, 0.5), box, 9)
        s.pic(self.frame(im, w, H, 9, "Closing image"), W - w, 0, 0, "Image - Closing")


# ====================================================================== 57  graffiti
class Graffiti(Style):
    key, name, folder, pptx, source = "graffiti", "Graffiti", "graffiti_40", "Graffiti_Template_40.pptx", "57"
    bg, fg, mute, line = (198, 194, 188), (16, 16, 16), (62, 60, 56), (120, 116, 110)
    card_c, acc, acc2, acc3, on_acc = (244, 242, 236), (255, 72, 20), (164, 255, 0), (255, 40, 150), (16, 16, 16)
    fonts = dict(H="Anton-Regular.ttf", HI="PermanentMarker-Regular.ttf", D="PermanentMarker-Regular.ttf",
                 B="Inter-Regular.ttf", BB="Inter-SemiBold.ttf", M="SpaceMono-Bold.ttf", HW="PermanentMarker-Regular.ttf")
    track = dict(H=0.0, D=-0.02)
    scale = dict(H=1.12, HI=0.92, D=1.05)
    shape = "poster"
    num_col = "fg"
    label_pos, label_neg = "PERSON!", "FALSE ALERT"
    wall = (198, 194, 188)

    @lru_cache(None)
    def _base(self):
        return art.concrete(W, H, self.wall, 7)

    def footer(self, s, yb):
        L = rgba(W, 60)
        art.spray_line(L, [(M, 30), (W * 0.45, 26), (W - M, 32)], self.fg, 9, s.n, False)
        s.g.paste(L, (0, yb - 30), L)

    def title_deco(self, s, x, y, w, size):
        L = rgba(w + 120, size)
        yy = size * 0.72
        art.spray_line(L, [(20, yy + 6), (w * 0.5, yy - 8), (w + 80, yy + 4)], self.acc2, size * 0.34, s.n, False)
        s.g.paste(L, (int(x - 40), int(y)), L)

    def divider_deco(self, s, x, y):
        L = rgba(1500, 1100)
        art.spray(L, lambda d: d.ellipse((120, 140, 1100, 1000), fill=255), self.acc3, 120, s.n, True, 1.4)
        s.g.paste(L, (int(x - 120), int(y - 80)), L)

    def _card(self, s, box, hi):
        x0, y0, x1, y1 = [int(v) for v in box]
        w, h = x1 - x0, y1 - y0
        col = self.acc if hi else self.card_c
        paperim = art.paper(w, h, col, s.n + x0, fiber=3, mottle=4)
        pc = art.print_card(paperim, seed=x0 + y0, torn=True, shadow=True, shadow_a=110)
        pad = (pc.width - w) // 2
        s.g.paste(pc, (x0 - pad, y0 - pad), pc)
        tp = art.tape(int(min(260, w * 0.4)), 64, x0)
        tp = tp.rotate(-4, Image.BICUBIC, expand=True)
        s.overlay(tp, x0 + w / 2 - tp.width / 2, y0 - 34)

    def tag(self, s, x, y, txt, size=40, hi=True):
        fo = fnt(self, "M", size)
        txt = txt.upper()
        tw = fo.getlength(txt)
        px, ph = int(size * 0.6), int(size * 1.8)
        L = rgba(tw + 2 * px + 40, ph + 40)
        art.spray(L, lambda d: d.rectangle((20, 20, 20 + tw + 2 * px, 20 + ph), fill=255),
                  self.acc if hi else self.fg, 14, x, False, 0.8)
        s.g.paste(L, (int(x - 20), int(y - 20)), L)
        bb = fo.getbbox("H")
        s.text(x + px, y + (ph - (bb[3] - bb[1])) / 2 - bb[1], txt, size, "M", self.on_acc if hi else self.bg)
        return x + tw + 2 * px

    def bar(self, s, x, y, w, h, frac, seed, label):
        bw = max(30, int(w * frac))
        s.d.rectangle((x, y, x + w, y + h), outline=self.fg, width=3)
        L = rgba(bw + 60, h + 60)
        art.spray(L, lambda d: d.rectangle((30, 30 + 6, 30 + bw, 30 + h - 6), fill=255),
                  [self.acc, self.acc3, self.acc2][seed % 3], 22, seed, frac > 0.3, 0.9)
        s.g.paste(L, (x - 30, y - 30), L)
        if label:
            if frac > 0.72:
                s.mono(x + bw - 30, y + (h - 28) // 2, label, 28, self.on_acc, anchor="ra")
            else:
                s.mono(x + bw + 40, y + (h - 28) // 2, label, 28, "fg")

    def inner(self, w, h):
        p = int(min(w, h) * 0.05)
        return w - 2 * p, h - 2 * p

    def treat(self, im, box, seed, kind):
        """Photocopied poster: high-contrast B&W, a touch of grain."""
        g = ImageOps.autocontrast(im.convert("L"), cutoff=2)
        g = ImageEnhance.Contrast(g).enhance(1.45)
        arr = np.asarray(g, np.float32)
        arr = art.grain(np.stack([arr] * 3, -1), np.random.default_rng(seed), 10)
        if kind == "th":
            Hm = art.heat(im, box)
            hot = np.array(self.acc3, np.float32)
            arr = arr * (1 - Hm[..., None] * 0.75) + hot * Hm[..., None] * 0.75
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def mark(self, im, box, seed):
        L = im.convert("RGBA")
        x0, y0, x1, y1 = self._pad(box, im, 0.25)
        wd = max(10, min(im.size) * 0.02)
        art.spray_rect(L, (x0, y0, x1, y1), self.acc2, wd, seed, True)
        fo = fnt(self, "HW", int(max(40, min(im.size) * 0.075)))
        tx = min(x1 + wd, im.width - fo.getlength(self.label_pos) - 20)
        ty = max(10, y0 - fo.size * 1.2)
        art.spray_text(L, (tx, ty), self.label_pos, fo, self.acc2, seed + 1, True, fo.size * 0.06)
        return L.convert("RGB")

    def mark_neg(self, im, seed):
        L = im.convert("RGBA")
        w, h = im.size
        rng = np.random.default_rng(seed)
        cx, cy, r = w * rng.uniform(0.4, 0.6), h * rng.uniform(0.4, 0.6), min(w, h) * 0.18
        wd = max(12, min(w, h) * 0.025)
        art.spray_line(L, [(cx - r, cy - r), (cx + r, cy + r)], self.acc3, wd, seed, True)
        art.spray_line(L, [(cx + r, cy - r), (cx - r, cy + r)], self.acc3, wd, seed + 1, True)
        fo = fnt(self, "HW", int(max(40, min(w, h) * 0.07)))
        art.spray_text(L, (max(10, cx - fo.getlength(self.label_neg) / 2), cy + r + 10), self.label_neg, fo,
                       self.acc3, seed + 2, False, fo.size * 0.05)
        return L.convert("RGB")

    def frame(self, im, w, h, seed, label):
        """Wheat-pasted print: torn edges, shadow, two strips of tape."""
        im = Style.frame(self, im, w, h, seed, label)
        pc = fit_into(art.print_card(im, seed=seed, torn=True, shadow=True, shadow_a=120), w, h)
        out = rgba(w, h)
        out.alpha_composite(pc, ((w - pc.width) // 2, (h - pc.height) // 2))
        for i, xx in enumerate((0.12, 0.78)):
            tp = art.tape(int(w * 0.16), int(max(40, h * 0.05)), seed + i).rotate([-8, 6][i], Image.BICUBIC, expand=True)
            out.alpha_composite(tp, (int(w * xx), 0))
        return out

    def trace_art(self, w, h, seed):
        tr = art.trace(seed)
        P = trace_xy(tr, w, h)
        im = art.paper(w, h, (244, 242, 236), seed, fiber=3).convert("RGBA")
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = P["box"]
        for i in range(int(x0), int(x1), 30):
            d.line((i, P["tau"], i + 14, P["tau"]), fill=(90, 90, 90), width=3)
        d.line(list(zip(P["xs"], P["raw"])), fill=(20, 20, 20), width=max(4, int(h * 0.01)))
        art.spray_line(im, list(zip(P["xs"][::3], P["ma"][::3])), self.acc2, max(10, h * 0.03), seed, False)
        fa = [i for i in np.nonzero(P["raw_alert"] & (P["gt"] == 0))[0]][::6][:5]
        for i in fa:
            cx, cy, r = P["xs"][i], P["raw"][i], h * 0.05
            art.spray_line(im, [(cx - r, cy - r), (cx + r, cy + r)], self.acc3, h * 0.018, seed + i, False)
            art.spray_line(im, [(cx + r, cy - r), (cx - r, cy + r)], self.acc3, h * 0.018, seed + i + 1, False)
        d = ImageDraw.Draw(im)
        self._ticks(d, P, h)
        return im.convert("RGB")

    def strip(self, w, h, seed):
        st = art.strips()[seed % 3]
        n = int(min(8, max(3, round(w / h))))
        gap = max(14, int(h * 0.05))
        tw = (w - gap * (n - 1)) / n
        out = rgba(w, h)
        for i, fr in enumerate(st["frames"][:: max(1, 8 // n)][:n]):
            im, box = art.fit(dict(id=fr["id"], box=fr["box"]), int(tw), h, 1.15, "rgb")
            im = self.mark(self.treat(im, box, seed + i, "rgb"), box, seed + i)
            out.alpha_composite(im.convert("RGBA"), (int(i * (tw + gap)), 0))
        return out

    def cover(self, s):
        """A pasted B&W poster, the person sprayed in acid green, a pink arrow from the title."""
        w, h = 2050, 1820
        x, y = W - w - 150, 170
        s.image((x, y, w, h), 2, "Cover image", "rgb", "desert", 1.2)
        L = rgba(W, H)
        art.spray_arrow(L, (1620, 1560), (x + 260, y + 1180), self.acc3, 26, 4, 0.22)
        dr = rgba(420, 420)
        art.spray(dr, lambda d: art.drone(d, 210, 210, 170, 255), self.fg, 10, 3, False, 0.6)
        L.alpha_composite(dr, (x + w - 520, y - 120))
        s.overlay(L, 0, 0)

    def thanks(self, s):
        w, h = 1750, 1700
        x, y = W - w - 170, 210
        s.image((x, y, w, h), 6, "Closing image", "neg", "snow", 1.0)


# ====================================================================== 58  pointillism
class Stipple(Style):
    key, name, folder, pptx, source = "stipple", "Pointillism", "pointillism_40", "Pointillism_Template_40.pptx", "58"
    bg, fg, mute, line = (246, 200, 214), (26, 18, 22), (112, 66, 84), (214, 160, 180)
    card_c, acc, acc2, acc3, on_acc = (246, 200, 214), (214, 40, 36), (214, 40, 36), (26, 18, 22), (246, 200, 214)
    fonts = dict(H="DMSerifDisplay-Regular.ttf", HI="DMSerifDisplay-Italic.ttf", D="DMSerifDisplay-Regular.ttf",
                 B="IBMPlexSans-Regular.ttf", BB="IBMPlexSans-SemiBold.ttf", M="IBMPlexMono-Medium.ttf")
    track = dict(H=-0.015, HI=-0.015, D=-0.02)
    shape = "stipple"
    label_pos, label_neg = "Person", "False alert"

    @lru_cache(None)
    def _base(self):
        return art.paper(W, H, self.bg, 3, fiber=3.5, mottle=5)

    def _dots_rect(self, d, box, density=0.08, r=2.2, seed=0, col=None):
        x0, y0, x1, y1 = box
        rng = np.random.default_rng(seed)
        n = int((x1 - x0) * (y1 - y0) * density / 30)
        for x, y in zip(rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)):
            d.ellipse((x - r, y - r, x + r, y + r), fill=col or self.fg)

    def footer(self, s, yb):
        for x in range(M, W - M, 14):
            s.d.ellipse((x - 3, yb - 3, x + 3, yb + 3), fill=self.fg)

    def _card(self, s, box, hi):
        x0, y0, x1, y1 = box
        if hi:
            s.d.rectangle(box, fill=self.fg)
            return
        self._dots_rect(s.d, (x0 + 22, y1, x1 + 22, y1 + 22), 0.9, 2.2, x0)
        self._dots_rect(s.d, (x1, y0 + 22, x1 + 22, y1 + 22), 0.9, 2.2, y0)
        s.d.rectangle(box, outline=self.fg, width=4)

    def divider_deco(self, s, x, y):
        rng = np.random.default_rng(s.n)
        for _ in range(4):
            art.sparkle(s.d, x + rng.uniform(700, 1500), y + rng.uniform(-40, 600), rng.uniform(40, 110), self.fg)

    def bar(self, s, x, y, w, h, frac, seed, label):
        bw = max(20, int(w * frac))
        s.d.rectangle((x, y, x + w, y + h), outline=self.fg, width=3)
        rng = np.random.default_rng(seed)
        n = int(bw * h / 9)
        xs, ys = rng.uniform(x, x + bw, n), rng.uniform(y, y + h, n)
        keep = rng.random(n) < 0.25 + 0.75 * ((xs - x) / max(1, bw))
        for xx, yy in zip(xs[keep], ys[keep]):
            s.d.ellipse((xx - 2.2, yy - 2.2, xx + 2.2, yy + 2.2), fill=self.acc if seed == 0 else self.fg)
        if label:
            s.mono(x + bw + 30, y + (h - 28) // 2, label, 28)

    def _D(self, im, box=None, kind="rgb"):
        g = im.convert("L").filter(ImageFilter.GaussianBlur(1.4)).filter(ImageFilter.UnsharpMask(6, 140, 3))
        L = np.asarray(g, np.float32) / 255
        D = art.stretch(1 - L, 8, 99.7, 1.9) if kind != "th" else art.heat(im, box) ** 1.6
        h, w = D.shape
        yy, xx = np.mgrid[0:h, 0:w]
        edge = np.minimum.reduce([xx / w, 1 - xx / w, yy / h, 1 - yy / h])
        return D * art.smooth(0.0, 0.12, edge)                      # dots thin out at the borders

    def treat(self, im, box, seed, kind):
        D = self._D(im, box, kind)
        w, h = im.size
        inside = np.zeros_like(D, bool)
        if box:                                                      # the person's own pixels: dark ones in the box
            x0, y0, x1, y1 = [int(v) for v in box]
            inside[y0:y1, x0:x1] = True
            inside &= D > np.percentile(D[y0:y1, x0:x1], 45) if inside.any() else inside
        out = art.stipple(np.where(inside, 0, D), self.fg, self.bg, density=2.2, rmin=1.3, rmax=3.0, seed=seed)
        if box:                                                      # ... drawn in red dots
            red_dots = art.stipple(np.where(inside, np.maximum(D, 0.55), 0), self.acc, self.bg, density=2.6,
                                   rmin=1.5, rmax=3.3, seed=seed + 1, alpha=True)
            out = out.convert("RGBA")
            out.alpha_composite(red_dots)
        return out.convert("RGB")

    def mark(self, im, box, seed):
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = self._pad(box, im, 0.3)
        for x in np.arange(x0, x1, 16):
            for yy in (y0, y1):
                d.ellipse((x - 3.5, yy - 3.5, x + 3.5, yy + 3.5), fill=self.acc)
        for y in np.arange(y0, y1, 16):
            for xx in (x0, x1):
                d.ellipse((xx - 3.5, y - 3.5, xx + 3.5, y + 3.5), fill=self.acc)
        r = max(26, (x1 - x0) * 0.22)
        art.sparkle(d, x1 + r * 0.3, y0 - r * 0.3, r, self.fg)
        art.sparkle(d, x0 - r * 0.2, y1 - r * 0.1, r * 0.6, self.fg)
        self._chip(im, x0, y0 - 14, self.label_pos, self.acc, self.bg, above=True)
        return im

    def frame(self, im, w, h, seed, label):
        im = im.convert("RGB")
        fo = fnt(self, "M", 26)
        ImageDraw.Draw(im).text((28, im.height - 60), f"( {label} )", font=fo, fill=self.fg)
        return im

    def trace_art(self, w, h, seed):
        tr = art.trace(seed)
        P = trace_xy(tr, w, h)
        im = Image.new("RGB", (w, h), self.bg)
        d = ImageDraw.Draw(im)
        rng = np.random.default_rng(seed)
        x0, y0, x1, y1 = P["box"]
        xs, step = P["xs"], P["xs"][1] - P["xs"][0]
        for i, g in enumerate(P["gt"]):
            if g:
                for _ in range(5):
                    xx, yy = xs[i] + rng.uniform(-step / 2, step / 2), rng.uniform(y0, y1)
                    d.ellipse((xx - 1.6, yy - 1.6, xx + 1.6, yy + 1.6), fill=self.line)
        for i in range(len(xs)):                                    # raw: a cloud of dots around the value
            for _ in range(6):
                yy = P["raw"][i] + rng.normal(0, h * 0.012)
                xx = xs[i] + rng.uniform(-step / 2, step / 2)
                d.ellipse((xx - 2.4, yy - 2.4, xx + 2.4, yy + 2.4), fill=self.fg)
        for i in range(0, len(xs), 1):                              # filtered: a crisp red dotted line
            d.ellipse((xs[i] - 4, P["ma"][i] - 4, xs[i] + 4, P["ma"][i] + 4), fill=self.acc)
        for x in np.arange(x0, x1, 18):
            d.ellipse((x - 2, P["tau"] - 2, x + 2, P["tau"] + 2), fill=self.mute)
        self._ticks(d, P, h)
        return im

    def cover(self, s):
        """A huge stippled snow scene bleeding off the right edge; the person in red dots."""
        w = int(W * 0.66)
        e = art.pick(2, "pos", "snow")
        im, box = art.fit(e, w, H, 1.15, "rgb")
        st = self.treat(im, box, 4, "rgb")
        st = self.mark(st, box, 4)
        s.pic(fade_alpha(st, "left", 0.0, 0.35), W - w, 0, 0, "Image - Cover")

    def thanks(self, s):
        w = int(W * 0.6)
        e = art.pick(5, "pos", "desert")
        im, box = art.fit(e, w, H, 1.05, "rgb")
        im = fade_alpha(self.mark(self.treat(im, box, 8, "th"), box, 8), "left", 0.0, 0.35)
        s.pic(im, W - w, 0, 0, "Image - Closing")


# ====================================================================== 59  mixed media
class Collage(Style):
    key, name, folder, pptx, source = "collage", "Mixed Media", "mixed_media_40", "MixedMedia_Template_40.pptx", "59"
    bg, fg, mute, line = (239, 233, 221), (20, 20, 20), (92, 88, 82), (196, 188, 174)
    card_c, acc, acc2, acc3, on_acc = (252, 250, 245), (255, 92, 28), (226, 34, 34), (140, 76, 255), (20, 20, 20)
    blue = (30, 110, 255)
    fonts = dict(H="PlayfairDisplay-Black.ttf", HI="PlayfairDisplay-BlackItalic.ttf", D="PlayfairDisplay-Black.ttf",
                 B="Inter-Regular.ttf", BB="Inter-SemiBold.ttf", M="CourierPrime-Bold.ttf", HW="Caveat-Bold.ttf")
    track = dict(H=-0.02, HI=-0.02, D=-0.03)
    scale = dict(HW=1.0)
    shape = "scrap"
    mono_upper = False
    label_pos, label_neg = "person?", "false alert!"

    @lru_cache(None)
    def _base(self):
        return art.paper(W, H, self.bg, 5, grid=(96, (226, 218, 203)), fiber=3)

    def footer(self, s, yb):
        rng = np.random.default_rng(s.n)
        xs = np.linspace(M, W - M, 60)
        s.d.line([(x, yb + rng.normal(0, 2.5)) for x in xs], fill=self.fg, width=4, joint="curve")

    def title_deco(self, s, x, y, w, size):
        rng = np.random.default_rng(s.n)
        L = art.brush_stroke(int(min(w * 0.6, 1000)), int(size * 0.34), [self.acc3, self.acc, self.blue][s.n % 3], s.n)
        s.g.paste(L, (int(x - 20), int(y + size * 0.98)), L)

    def divider_deco(self, s, x, y):
        art.concentric(s.d, x + 1120, y + 330, 300, self.blue, 7, 12)

    def _card(self, s, box, hi):
        x0, y0, x1, y1 = [int(v) for v in box]
        w, h = x1 - x0, y1 - y0
        col = self.acc if hi else self.card_c
        pc = art.print_card(art.paper(w, h, col, x0 + y0, fiber=2.5, mottle=3), seed=x0 * 3 + y0, torn=True,
                            shadow=True, shadow_a=70)
        pad = (pc.width - w) // 2
        s.g.paste(pc, (x0 - pad, y0 - pad), pc)
        tp = art.tape(int(min(220, w * 0.35)), 60, y0 + x0).rotate(5 if x0 % 2 else -5, Image.BICUBIC, expand=True)
        s.overlay(tp, x0 - 30, y0 - 30)

    def tag(self, s, x, y, txt, size=40, hi=True):
        fo = fnt(self, "M", size)
        tw = fo.getlength(txt)
        px, ph = int(size * 0.5), int(size * 1.6)
        s.d.rectangle((x, y, x + tw + 2 * px, y + ph), fill=self.fg if not hi else self.acc)
        bb = fo.getbbox("H")
        s.text(x + px, y + (ph - (bb[3] - bb[1])) / 2 - bb[1], txt, size, "M", self.bg if not hi else self.fg)
        return x + tw + 2 * px

    def bar(self, s, x, y, w, h, frac, seed, label):
        bw = max(40, int(w * frac))
        s.d.line((x, y + h, x + w, y + h), fill=self.line, width=2)
        L = art.brush_stroke(bw, h + 20, [self.acc3, self.acc, self.blue][seed % 3], seed)
        s.g.paste(L, (x, y - 10), L)
        if label:
            s.mono(x + bw + 30, y + (h - 28) // 2, label, 30)

    def inner(self, w, h):
        p = int(min(w, h) * 0.06)
        return w - 2 * p, h - 2 * p

    def treat(self, im, box, seed, kind):
        """Halftone print; pseudo-thermal becomes a two-colour riso print."""
        L = art.lum(im)
        D = art.stretch(1 - L, 2, 99.5, 1.05)
        cell = max(8, int(min(im.size) / 90))
        if kind == "th":
            Hm = art.heat(im, box)
            a = art.halftone(Hm, cell, 15, self.acc, (250, 246, 238))
            b = art.halftone(D * 0.7, cell, 75, self.blue, (250, 246, 238), alpha=True)
            a = a.convert("RGBA")
            a.alpha_composite(b)
            return a.convert("RGB")
        return art.halftone(D, cell, 45, (22, 22, 22), (250, 246, 238))

    def mark(self, im, box, seed):
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = self._pad(box, im, 0.45)
        wd = max(6, min(im.size) * 0.011)
        art.scribble_ellipse(d, (x0, y0, x1, y1), self.acc2, wd, seed)
        fo = fnt(self, "HW", int(max(44, min(im.size) * 0.08)))
        tx = x1 + 30 if x1 + 30 + fo.getlength(self.label_pos) < im.width else max(10, x0 - fo.getlength(self.label_pos) - 40)
        ty = max(10, y0 - fo.size * 0.9)
        d.text((tx, ty), self.label_pos, font=fo, fill=self.acc2)
        art.marker_arrow(d, (tx + fo.getlength(self.label_pos) * 0.3, ty + fo.size * 1.1),
                         ((x0 + x1) / 2 + (x1 - x0) * 0.35 * (1 if tx < x0 else -1) * -1, y0 + (y1 - y0) * 0.2),
                         self.acc2, wd * 0.8, 0.2)
        return im

    def mark_neg(self, im, seed):
        d = ImageDraw.Draw(im)
        w, h = im.size
        rng = np.random.default_rng(seed)
        cx, cy = w * rng.uniform(0.4, 0.6), h * rng.uniform(0.4, 0.55)
        r = min(w, h) * 0.13
        wd = max(6, min(w, h) * 0.011)
        art.scribble_ellipse(d, (cx - r, cy - r * 0.8, cx + r, cy + r * 0.8), self.acc2, wd, seed)
        d.line((cx - r * 1.2, cy - r, cx + r * 1.2, cy + r), fill=self.acc2, width=int(wd))
        fo = fnt(self, "HW", int(max(44, min(w, h) * 0.08)))
        d.text((max(10, cx - fo.getlength(self.label_neg) / 2), cy + r + 10), self.label_neg, font=fo, fill=self.acc2)
        return im

    def frame(self, im, w, h, seed, label):
        """A print with a white border, torn edge, tape and a typed label."""
        im = im.convert("RGB")
        fo = fnt(self, "M", 28)
        lab = f"fig. {label.lower()}"
        b = int(min(im.size) * 0.03)
        pc = fit_into(art.print_card(im, seed=seed, border=b, torn=True, shadow=True, shadow_a=80), w, h)
        out = rgba(w, h)
        out.alpha_composite(pc, ((w - pc.width) // 2, (h - pc.height) // 2))
        tp = art.tape(int(w * 0.22), int(max(44, h * 0.05)), seed).rotate(-6 if seed % 2 else 7, Image.BICUBIC, expand=True)
        out.alpha_composite(tp, (int(w * 0.39), 0))
        d = ImageDraw.Draw(out)
        tw = fo.getlength(lab)
        lx, ly = int(w * 0.06), int(h - h * 0.06 - 50)
        d.rectangle((lx, ly, lx + tw + 28, ly + 46), fill=(250, 248, 240), outline=self.fg, width=2)
        d.text((lx + 14, ly + 8), lab, font=fo, fill=self.fg)
        return out

    def strip(self, w, h, seed):
        st = art.strips()[seed % 3]
        n = int(min(6, max(3, round(w / h))))
        tw = w / n
        out = rgba(w, h)
        rng = np.random.default_rng(seed)
        for i, fr in enumerate(st["frames"][:: max(1, 8 // n)][:n]):
            im, box = art.fit(dict(id=fr["id"], box=fr["box"]), int(tw * 0.86), int(h * 0.82), 1.15, "rgb")
            im = self.treat(im, box, seed + i, "rgb")
            if i == n // 2 and box:
                im = self.mark(im, box, seed + i)
            pc = fit_into(art.print_card(im, seed=seed + i, border=10, torn=True, angle=rng.uniform(-3, 3),
                                         shadow=True, shadow_a=70), int(tw), h)
            out.alpha_composite(pc, (int(i * tw + (tw - pc.width) / 2), int((h - pc.height) / 2)))
        return out

    def trace_art(self, w, h, seed):
        tr = art.trace(seed)
        P = trace_xy(tr, w, h)
        im = art.paper(w, h, (250, 248, 242), seed, grid=(40, (214, 226, 236)), fiber=2)
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = P["box"]
        d.line([(x0, P["tau"]), (x1, P["tau"])], fill=self.blue, width=3)
        d.line(list(zip(P["xs"], P["raw"])), fill=(30, 30, 30), width=max(3, int(h * 0.008)))
        L = rgba(w, h)
        dd = ImageDraw.Draw(L)
        dd.line(list(zip(P["xs"], P["ma"])), fill=(*self.acc3, 255), width=max(8, int(h * 0.025)), joint="curve")
        im = im.convert("RGBA")
        im.alpha_composite(L)
        d = ImageDraw.Draw(im)
        fa = np.nonzero(P["raw_alert"] & (P["gt"] == 0))[0]
        if len(fa):
            i = fa[len(fa) // 2]
            r = h * 0.09
            art.scribble_ellipse(d, (P["xs"][i] - r, P["raw"][i] - r, P["xs"][i] + r, P["raw"][i] + r), self.acc2,
                                 max(5, h * 0.012), seed)
            fo = fnt(self, "HW", int(max(40, h * 0.1)))
            d.text((min(w - fo.getlength("flicker!") - 20, P["xs"][i] + r), max(5, P["raw"][i] - r * 2)), "flicker!",
                   font=fo, fill=self.acc2)
        self._ticks(d, P, h)
        return im.convert("RGB")

    def cover(self, s):
        """Collage: a halftone desert print, the drone render, a riso thermal print, brush, circles, marker."""
        d = s.d
        art.concentric(d, 3420, 330, 280, self.blue, 7, 12)
        L = art.brush_stroke(1500, 260, self.acc3, 3)
        s.g.paste(L, (1900, 1700), L)
        d.ellipse((2050, 140, 2470, 560), fill=self.acc)
        s.image((2120, 260, 1500, 1500), 2, "Cover image", "rgb", "desert", 1.25)
        s.image((1700, 1180, 900, 800), 4, "Thermal", "th", "desert", 1.25)
        s.image((3020, 1300, 700, 620), 1, "Scene", "scene", None, 1.6)

    def thanks(self, s):
        art.concentric(s.d, 3200, 1500, 420, self.blue, 8, 12)
        L = art.brush_stroke(1400, 240, self.acc, 9)
        s.g.paste(L, (2100, 350), L)
        s.image((2050, 420, 1500, 1350), 6, "Closing image", "neg", "snow", 1.0)


# ====================================================================== 55 x 58  ember stipple
class Ember(Stipple):
    key, name, folder, pptx, source = "ember", "Ember Stipple", "ember_stipple_40", "EmberStipple_Template_40.pptx", "55 + 58"
    bg, fg, mute, line = (10, 8, 8), (248, 236, 222), (150, 128, 116), (64, 46, 40)
    card_c, acc, acc2, acc3, on_acc = (10, 8, 8), (255, 110, 30), (255, 60, 40), (255, 200, 120), (20, 8, 4)
    fonts = dict(H="DMSerifDisplay-Regular.ttf", HI="DMSerifDisplay-Italic.ttf", D="DMSerifDisplay-Regular.ttf",
                 B="Inter-Regular.ttf", BB="Inter-SemiBold.ttf", M="JetBrainsMono-Medium.ttf")
    track = dict(H=-0.015, HI=-0.015, D=-0.02, M=0.02)
    label_pos, label_neg = "Person", "False alert"

    @lru_cache(None)
    def _base(self):
        rng = np.random.default_rng(1)
        arr = np.full((H, W, 3), self.bg, np.float32) + rng.normal(0, 3, (H, W))[..., None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def footer(self, s, yb):
        cols = art.apply_lut(np.linspace(0.35, 1, (W - 2 * M) // 14), "fire")
        for i, x in enumerate(range(M, W - M - 14, 14)):
            s.d.ellipse((x - 3, yb - 3, x + 3, yb + 3), fill=tuple(int(c) for c in cols[i]))

    def _card(self, s, box, hi):
        x0, y0, x1, y1 = box
        if hi:
            s.d.rectangle(box, fill=self.acc)
            return
        s.d.rectangle(box, outline=self.line, width=3)
        cols = art.apply_lut(np.linspace(0.4, 1, 12), "fire")
        for i in range(12):
            s.d.ellipse((x0 + 20 + i * 14 - 3, y0 + 20 - 3, x0 + 20 + i * 14 + 3, y0 + 20 + 3),
                        fill=tuple(int(c) for c in cols[i]))

    def bar(self, s, x, y, w, h, frac, seed, label):
        bw = max(20, int(w * frac))
        s.d.rectangle((x, y, x + w, y + h), outline=self.line, width=2)
        rng = np.random.default_rng(seed)
        n = int(bw * h / 8)
        xs, ys = rng.uniform(x, x + bw, n), rng.uniform(y, y + h, n)
        t = (xs - x) / max(1, w)
        keep = rng.random(n) < 0.3 + 0.7 * (xs - x) / max(1, bw)
        cols = art.apply_lut(0.4 + 0.6 * t, "fire")
        for xx, yy, c in zip(xs[keep], ys[keep], cols[keep]):
            s.d.ellipse((xx - 2.4, yy - 2.4, xx + 2.4, yy + 2.4), fill=tuple(int(v) for v in c))
        self.bar_label(s, x, y, w, h, bw, 0.0, label)

    def treat(self, im, box, seed, kind):
        """Heat as embers: dot density and colour both follow the pseudo-thermal value; a soft glow."""
        Hm = art.heat(im, box, 0.75) ** 1.8 if box else art.heat(im) ** 2.4     # scene: drone + rocks glow, sand stays dark
        h, w = Hm.shape
        yy, xx = np.mgrid[0:h, 0:w]
        edge = np.minimum.reduce([xx / w, 1 - xx / w, yy / h, 1 - yy / h])
        Hm = Hm * art.smooth(0.0, 0.1, edge)
        cl = art.lut("fire")
        fn = lambda v: tuple(int(c) for c in cl[int(min(255, 70 + v * 185))])
        dots = art.stipple(Hm, None, self.bg, density=2.4, rmin=1.2, rmax=3.6, seed=seed, color_fn=fn, gamma=1.2)
        a = np.asarray(dots, np.float32)
        glow = np.asarray(dots.filter(ImageFilter.GaussianBlur(10)), np.float32)
        out = a + glow * 0.9
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    def mark(self, im, box, seed):
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = self._pad(box, im, 0.3)
        for x in np.arange(x0, x1, 14):
            for yy in (y0, y1):
                d.ellipse((x - 2.5, yy - 2.5, x + 2.5, yy + 2.5), fill=self.fg)
        for y in np.arange(y0, y1, 14):
            for xx in (x0, x1):
                d.ellipse((xx - 2.5, y - 2.5, xx + 2.5, y + 2.5), fill=self.fg)
        r = max(26, (x1 - x0) * 0.22)
        art.sparkle(d, x1 + r * 0.3, y0 - r * 0.3, r, self.fg)
        self._chip(im, x0, y0 - 14, self.label_pos, self.fg, self.bg, above=True)
        return im

    def mark_neg(self, im, seed):
        return Style.mark_neg(self, im, seed)

    def frame(self, im, w, h, seed, label):
        return Style.frame(self, im, w, h, seed, label)

    def trace_art(self, w, h, seed):
        tr = art.trace(seed)
        P = trace_xy(tr, w, h)
        im = Image.new("RGB", (w, h), self.bg)
        d = ImageDraw.Draw(im)
        rng = np.random.default_rng(seed)
        xs, step = P["xs"], P["xs"][1] - P["xs"][0]
        cl = art.lut("fire")
        f = np.asarray(tr["fused"])
        for i in range(len(xs)):
            for _ in range(7):
                yy = P["raw"][i] + rng.normal(0, h * 0.014)
                xx = xs[i] + rng.uniform(-step / 2, step / 2)
                c = tuple(int(v) for v in cl[int(80 + f[i] * 175)])
                d.ellipse((xx - 2.6, yy - 2.6, xx + 2.6, yy + 2.6), fill=c)
        for i in range(len(xs)):
            d.ellipse((xs[i] - 3.5, P["ma"][i] - 3.5, xs[i] + 3.5, P["ma"][i] + 3.5), fill=self.fg)
        glow = np.asarray(im.filter(ImageFilter.GaussianBlur(8)), np.float32)
        im = Image.fromarray(np.clip(np.asarray(im, np.float32) + glow * 0.8, 0, 255).astype(np.uint8), "RGB")
        self._ticks(ImageDraw.Draw(im), P, h)
        return im

    def divider_deco(self, s, x, y):
        rng = np.random.default_rng(s.n)
        for _ in range(3):
            art.sparkle(s.d, x + rng.uniform(800, 1500), y + rng.uniform(-40, 600), rng.uniform(40, 100), self.acc3)

    def cover(self, s):
        w = int(W * 0.64)
        e = art.pick(2, "pos", "snow")
        im, box = art.fit(e, w, H, 1.1, "rgb")
        im = self.mark(self.treat(im, box, 4, "th"), box, 4)
        s.pic(fade_alpha(im, "left", 0.0, 0.3), W - w, 0, 0, "Image - Cover")

    def thanks(self, s):
        w = int(W * 0.58)
        e = art.pick(5, "pos", "forest")
        im, box = art.fit(e, w, H, 1.05, "rgb")
        im = self.mark(self.treat(im, box, 8, "th"), box, 8)
        s.pic(fade_alpha(im, "left", 0.0, 0.3), W - w, 0, 0, "Image - Closing")


# ====================================================================== 56 x 58  dot matrix
class DotMatrix(Pixel):
    key, name, folder, pptx, source = "dotmatrix", "Dot Matrix", "dot_matrix_40", "DotMatrix_Template_40.pptx", "56 + 58"
    bg, fg, mute, line = (244, 242, 236), (16, 16, 22), (110, 108, 104), (210, 206, 198)
    card_c, acc, acc2, on_acc = (244, 242, 236), (36, 80, 255), (255, 40, 130), (255, 255, 255)
    fonts = dict(H="SpaceGrotesk-Bold.ttf", HI="SpaceGrotesk-Regular.ttf", D="DotGothic16-Regular.ttf",
                 B="SpaceGrotesk-Regular.ttf", BB="SpaceGrotesk-Bold.ttf", M="SpaceMono-Regular.ttf")
    track = dict(H=-0.03, HI=-0.03, D=-0.02, BB=-0.01)
    shape, png, real_thermal = "dots", False, True
    cell = 26
    bullet = "●"

    @lru_cache(None)
    def _base(self):
        im = Image.new("RGB", (W, H), self.bg)
        d = ImageDraw.Draw(im)
        for y in range(0, H, 52):
            for x in range(0, W, 52):
                d.ellipse((x + 24, y + 24, x + 28, y + 28), fill=(226, 223, 216))
        return im

    def footer(self, s, yb):
        n = (W - 2 * M) // 18
        cols = art.apply_lut(np.linspace(0, 1, n), ["rainbow", "blue", "pink", "sunset"][s.n % 4])
        rng = np.random.default_rng(s.n)
        for i in range(n):
            r = 5 * (0.35 + 0.65 * (i / n)) if rng.random() < 0.9 else 0
            x = M + i * 18 + 9
            if r:
                s.d.ellipse((x - r, yb - r, x + r, yb + r), fill=tuple(int(c) for c in cols[i]))

    def _card(self, s, box, hi):
        x0, y0, x1, y1 = box
        for y in range(int(y0) + 26, int(y1) + 22, 14):
            for x in range(int(x0) + 26, int(x1) + 22, 14):
                if x > x1 - 2 or y > y1 - 2:
                    s.d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=self.fg)
        s.d.rectangle(box, fill=self.acc if hi else (255, 255, 255), outline=self.fg, width=4)

    def treat(self, im, box, seed, kind):
        w, h = im.size
        gw, gh = math.ceil(w / self.cell), math.ceil(h / self.cell)
        keep = art.edge_keep(gw, gh, ["top", "left", "right", "bottom"][seed % 4], 0.0, 0.3)
        return art.dotgrid(im, self.cell, self.bg, keep, seed)

    def mark(self, im, box, seed):
        d = ImageDraw.Draw(im)
        c = self.cell
        x0, y0, x1, y1 = self._pad(box, im, 0.15)
        for x in np.arange(x0, x1 + 1, c / 2):
            for yy in (y0, y1):
                d.ellipse((x - 6, yy - 6, x + 6, yy + 6), fill=self.acc)
        for y in np.arange(y0, y1 + 1, c / 2):
            for xx in (x0, x1):
                d.ellipse((xx - 6, y - 6, xx + 6, y + 6), fill=self.acc)
        self._chip(im, x0, y0 - 10, self.label_pos, self.acc, (255, 255, 255), above=True)
        return im

    def bar(self, s, x, y, w, h, frac, seed, label):
        c = 22
        n, rows = int(w * frac // c), max(1, h // c)
        cols = art.apply_lut(np.linspace(0.2, 0.2 + 0.8 * frac, max(1, n)), ["rainbow", "blue", "pink", "sunset"][seed % 4])
        for r in range(rows):
            for i in range(int(w // c)):
                cx, cy = x + i * c + c / 2, y + r * c + c / 2
                if i < n:
                    s.d.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=tuple(int(v) for v in cols[i]))
                else:
                    s.d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=self.line)
        if label:
            s.mono(x + n * c + 30, y + (h - 28) // 2, label, 28)

    def trace_art(self, w, h, seed):
        tr = art.trace(seed)
        P = trace_xy(tr, w, h, top=0.06, bottom=0.3)
        im = Image.new("RGB", (w, h), self.bg)
        d = ImageDraw.Draw(im)
        c = max(12, int(h * 0.045))
        x0, y0, x1, y1 = P["box"]
        ncol = int((x1 - x0) // c)
        f = np.asarray(tr["fused"])
        lutc = art.lut("rainbow")
        for j in range(ncol):
            a, b = int(j / ncol * len(f)), int((j + 1) / ncol * len(f))
            v = f[a:b].max() if b > a else 0
            k = int(round(v * (y1 - y0) / c))
            for r in range(int((y1 - y0) // c)):
                cx, cy = x0 + j * c + c / 2, y1 - r * c - c / 2
                if r < k:
                    col = tuple(int(q) for q in lutc[int(min(255, r * c / (y1 - y0) * 255))])
                    d.ellipse((cx - c * 0.42, cy - c * 0.42, cx + c * 0.42, cy + c * 0.42), fill=col)
                else:
                    d.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=self.line)
        d.line([(x0, P["tau"]), (x1, P["tau"])], fill=self.fg, width=3)
        self._ticks(d, P, h)
        return im

    def cover(self, s):
        w = int(W * 0.62)
        e = art.pick(2, "pos", "snow")
        im, box = art.fit(e, w, H, 1.05, "rgb")
        gw, gh = math.ceil(w / self.cell), math.ceil(H / self.cell)
        im = art.dotgrid(im, self.cell, self.bg, art.edge_keep(gw, gh, "left", 0.0, 0.55), 3)
        s.pic(self.frame(self.mark(im, box, 3), w, H, 3, "Cover image"), W - w, 0, 0, "Image - Cover")

    def thanks(self, s):
        w = int(W * 0.55)
        e = art.pick(5, "pos", "forest")
        im, box = art.fit(e, w, H, 1.05, "th")
        gw, gh = math.ceil(w / self.cell), math.ceil(H / self.cell)
        im = art.dotgrid(im, self.cell, self.bg, art.edge_keep(gw, gh, "left", 0.0, 0.55), 9)
        s.pic(self.frame(self.mark(im, box, 9), w, H, 9, "Closing image"), W - w, 0, 0, "Image - Closing")


# ====================================================================== 57 x 59  street collage
class Street(Graffiti):
    key, name, folder, pptx, source = "street", "Street Collage", "street_collage_40", "StreetCollage_Template_40.pptx", "57 + 59"
    wall = (168, 164, 158)
    bg, mute = (168, 164, 158), (48, 46, 44)
    fonts = dict(H="Anton-Regular.ttf", HI="PermanentMarker-Regular.ttf", D="Anton-Regular.ttf",
                 B="Inter-Regular.ttf", BB="Inter-SemiBold.ttf", M="CourierPrime-Bold.ttf", HW="PermanentMarker-Regular.ttf")
    track = dict(H=0.0, D=0.0)
    scale = dict(H=1.12, HI=0.92, D=1.1)
    blue = (30, 110, 255)
    label_pos = "PERSON?"

    @lru_cache(None)
    def _base(self):
        return art.concrete(W, H, self.wall, 11)

    def treat(self, im, box, seed, kind):
        """Halftone photocopy, pseudo-thermal as an orange riso layer."""
        D = art.stretch(1 - art.lum(im), 2, 99.5, 1.05)
        cell = max(8, int(min(im.size) / 95))
        if kind == "th":
            a = art.halftone(art.heat(im, box), cell, 15, self.acc, (244, 240, 232)).convert("RGBA")
            a.alpha_composite(art.halftone(D * 0.7, cell, 75, (20, 20, 20), (0, 0, 0), alpha=True))
            return a.convert("RGB")
        return art.halftone(D, cell, 45, (20, 20, 20), (244, 240, 232))

    def mark(self, im, box, seed):
        im = Graffiti.mark(self, im, box, seed) if seed % 2 == 0 else Collage.mark(self, im, box, seed)
        return im

    acc2_marker = (226, 34, 34)

    def divider_deco(self, s, x, y):
        art.concentric(s.d, x + 1120, y + 330, 300, self.blue, 6, 14)
        Graffiti.divider_deco(self, s, x, y)

    def cover(self, s):
        """Three pasted prints - desert, snow, thermal - sprayed and scribbled over."""
        s.image((1900, 180, 1250, 1180), 2, "Cover image", "rgb", "desert", 1.25)
        s.image((2800, 640, 860, 900), 3, "Snow", "rgb", "snow", 1.25)
        s.image((2050, 1300, 1000, 640), 4, "Thermal", "th", "forest", 1.3)
        L = rgba(W, H)
        art.spray_arrow(L, (1560, 1580), (2060, 1080), self.acc3, 24, 4, 0.25)
        fo = fnt(self, "HW", 120)
        art.spray_text(L, (3060, 1640), "FOUND?", fo, self.acc2, 5, True, 8)
        s.overlay(L, 0, 0)


# ====================================================================== 55 56 57 58 59  signal collage
class Signal(Style):
    key, name, folder, pptx, source = "signal", "Signal Collage", "signal_collage_40", "SignalCollage_Template_40.pptx", "55 + 56 + 57 + 58 + 59"
    bg, fg, mute, line = (27, 25, 24), (244, 236, 224), (150, 140, 128), (70, 64, 58)
    card_c, acc, acc2, acc3, on_acc = (16, 15, 14), (255, 98, 26), (164, 255, 0), (255, 40, 150), (16, 15, 14)
    fonts = dict(H="Anton-Regular.ttf", HI="InstrumentSerif-Italic.ttf", D="Silkscreen-Regular.ttf",
                 B="Inter-Regular.ttf", BB="Inter-SemiBold.ttf", M="JetBrainsMono-Medium.ttf", HW="PermanentMarker-Regular.ttf")
    track = dict(H=0.0, HI=-0.01, D=-0.12, M=0.02)
    scale = dict(H=1.1, HI=1.35)
    shape = "scrap"
    label_pos, label_neg = "PERSON", "FALSE ALERT"

    def __init__(self):
        self._burn, self._pb, self._st, self._co = Burn(), PixelBurn(), Ember(), Collage()

    @lru_cache(None)
    def _base(self):
        im = art.paper(W, H, self.bg, 9, fiber=4, mottle=5)
        leak = art.leak_field(W, H, 77, "leak", ("left", 0.25, 0.95))     # a faint film-burn glow, top right
        arr = np.asarray(im, np.float32) + art.apply_lut(leak * 0.55, "fire") * 0.35
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def footer(self, s, yb):
        c, n = 16, (W - 2 * M) // 16
        rng = np.random.default_rng(s.n)
        cols = art.apply_lut(np.linspace(0.35, 1, n), "fire")
        for i in range(n):
            if rng.random() < (i / n) ** 0.6 + 0.08:
                s.d.rectangle((M + i * c + 1, yb - 4, M + i * c + c - 2, yb + c - 7), fill=tuple(int(v) for v in cols[i]))

    def title_deco(self, s, x, y, w, size):
        L = rgba(w + 120, size)
        yy = size * 0.74
        art.spray_line(L, [(20, yy + 6), (w * 0.5, yy - 8), (w + 80, yy + 4)], self.acc, size * 0.22, s.n, False)
        s.g.paste(L, (int(x - 40), int(y)), L)

    def divider_deco(self, s, x, y):
        L = rgba(1400, 1000)
        art.spray(L, lambda d: d.ellipse((140, 140, 1000, 900), outline=255, width=60), self.acc2, 60, s.n, True)
        s.g.paste(L, (int(x - 120), int(y - 80)), L)

    def _card(self, s, box, hi):
        x0, y0, x1, y1 = [int(v) for v in box]
        w, h = x1 - x0, y1 - y0
        col = self.acc if hi else self.card_c
        pc = art.print_card(art.paper(w, h, col, x0 + y0, fiber=3, mottle=3), seed=x0 * 3 + y0, torn=True,
                            shadow=True, shadow_a=160)
        pad = (pc.width - w) // 2
        s.g.paste(pc, (x0 - pad, y0 - pad), pc)
        tp = art.tape(int(min(220, w * 0.35)), 60, y0 + x0, (210, 200, 176), 150)
        s.overlay(tp.rotate(5 if x0 % 2 else -5, Image.BICUBIC, expand=True), x0 + w - 200, y0 - 30)

    def tag(self, s, x, y, txt, size=40, hi=True):
        return Graffiti.tag(self, s, x, y, txt, size, hi)

    def bar(self, s, x, y, w, h, frac, seed, label):
        PixelBurn.bar(self, s, x, y, w, h, frac, seed, label)

    def inner(self, w, h):
        p = int(min(w, h) * 0.05)
        return w - 2 * p, h - 2 * p

    def treat(self, im, box, seed, kind):
        """Each print uses a different medium of the five mood boards."""
        k = seed % 4
        if kind == "th":
            k = 1 if seed % 2 else 2
        if k == 0:
            return self._burn.treat(im, box, seed, kind)
        if k == 1:
            return self._pb._heat_cells(im, box, seed, None)
        if k == 2:
            return self._st.treat(im, box, seed, "th")
        return self._co.treat(im, box, seed, "rgb")

    def mark(self, im, box, seed):
        return Graffiti.mark(self, im, box, seed)

    def mark_neg(self, im, seed):
        return Graffiti.mark_neg(self, im, seed)

    def frame(self, im, w, h, seed, label):
        im = Style.frame(self, im, w, h, seed, label)
        pc = fit_into(art.print_card(im, seed=seed, torn=True, shadow=True, shadow_a=170), w, h)
        out = rgba(w, h)
        out.alpha_composite(pc, ((w - pc.width) // 2, (h - pc.height) // 2))
        tp = art.tape(int(w * 0.2), int(max(40, h * 0.05)), seed, (214, 204, 180), 160)
        out.alpha_composite(tp.rotate(-6 if seed % 2 else 7, Image.BICUBIC, expand=True), (int(w * 0.06), 0))
        return out

    acc2_marker = (164, 255, 0)
    wall = bg

    def strip(self, w, h, seed):
        return Graffiti.strip(self, w, h, seed)

    def trace_art(self, w, h, seed):
        return self._burn.trace_art(w, h, seed)

    def cover(self, s):
        """All five boards on one wall: burnt RGB print, pixel thermal, ember stipple, halftone, spray."""
        # seed % 4 picks the medium: 4 -> burn, 1 -> pixel heat, 8 -> burn, 2 -> ember (thermal)
        s.image((1880, 150, 1300, 1150), 4, "Cover image", "rgb", "desert", 1.3)
        s.image((2960, 250, 720, 820), 1, "Thermal", "th", "snow", 1.3)
        s.image((2700, 1160, 980, 760), 8, "Burn", "rgb", "forest", 1.3)
        s.image((1960, 1350, 760, 600), 2, "Ember", "th", "desert", 1.3)
        L = rgba(W, H)
        art.spray_arrow(L, (1560, 1560), (1980, 1120), self.acc3, 24, 4, 0.25)
        s.overlay(L, 0, 0)

    def thanks(self, s):
        s.image((2050, 300, 1500, 1500), 5, "Closing image", "rgb", "snow", 1.2)


STYLES = {c.key: c for c in (Burn, Pixel, PixelBurn, Graffiti, Stipple, Collage, Ember, DotMatrix, Street, Signal)}
