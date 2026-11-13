"""Slide model + exporters.

A slide is three layers, kept apart so PowerPoint stays editable:
  g      background canvas (surface texture, cards, rules, chart bars) - one picture in the PPTX
  pics   image slots - one picture each, so right-click > Change Picture works
  over   decorations that sit above the images (tape, spray tags) - small transparent PNGs
  texts  every word, as live text boxes in the style's own fonts
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONTS = ROOT / "fonts"
CALIB = HERE / "calib.json"
W, H = 3840, 2160
M = 180
N = 40
EMU_PX = 12192000 / W          # 3175 EMU per px
PT_PX = 13.333333 * 72 / W     # 0.25 pt per px
LINE = 1.2                     # line pitch of single-line boxes (em)


# ====================================================================== fonts
@lru_cache(None)
def face(file):
    """PowerPoint face name + bold/italic flags straight from the font's name table."""
    from fontTools.ttLib import TTFont
    n = TTFont(FONTS / file)["name"]
    fam, sub = n.getDebugName(1), (n.getDebugName(2) or "").lower()
    return dict(file=str(FONTS / file), face=fam, bold="bold" in sub, italic="italic" in sub)


_FC = {}


def fnt(st, key, size):
    p = face(st.fonts[key])["file"]
    k = (p, int(size))
    if k not in _FC:
        _FC[k] = ImageFont.truetype(p, int(size))
    return _FC[k]


def tlen(s, fo, track, size):
    return fo.getlength(s) + track * size * max(0, len(s) - 1)


def wrap(text, fo, maxw):
    """Greedy word wrap that keeps runs of spaces exactly as typed (PowerPoint keeps them too)."""
    import re
    out = []
    for para in text.split("\n"):
        line = ""
        for w_ in re.findall(r"\S+\s*|\s+", para):
            t = line + w_
            if fo.getlength(t.rstrip()) <= maxw or not line.strip():
                line = t
            else:
                out.append(line.rstrip())
                line = w_
        out.append(line.rstrip())
    return out


# ====================================================================== slide
class Slide:
    def __init__(self, st, n, section="", chrome=True):
        self.T, self.n = st, n
        self.g = st.base(n).copy()
        self.d = ImageDraw.Draw(self.g)
        self.texts, self.pics, self.over = [], [], []
        if chrome:
            st.chrome(self, section)

    def c(self, col):
        return getattr(self.T, col) if isinstance(col, str) else col

    # ---------------------------------------------------------------- text
    def text(self, x, y, s, size, f="B", fill="fg", anchor="la", track=None):
        if not s:
            return 0
        T = self.T
        size = int(size * T.scale.get(f, 1))
        fo = fnt(T, f, size)
        tr = T.track.get(f, 0) if track is None else track
        w = tlen(s, fo, tr, size)
        if anchor[0] == "r":
            x -= w
        elif anchor[0] == "m":
            x -= w / 2
        self.texts.append(dict(kind="line", x=x, y=y, s=s, size=size, f=f, fill=self.c(fill), track=tr,
                               w=w, align={"r": "r", "m": "m"}.get(anchor[0], "l")))
        return w

    def head(self, x, y, s, size=150, fill="fg", f="H"):
        return self.text(x, y, s, size, f, fill)

    def mono(self, x, y, s, size=34, fill="mute", anchor="la"):
        return self.text(x, y, s.upper() if self.T.mono_upper else s, size, "M", fill, anchor)

    def hw(self, x, y, s, size=70, fill="acc2", anchor="la"):
        """Handwritten annotation (only in styles that have a hand font)."""
        if "HW" in self.T.fonts:
            return self.text(x, y, s, size, "HW", fill, anchor)
        return 0

    def block(self, x, y, w, s, size=46, f="B", lh=1.38, fill="fg"):
        size = int(size * self.T.scale.get(f, 1))
        lines = wrap(s, fnt(self.T, f, size), w)
        self.texts.append(dict(kind="block", x=x, y=y, w=w, s=s, lines=lines, size=size, f=f, lh=lh,
                               fill=self.c(fill), track=0))
        return y + int(size * lh) * len(lines)

    def bullets(self, x, y, w, items, size=46, gap=0.55, fill="fg", lh=1.36):
        fo = fnt(self.T, "B", size)
        ind = int(size * 0.95)
        lay, yy = [], y
        for it in items:
            lines = wrap(it, fo, w - ind)
            lay.append((yy, lines))
            yy += int(size * lh) * len(lines) + int(size * gap)
        self.texts.append(dict(kind="bullets", f="B", x=x, y=y, w=w, items=items, lay=lay, size=size, gap=gap,
                               lh=lh, ind=ind, fill=self.c(fill), track=0))
        return yy

    def title(self, s, kicker=None, y=210, size=150, x=M):
        if kicker:
            self.mono(x, y, kicker, 34, "acc")
        sz = int(size * self.T.scale.get("H", 1))
        w = tlen(s, fnt(self.T, "H", sz), self.T.track.get("H", 0), sz)
        self.T.title_deco(self, x, y + 66, w, sz)
        self.head(x, y + 66, s, size)
        return y + 66 + int(sz * 1.25)

    # ---------------------------------------------------------------- shapes
    def card(self, box, hi=False):
        self.T.card(self, box, hi)

    def tag(self, x, y, s, size=40, hi=True):
        return self.T.tag(self, x, y, s, size, hi)

    def rule(self, x0, y, x1, col="line", wd=3):
        self.d.line((x0, y, x1, y), fill=self.c(col), width=wd)

    def vrule(self, x, y0, y1, col="line", wd=3):
        self.d.line((x, y0, x, y1), fill=self.c(col), width=wd)

    def bar(self, x, y, w, h, frac, seed=0, label=None):
        self.T.bar(self, x, y, w, h, frac, seed, label)

    # ---------------------------------------------------------------- pictures
    def pic(self, im, x, y, radius=0, name=None):
        """An image slot. RGBA prints are flattened onto the surface, so the PPTX picture is a plain JPEG."""
        x, y = int(x), int(y)
        if im.mode == "RGBA":
            under = self.g.crop((x, y, x + im.width, y + im.height)).convert("RGBA")
            under.alpha_composite(im)
            im = under.convert("RGB")
        self.pics.append(dict(x=x, y=y, w=im.width, h=im.height, r=radius, im=im, name=name))

    def overlay(self, im, x, y):
        """Decoration above the images (RGBA); cropped to its visible part."""
        bb = im.getchannel("A").getbbox()
        if bb:
            self.over.append(dict(x=int(x) + bb[0], y=int(y) + bb[1], im=im.crop(bb)))

    def image(self, box, seed, label="Image", kind="rgb", biome=None, zoom=None, radius=None):
        x, y, w, h = [int(v) for v in box]
        im = self.T.photo(self, w, h, seed, kind, label, biome, zoom)
        r = self.T.radius if radius is None else radius
        self.pic(im, x, y, r, f"Image - {label}")
        return (x, y, w, h)

    # ---------------------------------------------------------------- export
    def composite(self):
        im = self.g.copy()
        for p in self.pics:
            if p["r"]:
                m = Image.new("L", p["im"].size, 0)
                ImageDraw.Draw(m).rounded_rectangle((0, 0, p["w"] - 1, p["h"] - 1), p["r"], fill=255)
                im.paste(p["im"], (p["x"], p["y"]), m)
            else:
                im.paste(p["im"], (p["x"], p["y"]))
        im = im.convert("RGBA")
        for o in self.over:                     # paste onto a full-size layer so off-slide parts are clipped
            layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
            layer.paste(o["im"], (o["x"], o["y"]))
            im.alpha_composite(layer)
        return im.convert("RGB")

    def render(self):
        im = self.composite()
        d = ImageDraw.Draw(im)
        T = self.T
        for t in self.texts:
            fo = fnt(T, t["f"], t["size"])
            if t["kind"] == "line":
                if t["track"]:
                    for i, ch in enumerate(t["s"]):
                        d.text((t["x"] + fo.getlength(t["s"][:i]) + i * t["track"] * t["size"], t["y"]), ch,
                               font=fo, fill=t["fill"])
                else:
                    d.text((t["x"], t["y"]), t["s"], font=fo, fill=t["fill"])
            elif t["kind"] == "block":
                for i, ln in enumerate(t["lines"]):
                    d.text((t["x"], t["y"] + i * int(t["size"] * t["lh"])), ln, font=fo, fill=t["fill"])
            else:
                sz = t["size"]
                for yy, lines in t["lay"]:
                    cy, r = yy + sz * 0.56, sz * 0.17
                    shape = d.ellipse if T.bullet in ("●", "•") else d.rectangle
                    shape((t["x"], cy - r, t["x"] + 2 * r, cy + r), fill=T.acc)
                    for i, ln in enumerate(lines):
                        d.text((t["x"] + t["ind"], yy + i * int(sz * t["lh"])), ln, font=fo, fill=t["fill"])
        return im

    def to_pptx(self, prs):
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        add_pic(sl, self.g, 0, 0, W, H, name="Background", png=self.T.png)
        for p in self.pics:
            x0, y0 = max(0, p["x"]), max(0, p["y"])
            x1, y1 = min(W, p["x"] + p["w"]), min(H, p["y"] + p["h"])
            im = p["im"].crop((x0 - p["x"], y0 - p["y"], x1 - p["x"], y1 - p["y"]))
            add_pic(sl, im, x0, y0, x1 - x0, y1 - y0, radius=p["r"], name=p["name"], png=self.T.png)
        for o in self.over:
            add_pic(sl, o["im"], o["x"], o["y"], o["im"].width, o["im"].height, name="Decoration", png=True)
        for t in self.texts:
            add_text(sl, t, self.T)
        return sl


# ====================================================================== pptx helpers
def E(px):
    from pptx.util import Emu
    return Emu(int(round(px * EMU_PX)))


def add_pic(sl, im, x, y, w, h, radius=0, name=None, png=False):
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.oxml.ns import qn
    from lxml import etree
    buf = io.BytesIO()
    if im.mode == "RGBA":
        im.save(buf, "PNG", optimize=True)
    elif png:
        im.convert("RGB").save(buf, "PNG", optimize=True)
    else:
        im.convert("RGB").save(buf, "JPEG", quality=88)
    buf.seek(0)
    p = sl.shapes.add_picture(buf, E(x), E(y), E(w), E(h))
    if name:
        p.name = name
    if radius:
        p.auto_shape_type = MSO_SHAPE.ROUNDED_RECTANGLE      # keeps the rounding after Change Picture
        geom = p._element.spPr.find(qn("a:prstGeom"))
        av = geom.find(qn("a:avLst"))
        if av is None:
            av = etree.SubElement(geom, qn("a:avLst"))
        gd = etree.SubElement(av, qn("a:gd"))
        gd.set("name", "adj")
        gd.set("fmla", "val %d" % round(radius / min(w, h) * 100000))
    return p


_CAL = None


def dy(st, f, lh):
    """vertical correction (em) between PIL's text origin and the PowerPoint box top (see calibrate)."""
    global _CAL
    if _CAL is None:
        _CAL = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    c = _CAL.get(st.fonts[f])
    if not c:
        return -0.1
    return c["a"] + c["b"] * lh


def _run(p, s, t, st, size, fkey=None):
    from pptx.util import Pt
    from pptx.dml.color import RGBColor
    from pptx.oxml.ns import qn
    from lxml import etree
    fd = face(st.fonts[fkey or t.get("f", "B")])
    r = p.add_run()
    r.text = s
    r.font.name = fd["face"]
    r.font.bold = fd["bold"]
    r.font.italic = fd["italic"]
    r.font.size = Pt(size * PT_PX)
    r.font.color.rgb = RGBColor(*t["fill"])
    tr = t.get("track", 0)
    if tr:
        r.font._rPr.set("spc", str(int(round(tr * size * PT_PX * 100))))
    for tag in ("a:latin", "a:ea", "a:cs"):          # pin the face for every script
        el = r.font._rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(r.font._rPr, qn(tag))
        el.set("typeface", fd["face"])
    return r


def _frame(sl, x, y, w, h, wrap_):
    from pptx.enum.text import MSO_AUTO_SIZE
    tb = sl.shapes.add_textbox(E(x), E(y), E(w), E(h))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = wrap_
    tf.auto_size = MSO_AUTO_SIZE.NONE
    return tf


def add_text(sl, t, st):
    from pptx.util import Pt
    from pptx.enum.text import PP_ALIGN
    from pptx.oxml.ns import qn
    from lxml import etree
    sz = t["size"]
    f = t.get("f", "B")
    if t["kind"] == "line":
        pitch = sz * LINE
        slack = sz * 0.5
        x = t["x"] - {"r": slack, "m": slack / 2}.get(t["align"], 0)
        tf = _frame(sl, x, t["y"] + dy(st, f, LINE) * sz, t["w"] + slack, pitch, False)
        p = tf.paragraphs[0]
        p.line_spacing = Pt(pitch * PT_PX)
        p.alignment = {"r": PP_ALIGN.RIGHT, "m": PP_ALIGN.CENTER}.get(t["align"], PP_ALIGN.LEFT)
        _run(p, t["s"], t, st, sz)
    elif t["kind"] == "block":
        pitch = int(sz * t["lh"])
        tf = _frame(sl, t["x"], t["y"] + dy(st, f, t["lh"]) * sz, t["w"] + sz * 0.3, pitch * len(t["lines"]), True)
        p = tf.paragraphs[0]
        p.line_spacing = Pt(pitch * PT_PX)
        _run(p, t["s"], t, st, sz)
    else:
        pitch = int(sz * t["lh"])
        last_y = t["lay"][-1][0] + pitch * len(t["lay"][-1][1])
        tf = _frame(sl, t["x"], t["y"] + dy(st, "B", t["lh"]) * sz, t["w"] + sz * 0.3, last_y - t["y"], True)
        for i, it in enumerate(t["items"]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.line_spacing = Pt(pitch * PT_PX)
            if i:
                p.space_before = Pt(int(sz * t["gap"]) * PT_PX)
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(t["ind"] * EMU_PX)))
            pPr.set("indent", str(-int(t["ind"] * EMU_PX)))
            clr = etree.SubElement(pPr, qn("a:buClr"))
            etree.SubElement(clr, qn("a:srgbClr")).set("val", "%02X%02X%02X" % st.acc)
            etree.SubElement(pPr, qn("a:buSzPct")).set("val", "60000")
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
            etree.SubElement(pPr, qn("a:buChar")).set("char", st.bullet)
            _run(p, it, dict(t, f="B"), st, sz)


# ====================================================================== calibration
def calibrate(files):
    """Place 'HHH' in each font in PowerPoint at two line pitches, export, and measure where the ink lands."""
    import tempfile
    import win32com.client
    from pptx import Presentation
    from pptx.util import Emu, Pt

    class _S:
        fonts = {}
    size, y0, LHS = 200, 600, (1.2, 1.4)       # the pitches the slides actually use (lines 1.2, blocks 1.2-1.38)
    files = sorted(set(files))
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
    for f in files:
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        _S.fonts = {"X": f}
        for k, lh in enumerate(LHS):
            tf = _frame(sl, 200 + k * 1800, y0, 1500, int(size * lh), True)
            p = tf.paragraphs[0]
            p.line_spacing = Pt(int(size * lh) * PT_PX)
            _run(p, "HHH", dict(fill=(0, 0, 0), f="X"), _S, size)
    tmp = Path(tempfile.mkdtemp())
    pp = tmp / "cal.pptx"
    prs.save(pp)
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(pp), True, False, False)
    out = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    for i, f in enumerate(files, 1):
        png = tmp / f"cal{i}.png"
        pres.Slides(i).Export(str(png), "PNG", W, H)
        a = np.asarray(Image.open(png).convert("L")) < 128
        fo = ImageFont.truetype(str(FONTS / f), size)
        pil_top = y0 + fo.getbbox("H")[1]
        vals = []
        for k in range(2):
            rows = np.where(a[:, 200 + k * 1800: 200 + k * 1800 + 1500].any(1))[0]
            vals.append((pil_top - rows[0]) / size)
        b = (vals[1] - vals[0]) / (LHS[1] - LHS[0])
        out[f] = dict(a=vals[0] - b * LHS[0], b=b)
        print(f, ["%.3f" % v for v in vals])
    pres.Close()
    CALIB.write_text(json.dumps(out, indent=1))
