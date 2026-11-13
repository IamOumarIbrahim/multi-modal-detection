"""Restyle the finished talk (D:/Downloads/Causal Temporal Filtering.pptx) in the Mixed Media look - in place.

Nothing of the talk is rebuilt: every text box, equation (OMML), video, bullet and position stays as authored.
Only the look changes:
  * slide background  -> Mixed Media paper (grid, footer stroke, small collage marks in empty corners)
  * fonts             -> headings Playfair Display Black; body text keeps its own font, equations Cambria Math
  * title subtitle    -> the blue second title line becomes the look's blue, in italic, with a brush underline
  * charts/diagrams   -> printed onto paper (multiply: white becomes paper, colours untouched)
  * photos, videos,   -> pixels untouched; mounted as torn paper prints with tape
    QR code
  * cover             -> the Mixed Media collage cover (halftone tile, riso thermal, a rendered drone print)

Run:  python -I mm_restyle.py [path/to/talk.pptx]
"""
import io
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from lxml import etree
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import art                                                    # noqa: E402
from engine import Slide, W, H, M, ROOT, EMU_PX, add_pic, add_text, fnt, tlen   # noqa: E402
from styles import Collage, fit_into, rgba                    # noqa: E402

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:/Downloads/Causal Temporal Filtering.pptx")
OUT = ROOT / "talk" / "Causal Temporal Filtering - Mixed Media.pptx"
PREV = ROOT / "talk" / "mixed_media"
DRONE = ROOT / "assets" / "drone_render.png"
LOGO_U = ROOT / "talk" / "source" / "s01_23.png"
LOGO_I = ROOT / "talk" / "source" / "s01_17.png"

NS = dict(p="http://schemas.openxmlformats.org/presentationml/2006/main",
          a="http://schemas.openxmlformats.org/drawingml/2006/main",
          r="http://schemas.openxmlformats.org/officeDocument/2006/relationships")
q = lambda t: "{%s}%s" % (NS[t.split(":")[0]], t.split(":")[1])

T = Collage()
BLUE = "%02X%02X%02X" % T.blue
HEAD, BODY = "Playfair Display Black", "Inter"
# what each picture is, by slide (photos/videos/QR keep their pixels; charts are printed on paper)
CHART_SLIDES = {2, 6, 7, 8, 9, 11, 12, 14}


# ====================================================================== helpers
def px(emu):
    return int(round(emu / EMU_PX))


def png_bytes(im, fmt="PNG"):
    b = io.BytesIO()
    im.save(b, fmt, **({"quality": 90} if fmt == "JPEG" else {"optimize": True}))
    b.seek(0)
    return b


def set_bg(slide, im):
    _, rid = slide.part.get_or_add_image_part(png_bytes(im.convert("RGB"), "JPEG"))
    cSld = slide._element.find(q("p:cSld"))
    old = cSld.find(q("p:bg"))
    if old is not None:
        cSld.remove(old)
    bg = etree.fromstring(
        f'<p:bg xmlns:p="{NS["p"]}" xmlns:a="{NS["a"]}" xmlns:r="{NS["r"]}"><p:bgPr>'
        f'<a:blipFill dpi="0" rotWithShape="1"><a:blip r:embed="{rid}"/><a:srcRect/><a:stretch><a:fillRect/>'
        f'</a:stretch></a:blipFill><a:effectLst/></p:bgPr></p:bg>')
    cSld.insert(0, bg)


def shape_box(el):
    off = el.find(".//" + q("a:xfrm") + "/" + q("a:off"))
    ext = el.find(".//" + q("a:xfrm") + "/" + q("a:ext"))
    if off is None or ext is None:
        return None
    return px(int(off.get("x"))), px(int(off.get("y"))), px(int(ext.get("cx"))), px(int(ext.get("cy")))


def is_video(el):
    return el.find(".//" + q("a:videoFile")) is not None


def restyle_text(slide):
    root = slide._element
    for c in root.iter(q("a:srgbClr")):
        if c.get("val") == "0070C0":
            c.set("val", BLUE)
            rpr = c.getparent().getparent()
            if rpr.tag in (q("a:rPr"), q("a:endParaRPr")):
                rpr.set("i", "1")
    for sp in root.iter(q("p:sp")):                      # titles: Playfair Black has its own weight - no fake bold
        ph = sp.find(".//" + q("p:ph"))
        if ph is not None and ph.get("type") in ("title", "ctrTitle"):
            for r in sp.iter(q("a:rPr"), q("a:endParaRPr")):
                r.set("b", "0")
                r.set("spc", "-20")
            # PowerPoint's renderer fails on the upright Playfair Black '&' at title size; the italic one is
            # fine (and the classic typographic choice), so '&' gets a run of its own in italic
            import copy
            for run in list(sp.iter(q("a:r"))):
                t = run.find(q("a:t"))
                if t is None or "&" not in (t.text or "") or len(t.text) == 1:
                    continue
                parts = t.text.split("&")
                for k, part in enumerate(parts):
                    for txt, ital in ([(part, None)] + ([("&", "1")] if k < len(parts) - 1 else [])):
                        if not txt:
                            continue
                        nr = copy.deepcopy(run)
                        nr.find(q("a:t")).text = txt
                        if ital:
                            nr.find(q("a:rPr")).set("i", ital)
                        run.addprevious(nr)
                run.getparent().remove(run)


def paper_print(blob, seed):
    """Print a chart onto paper: multiply, so white becomes paper and every colour stays where it was."""
    im = Image.open(io.BytesIO(blob)).convert("RGBA")
    wb = Image.new("RGBA", im.size, (255, 255, 255, 255))
    wb.alpha_composite(im)
    a = np.asarray(wb.convert("RGB"), np.float32)
    pap = np.asarray(art.paper(im.width, im.height, (252, 250, 245), seed, fiber=2.0, mottle=2.5), np.float32)
    return Image.fromarray(np.clip(pap * a / 255, 0, 255).astype(np.uint8), "RGB")


def mount(slide, el, seed, tape=True, avoid=None):
    """A torn paper print behind a picture/video, tape over its top edge. The picture itself is not touched."""
    x, y, w, h = shape_box(el)
    b = int(min(w, h) * 0.035) + 12
    pc = art.print_card(art.paper(w + 2 * b, h + 2 * b, (252, 250, 245), seed, fiber=2.5, mottle=3), seed=seed,
                        torn=True, shadow=True, shadow_a=70)
    pad = (pc.width - (w + 2 * b)) // 2
    card = add_pic(slide, pc, x - b - pad, y - b - pad, pc.width, pc.height, name="Decoration - print")
    el.addprevious(card._element)
    if tape:
        tw = int(min(320, max(150, w * 0.24)))
        tp = art.tape(tw, 62, seed).rotate(-6 if seed % 2 else 6, Image.BICUBIC, expand=True)
        ty = y - b - tp.height // 2 + 6
        for tx in (x + (w - tp.width) // 2, x + w - tp.width + b, x - b):      # centre, else a corner
            if avoid is None or avoid((tx, ty, tx + tp.width, ty + tp.height)):
                break
        else:
            return
        t = add_pic(slide, tp, tx, ty, tp.width, tp.height, name="Decoration - tape")
        el.addnext(t._element)


def free(rect, boxes, m=40):
    x0, y0, x1, y1 = rect
    for (x, y, w, h) in boxes:
        if x < x1 + m and x + w > x0 - m and y < y1 + m and y + h > y0 - m:
            return False
    return True


def background(n, boxes, brush=None):
    s = Slide(T, n, chrome=False)
    T.footer(s, H - 128)
    d = s.d
    if free((W - 640, 0, W, 560), boxes):
        art.concentric(d, W - 250, 250, 230, T.blue, 6, 11)
        d.ellipse((W - 600, 300, W - 420, 480), fill=T.acc)
    elif free((W - 520, H - 560, W, H - 150), boxes):
        art.concentric(d, W - 260, H - 360, 170, T.blue, 5, 10)
    if brush:                                            # under the blue title line, found in a first export
        x0, y0, x1, y1 = brush
        L = art.brush_stroke(int(min(1100, (x1 - x0) * 1.05)), int((y1 - y0) * 0.30), T.acc, n, a=200)
        s.g.paste(L, (int(x0 - 16), int(y1 - (y1 - y0) * 0.02)), L)
    return s


def page_no(slide, n, total):
    s = Slide(T, n, chrome=False)
    s.mono(W - M, H - 86, f"{n:02d} / {total}", 28, anchor="ra")
    s.mono(M, H - 86, "Causal Temporal Filtering", 28)
    for t in s.texts:
        add_text(slide, t, T)


# ====================================================================== cover
def drone_print(w, h, seed):
    """The rendered quadcopter as a Mixed Media print: black halftone body, blue riso propeller discs."""
    dr = Image.open(DRONE).convert("RGBA")
    dr = fit_into(dr, int(w * 0.9), int(h * 0.8)) if dr.width > w * 0.9 or dr.height > h * 0.8 else \
        dr.resize((int(dr.width * min(w * 0.9 / dr.width, h * 0.8 / dr.height)),
                   int(dr.height * min(w * 0.9 / dr.width, h * 0.8 / dr.height))), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (250, 246, 238, 255))
    canvas.alpha_composite(dr, ((w - dr.width) // 2, (h - dr.height) // 2 + int(h * 0.02)))
    al = np.zeros((h, w), np.float32)
    al[(h - dr.height) // 2 + int(h * 0.02):(h - dr.height) // 2 + int(h * 0.02) + dr.height,
       (w - dr.width) // 2:(w - dr.width) // 2 + dr.width] = np.asarray(dr.getchannel("A"), np.float32) / 255
    rgb = canvas.convert("RGB")
    D = art.stretch(1 - art.lum(rgb), 1, 99.8, 0.95) * (al > 0.95)
    cell = max(7, int(min(w, h) / 85))
    out = art.halftone(D, cell, 45, (22, 22, 22), (250, 246, 238)).convert("RGBA")
    prop = np.clip((al > 0.05) & (al < 0.95), 0, 1) * 0.55
    out.alpha_composite(art.halftone(prop, cell, 75, T.blue, (0, 0, 0), alpha=True))
    return out.convert("RGB")


def logos(s, x, y, hgt):
    u = Image.open(LOGO_U).convert("RGBA")
    a = np.asarray(u).copy()
    a[a[..., :3].min(-1) > 200, :3] = T.fg                # white parts -> ink on the light paper
    u = Image.fromarray(a, "RGBA")
    u = u.resize((int(u.width * hgt / u.height), hgt), Image.LANCZOS)
    s.pic(u, x, y, 0, "Logo - University of Sharjah")
    ic = Image.open(LOGO_I).convert("RGBA")
    wb = Image.new("RGBA", ic.size, (255, 255, 255, 255))
    wb.alpha_composite(ic)
    ih = int(hgt * 0.62)
    ic = wb.convert("RGB").resize((int(ic.width * ih / ic.height), ih), Image.LANCZOS)
    plate = Image.new("RGB", (ic.width + 40, ic.height + 30), (255, 255, 255))
    plate.paste(ic, (20, 15))
    s.pic(plate, x + u.width + 60, y + (hgt - plate.height) // 2, 0, "Logo - ICSPIS")


def cover_slide():
    s = Slide(T, 1, chrome=False)
    d = s.d
    art.concentric(d, 3420, 330, 280, T.blue, 7, 12)
    L = art.brush_stroke(1500, 260, T.acc3, 3)
    s.g.paste(L, (1900, 1700), L)
    d.ellipse((2050, 140, 2470, 560), fill=T.acc)
    s.image((2120, 260, 1500, 1500), 2, "Cover image", "rgb", "desert", 1.25)
    s.image((1700, 1180, 900, 800), 4, "Thermal", "th", "desert", 1.25)
    w, h = 900, 700
    im = T.frame(drone_print(*T.inner(w, h), 5), w, h, 5, "drone")
    im = im.rotate(-4, Image.BICUBIC, expand=True)
    s.pic(im, 2920 - (im.width - w) // 2, 1240 - (im.height - h) // 2, 0, "Image - Drone")
    mw = 1420
    s.mono(M, 150, "Department of Computer Engineering", 32, "fg")
    s.mono(M, 198, "University of Sharjah, Sharjah, UAE", 32)
    sz = 200
    while max(tlen(t, fnt(T, f, sz), T.track.get(f, 0), sz) for t, f in (("Causal Temporal", "H"),
                                                                        ("Filtering", "HI"))) > mw:
        sz -= 4
    y2 = 470 + int(sz * 1.12)
    s.head(M, 470, "Causal Temporal", sz)
    s.head(M, y2, "Filtering", sz, f="HI")
    yy = s.block(M, y2 + int(sz * 1.32), mw, "for Reliable Alerts in Dual-Stream UAV Search and Rescue Detection",
                 62, "BB", lh=1.22)
    s.block(M, yy + 70, mw, "Oumar Mamoun Ibrahim\nMohamad Khairi bin Ishak", 44, lh=1.35)
    logos(s, M, 1720, 200)
    return s


def put_cover(slide):
    tree = slide.shapes._spTree
    for el in list(tree):
        if el.tag not in (q("p:nvGrpSpPr"), q("p:grpSpPr")):
            tree.remove(el)
    s = cover_slide()
    set_bg(slide, s.g)
    for p_ in s.pics:
        add_pic(slide, p_["im"], p_["x"], p_["y"], p_["w"], p_["h"], name=p_["name"])
    for o in s.over:
        add_pic(slide, o["im"], o["x"], o["y"], o["im"].width, o["im"].height, name="Decoration", png=True)
    for t in s.texts:
        add_text(slide, t, T)
    return s


# ====================================================================== theme
def set_theme_fonts(prs):
    for part in prs.part.package.iter_parts():
        if str(part.partname).startswith("/ppt/theme/theme"):
            x = etree.fromstring(part.blob)
            for tag, face in (("a:majorFont", HEAD),):     # body font stays the talk's own
                for f in x.iter(q(tag)):
                    f.find(q("a:latin")).set("typeface", face)
            part._blob = etree.tostring(x, xml_declaration=True, encoding="UTF-8", standalone=True)


# ====================================================================== build
def build(brushes=None, inks=None):
    from pptx import Presentation
    tmp = Path(tempfile.mkdtemp()) / "src.pptx"
    shutil.copy(SRC, tmp)                                 # never open the user's own file for writing
    prs = Presentation(tmp)
    set_theme_fonts(prs)
    total = len(prs.slides)
    for n, slide in enumerate(prs.slides, 1):
        if n == 1:
            put_cover(slide)
            continue
        restyle_text(slide)
        tree = slide.shapes._spTree
        boxes = [b for el in tree if (b := shape_box(el))]
        ink = (inks or {}).get(n)

        def no_text(r, ink=ink):                         # tape must not cover words: check the first export
            if ink is None:
                return True
            x0, y0, x1, y1 = [int(v) for v in r]
            reg = ink[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
            return reg.size == 0 or (reg < 90).mean() < 0.003
        pics = [el for el in tree if el.tag == q("p:pic")]
        for i, el in enumerate(pics):
            seed = n * 10 + i
            if not is_video(el) and n in CHART_SLIDES:
                blip = el.find(".//" + q("a:blip"))
                rid = blip.get(q("r:embed"))
                blob = slide.part.related_part(rid).blob
                _, nrid = slide.part.get_or_add_image_part(png_bytes(paper_print(blob, seed)))
                blip.set(q("r:embed"), nrid)
            mount(slide, el, seed, avoid=no_text)
        set_bg(slide, background(n, boxes, (brushes or {}).get(n)).g)
        page_no(slide, n, total)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


def export(pptx, folder, size=(3840, 2160)):
    import win32com.client
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("*.jpg"):
        f.unlink()
    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(pptx), True, False, False)
    outs = []
    try:
        for i in range(1, pres.Slides.Count + 1):
            f = folder / f"{i:02d}.png"
            f.unlink(missing_ok=True)
            pres.Slides(i).Export(str(f), "PNG", *size)
            outs.append(f)
    finally:
        try:
            pres.Close()                                  # only the copy we opened
        except Exception:
            pass
    return outs


def find_blue_lines(pngs):
    """Box of the blue (italic) title line on each slide, in slide px."""
    out = {}
    for i, f in enumerate(pngs, 1):
        a = np.asarray(Image.open(f).convert("RGB"), np.int16)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        m = (b > 200) & (r < 90) & (g > 70) & (g < 150)
        m[int(H * 0.45):] = False
        m[:, int(W * 0.62):] = False
        ys, xs = np.nonzero(m)
        if len(xs) > 400:
            out[i] = (int(np.percentile(xs, 0.5)), int(np.percentile(ys, 0.5)),
                      int(np.percentile(xs, 99.5)), int(np.percentile(ys, 99.5)))
    return out


if __name__ == "__main__":
    work = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "talk" / "_work"
    p = build()
    first = export(p, work / "pass1")
    blues = find_blue_lines(first)
    inks = {i: np.asarray(Image.open(f).convert("L")) for i, f in enumerate(first, 1)}
    p = build(blues, inks)
    pngs = export(p, work / "pass2")
    PREV.mkdir(parents=True, exist_ok=True)
    for f in PREV.glob("*.jpg"):
        f.unlink()
    for f in pngs:
        Image.open(f).convert("RGB").save(PREV / f"{f.stem}.jpg", quality=90)
    print("saved", p)
