"""The 40 slide layouts, shared by every style.

Image slots ask for a kind of the paper's material and the style renders it in its medium:
  rgb / th   a UAV tile (RGB or pseudo-thermal) with the person marked
  neg        an empty tile with a crossed-out false alert
  strip      consecutive frames of one clip (the temporal view)
  trace      the real per-frame confidence: raw flicker vs the filtered signal, alert ticks below
  scene      the drone-footprint render
"""
from engine import Slide, W, H, M, N

CT = 560            # content top
CB = 1900           # content bottom
SEC = ["Section title one", "Section title two", "Section title three", "Section title four",
       "Section title five", "Section title six"]


def cols(n, gap=60, x0=M, x1=W - M):
    w = (x1 - x0 - (n - 1) * gap) / n
    return [(int(x0 + i * (w + gap)), int(w)) for i in range(n)]


def divider(T, n, num, title, sub, seed, kind="rgb", biome=None):
    s = Slide(T, n, title)
    if getattr(T, "divider_full", False):
        s.image((1980, 0, W - 1980, H - 160), seed, "Section image", kind, biome, radius=0)
    else:
        s.image((1980, 160, W - 1980 - M, H - 360), seed, "Section image", kind, biome)
    T.divider_deco(s, M, 330)
    s.text(M - 14, 330, num, 720, "D", getattr(T, "num_col", "acc"))
    s.head(M, 1290, title, 190)
    s.block(M, 1560, 1600, sub, 46, fill="mute")
    return s


def numbered_rows(s, x, y, w, items, rh=180, num_size=58, size=58):
    for i, (a, b) in enumerate(items):
        yy = y + i * rh
        s.text(x, yy + 24, f"{i + 1:02d}", num_size, "D", "acc")
        s.text(x + 170, yy + 18, a, size, "BB")
        s.mono(x + 172, yy + 24 + int(size * 1.3), b, 28)
        s.rule(x, yy + rh - 2, x + w)
    return y + len(items) * rh


def table(s, x, y, headers, rows, widths, rh=140, size=44, hi_row=None):
    cx = [x]
    for w_ in widths:
        cx.append(cx[-1] + w_)
    for j, c in enumerate(headers):
        s.mono(cx[j], y, c, 30)
    y += 66
    s.rule(x, y, cx[-1], "fg", 4)
    for i, r in enumerate(rows):
        hi = hi_row == i
        if hi:
            s.d.rectangle((x, y + 4, cx[-1], y + rh), fill=s.T.acc)
        for j, v in enumerate(r):
            s.block(cx[j] + (24 if hi and j == 0 else 0), y + int((rh - size * 1.2) / 2) - 2, widths[j] - 40, v,
                    size, "BB" if j == 0 else "B", lh=1.2, fill="on_acc" if hi else "fg")
        y += rh
        s.rule(x, y, cx[-1])
    return y


def big_number(s, x, y, num, unit, label, size=260, w=900):
    nw = s.text(x - 6, y, num, size, "D", "fg")
    if unit:
        s.mono(x + nw + 24, y + int(size * 0.12), unit, 34, "acc")
    sz = int(size * s.T.scale.get("D", 1))
    s.block(x, y + int(sz * 1.22), w, label, 40, fill="mute")


# ====================================================================== the 40 slides
def s01(T):
    s = Slide(T, 1, chrome=False)
    T.cover(s)
    s.mono(M, 150, "Course / event name", 34, "fg")
    s.mono(M, 200, "Team  /  Term  /  Organisation", 34)
    s.head(M, 560, "Presentation", 270)
    s.head(M, 560 + 300, "title goes", 270, f="HI")
    s.head(M, 560 + 600, "here.", 270)
    s.tag(M + 6, 1650, "With a short subtitle line", 38)
    s.block(M, 1830, 1700, "Presenter One  /  Presenter Two  /  Presenter Three  /  Presenter Four", 38, fill="fg")
    s.mono(M, 1925, "Supervisor: Name  //  Department", 28)
    return s


def s02(T):
    s = Slide(T, 2, "Agenda")
    s.title("Agenda", "Today")
    numbered_rows(s, M, CT + 10, 1900, [(t.capitalize(), "One-line description of the section") for t in SEC], 210)
    s.image((2300, CT - 230, W - M - 2300, 1560), 1, "Agenda image", "rgb", "forest")
    return s


def s03(T):
    return divider(T, 3, "01", "Section title", "A short subtitle that sets up what this section covers.", 31,
                   "rgb", "desert")


def s04(T):
    s = Slide(T, 4, SEC[0])
    s.mono(M, 300, "Kicker label", 34, "acc")
    s.head(M, 380, "A big statement that", 150)
    s.head(M, 380 + 180, "frames the slide in", 150, f="HI")
    s.head(M, 380 + 360, "three short lines.", 150)
    s.bullets(M, 1180, 1900, ["Supporting point one, written as a full sentence that can wrap onto a second line.",
                              "Supporting point two, kept short.",
                              "Supporting point three, explaining the consequence."], 48)
    s.image((2350, 300, W - M - 2350, 1000), 2, "Image", "rgb", "snow")
    s.image((2700, 1360, W - M - 2700, 520), 3, "Detail", "neg", "desert")
    s.hw(2350, 1420, "add a note here", 64)
    return s


def s05(T):
    s = Slide(T, 5, SEC[0])
    s.title("Key numbers", "Kicker label")
    for i, (x, w) in enumerate(cols(3, 80)):
        if i:
            s.vrule(x - 40, CT, CT + 640)
        big_number(s, x + (40 if i else 0), CT + 20, ["00%", "00×", "00.0"][i], "unit",
                   "Short label that explains what this number means", 280, w - 80)
    s.image((M, 1420, W - 2 * M, 440), 4, "Wide image", "trace")
    return s


def s06(T):
    s = Slide(T, 6, SEC[0])
    s.title("Four-card grid", "Kicker label")
    cw, chh, g = 1060, 600, 60
    for i in range(4):
        x, y = M + (i % 2) * (cw + g), CT + (i // 2) * (chh + g)
        s.card((x, y, x + cw, y + chh))
        s.mono(x + 56, y + 56, f"0{i + 1}", 30, "acc")
        s.head(x + 56, y + 120, f"Card title {i + 1}", 72, f="BB")
        s.block(x + 56, y + 250, cw - 112, "Two short sentences that explain this point and why it matters to the "
                                           "audience.", 42, fill="mute")
    s.image((M + 2 * cw + g + 100, CT, W - M - (M + 2 * cw + g + 100), 2 * chh + g), 6, "Image", "th", "forest")
    return s


def s07(T):
    s = Slide(T, 7, SEC[0])
    s.title("Three pillars headline", "Kicker label")
    for i, (x, w) in enumerate(cols(3, 80)):
        s.image((x, CT, w, 700), 7 + i, f"Pillar {i + 1}", "rgb", ["desert", "forest", "snow"][i])
        s.mono(x, CT + 760, f"0{i + 1}  /  Pillar", 30, "acc")
        s.head(x, CT + 820, "Pillar headline", 72, f="BB")
        s.block(x, CT + 940, w - 40, "Two lines of supporting text describing this pillar in plain words.", 42,
                fill="mute")
    return s


def s08(T):
    s = Slide(T, 8, SEC[0])
    s.image((M, 240, 1700, CB - 240), 10, "Image", "rgb", "desert")
    x = M + 1700 + 140
    s.title("Image + numbered list", "Kicker label", x=x)
    numbered_rows(s, x, CT + 30, W - M - x, [(f"List item {i + 1}", "Short supporting line for this item")
                                              for i in range(5)], 240)
    return s


def s09(T):
    return divider(T, 9, "02", "Section title", "A short subtitle that sets up what this section covers.", 41,
                   "th", "snow")


def s10(T):
    s = Slide(T, 10, SEC[1])
    s.title("Feature matrix", "Kicker label")
    x0, lw, cw = M, 900, (W - 2 * M - 900) // 6
    for j in range(6):
        s.mono(x0 + lw + j * cw + cw // 2, CT, f"Criterion {j + 1}", 28, anchor="ma")
    y = CT + 70
    s.rule(x0, y, W - M, "fg", 4)
    marks = [[2, 0, 1, 0, 0, 2], [0, 1, 2, 0, 0, 0], [1, 2, 1, 0, 0, 2], [2, 2, 3, 3, 0, 0],
             [2, 1, 3, 3, 0, 0], [2, 2, 2, 2, 2, 2]]
    rh = 170
    T_ = s.T
    for i, row in enumerate(marks):
        hi = i == 5
        if hi:
            s.d.rectangle((x0, y + 4, W - M, y + rh), fill=T_.acc)
        s.text(x0 + (24 if hi else 0), y + 56, "Our option" if hi else f"Option {'ABCDE'[i]}", 50, "BB",
               "on_acc" if hi else "fg")
        for j, m in enumerate(row):
            cx, cy, r = x0 + lw + j * cw + cw // 2, y + rh // 2, 30
            col = T_.on_acc if hi else T_.fg
            box = (cx - r, cy - r, cx + r, cy + r)
            if m == 2:
                s.d.rectangle(box, fill=col)
            elif m == 1:
                s.d.rectangle(box, outline=col, width=5)
                s.d.rectangle((cx - r, cy - r, cx, cy + r), fill=col)
            elif m == 0:
                s.d.rectangle(box, outline=col, width=5)
            else:
                s.text(cx, cy - 34, "?", 60, "BB", col, "ma")
        y += rh
        s.rule(x0, y, W - M)
    for k, lab in enumerate(["Yes", "Partly", "No", "Not public"]):
        lx, cy, col = M + k * 520, y + 110, T_.fg
        b = (lx, cy - 18, lx + 36, cy + 18)
        if k == 0:
            s.d.rectangle(b, fill=col)
        elif k == 1:
            s.d.rectangle(b, outline=col, width=4)
            s.d.rectangle((lx, cy - 18, lx + 18, cy + 18), fill=col)
        elif k == 2:
            s.d.rectangle(b, outline=col, width=4)
        else:
            s.text(lx + 6, cy - 26, "?", 44, "BB")
        s.mono(lx + 70, cy - 14, lab, 28)
    return s


def s11(T):
    s = Slide(T, 11, SEC[1])
    s.image((M, 240, 1600, CB - 240 - 60), 12, "Image", "rgb", "snow")
    s.mono(M, CB - 30, "Image caption", 28)
    x = M + 1600 + 160
    s.title("Image + table", "Kicker label", x=x)
    y = table(s, x, CT + 20, ["Name", "Detail", "Note"], [[f"Item {i + 1}", "Attribute", "Why it falls short"]
                                                         for i in range(4)], [500, 600, W - M - x - 1100], 170)
    s.tag(x, y + 80, "Key takeaway")
    return s


def s12(T):
    s = Slide(T, 12, SEC[1])
    s.title("Concept explainer", "Kicker label")
    s.image((M, CT, 1450, CB - CT - 60), 13, "Concept image", "th", "desert")
    x = M + 1450 + 120
    s.image((x, CT, W - M - x, 480), 14, "Diagram strip", "trace")
    for i, (cx, cw) in enumerate(cols(4, 40, x, W - M)):
        s.card((cx, CT + 540, cx + cw, CT + 920), hi=i == 0)
        s.mono(cx + 36, CT + 580, f"Stage {i + 1}", 30, "on_acc" if i == 0 else "acc")
        s.block(cx + 36, CT + 650, cw - 72, "One line on what the stage does", 40, fill="on_acc" if i == 0 else "fg")
    for i, (cx, cw) in enumerate(cols(3, 40, x, W - M)):
        big_number(s, cx, CT + 1000, ["0.0M", "0.0", "00.0"][i], "", "Label for this number", 150, cw)
    return s


def s13(T):
    s = Slide(T, 13, SEC[1])
    s.title("Three formula cards", "Kicker label")
    for i, (x, w) in enumerate(cols(3, 70)):
        s.card((x, CT, x + w, CB - 40))
        s.mono(x + 60, CT + 60, f"Idea {i + 1}", 30, "acc")
        s.head(x + 60, CT + 120, "Name", 100)
        fb = (x + 60, CT + 330, x + w - 60, CT + 560)
        s.d.rectangle(fb, fill=s.T.acc)
        s.text(x + 100, CT + 375, "formula line one", 44, "M", "on_acc")
        s.text(x + 100, CT + 455, "formula line two", 44, "M", "on_acc")
        s.block(x + 60, CT + 640, w - 120, "Two sentences that explain the idea in plain words, and why it is "
                                          "needed here.", 44, fill="mute")
    return s


def s14(T):
    return divider(T, 14, "03", "Section title", "A short subtitle that sets up what this section covers.", 51,
                   "rgb", "forest")


def s15(T):
    s = Slide(T, 15, SEC[2])
    s.title("Specification table", "Kicker label")
    table(s, M, CT, ["Metric", "Target", "How it is measured"],
          [[f"Metric {i + 1}", "≥ 00.0 unit", "Method and conditions used to measure it"] for i in range(7)],
          [1100, 800, W - 2 * M - 1900], 168)
    return s


def s16(T):
    s = Slide(T, 16, SEC[2])
    s.title("Comparison table", "Kicker label")
    y, rh = CT, 230
    for i in range(5):
        hi = i == 0
        if hi:
            s.card((M, y, W - M, y + rh - 30), hi=True)
        fc = "on_acc" if hi else "fg"
        s.text(M + 50, y + 30, "ABCDE"[i], 120, "D", fc if hi else "acc")
        s.text(M + 300, y + 40, "Option one" if hi else f"Option {'ABCDE'[i]}", 62, "BB", fc)
        s.mono(M + 302, y + 130, "Short description of how it works", 28, fc if hi else "mute")
        s.text(W - M - 50, y + 40, "0.00", 90, "D", fc, "ra")
        if not hi:
            s.rule(M, y + rh - 30, W - M)
        y += rh
    s.mono(M, y + 20, "Footnote explaining how the scores were calculated", 28)
    return s


def s17(T):
    s = Slide(T, 17, SEC[2])
    s.title("Three product columns", "Kicker label")
    for i, (x, w) in enumerate(cols(3, 80)):
        s.image((x, CT, w, 640), 17 + i, f"Product {i + 1}", ["rgb", "th", "neg"][i], ["snow", "snow", "forest"][i])
        s.mono(x, CT + 700, f"Part {i + 1}", 30, "acc")
        s.head(x, CT + 760, "Choice", 90)
        for k in range(3):
            s.mono(x, CT + 900 + k * 50, f"Spec {['one', 'two', 'three'][k]}  ·  value", 30, "fg")
        s.block(x, CT + 1080, w - 40, "What else was considered and why it lost.", 40, fill="mute")
        s.tag(x, CT + 1230, "Status", 30, hi=i == 0)
    return s


def s18(T):
    s = Slide(T, 18, SEC[2])
    s.title("System overview", "Kicker label")
    cs = cols(6, 70)
    for i, (x, w) in enumerate(cs):
        s.card((x, CT, x + w, CT + 300), hi=i == 2)
        fc = "on_acc" if i == 2 else "fg"
        s.mono(x + 36, CT + 40, f"0{i + 1}", 28, fc if i == 2 else "acc")
        s.text(x + 36, CT + 110, f"Step {i + 1}", 62, "BB", fc)
        s.mono(x + 36, CT + 210, "Short note", 28, fc if i == 2 else "mute")
        if i < 5:
            ax = x + w + 12
            s.d.polygon([(ax, CT + 135), (ax + 44, CT + 150), (ax, CT + 165)], fill=s.T.acc)
    for k, (x, w) in enumerate(cols(2, 120)):
        s.mono(x, CT + 400, ["Zone A", "Zone B"][k], 30, "acc")
        s.rule(x, CT + 460, x + w)
        s.bullets(x, CT + 500, w, [f"{['Edge', 'Cloud'][k]} detail {n}" for n in ("one", "two", "three")], 44)
    s.image((M, 1500, W - 2 * M, 360), 18, "Diagram strip", "strip")
    return s


def s19(T):
    s = Slide(T, 19, SEC[2])
    s.image((M, 240, 1700, CB - 240), 19, "Product image", "scene", zoom=1.4)
    x = M + 1700 + 140
    y = s.title("Hardware", "Kicker label", x=x)
    s.block(x, y, W - M - x, "One sentence that introduces the product or diagram on the left.", 46, fill="mute")
    yy = y + 140
    for i in range(6):
        s.mono(x, yy + 30, f"0{i + 1}", 30, "acc")
        s.text(x + 120, yy + 18, f"Component {i + 1}", 50, "BB")
        s.mono(x + 1000, yy + 30, "What it does", 30)
        yy += 130
        s.rule(x, yy, W - M)
    s.tag(x, yy + 60, "Key figure")
    return s


def s20(T):
    s = Slide(T, 20, SEC[2])
    s.title("Bar chart", "Kicker label")
    for i, f in enumerate([0.92, 0.62, 0.48, 0.4, 0.3, 0.2, 0.08]):
        y = CT + 20 + i * 180
        s.text(M, y + 22, f"Item {i + 1}", 46, "BB")
        s.bar(M + 500, y, 1900, 96, f, i, "0.0 s  /  peak 0.0")
    x = 3020
    big_number(s, x, CT + 20, "00.0", "unit", "Label", 200, 600)
    big_number(s, x, CT + 520, "0.0", "unit", "One sentence explaining the constraint behind the numbers.", 200, 640)
    return s


def s21(T):
    s = Slide(T, 21, SEC[2])
    s.title("Three-step headline", "Kicker label")
    for i, (x, w) in enumerate(cols(3, 70)):
        hi = i == 2
        s.card((x, CT, x + w, CT + 640), hi=hi)
        fc = "on_acc" if hi else "fg"
        s.mono(x + 60, CT + 60, f"Step {i + 1}", 30, fc if hi else "acc")
        for k in range(3):
            bx = x + w - 60 - (3 - k) * 46
            if k <= i:
                s.d.rectangle((bx, CT + 62, bx + 30, CT + 92), fill=s.c(fc))
            else:
                s.d.rectangle((bx, CT + 62, bx + 30, CT + 92), outline=s.c(fc), width=4)
        s.head(x + 60, CT + 140, "Step name", 100, fill=fc)
        s.block(x + 60, CT + 330, w - 120, "Describe what happens at this step in one or two sentences.", 44,
                fill=fc)
    s.mono(M, CT + 700, "Footnote with rules or conditions", 28)
    s.image((M, 1460, W - 2 * M, 400), 21, "Wide image", "strip")
    return s


def s22(T):
    s = Slide(T, 22, SEC[2])
    s.title("Four-up gallery", "Kicker label")
    kinds = [("rgb", "desert"), ("th", "desert"), ("rgb", "snow"), ("neg", "forest")]
    for i, (x, w) in enumerate(cols(4, 60)):
        s.image((x, CT, w, 1050), 22 + i, f"Class {i + 1}", *kinds[i])
        s.head(x, CT + 1100, f"Class {i + 1}", 72, f="BB")
        s.mono(x, CT + 1200, "Detail  //  detail", 28)
    return s


def s23(T):
    s = Slide(T, 23, SEC[2])
    s.title("Detail table", "Kicker label")
    table(s, M, CT, ["Row", "Col", "Col", "Col", "Column", "Long note"],
          [[f"Row {i + 1}", "0.00", "0.0", "0.0", "value", "A longer text cell that can wrap onto a second line."]
           for i in range(5)], [500, 360, 360, 360, 500, W - 2 * M - 2080], 220)
    s.mono(M, 1820, "Footnote", 28)
    return s


def s24(T):
    s = Slide(T, 24, SEC[2])
    s.image((M, 240, 1750, 1200), 24, "Screen", "th", "snow")
    s.card((M, 1520, M + 1750, CB), hi=True)
    s.text(M + 60, 1580, "Formula or key line", 54, "M", "on_acc")
    s.mono(M + 60, 1700, "Caption", 28, "on_acc")
    x = M + 1750 + 160
    s.mono(x, 280, "Kicker label", 34, "acc")
    s.head(x, 346, "Steps + screen", 140)
    s.head(x, 346 + 165, "headline line two", 140, f="HI")
    y = 820
    for i in range(5):
        s.d.rectangle((x, y + 6, x + 64, y + 70), outline=s.T.acc, width=4)
        s.text(x + 32, y + 12, str(i + 1), 40, "BB", "acc", "ma")
        s.text(x + 110, y, f"Step {i + 1}", 52, "BB")
        s.mono(x + 112, y + 76, "One-line description of the step", 28)
        y += 210
    return s


def s25(T):
    s = Slide(T, 25, SEC[2])
    s.title("Two screens", "Kicker label")
    for i, (x, w) in enumerate(cols(2, 100)):
        s.image((x, CT, w, 960), 25, f"Screen {i + 1}", ["rgb", "th"][i], "forest")      # same tile, two streams
        s.mono(x, CT + 1020, f"Screen {['one', 'two'][i]}  //  content", 30, "acc")
        s.block(x, CT + 1080, w - 100, "Two sentences describing what this screen shows and when the user sees it.",
                44, fill="mute")
    return s


def s26(T):
    s = Slide(T, 26, SEC[2])
    s.mono(M, 300, "Kicker label", 34, "acc")
    s.head(M, 380, "A short, bold", 200)
    s.head(M, 380 + 240, "statement.", 200, f="HI")
    s.image((2200, 240, W - M - 2200, 1050), 26, "Image", "rgb", "desert")
    for i, (x, w) in enumerate(cols(3, 100)):
        s.rule(x, 1420, x + w, "fg", 4)
        s.mono(x, 1470, f"Point {i + 1}", 30, "acc")
        s.block(x, 1540, w - 60, "Two or three sentences explaining the point in plain language.", 44, fill="mute")
    return s


def s27(T):
    return divider(T, 27, "04", "Section title", "A short subtitle that sets up what this section covers.", 61,
                   "neg", "snow")


def s28(T):
    s = Slide(T, 28, SEC[3])
    s.title("Four findings", "Kicker label")
    for i, (x, w) in enumerate(cols(4, 50)):
        hi = i == 3
        s.card((x, CT, x + w, CB - 40), hi=hi)
        fc = "on_acc" if hi else "fg"
        s.mono(x + 50, CT + 50, f"Study {i + 1}", 30, fc if hi else "acc")
        s.text(x + 44, CT + 110, "00.0", 200, "D", fc)
        s.mono(x + 50, CT + 360, "Unit", 30, fc if hi else "mute")
        s.block(x + 50, CT + 450, w - 100, "One or two sentences on what was measured and what was found.", 42,
                fill=fc)
        s.rule(x + 50, CB - 200, x + w - 50, fc if hi else "line")
        s.mono(x + 50, CB - 150, "→ Resulting decision", 28, fc if hi else "acc")
    return s


def s29(T):
    s = Slide(T, 29, SEC[3])
    s.image((M, 240, 1750, CB - 240 - 60), 29, "Image", "th", "forest")
    s.mono(M, CB - 30, "Image caption", 28)
    x = M + 1750 + 160
    y = s.title("Image + key numbers", "Kicker label", x=x)
    for i, (n, l) in enumerate([("0.00", "What this number means"), ("0.00–0.00", "What this range means"),
                                ("0 in 0", "What this ratio means")]):
        big_number(s, x, y + 40 + i * 380, n, "", l, 190, W - M - x)
    s.tag(x, CB - 70, "Resulting decision")
    return s


def s30(T):
    s = Slide(T, 30, SEC[3])
    s.title("Big numbers", "Kicker label")
    for i in range(4):
        x = M + (i % 2) * 1760
        y = CT + 20 + (i // 2) * 640
        s.rule(x, y, x + 1600, "fg", 4)
        big_number(s, x, y + 60, "00", "unit", "What this number means", 300, 1500)
    return s


def s31(T):
    s = Slide(T, 31, SEC[3])
    s.title("Strip + numbers", "Kicker label")
    s.image((M, CT, W - 2 * M, 640), 31, "Image strip", "strip")
    s.mono(M, CT + 680, "Caption", 28)
    for i, (x, w) in enumerate(cols(4, 80)):
        if i:
            s.vrule(x - 40, CT + 800, CB - 40)
        big_number(s, x + (30 if i else 0), CT + 820, ["0.0 h", "00,000", "0 unit", "000"][i], "", "Label", 150,
                   w - 60)
    return s


def s32(T):
    return divider(T, 32, "05", "Section title", "A short subtitle that sets up what this section covers.", 71,
                   "th", "forest")


def s33(T):
    s = Slide(T, 33, SEC[4])
    s.title("What exists today", "Kicker label")
    s.image((M, CT, 2000, CB - CT - 40), 33, "Image", "scene", zoom=1.1)
    x = M + 2000 + 140
    y = CT
    for i in range(6):
        s.mono(x, y + 28, f"0{i + 1}", 30, "acc")
        s.text(x + 110, y + 14, f"Module {i + 1}", 52, "BB")
        s.mono(x + 112, y + 96, "Technology  //  note", 28)
        y += 190
        s.rule(x, y - 20, W - M)
    s.tag(x, y + 30, "Key result")
    return s


def s34(T):
    s = Slide(T, 34, SEC[4])
    s.title("Results", "Kicker label")
    for i, n in enumerate(["0/0", "0/0", "0"]):
        big_number(s, M + i * 680, CT + 10, n, "", "Metric label", 260, 600)
    s.rule(M, CT + 560, M + 1950, "fg", 4)
    s.head(M, CT + 620, "A short finding headline.", 96, f="BB")
    s.block(M, CT + 780, 1900, "Two or three sentences explaining the finding, what changed, and what it means for "
                               "the next phase.", 46, fill="mute")
    s.hw(M, CT + 1060, "← the number that matters", 70)
    s.image((2400, 240, W - M - 2400, CB - 240), 34, "Image", "rgb", "snow")
    return s


def s35(T):
    s = Slide(T, 35, SEC[4])
    s.title("Bar chart", "Kicker label")
    for i, f in enumerate([0.95, 0.4, 0.16, 0.08, 0.05]):
        y = CT + 20 + i * 200
        s.text(M, y + 26, f"Stage {i + 1}", 46, "BB")
        s.bar(M + 440, y, 1500, 110, f, i + 3, "00 ms")
    s.block(M, CT + 1100, 2000, "Two sentences explaining what the chart shows and what was done about it.", 44,
            fill="mute")
    s.image((2500, CT, W - M - 2500, CB - CT - 40), 35, "Image", "neg", "desert")
    return s


def s36(T):
    s = Slide(T, 36, SEC[4])
    y = s.title("Photo strip", "Kicker label")
    s.block(M, y - 10, 2600, "One or two sentences describing what these images show.", 46, fill="mute")
    for i, (x, w) in enumerate(cols(4, 50)):
        s.image((x, 760, w, 1000), 36 + i, f"Photo {i + 1}", ["rgb", "th", "rgb", "neg"][i],
                ["forest", "forest", "desert", "snow"][i])
        s.mono(x, 1800, "Caption", 28)
    return s


def s37(T):
    return divider(T, 37, "06", "Section title", "A short subtitle that sets up what this section covers.", 81,
                   "rgb", "snow")


def s38(T):
    s = Slide(T, 38, SEC[5])
    s.title("Numbers + 2×2 grid", "Kicker label")
    for i, n in enumerate(["0000", "0000", "000"]):
        big_number(s, M, CT + i * 430, n, "unit", "Label", 200, 1000)
    x0, g = 1500, 50
    cw, chh = (W - M - x0 - g) // 2, (CB - 40 - CT - g) // 2
    for i in range(4):
        x, y = x0 + (i % 2) * (cw + g), CT + (i // 2) * (chh + g)
        s.card((x, y, x + cw, y + chh), hi=i == 0)
        fc = "on_acc" if i == 0 else "fg"
        s.mono(x + 50, y + 50, f"Quadrant {i + 1}", 30, fc if i == 0 else "acc")
        s.head(x + 50, y + 120, ["Strengths", "Weaknesses", "Opportunities", "Threats"][i], 80, fill=fc, f="BB")
        s.block(x + 50, y + 260, cw - 100, "Two sentences for this quadrant.", 42, fill=fc)
    return s


def s39(T):
    s = Slide(T, 39, SEC[5])
    s.title("Timeline + risks", "Kicker label")
    y = CT + 80
    s.rule(M, y, W - M, "fg", 4)
    for i, (x, w) in enumerate(cols(5, 60)):
        hi = i == 2
        r = 26
        if hi:
            s.d.rectangle((x, y - r, x + 2 * r, y + r), fill=s.T.acc)
        else:
            s.d.rectangle((x, y - r, x + 2 * r, y + r), fill=s.T.bg, outline=s.T.fg, width=5)
        s.text(x, y - 130, f"M{i + 1}", 64, "D", "acc" if hi else "fg")
        s.block(x, y + 70, w - 40, "Milestone description", 40, fill="mute")
    yy = CT + 460
    s.mono(M, yy, "Risks", 30, "acc")
    table(s, M, yy + 70, ["#", "Risk", "Mitigation"], [[f"0{i + 1}", "Risk name", "Mitigation in one line"]
                                                     for i in range(3)], [300, 1300, W - 2 * M - 1600], 170)
    return s


def s40(T):
    s = Slide(T, 40, chrome=False)
    T.thanks(s)
    s.mono(M, 150, "Project name  /  Team", 34, "fg")
    s.head(M, 600, "Thank", 380)
    s.head(M, 600 + 430, "you.", 380, f="HI")
    s.tag(M + 6, 1720, "Questions welcome", 38)
    s.mono(M, 1900, "Contact  //  Email  //  Repository", 30, "fg")
    s.hw(M + 1250, 1200, "questions?", 90)
    return s


SLIDES = [globals()[f"s{i:02d}"] for i in range(1, N + 1)]
NAMES = ["cover", "agenda", "section_01", "statement", "big_numbers", "four_card_grid", "three_pillars",
         "image_numbered_list", "section_02", "feature_matrix", "image_table", "concept_explainer",
         "three_formula_cards", "section_03", "spec_table", "comparison", "three_product_columns", "process",
         "product_list", "bar_chart", "three_steps", "four_up_gallery", "detail_table", "steps_screen",
         "two_screens", "statement_three_columns", "section_04", "four_cards", "image_key_numbers", "targets",
         "strip_numbers", "section_05", "image_list", "results", "bar_chart_side_image", "photo_strip",
         "section_06", "numbers_2x2_grid", "timeline_risks", "thank_you"]
