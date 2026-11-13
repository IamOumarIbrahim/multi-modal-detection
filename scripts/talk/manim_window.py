#   MMSAR - animated explainer: a 5-frame causal window sliding over detector confidences
#        Oumar Ibrahim U22200741

# render (needs manim >= 0.18, run from the repo root):
#   manim -qh --media_dir scratch/manim -o window_filter scripts/talk/manim_window.py WindowFilter
#   then copy scratch/manim/videos/manim_window/1080p60/window_filter.mp4 to figures/
# uses the same ten-frame example as tests/test_filters.py and figures/fig_filters_toy.png

from manim import (DOWN, RIGHT, UP, Create, FadeIn, FadeOut, Line, Rectangle, Scene, Text, Transform,
                   VGroup, Write, config)

config.background_color = "#ffffff"

CONF = [0.1, 0.2, 0.8, 0.1, 0.2, 0.6, 0.7, 0.8, 0.2, 0.1]
TAU = 0.5
W = 5
RAW = "#eb6834"
FILT = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
GREY = "#8a8984"
EMPTY = "#eeede8"

BAR_W = 0.62
STEP = 0.95
H = 3.2          # height of confidence 1.0
X0 = -4.3
BASE = -0.6


def x_of(i):
    return X0 + i * STEP


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


class WindowFilter(Scene):
    def construct(self):
        title = Text("A causal filter only looks back: the last 5 frames", font="Segoe UI", weight="BOLD",
                     color=INK).scale(0.6).to_edge(UP, buff=0.35)
        self.play(Write(title), run_time=0.8)

        # confidence bars
        bars = VGroup()
        for i, c in enumerate(CONF):
            r = Rectangle(width=BAR_W, height=max(c * H, 0.02), fill_color=RAW if c >= TAU else GREY,
                          fill_opacity=1, stroke_width=0)
            r.move_to([x_of(i), BASE + c * H / 2, 0])
            bars.add(r)
        idx = VGroup(*[Text(str(i), font="Segoe UI", color=INK2).scale(0.38).move_to([x_of(i), BASE - 0.25, 0])
                       for i in range(len(CONF))])
        tau = Line([x_of(0) - 0.5, BASE + TAU * H, 0], [x_of(9) + 0.5, BASE + TAU * H, 0], color=INK2, stroke_width=2)
        tau_lab = Text("τ = 0.5", font="Segoe UI", color=INK2).scale(0.4).next_to(tau, RIGHT, buff=0.1)
        self.play(FadeIn(bars, shift=UP * 0.2), FadeIn(idx), Create(tau), FadeIn(tau_lab), run_time=1.2)
        spike = Text("one-frame spike", font="Segoe UI", color=INK).scale(0.4).move_to([x_of(2), BASE + H + 0.6, 0])
        self.play(FadeIn(spike), run_time=0.5)

        # decision rows
        rows_y = {"raw": -2.15, "med": -2.95}
        labels = VGroup(
            Text("Raw threshold", font="Segoe UI", color=INK).scale(0.4).move_to([x_of(0) - 1.55, rows_y["raw"], 0]),
            Text("Median of 5", font="Segoe UI", color=INK).scale(0.4).move_to([x_of(0) - 1.55, rows_y["med"], 0]),
        )
        cells = {k: VGroup(*[Rectangle(width=BAR_W, height=0.48, fill_color=EMPTY, fill_opacity=1, stroke_width=0)
                             .move_to([x_of(i), y, 0]) for i in range(len(CONF))]) for k, y in rows_y.items()}
        self.play(FadeIn(labels), FadeIn(cells["raw"]), FadeIn(cells["med"]), run_time=0.6)

        window = Rectangle(width=STEP * 1 - 0.1, height=H + 0.5, stroke_color=FILT, stroke_width=4)
        window.move_to([x_of(0), BASE + (H + 0.5) / 2 - 0.15, 0])
        readout = Text("", font="Segoe UI").scale(0.42)
        self.play(Create(window), run_time=0.4)

        for i in range(len(CONF)):
            start = max(0, i - W + 1)
            span = i - start + 1
            new_w = Rectangle(width=STEP * span - 0.1, height=H + 0.5, stroke_color=FILT, stroke_width=4)
            new_w.move_to([(x_of(start) + x_of(i)) / 2, BASE + (H + 0.5) / 2 - 0.15, 0])
            m = median(CONF[start: i + 1])
            new_read = Text(f"frame {i}: median of last {span} = {m:.2f}", font="Segoe UI",
                            color=FILT).scale(0.42).to_edge(RIGHT, buff=0.4).shift(UP * 2.3)
            raw_on = CONF[i] >= TAU
            med_on = m >= TAU
            anims = [Transform(window, new_w), Transform(readout, new_read),
                     cells["raw"][i].animate.set_fill(RAW if raw_on else EMPTY),
                     cells["med"][i].animate.set_fill(FILT if med_on else EMPTY)]
            self.play(*anims, run_time=0.55 if i not in (2, 5, 6) else 0.9)
            if i == 2:
                note = Text("raw alerts on the spike, the median does not", font="Segoe UI",
                            color=INK).scale(0.42).to_edge(DOWN, buff=0.25)
                self.play(FadeIn(note), run_time=0.4)
                self.wait(1.2)
                self.play(FadeOut(note), run_time=0.3)
            if i == 6:
                note = Text("a real target lasts: the median follows it one frame later", font="Segoe UI",
                            color=INK).scale(0.42).to_edge(DOWN, buff=0.25)
                self.play(FadeIn(note), run_time=0.4)
                self.wait(1.2)
                self.play(FadeOut(note), run_time=0.3)

        end = Text("Cost: 5 numbers of memory and well under 1 µs per frame", font="Segoe UI", weight="BOLD",
                   color=INK).scale(0.48).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(readout), FadeIn(end), run_time=0.6)
        self.wait(2.0)
