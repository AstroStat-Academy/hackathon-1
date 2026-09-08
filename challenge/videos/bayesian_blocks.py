"""Run `python videos/bayesian_blocks.py` to render bayesian_blocks.mp4 beside it.

Requires Manim Community and NumPy; no LaTeX installation is needed.
The animation uses exact dynamic programming for Gaussian measurements with
known, equal errors. Each prefix is optimized globally, not greedily split.
"""
from pathlib import Path
import shutil
import tempfile

import numpy as np
from manim import Axes, Dot, FadeIn, FadeOut, Line, Scene, Text, VGroup, tempconfig

BLUE = "#65B8FF"
GREEN = "#72E0B5"
ORANGE = "#FFAE67"
MUTED = "#A4B2C7"


def observations():
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0, 100, 70))
    truth = np.where((t >= 30) & (t < 60), 1.5,
                     np.where((t >= 75) & (t < 84), 0.55, 0.0))
    return t, truth + rng.normal(0, 0.20, len(t)), 0.20


def dynamic_program(y, sigma, penalty=7.0):
    """Return A(n) and remembered one-based r(n), with A(0) = 0."""
    sums = np.r_[0, np.cumsum(y)]
    squares = np.r_[0, np.cumsum(y * y)]
    best = np.zeros(len(y) + 1)
    remembered = np.zeros(len(y) + 1, dtype=int)
    for n in range(1, len(y) + 1):
        starts = np.arange(n)
        residual = (squares[n] - squares[starts]
                    - (sums[n] - sums[starts]) ** 2 / (n - starts))
        scores = best[starts] - residual / (2 * sigma ** 2) - penalty
        start = int(np.argmax(scores))
        best[n] = scores[start]
        remembered[n] = start + 1
    return best, remembered


def caption(words, y, size=24, color=MUTED):
    text = Text(words, font="DejaVu Sans", font_size=size, color=color)
    if text.width > 12.5:
        text.scale_to_fit_width(12.5)
    return text.move_to([0, y, 0])


class BayesianBlocks(Scene):
    def construct(self):
        self.camera.background_color = "#101827"
        t, y, sigma = observations()
        _, remembered = dynamic_program(y, sigma)
        N = len(t)
        edges = np.r_[t[0], (t[1:] + t[:-1]) / 2, t[-1]]
        axes = Axes(x_range=[0, 100, 20], y_range=[-0.6, 2.2, 0.5],
                    x_length=8.6, y_length=4.6, tips=False,
                    axis_config={"color": MUTED, "include_ticks": False})
        axes.move_to([-2.0, -0.55, 0])
        self.add(caption("Fitting Bayesian Blocks", 3.35, 34, "#FFFFFF"))
        subtitle = caption("1. Sweep n = 1 to N; remember the best r", 2.65)
        self.add(subtitle, axes)
        self.add(Text("Time", font_size=19, color=MUTED).move_to([-2, -3.4, 0]))
        for x in range(0, 101, 20):
            self.add(Text(str(x), font_size=16, color=MUTED).move_to(
                axes.c2p(x, -0.6) + np.array([0, -0.2, 0])))
        dots = VGroup(*[Dot(axes.c2p(tx, yy), radius=0.035, color=BLUE)
                        for tx, yy in zip(t, y)])
        errors = VGroup(*[Line(axes.c2p(tx, yy - sigma), axes.c2p(tx, yy + sigma),
                              color=BLUE, stroke_width=1, stroke_opacity=0.35)
                          for tx, yy in zip(t, y)])
        self.add(errors, dots)
        self.add(Line([2.8, 2.15, 0], [2.8, -3.25, 0], color=MUTED,
                      stroke_width=1, stroke_opacity=0.4))
        self.add(Text("Remembered r", font_size=23, color="WHITE").move_to([4.7, 2.05, 0]),
                 Text("r = first point of final block", font_size=15, color=MUTED)
                 .move_to([4.7, 1.67, 0]),
                 Text("n", font_size=20, color=MUTED).move_to([4.0, 1.25, 0]),
                 Text("r", font_size=20, color=MUTED).move_to([5.3, 1.25, 0]))

        def table(known, selected, color=ORANGE):
            # A scrolling column keeps every saved pair legible at 720p.
            first = max(1, selected - 12)
            last = min(known, first + 13)
            rows = VGroup()
            for row, k in enumerate(range(first, last + 1)):
                shade = color if k == selected else MUTED
                ypos = 0.88 - row * 0.30
                rows.add(Text(str(k), font="DejaVu Sans Mono", font_size=18,
                              color=shade).move_to([4.0, ypos, 0]),
                         Text(str(remembered[k]), font="DejaVu Sans Mono", font_size=18,
                              color=shade).move_to([5.3, ypos, 0]))
            if first > 1:
                rows.add(Text("...", font_size=16, color=MUTED).move_to([6.0, 1.0, 0]))
            if last < known:
                rows.add(Text("...", font_size=16, color=MUTED).move_to([6.0, -3.05, 0]))
            return rows

        def block(r, n, color, final=False):
            left = edges[r - 1]
            right = edges[n] if final else t[n - 1]
            level = float(np.mean(y[r - 1:n]))
            shape = VGroup()
            if right > left:
                shape.add(Line(axes.c2p(left, level), axes.c2p(right, level),
                               color=color, stroke_width=5))
            else:
                shape.add(Dot(axes.c2p(left, level), radius=0.06, color=color))
            return shape

        def status(words, color=ORANGE):
            return Text(words, font="DejaVu Sans", font_size=22, color=color).move_to([-2, 2.04, 0])

        rows, current_block, cursor, counter = VGroup(), VGroup(), VGroup(), VGroup()
        for n in range(1, N + 1):
            r = int(remembered[n])
            next_rows = table(n, n)
            next_block = block(r, n, ORANGE)
            next_cursor = Line(axes.c2p(t[n - 1], -0.6), axes.c2p(t[n - 1], 2.2),
                               color=ORANGE, stroke_width=1.4, stroke_opacity=0.7)
            next_counter = status(f"n = {n} / {N}     r = {r}")
            # Future observations stay visible; the cursor marks the prefix
            # being scored, so no final segmentation is implied during sweep.
            self.remove(rows, current_block, cursor, counter)
            self.add(next_rows, next_block, next_cursor, next_counter)
            rows, current_block, cursor, counter = next_rows, next_block, next_cursor, next_counter
            self.wait(1.15 if n <= 8 else 0.23)
        self.wait(1.5)
        new_subtitle = caption("2. Backtrack from N: read r, then set n = r - 1", 2.65)
        self.play(FadeOut(subtitle), FadeIn(new_subtitle),
                  FadeOut(current_block), FadeOut(cursor), FadeOut(counter), run_time=0.6)
        n = N
        recovered = VGroup()
        trail = [N]
        while n > 0:
            r = int(remembered[n])
            next_rows = table(N, n, GREEN)
            next_counter = status(f"n = {n}   →   r = {r}   →   n = {r - 1}", GREEN)
            self.remove(rows)
            self.add(next_rows)
            rows = next_rows
            self.play(FadeIn(next_counter), run_time=0.3)
            self.wait(0.8)
            chosen = block(r, n, GREEN, final=True)
            if r > 1:
                chosen.add(Line(axes.c2p(edges[r - 1], -0.6),
                                axes.c2p(edges[r - 1], 2.2), color=GREEN,
                                stroke_width=1.2, stroke_opacity=0.6))
            self.play(FadeIn(chosen), run_time=0.7)
            recovered.add(chosen)
            self.wait(1.5)
            self.play(FadeOut(next_counter), run_time=0.3)
            n = r - 1
            trail.append(n)
        final_subtitle = caption(f"{len(recovered)} blocks recovered", 2.65)
        self.play(FadeOut(new_subtitle), FadeIn(final_subtitle), run_time=0.5)
        self.add(status(" → ".join(map(str, trail)), GREEN))
        self.wait(5)


if __name__ == "__main__":
    destination = Path(__file__).resolve().with_suffix(".mp4")
    with tempfile.TemporaryDirectory(prefix="bayesian-blocks-manim-") as media:
        with tempconfig({"media_dir": media, "pixel_width": 1280,
                         "pixel_height": 720, "frame_rate": 30,
                         "disable_caching": True, "verbosity": "WARNING",
                         "output_file": destination.stem, "preview": False}):
            scene = BayesianBlocks()
            scene.render()
            shutil.copy2(scene.renderer.file_writer.movie_file_path, destination)
    print(f"Saved {destination}")
