"""Animate a lag scan of two gapped light curves using Manim Community.

Run from any directory:
    python cross_correlation.py

Requires manim and numpy. Produces cross_correlation.mp4 beside this script;
intermediate render files are kept in a temporary directory. Uses the same
seeded toy observations as methods.ipynb. Positive lag means B follows A:
we draw B at t - lag and correlate A(t) with B(t + lag), ignoring missing pairs.
"""
from pathlib import Path
import shutil
import tempfile

import numpy as np
from manim import (
    Axes, Create, DashedLine, Dot, FadeIn, FadeOut, Flash, Line, Scene,
    Text, Transform, VGroup, VMobject, config, linear, tempconfig,
)

BLUE = "#65B8FF"
ORANGE = "#FFAE67"
GREEN = "#72E0B5"
MUTED = "#A4B2C7"


def make_data():
    """Return daily timestamps and two noisy, gapped curves delayed by 12 days."""
    rng = np.random.default_rng(42)
    t = np.arange(221.0)

    def signal(x):
        return (1.2 * np.exp(-0.5 * ((x - 35) / 7) ** 2)
                - 0.8 * np.exp(-0.5 * ((x - 88) / 12) ** 2)
                + 1.5 * np.exp(-0.5 * ((x - 148) / 9) ** 2)
                + 0.5 * np.exp(-0.5 * ((x - 190) / 5) ** 2))

    a = signal(t) + rng.normal(0, 0.08, t.size)
    b = 1.3 * signal(t - 12) + 0.7 + rng.normal(0, 0.08, t.size)
    for values, gaps in [(a, [(50, 70), (112, 132)]),
                         (b, [(20, 34), (100, 119), (174, 190)])]:
        for start, stop in gaps:
            values[(t >= start) & (t <= stop)] = np.nan
        values[rng.random(t.size) < 0.12] = np.nan
    return t, a, b


def correlation_at_lag(a, b, lag):
    """Return overlap Pearson r and pair count for an integer daily lag."""
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:lag]
    else:
        x, y = a, b
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 40 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, x.size
    return float(np.corrcoef(x, y)[0, 1]), x.size


def label(text, x, y, size=22, color=MUTED):
    """Create text without requiring a LaTeX installation."""
    return Text(text, font="DejaVu Sans", font_size=size, color=color).move_to([x, y, 0])


def curve(axes, t, values, color):
    """Draw observed samples, connecting only consecutive finite samples."""
    parts = VGroup()
    indices = np.flatnonzero(np.isfinite(values))
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    for run in runs:
        if len(run) > 1:
            line = VMobject(color=color, stroke_width=2)
            line.set_points_as_corners([axes.c2p(t[i], values[i]) for i in run])
            parts.add(line)
    for i in indices:
        parts.add(Dot(axes.c2p(t[i], values[i]), radius=0.018, color=color))
    return parts


class CrossCorrelation(Scene):
    """Shift B at each daily lag, then add the corresponding correlation point."""

    def construct(self):
        self.camera.background_color = "#101827"
        t, a, b = make_data()
        lags = np.arange(-30, 31)
        results = [correlation_at_lag(a, b, int(k)) for k in lags]
        best = int(np.nanargmax([r for r, _ in results]))
        best_lag = int(lags[best])
        # Whole-curve normalization is for display only; r uses each overlap.
        a_display = (a - np.nanmedian(a)) / np.nanstd(a)
        b_display = (b - np.nanmedian(b)) / np.nanstd(b)

        self.add(label("Building the Cross-Correlation plot", 0, 3.25, 34, "#FFFFFF"))
        left = Axes(x_range=[-35, 255, 50], y_range=[-2.2, 3.5, 1],
                    x_length=6.0, y_length=3.4, tips=False,
                    axis_config={"color": MUTED, "stroke_width": 1.3,
                                 "include_ticks": False}).move_to([-3.65, -0.10, 0])
        right = Axes(x_range=[-33, 33, 10], y_range=[-0.6, 1.1, 0.2],
                     x_length=5.55, y_length=3.4, tips=False,
                     axis_config={"color": MUTED, "stroke_width": 1.3,
                                  "include_ticks": False}).move_to([3.55, -0.10, 0])
        self.add(left, right)
        self.add(label("A: fixed", -5.15, 1.94, 18, BLUE),
                 label("B: shifted by minus lag", -2.65, 1.94, 18, ORANGE))
        self.add(label("Time [days]", -3.65, -2.25, 19),
                 label("Trial lag [days]", 3.55, -2.25, 19))
        self.add(label("Correlation r", 3.55, 1.94, 18, GREEN))
        for x in [0, 50, 100, 150, 200]:
            pos = left.c2p(x, -2.2)
            self.add(label(str(x), pos[0], pos[1] - 0.19, 15))
        for x in [-30, -20, -10, 0, 10, 20, 30]:
            pos = right.c2p(x, -0.6)
            self.add(label(str(x), pos[0], pos[1] - 0.19, 15))
        for y in [-0.5, 0, 0.5, 1]:
            pos = right.c2p(-33, y)
            self.add(label(f"{y:g}", pos[0] - 0.20, pos[1], 14))
        for y in [-0.5, 0.5, 1]:
            self.add(Line(right.c2p(-33, y), right.c2p(33, y),
                          color=MUTED, stroke_width=0.6, stroke_opacity=0.2))

        a_curve = curve(left, t, a_display, BLUE)
        b_curve = curve(left, t, b_display, ORANGE)
        self.play(FadeIn(a_curve), FadeIn(b_curve), run_time=0.8)
        status = label("Lag 0 days", -1.75, 1.52, 21, "#FFFFFF")
        self.add(status)
        self.wait(2)

        unit = left.c2p(1, 0) - left.c2p(0, 0)
        current_lag = 0
        previous_point = None
        for j, lag in enumerate(lags):
            lag = int(lag)
            r, count = results[j]
            target = label(f"Lag {lag:+d} days", -1.75, 1.52, 21, "#FFFFFF")
            # Shift first, calculate/add the point second, to make the loop visible.
            self.play(b_curve.animate.shift(-(lag - current_lag) * unit),
                      FadeOut(status), run_time=0.55 if j < 3 else 0.18,
                      rate_func=linear)
            status = target
            self.add(status)
            position = right.c2p(lag, r)
            point = Dot(position, radius=0.034, color=GREEN)
            animations = [FadeIn(point, scale=2)]
            if previous_point is not None:
                animations.append(Create(Line(previous_point, position,
                                              color=GREEN, stroke_width=2)))
            self.play(*animations, run_time=0.30 if j < 3 else 0.14)
            previous_point = position
            current_lag = lag
            if j == best:
                self.play(Flash(point, color=ORANGE, flash_radius=0.20), run_time=0.8)
                self.wait(2.5)
            elif j < 3:
                self.wait(0.35)

        self.wait(0.8)
        self.play(b_curve.animate.shift(-(best_lag - current_lag) * unit),
                  FadeOut(status), run_time=1.2)
        peak = right.c2p(best_lag, results[best][0])
        self.play(Create(DashedLine(right.c2p(best_lag, -0.6), peak,
                                    color=ORANGE, stroke_width=2)),
                  FadeIn(Dot(peak, color=ORANGE, radius=0.08)), run_time=0.6)
        self.play(Flash(peak, color=ORANGE, flash_radius=0.24), run_time=0.9)
        self.wait(6)



if __name__ == "__main__":
    destination = Path(__file__).resolve().with_suffix(".mp4")
    with tempfile.TemporaryDirectory(prefix="cross-correlation-manim-") as media:
        with tempconfig({"media_dir": media, "pixel_width": 1280,
                         "pixel_height": 720, "frame_rate": 30,
                         "disable_caching": True, "verbosity": "WARNING",
                         "output_file": "cross_correlation", "preview": False}):
            scene = CrossCorrelation()
            scene.render()
            shutil.copy2(scene.renderer.file_writer.movie_file_path, destination)
    print(f"Saved {destination}")
