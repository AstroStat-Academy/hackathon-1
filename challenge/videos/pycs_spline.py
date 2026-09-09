"""Render pycs_spline.mp4 beside this script: python3 videos/pycs_spline.py.

Requires Manim Community, NumPy and SciPy. Uses the observations from
images/plot_pycs_alignment.py. This illustrative profile fit uses fixed cubic
B-spline knots and the known magnitude offset; it is not a full PyCS fit.
At every displayed delay the shared spline is refitted to both light curves.
"""
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar
from manim import (
    Axes, Create, DashedLine, Dot, FadeIn, Line,
    Scene, Text, VGroup, VMobject, ValueTracker, always_redraw, tempconfig,
)

BLUE, ORANGE, GREEN = '#65B8FF', '#FFAE67', '#72E0B5'
MUTED = '#A4B2C7'


def observations():
    rng = np.random.default_rng(18)
    def signal(t):
        return 20 - .5*np.exp(-.5*((t-38)/8)**2) - .3*np.exp(-.5*((t-83)/11)**2) + .065*np.sin(t/8)
    ta = np.sort(rng.uniform(5, 110, 65))
    tb = np.sort(rng.uniform(22, 130, 65))
    a = signal(ta) + rng.normal(0, .027, len(ta))
    b = signal(tb-20) + .45 + rng.normal(0, .027, len(tb))
    return ta, 20-a, tb, 20-(b-.45)


class SplineFit:
    def __init__(self):
        self.ta, self.a, self.tb, self.b = observations()
        self.knots = np.r_[[-15.]*4, np.arange(-5., 130., 10.), [140.]*4]
        self.y = np.r_[self.a, self.b]

    def fit(self, delay):
        x = np.r_[self.ta, self.tb-delay]
        design = BSpline.design_matrix(x, self.knots, 3).toarray()
        coef = np.linalg.lstsq(design, self.y, rcond=None)[0]
        spline = BSpline(self.knots, coef, 3)
        chi2 = float(np.sum(((self.y-spline(x))/.027)**2))
        return spline, chi2


def label(value, position, size=23, color=MUTED):
    return Text(value, font='DejaVu Sans', font_size=size, color=color).move_to(position)


class PyCSSpline(Scene):
    def construct(self):
        self.camera.background_color = '#101827'
        model = SplineFit()
        best = minimize_scalar(lambda d: model.fit(d)[1], bounds=(10, 28),
                               method='bounded').x
        print(f'Fitted delay: {best:.3f} days; chi2: {model.fit(best)[1]:.2f}')
        axes = Axes(x_range=[-10, 140, 20], y_range=[-.15, .75, .15],
                    x_length=11.7, y_length=4.55, tips=False,
                    axis_config={'color': MUTED, 'stroke_width': 1.2,
                                 'include_ticks': False}).move_to([.25, -.2, 0])
        self.add(axes)
        for x in range(0, 141, 20):
            p = axes.c2p(x, -.15)
            self.add(label(str(x), p+[0, -.25, 0], 18))
        for y in [0, .2, .4, .6]:
            p = axes.c2p(-10, y)
            self.add(label(f'{y:.1f}', p+[-.3, 0, 0], 17))
            self.add(Line(p, axes.c2p(140, y), stroke_width=.6,
                          color=MUTED, stroke_opacity=.15))
        self.add(label('Aligned time [days]', [0, -3.08, 0], 23),
                 label('−Δmag', [-5.6, 2.48, 0], 23),
                 label('A', [-4.3, 3.25, 0], 25, BLUE),
                 label('B  (t − τ)', [-1.8, 3.25, 0], 25, ORANGE),
                 label('Common spline', [2.35, 3.25, 0], 25, GREEN),
                 label('cB = 0.45 mag', [4.75, -3.08, 0], 19))
        delay = ValueTracker(0)
        def points(t, y, color):
            group = VGroup()
            for x, value in zip(t, y):
                group.add(Line(axes.c2p(x, value-.027), axes.c2p(x, value+.027),
                               color=color, stroke_width=1, stroke_opacity=.55),
                          Dot(axes.c2p(x, value), radius=.033, color=color))
            return group
        a = points(model.ta, model.a, BLUE)
        b = points(model.tb, model.b, ORANGE)
        original_b = b.copy()
        unit = axes.c2p(1, 0)-axes.c2p(0, 0)
        b.add_updater(lambda m: m.become(original_b.copy().shift(-delay.get_value()*unit)))
        self.play(FadeIn(a), FadeIn(b), run_time=1)

        # Cache one numerical fit per frame for the curve, residuals and readout.
        cache = {}
        def current():
            d = delay.get_value()
            if cache.get('delay') != d:
                cache['delay'] = d
                cache['fit'] = model.fit(d)
            return cache['fit']
        def spline_path():
            spline, _ = current()
            x = np.linspace(max(model.ta.min(), (model.tb-delay.get_value()).min()),
                            min(model.ta.max(), (model.tb-delay.get_value()).max()), 420)
            path = VMobject(color=GREEN, stroke_width=3.5)
            path.set_points_as_corners([axes.c2p(t, y) for t, y in zip(x, spline(x))])
            return path
        spline = always_redraw(spline_path)
        status = always_redraw(lambda: VGroup(
            label(f'τ = {delay.get_value():04.1f} days', [-2, -3.65, 0], 25, ORANGE),
            label(f'χ² = {current()[1]:.0f}', [2, -3.65, 0], 25, GREEN)))
        self.play(Create(spline), FadeIn(status), run_time=1.2)
        self.wait(1.3)

        # Pair the two prominent brightness peaks; the B guides follow its shift.
        guides = always_redraw(lambda: VGroup(*[
            DashedLine(axes.c2p(p+s, -.10), axes.c2p(p+s, .65),
                       color=color, stroke_width=1.6, dash_length=.08,
                       stroke_opacity=.6)
            for p in [39, 85]
            for s, color in [(0, BLUE), (20-delay.get_value(), ORANGE)]
        ]))
        self.play(FadeIn(guides), run_time=.7)
        self.play(delay.animate.set_value(10), run_time=3)
        self.wait(.6)
        self.play(delay.animate.set_value(16), run_time=2.5)
        self.wait(.6)
        self.play(delay.animate.set_value(24), run_time=3)
        self.wait(.7)
        self.play(delay.animate.set_value(best), run_time=3)
        self.wait(.7)
        self.wait(3)


if __name__ == '__main__':
    destination = Path(__file__).resolve().with_suffix('.mp4')
    with tempfile.TemporaryDirectory(prefix='pycs-spline-manim-') as media:
        with tempconfig({'media_dir': media, 'pixel_width': 1920,
                         'pixel_height': 1080, 'frame_rate': 30,
                         'disable_caching': True, 'verbosity': 'WARNING',
                         'output_file': destination.stem, 'preview': False}):
            scene = PyCSSpline()
            scene.render()
            shutil.copy2(scene.renderer.file_writer.movie_file_path, destination)
    print(f'Saved {destination}')
