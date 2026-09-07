"""
Preprocessing the lensed-quasar light curves — animation.

Same visual language as the notebook video (`preprocessing.mp4`): the macro
image fades up, rings mark the four observed images, the panel zooms out and
parks bottom-left, then the right-hand panel walks through the pipeline:
  1. raw light curves of the observed images
  2. Step 1 — subtract each curve's median (macro-magnification offset)
  3. Step 2 — subtract a degree-N polynomial in time (microlensing drift)
  4. Step 3 — Bayesian Blocks denoising

Run:  python preprocessing_video.py              -> analysis_bayesian_blocks.mp4
      python preprocessing_video.py --no-blocks  -> analysis_detrended.mp4  (steps 1-2 only)
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import image as mpimg
from matplotlib.patches import Circle
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import label, center_of_mass
from astropy.io import fits
from astropy.stats import bayesian_blocks

# --------------------------------------------------------------------------
# "tron" theme — inlined (identical to the notebook) so the two videos match
# --------------------------------------------------------------------------
_FG, _BG = "#CDECF7", "#0A1721"
TRON_CYCLE = ["#00E8F8", "#FFA300", "#7DF9FF", "#37C0DC", "#EC7600", "#B7F7FF", "#2A6F97"]

plt.style.use("dark_background")
mpl.rcParams.update({
    "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
    "text.color": _FG, "axes.labelcolor": _FG, "axes.titlecolor": _FG,
    "axes.edgecolor": "#2E6E8E", "xtick.color": _FG, "ytick.color": _FG,
    "axes.prop_cycle": mpl.cycler(color=TRON_CYCLE),
    "axes.grid": True, "grid.color": "#12384A", "grid.linewidth": 0.8, "grid.alpha": 0.7,
    "lines.linewidth": 2.0, "lines.markersize": 5, "errorbar.capsize": 3,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"],
    "font.weight": "medium",
    "axes.titleweight": "bold", "axes.labelweight": "bold", "figure.titleweight": "bold",
    "axes.titlepad": 10, "axes.labelpad": 6, "mathtext.fontset": "cm",
    "font.size": 13, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 13, "legend.title_fontsize": 13, "figure.titlesize": 18,
    "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.facecolor": _BG, "legend.edgecolor": "#00E8F8", "legend.fancybox": True,
    "legend.borderpad": 0.6, "legend.labelspacing": 0.5, "legend.handlelength": 1.8,
})

# one clearly distinct hue per image — the theme cycle reuses near-identical
# blues/oranges, so images 0 & 3 (and 1 & 4) would otherwise be indistinguishable
PT_COLOR = {0: "#00E8F8",   # cyan
            1: "#FFA300",   # amber
            3: "#C77DFF",   # violet
            4: "#4DF0A0"}   # mint

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent            # .../hackathon-1/challenge/videos
_ROOT = _HERE.parent                               # data files live one level up

FITS_PATH       = _ROOT / "mock_data" / "OBS_testCAM-i_ps_macro_noiseless.fits"   # lens-system image
LC_JSON_PATH    = _ROOT / "mock_data" / "testCAM-i_LC_sampled.json"               # sampled light curves
TRUTH_JSON_PATH = _ROOT / "mock_data" / "multiple_images.json"                    # image positions -> ring colours

KEEP = [0, 1, 3, 4]        # JSON index 2 is the central (unobserved) image

POLY_DEG = 4               # microlensing-trend polynomial degree
BB_SIGMA = 0.02            # assumed per-point scatter for Bayesian Blocks
BB_P0    = 0.05            # Bayesian Blocks false-alarm probability

WITH_BLOCKS = "--no-blocks" not in sys.argv      # pass --no-blocks to drop Step 3
OUT_MP4   = str(_HERE / ("analysis_bayesian_blocks.mp4" if WITH_BLOCKS else "analysis_detrended.mp4"))
LOGO_PATH = _ROOT.parent / "assets" / "logo_w_text_w_highres.png"  # bottom-left watermark
FPS     = 30
DPI     = 140
FIG_W, FIG_H = 12.5, 7.0


# --------------------------------------------------------------------------
# load & preprocess
# --------------------------------------------------------------------------
def _dedup(t, y):
    """Average measurements that share a timestamp."""
    u, inv = np.unique(t, return_inverse=True)
    ym = np.zeros_like(u)
    np.add.at(ym, inv, y)
    ym /= np.bincount(inv)
    return u, ym


def preprocess_light_curves(lc_raw, keep, deg, bb_sigma, bb_p0):
    out = {}
    for i in keep:
        t, y = _dedup(np.asarray(lc_raw[i]["time"], float),
                      np.asarray(lc_raw[i]["signal"], float))
        s1    = y - np.median(y)
        coef  = np.polyfit(t - t.mean(), s1, deg)
        trend = np.polyval(coef, t - t.mean())
        s2    = s1 - trend
        edges = bayesian_blocks(t, s2, sigma=bb_sigma, fitness="measures", p0=bb_p0)
        b     = np.clip(np.digitize(t, edges[1:-1]), 0, len(edges) - 2)
        level = np.array([s2[b == k].mean() for k in range(len(edges) - 1)])
        out[i] = dict(t=t, raw=y, s1=s1, trend=trend, s2=s2,
                      edges=edges, level=level)
    return out


LC_RAW = json.loads(Path(LC_JSON_PATH).read_text())
IMG = preprocess_light_curves(LC_RAW, KEEP, POLY_DEG, BB_SIGMA, BB_P0)

with fits.open(FITS_PATH) as h:
    LENS_IMG = np.asarray(h[0].data, float)
    _hd = h[0].header
    try:
        LENS_EXTENT = [float(_hd[k]) for k in ("XMIN", "XMAX", "YMIN", "YMAX")]
    except KeyError:
        LENS_EXTENT = [-1.75, 1.75, -1.75, 1.75]


# --------------------------------------------------------------------------
# locate the point-source images -> one ring per observed image
# --------------------------------------------------------------------------
_ny, _nx = LENS_IMG.shape


def _pix_to_sky(r, c):
    x = LENS_EXTENT[0] + (c + 0.5) * (LENS_EXTENT[1] - LENS_EXTENT[0]) / _nx
    y = LENS_EXTENT[2] + (r + 0.5) * (LENS_EXTENT[3] - LENS_EXTENT[2]) / _ny
    return x, y


_lab, _n = label(LENS_IMG > 4.0)
SOURCES  = [_pix_to_sky(r, c) for r, c in center_of_mass(LENS_IMG, _lab, range(1, _n + 1))]

_TRUTH = json.loads(Path(TRUTH_JSON_PATH).read_text())
_TXY   = {i: (t["x"], t["y"]) for i, t in enumerate(_TRUTH)}
_CEN   = min(_TXY, key=lambda i: np.hypot(*_TXY[i]))          # central demagnified image

RINGS = []
for sx, sy in SOURCES:
    idx = min(_TXY, key=lambda i: np.hypot(sx - _TXY[i][0], sy - _TXY[i][1]))
    if idx == _CEN:
        continue
    RINGS.append((sx, sy, PT_COLOR.get(idx, _FG)))

RING_R = 0.14


# --------------------------------------------------------------------------
# storyboard
# --------------------------------------------------------------------------
_ease = lambda u: u * u * (3 - 2 * u)               # ease in / out
_lerp = lambda a, b, u: a + (b - a) * u

# intro — mirrors the notebook video
FADE_IN    = int(0.7 * FPS)                 # macro image fades up
HOLD       = int(0.5 * FPS)
STAGGER    = int(0.18 * FPS)                # gap between successive rings
POP        = int(0.45 * FPS)               # one ring's pop-in
RINGS_HOLD = int(0.8 * FPS)
MOVE       = int(2.2 * FPS)                 # zoom out + glide to bottom-left
REVEAL     = int(4.0 * FPS)                 # light curves arrive "in real time"

_RINGS_END    = FADE_IN + HOLD + STAGGER * (len(RINGS) - 1) + POP
_MOVE_START   = _RINGS_END + RINGS_HOLD
_REVEAL_START = _MOVE_START + MOVE
_PHASES_START = _REVEAL_START + REVEAL

# analysis — unchanged pipeline steps
PHASES = [
    ("Raw light curves — 4 observed images",                     int(1.6 * FPS), "hold",        "raw"),
    ("Step 1 — subtract each curve's median",                    int(2.0 * FPS), "morph",       ("raw", "s1")),
    ("Step 1 — macro-magnification offset removed",              int(1.2 * FPS), "hold",        "s1"),
    (f"Step 2 — fit a degree-{POLY_DEG} polynomial (microlensing drift)",
                                                                 int(1.8 * FPS), "trend_in",    "s1"),
    ("Step 2 — subtract the microlensing drift",                 int(2.0 * FPS), "morph",       ("s1", "s2")),
    ("Step 2 — one shared intrinsic signal, time-shifted",       int(1.2 * FPS), "hold",        "s2"),
]
if WITH_BLOCKS:
    PHASES += [
        ("Step 3 — Bayesian Blocks denoising",                   int(2.0 * FPS), "blocks_in",   "s2"),
        ("Preprocessed — ready for peak-finding & Δt",           int(1.8 * FPS), "blocks_hold", "s2"),
    ]
else:
    PHASES += [
        ("Preprocessed — ready for peak-finding & Δt",           int(2.6 * FPS), "hold",        "s2"),
    ]
N_FRAMES = _PHASES_START + sum(p[1] for p in PHASES)


def _phase_at(f):
    acc = _PHASES_START
    for title, n, kind, arg in PHASES:
        if f < acc + n:
            return title, kind, arg, _ease((f - acc) / max(n - 1, 1))
        acc += n
    t, _, k, a = PHASES[-1]
    return t, k, a, 1.0


# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------
def _ylim(key, pad=0.12):
    v = np.concatenate([IMG[i][key] for i in KEEP])
    lo, hi = v.min(), v.max()
    m = (hi - lo) * pad
    return (hi + m, lo - m)              # inverted: magnitudes brighten upward


YLIM = {k: _ylim(k) for k in ("raw", "s1", "s2")}
_TALL  = np.concatenate([IMG[i]["t"] for i in KEEP])
T0, T1 = _TALL.min(), _TALL.max()


def make_animation():
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # left panel: the macro image — starts large & central, ends parked bottom-left
    ax = fig.add_axes([0.24, 0.09, 0.40, 0.84])
    _img = ax.imshow(np.sqrt(LENS_IMG), origin="lower", extent=LENS_EXTENT,
                     cmap="magma", interpolation="bilinear")
    ax.set_xlabel("x  [arcsec]")
    ax.set_ylabel("y  [arcsec]")
    ax.grid(False)
    ax.set_anchor("C")
    cbar = fig.colorbar(_img, ax=ax, fraction=0.046, pad=0.04, label=r"$\sqrt{\mathrm{flux}}$")
    fig.canvas.draw()

    AX0 = ax.get_position().bounds
    CB0 = cbar.ax.get_position().bounds

    H1 = 0.46                          # parked thumbnail: bigger, since it loses ticks + labels
    W1 = H1 * FIG_H / FIG_W
    L1 = 0.020
    B1 = (1.0 - H1) / 2
    AX1 = (L1, B1, W1, H1)
    CB1 = (L1 + W1 + 0.006, B1, 0.013, H1)

    FX = (LENS_EXTENT[0], LENS_EXTENT[1])
    FY = (LENS_EXTENT[2], LENS_EXTENT[3])
    cx, cy = sum(FX) / 2, sum(FY) / 2
    Z = 1.55
    ZX = (cx + (FX[0] - cx) * Z, cx + (FX[1] - cx) * Z)
    ZY = (cy + (FY[0] - cy) * Z, cy + (FY[1] - cy) * Z)

    rings = [Circle((x, y), RING_R, fill=False, lw=2.4, ec=col, visible=False)
             for x, y, col in RINGS]
    for c in rings:
        ax.add_patch(c)

    # right panel: the analysis
    ax2 = fig.add_axes([0.33, 0.12, 0.64, 0.80])
    ax2.set_visible(False)

    # watermark logo, bottom-left (fixed, drawn once, never cleared)
    logo = mpimg.imread(LOGO_PATH).astype(float)
    if logo.max() > 1.0:
        logo /= 255.0
    if logo.shape[-1] == 4:
        logo[..., 3] *= 0.55                                    # semi-transparent
    lh, lw = logo.shape[:2]
    wm_w = 0.135
    wm_h = wm_w * (lh / lw) * (FIG_W / FIG_H)                   # keep the logo's aspect
    wm_ax = fig.add_axes([0.012, 0.012, wm_w, wm_h], zorder=12)
    wm_ax.imshow(logo)
    wm_ax.axis("off")
    wm_ax.patch.set_alpha(0.0)

    def prettify(a):
        a.spines[["top", "right"]].set_visible(False)
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_minor_locator(AutoMinorLocator())
        a.tick_params(which="minor", length=0)
        a.margins(x=0.01)

    def fade_cbar(al):
        cbar.solids.set_alpha(al)
        cbar.outline.set_alpha(al)
        cbar.ax.yaxis.label.set_alpha(al)
        for t in cbar.ax.get_yticklabels():
            t.set_alpha(al)
        cbar.ax.tick_params(length=2.5 * al)

    def fade_ax_frame(al):
        for sp in ax.spines.values():
            sp.set_alpha(al)
        ax.xaxis.label.set_alpha(al)
        ax.yaxis.label.set_alpha(al)
        for t in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            t.set_alpha(al)
        ax.tick_params(length=4.5 * al, width=1.1 * al)

    def animate(f):
        # ---- left panel: fade up -> rings pop in -> zoom out & park ----
        _img.set_alpha(_ease(np.clip(f / FADE_IN, 0, 1)))

        r0 = FADE_IN + HOLD
        for k, c in enumerate(rings):
            u = _ease(np.clip((f - (r0 + k * STAGGER)) / POP, 0, 1))
            c.set_visible(u > 0)
            if u > 0:
                c.set_radius(RING_R * (2.4 - 1.4 * u))
                c.set_alpha(u)

        m = _ease(np.clip((f - _MOVE_START) / MOVE, 0, 1))
        ax.set_position([_lerp(a, b, m) for a, b in zip(AX0, AX1)])
        cbar.ax.set_position([_lerp(a, b, m) for a, b in zip(CB0, CB1)])
        fade_cbar(1 - m)
        fade_ax_frame(1 - m)                    # spines, "x/y [arcsec]", ticks & numbers all go
        ax.set_xlim(_lerp(FX[0], ZX[0], m), _lerp(FX[1], ZX[1], m))
        ax.set_ylim(_lerp(FY[0], ZY[0], m), _lerp(FY[1], ZY[1], m))

        # ---- right panel ----
        if f < _REVEAL_START:
            return ()

        ax2.set_visible(True)
        ax2.clear()

        trend_a = blocks_a = 0.0
        cursor = None

        if f < _PHASES_START:                                   # points arrive "in real time"
            cursor = T0 + (T1 - T0) * _ease(np.clip((f - _REVEAL_START) / REVEAL, 0, 1))
            yk = {i: IMG[i]["raw"] for i in KEEP}
            yl = YLIM["raw"]
            title, raw_axis = "Sampled light curves", True
        else:
            title, kind, arg, u = _phase_at(f)
            raw_axis = False
            if kind == "hold":
                yk = {i: IMG[i][arg] for i in KEEP}
                yl = YLIM[arg]; raw_axis = (arg == "raw")
            elif kind == "morph":
                k0, k1 = arg
                yk = {i: _lerp(IMG[i][k0], IMG[i][k1], u) for i in KEEP}
                yl = tuple(_lerp(np.array(YLIM[k0]), np.array(YLIM[k1]), u))
                raw_axis = (k0 == "raw" and u < 0.5)
                trend_a = (1 - u) if (k0, k1) == ("s1", "s2") else 0.0
            elif kind == "trend_in":
                yk = {i: IMG[i]["s1"] for i in KEEP}
                yl = YLIM["s1"]; trend_a = u
            elif kind == "blocks_in":
                yk = {i: IMG[i]["s2"] for i in KEEP}
                yl = YLIM["s2"]; blocks_a = u
            else:  # blocks_hold
                yk = {i: IMG[i]["s2"] for i in KEEP}
                yl = YLIM["s2"]; blocks_a = 1.0

        tc = cursor if cursor is not None else T1 + 1
        for i in KEEP:
            col = PT_COLOR[i]
            t = IMG[i]["t"]
            sel = t <= tc
            ax2.scatter(t[sel], yk[i][sel], s=14, color=col, alpha=0.9,
                        linewidths=0, label=f"image {i}")
            if trend_a > 0:
                ax2.plot(t, IMG[i]["trend"], color=col, lw=2.2, alpha=trend_a)
            if blocks_a > 0:
                ax2.stairs(IMG[i]["level"], IMG[i]["edges"], color=col, lw=1.8, alpha=blocks_a)

        if cursor is not None:
            ax2.axvline(cursor, color=_FG, lw=1.0, alpha=0.30)

        ax2.set_xlim(T0 - 40, T1 + 40)
        ax2.set_ylim(yl)
        ax2.set_xlabel("time  [days]")
        ax2.set_ylabel("magnitude  (i band)" if raw_axis else r"$\Delta$ mag")
        ax2.set_title(title)
        prettify(ax2)
        ax2.legend(loc="upper right", ncol=2, fontsize=10)
        return ()

    return animation.FuncAnimation(fig, animate, frames=N_FRAMES,
                                   interval=1000 / FPS, blit=False)


if __name__ == "__main__":
    print(f"{N_FRAMES} frames  ->  {N_FRAMES / FPS:.1f} s at {FPS} fps")
    anim = make_animation()
    anim.save(OUT_MP4, writer="ffmpeg", fps=FPS, dpi=DPI)
    plt.close("all")
    print("wrote", OUT_MP4)
