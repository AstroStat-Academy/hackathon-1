# Lensing Time-Delay Hackathon — Petnica (Serbia) 2026

<p align="center">
  <img src="challenge/videos/preprocessing.gif" alt="How repeated observations of a lensed quasar build up a set of light curves" width="820">
</p>


[Presentation](https://docs.google.com/presentation/d/1QdTCr_AhTNxT4i0USVuw67Ui06PHa4WUhkZEFXKfqKc/edit?usp=sharing)

Hackathon materials for **TALES School II**. A hands-on workshop on machine-learning
applications to astrophysics, held **13–18 September 2026 at the Petnica Science Center,
Serbia**.

## The challenge

Strongly lensed quasars produce two or four images of the same source. Because light
travels different paths through the lens, the images show the same intrinsic brightness
variations **shifted in time** by a few days to a few weeks. These *time delays* are a
direct probe of the Hubble constant and of the lens mass distribution.

**Your task:** given the light curves of a set of lensed quasars, estimate the time delay
between every pair of images.

- Each system is a **double** (2 light curves) or a **quad** (4 light curves).
- Report delays relative to the leading image (label `0`): `Δt₀₁` for doubles;
  `Δt₀₁`, `Δt₀₂`, `Δt₀₃` for quads.
- **Training** data comes with ground-truth delays; **test** data does not.
- Submissions are ranked by the **RMSE** (in days) between predicted and true delays,
  pooled across all systems and all pairs. The exact `score()` function is provided in
  the challenge notebook.

Deliverable: fill in the `Δt` predictions in `test.csv` and submit it as `results.csv`.

## Repository layout

| Path | Contents |
| --- | --- |
| [challenge/challenge.ipynb](challenge/challenge.ipynb) | The brief: data-generating process, light-curve / catalog file formats, the reader/plotting helpers, and the official scoring function. **Start here.** |
| [challenge/methods.ipynb](challenge/methods.ipynb) | Walk-through of suggested approaches — cross-correlation, Bayesian Blocks + cross-correlation, and [PyCS3](https://gitlab.com/cosmograil/PyCS3) curve shifting — plus notes on what to watch for and relevant libraries. |
| [challenge/mock_data/](challenge/mock_data/) | Small worked examples: sample light-curve `.json` files, a lens-model FITS image, and `truth.csv` / `results.csv` illustrating the submission format. |
| [challenge/videos/](challenge/videos/) | Explainer animations (and the scripts that build them) used in the notebooks. |
| [environment.yaml](environment.yaml) | Conda environment specification. |
| [data/](data/) | Drop the distributed training/test data here. |

## Data format

Light curves are stored as a JSON list of 2 or 4 dictionaries — **order matters** and
defines the image labels `0, 1, 2, 3`. Each dictionary holds equal-length arrays:

- `time` — timestamp in days from an arbitrary reference
- `signal` — flux in magnitudes, arbitrary optical filter
- `dsignal` — uncertainty on the flux

The catalog (`train.csv` / `test.csv`) has one row per lens: `id`, `flag` (2 or 4), and
the delay columns `01`, `02`, `03` (`NA` where not applicable or to be predicted).

## Getting started

```bash
conda env create -f environment.yaml
conda activate hackathon-1
jupyter lab
```

Then open [challenge/challenge.ipynb](challenge/challenge.ipynb), read the brief, run the
helper cells, and try the scoring function on the mock data. When you are comfortable,
move to [challenge/methods.ipynb](challenge/methods.ipynb) for method ideas and build your
own pipeline from there.

Start simple (e.g. cross-correlation on interpolated curves), compare notes with your
team, and iterate. Happy hacking!

<!-- ## License

Original content is licensed under the
[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
External material is credited to its sources in the notebooks. -->
