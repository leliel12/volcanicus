# Volcanicus

<img src="res/cover.jpg" alt="Volcanicus" width="50%">

**A small toolkit to load, normalize and plot volcano deformation/residual measurements**

<!-- BODY -->

[![License](https://img.shields.io/badge/license-BSD--3-blue.svg)](https://www.tldrlegal.com/l/bsd3)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

**Volcanicus** wraps per-volcano residual/deformation measurements (one row
per date/direction, one column per distance bin) in a `Volcano` object, taking
care of dtype normalization and offering quick summary and plotting helpers
on top of the pandas stack.

## 📦 Installation

From the project root (this directory):

```bash
pip install -e .
```

### From GitHub:

```bash
pip install https://github.com/leliel12/volcanicus/archive/refs/heads/master.zip
```

### Development dependencies

```bash
pip install -r requirements_dev.txt
```


## 🚀 Usage

### Bundled datasets

Volcanicus ships 186 example datasets, loadable by name. Each one carries
the volcano's descriptive metadata (GVP identification, location,
classification, morphology and PyVOLCANS analogues):

```python
from volcanicus import datasets

datasets.available()  # ["agung", "aira", "akan", "alaid", ..., "zubair_group"]
etna = datasets.load("etna")

etna.gvp_lat, etna.gvp_lon  # (37.7442, 15.0107), GVP reference location
etna.metadata.country       # "Italy"
etna.m.volcano_type_class   # `.m` is a shorthand for `.metadata`
```

### Working with a `Volcano`

```python
# normalized, read-only copy of the measurements
etna.to_dataframe()

# mean residual per distance bin, per direction
etna.radial_profile()

# residual vs. distance, one line per direction, with an aggregate median line
etna.plot.radial_profile()

# how the missing values are spread across directions (default) or
# dates: a DataFrame with `total`/`n_missing`/`proportion` columns and
# a trailing "TOTAL" row
etna.describe()
etna.describe(by="date")

# fill the gaps; the new instance keeps the original metadata and
# records what was done
imputed = etna.impute()
imputed.m.imputed  # True
```

### Your own measurements

Build a `Volcano` directly from a `DataFrame` with a `date` column,
`center_lat`/`center_long`, a `direction` column and one column per distance
bin, plus the volcano's reference location from the Smithsonian's
[Global Volcanism Program](https://volcano.si.edu) (GVP):

```python
import pandas as pd
from volcanicus import Volcano

df = pd.read_csv("measurements.csv")
volcano = Volcano(
    "copahue",
    df,
    gvp_lat=-37.8561,
    gvp_lon=-71.1605,
    metadata={"country": "Argentina"},
)
```

## 🔭 Views of the data

The raw data is a long table: one row per **date × direction** and one
column per **distance** (8 directions, 30 bins from 15 to 885 m). Volcanicus
offers five *views* of it; each one changes the unit of analysis and answers
a different question. Every view is an accessor on the `Volcano`
(`v.scenes`, `v.descriptors`, `v.rays`, `v.anomalies`, `v.episodes`), and
calling it runs its main method: `v.rays()` is `v.rays.descriptors()`.

```
raw data ──► scenes ──► descriptors
    │           │
    └──► rays   └──► anomalies ──► episodes
```

| View | Unit | Answers | Main API |
|---|---|---|---|
| **Scene** | one date (8 directions × 30 distances) | What did the volcano look like that day? | `scenes()`, `scenes.reduce()`, `plot.scene()` |
| **Descriptors** | one scene condensed into 5 numbers | How hot, how far-reaching, and towards where? | `descriptors()`, `plot.polar()`, `plot.azimuth_rose()`, `plot.centers_map()` |
| **Ray** | one direction of one date | Is something happening in a single direction? | `rays()`, `rays.flag()`, `rays.mask()`, `plot.rays_rose()` |
| **Anomaly** | one scene against the volcano's usual state | Is this unusual *for this volcano*? | `anomalies.baseline()`, `anomalies()`, `plot.baseline()` |
| **Episode** | a run of consecutive anomalous scenes | When was there unusual activity, for how long, how intense? | `episodes.score()`, `episodes()`, `plot.episodes()` |

A typical flow:

```python
v = datasets.load("kadovar").rays.mask()  # 1. mask corrupted rays
v.descriptors()                           # 2. what happened on each date
v.episodes().query("n_scenes >= 2")       # 3. when there were events
```

## 📓 Tutorials

- [tutorials/00_quickstart.ipynb](tutorials/00_quickstart.ipynb) — full `Volcano` API
  walkthrough: loading datasets, plotting, the `.stats` accessor,
  missing-value reporting and imputation.
- [tutorials/01_comparison.ipynb](tutorials/01_comparison.ipynb) — compares
  several bundled volcanoes side by side (and overlaid).
- [tutorials/02_scenes.ipynb](tutorials/02_scenes.ipynb) — reshapes measurements
  into one scene per date (`scenes()`) and collapses directions
  (`scenes.reduce()`, `scenes.median()`, `scenes.max()`).
- [tutorials/03_scene_descriptors.ipynb](tutorials/03_scene_descriptors.ipynb) —
  condenses every scene into five descriptors (`descriptors()`),
  with caveats on corrupted values (the Kadovar case) and far-field weight.
- [tutorials/04_rays.ipynb](tutorials/04_rays.ipynb) — describes every direction
  of every scene (`rays()`), residual saturation, and flagging
  and masking corrupted rays (`rays.flag()`, `rays.mask()`).
- [tutorials/05_anomalies.ipynb](tutorials/05_anomalies.ipynb) — measures every
  scene against the volcano's usual state (`anomalies.baseline()`,
  `anomalies()`), making scenes comparable across volcanoes.
- [tutorials/06_episodes.ipynb](tutorials/06_episodes.ipynb) — finds periods of
  unusual activity as runs of anomalous scenes (`episodes.score()`,
  `episodes()`).

## 📜 License

Volcanicus is under
[The 3-Clause BSD License](LICENSE.txt)

This license allows unlimited redistribution for any purpose as long as
its copyright notices and the license's disclaimers of warranty are
maintained.

## 💬 Contact

**You can contact me at:** <hrosenfeld@gl.fcen.uba.ar>
