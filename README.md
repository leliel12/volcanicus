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

etna.metadata.country       # "Italy"
etna.m.latitude, etna.m.longitude  # `.m` is a shorthand for `.metadata`
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
bin:

```python
import pandas as pd
from volcanicus import Volcano

df = pd.read_csv("measurements.csv")
volcano = Volcano("copahue", df, metadata={"country": "Argentina"})
```

## 📓 Tutorials

- [notebooks/tutorial.ipynb](notebooks/tutorial.ipynb) — full `Volcano` API
  walkthrough: loading datasets, plotting, the `.stats` accessor,
  missing-value reporting and imputation.
- [notebooks/comparison.ipynb](notebooks/comparison.ipynb) — compares
  several bundled volcanoes side by side (and overlaid).

## 📜 License

Volcanicus is under
[The 3-Clause BSD License](LICENSE.txt)

This license allows unlimited redistribution for any purpose as long as
its copyright notices and the license's disclaimers of warranty are
maintained.

## 💬 Contact

**You can contact me at:** <hrosenfeld@gl.fcen.uba.ar>
