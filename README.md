# Volcanicus

<img src="res/cover.jpg" alt="Volcanicus" width="320">

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

From GitHub:

```bash
pip install https://github.com/leliel12/volcanicus/archive/refs/heads/master.zip
```

## 🚀 Usage

```python
from volcanicus import Volcano

volcano = Volcano.from_csv("measurements.csv", volcano_name="Copahue")

# normalized, read-only copy of the measurements
volcano.dataframe

# mean residual per distance bin, per direction
volcano.radial_profile()

# residual vs. distance, one line per direction, with an aggregate median line
volcano.plots.radial_profile()
```

### Bundled datasets

Volcanicus ships a few example datasets, loadable by name:

```python
from volcanicus import datasets

datasets.available()  # ["etna"]
etna = datasets.load("etna")
```

## 📓 Tutorial

See [notebooks/tutorial.ipynb](notebooks/tutorial.ipynb) for a full walkthrough:
loading datasets, plotting, the `.stats` accessor, and missing-value
reporting.

## 📜 License

Volcanicus is under
[The 3-Clause BSD License](LICENSE.txt)

This license allows unlimited redistribution for any purpose as long as
its copyright notices and the license's disclaimers of warranty are
maintained.

## 💬 Contact

**You can contact me at:** <jbcabral@unc.edu.ar>
