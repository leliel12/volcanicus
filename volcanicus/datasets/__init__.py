#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The :mod:`volcanicus.datasets` module includes utilities to load \
bundled example :class:`~volcanicus.core.Volcano` datasets.

Each dataset lives in its own directory, ``<name>/``, holding the
measurements (``<name>.csv``) and the volcano's descriptive metadata
(``<name>.json``).

"""

# =============================================================================
# IMPORTS
# =============================================================================

import functools
import json
import os
import pathlib

import pandas as pd

from ..core import Volcano

# =============================================================================
# CONSTANTS
# =============================================================================

_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# =============================================================================
# FUNCTIONS
# =============================================================================


@functools.cache
def available() -> list[str]:
    """List the volcano names bundled with :mod:`volcanicus.datasets`.

    Returns
    -------
    list of str
        Names accepted by :func:`load`, sorted alphabetically.

    """
    return sorted(
        p.parts[-2]
        for p in _PATH.glob("*/*.csv")
        if not p.parts[-2].startswith(".")
    )


def load(volcano_name):
    """Load a bundled example :class:`~volcanicus.core.Volcano` dataset.

    Parameters
    ----------
    volcano_name : str
        Name of the bundled volcano to load (see :func:`available` for the
        valid values). Matched case-sensitively against the dataset
        directory name.

    Returns
    -------
    Volcano
        A new instance built from the bundled ``<name>.csv`` measurements,
        with the contents of ``<name>.json`` as its
        :attr:`~volcanicus.core.Volcano.metadata`.

    Raises
    ------
    ValueError
        If ``volcano_name`` is not a bundled dataset.

    """
    data_path = _PATH / volcano_name / f"{volcano_name}.csv"
    metadata_path = _PATH / volcano_name / f"{volcano_name}.json"
    if not data_path.exists():
        raise ValueError(f"Unknown volcano {volcano_name!r}")

    data = pd.read_csv(data_path)
    metadata = json.loads(metadata_path.read_text())

    return Volcano(name=volcano_name, df=data, metadata=metadata)
