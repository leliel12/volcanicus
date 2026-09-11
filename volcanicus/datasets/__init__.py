#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The :mod:`volcanicus.datasets` module includes utilities to load \
bundled example :class:`~volcanicus.core.Volcano` datasets."""

# =============================================================================
# IMPORTS
# =============================================================================

import functools
import os
import pathlib

from ..core import Volcano

# =============================================================================
# CONSTANTS
# =============================================================================

_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# =============================================================================
# FUNCTIONS
# =============================================================================


@functools.cache
def available():
    """List the volcano names bundled with :mod:`volcanicus.datasets`.

    Returns
    -------
    list of str
        Names accepted by :func:`load`, sorted alphabetically.

    """
    return sorted(p.stem for p in _PATH.glob("*.csv"))


def load(volcano_name):
    """Load a bundled example :class:`~volcanicus.core.Volcano` dataset.

    Parameters
    ----------
    volcano_name : str
        Name of the bundled volcano to load (see :func:`available` for the
        valid values). Matched case-sensitively against the CSV file name.

    Returns
    -------
    Volcano
        A new instance built from the bundled CSV contents.

    """
    path = _PATH / f"{volcano_name}.csv"
    if not path.exists():
        raise ValueError(f"Unknown volcano {volcano_name!r}. ")
    return Volcano.from_csv(path, volcano_name)
