#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Volcanicus is a small toolkit to load, normalize and plot volcano \
deformation/residual measurements."""

# =============================================================================
# IMPORTS
# =============================================================================

from . import datasets
from .accessors import PlotAccessor, StatsAccessor
from .core import Volcano, normalize_dataframe, read_csv

# =============================================================================
# CONSTANTS
# =============================================================================

__all__ = [
    "Volcano",
    "normalize_dataframe",
    "read_csv",
    "PlotAccessor",
    "StatsAccessor",
    "datasets",
]
