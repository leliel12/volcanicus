#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Accessors of :class:`~volcanicus.core.Volcano` instances.

Every accessor is a property of the volcano (``volcano.plot``,
``volcano.scenes``, ...) and follows :class:`AccessorABC`: calling it runs
its default method.

- Tools: :class:`PlotAccessor` (``volcano.plot``) and
  :class:`StatsAccessor` (``volcano.stats``).
- Views of the data: :class:`ScenesAccessor` (``volcano.scenes``),
  :class:`DescriptorsAccessor` (``volcano.descriptors``),
  :class:`RaysAccessor` (``volcano.rays``), :class:`AnomaliesAccessor`
  (``volcano.anomalies``) and :class:`EpisodesAccessor`
  (``volcano.episodes``).

"""

# =============================================================================
# IMPORTS
# =============================================================================

from .anomalies import AnomaliesAccessor
from .base import AccessorABC
from .descriptors import DescriptorsAccessor
from .episodes import EpisodesAccessor
from .plot import PlotAccessor
from .rays import RaysAccessor
from .scenes import ScenesAccessor
from .stats import StatsAccessor

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "AccessorABC",
    "PlotAccessor",
    "StatsAccessor",
    "ScenesAccessor",
    "DescriptorsAccessor",
    "RaysAccessor",
    "AnomaliesAccessor",
    "EpisodesAccessor",
]
