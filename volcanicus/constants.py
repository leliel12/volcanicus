#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Constants shared across the :mod:`volcanicus` package.

Centralized here (instead of living next to their main user) so
:mod:`volcanicus.core` and the :mod:`volcanicus.accessors` can all import
them without running into a circular import.

"""

# =============================================================================
# CONSTANTS
# =============================================================================

#: Columns in the source CSV that are not distance bins.
NON_DISTANCE_COLUMNS = ("date", "center_lat", "center_long", "direction")

#: Compass directions in clockwise order, used by
#: :meth:`~volcanicus.core.Volcano.impute` to find the two directions
#: adjacent to a fully-missing one.
DIRECTION_ORDER = ("N", "NE", "E", "SE", "S", "SO", "O", "NO")

#: Columns that measurements can be grouped by: shared by
#: :meth:`~volcanicus.core.Volcano.describe` (its ``by`` parameter) and
#: :class:`~volcanicus.accessors.stats.StatsAccessor` (its ``groupby``
#: parameter).
GROUPBY_WHITELIST = ("date", "direction")

#: Reductions :meth:`~volcanicus.accessors.scenes.ScenesAccessor.reduce`
#: accepts for its ``how`` parameter.
REDUCE_DIRECTIONS_WHITELIST = (
    "count",
    "max",
    "mean",
    "median",
    "min",
    "std",
    "sum",
    "var",
)

#: Level at which positive residuals saturate: the bundled data caps them at
#: exactly ``1.0`` (thousands of values sit right there, only a handful just
#: below), so a residual at this level means "at least this hot".
SATURATION_LEVEL = 1.0

#: Meters per degree of latitude (and of longitude at the equator). Enough
#: to place a volcano's centers, a few hundred meters apart, on a flat earth.
METERS_PER_DEGREE = 111_320.0
