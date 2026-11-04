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

#: Per-pixel bands a parquet-backed dataset can be fetched or transformed
#: as; shared by :func:`~volcanicus.datasets.fetch` (its ``band`` parameter)
#: and :meth:`~volcanicus.core.Volcano.to_band`.
BANDS = ("tir", "lst", "swir1", "swir2", "fused")

# -----------------------------------------------------------------------
# Pivoting a per-pixel parquet into the wide ring tables fetch() returns
# (see volcanicus.datasets).
# -----------------------------------------------------------------------

#: Width of each distance ring, in meters.
RING_WIDTH_M = 30

#: Outer radius of the profile, in meters.
MAX_RADIUS_M = 900

#: Columns that are not distance rings.
METADATA_COLUMNS = ("date", "center_lat", "center_long", "direction")

#: Sector index (0 = N, clockwise in 45° steps) to compass direction.
SECTOR_TO_DIRECTION = {
    0: "N",
    1: "NE",
    2: "E",
    3: "SE",
    4: "S",
    5: "SO",
    6: "O",
    7: "NO",
}

#: Pixel columns the tables are built from.
REQUIRED_COLUMNS = (
    "stem",
    "date",
    "sector",
    "distance_m",
    "focus_lat",
    "focus_lon",
    "sat",
    "lst_k",
    "bg_tir_k",
    "sat_tir_final",
    "resn_tir",
    "lst_corr_k",
    "sat_swir1",
    "sat_swir2",
    "resn_swir1",
    "resn_swir2",
    "p_tir_w",
    "p_swir_w",
)

#: Fusion parameters. The first three are only defaults: ``fetch`` and
#: ``_band_table`` let the caller override them.
#: Temperature (K) at which the TIR of each Landsat sensor (``sat`` column)
#: saturates, as in ``LST_SAT_POR_SENSOR`` of the data collection script. It
#: gives the lower bound of the saturated pixels in the ``"sum"`` fusion and
#: maps to ``lst_norm = 1`` in the ``"blend"`` one.
LST_SATURATION_K = {"L7": 347.5, "L8": 373.0, "L9": 373.0}
#: How TIR and SWIR are combined: ``"sum"`` (TIR + SWIR powers) or the old
#: ``"blend"`` (sigmoid-weighted average, needs ``tir`` and ``swir`` columns).
FUSION_METHOD = "sum"
#: Pixel area (m2), TIR emissivity and Stefan-Boltzmann constant, as in the
#: data collection script (``p_tir_w``).
PIXEL_AREA_M2 = 900.0
EMISSIVITY_TIR = 0.95
SIGMA_SB = 5.670374e-8
#: ``lst_norm`` at which TIR and SWIR weigh the same (only for ``"blend"``).
FUSION_CENTER = 0.82
#: Width of the TIR -> SWIR transition, in ``lst_norm`` units.
FUSION_SCALE = 0.055
#: Only pixels below this ``lst_norm`` calibrate SWIR against TIR.
CALIBRATION_MAX_LST_NORM = 0.90
#: Minimum number of pixels needed to calibrate; otherwise identity.
CALIBRATION_MIN_PIXELS = 50
#: Bounds for the calibration slope and intercept.
CALIBRATION_SLOPE_RANGE = (0.25, 4.0)
CALIBRATION_INTERCEPT_RANGE = (-8.0, 8.0)
