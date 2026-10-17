#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Statistics accessor for :class:`~volcanicus.core.Volcano` instances \
(``volcano.stats``)."""

# =============================================================================
# IMPORTS
# =============================================================================

from .base import AccessorABC
from ..constants import (
    GROUPBY_WHITELIST,
    NON_DISTANCE_COLUMNS as _NON_DISTANCE_COLUMNS,
)

# =============================================================================
# CLASSES
# =============================================================================


class StatsAccessor(AccessorABC):
    """Calculate basic statistics of the volcano measurements \
    (used as ``volcano.stats``).

    Kind of statistic to produce:

    - 'cov' : Compute pairwise covariance of distance-bin columns,
      excluding NA/null values.
    - 'describe' : Generate descriptive statistics.
    - 'kurtosis' : Return unbiased kurtosis over requested axis.
    - 'max' : Return the maximum of the values over the requested axis.
    - 'mean' : Return the mean of the values over the requested axis.
    - 'median' : Return the median of the values over the requested
      axis.
    - 'min' : Return the minimum of the values over the requested axis.
    - 'pct_change' : Percentage change between the current and a prior
      element.
    - 'quantile' : Return values at the given quantile over requested
      axis.
    - 'sem' : Return unbiased standard error of the mean over requested
      axis.
    - 'skew' : Return unbiased skew over requested axis.
    - 'std' : Return sample standard deviation over requested axis.
    - 'var' : Return unbiased variance over requested axis.

    Every statistic is computed per group: the mandatory first parameter
    ``groupby`` (either ``"date"`` or ``"direction"``) selects which column
    to group the measurements by before the underlying pandas method is
    applied.

    Parameters
    ----------
    volcano : Volcano
        The :class:`~volcanicus.core.Volcano` instance this accessor belongs
        to.

    """

    # The list of methods that can be accessed of the subjacent dataframe.
    _DF_WHITELIST = (
        "cov",
        "describe",
        "kurtosis",
        "max",
        "mean",
        "median",
        "min",
        "pct_change",
        "quantile",
        "sem",
        "skew",
        "std",
        "var",
    )
    _default_kind = "describe"

    def __init__(self, volcano):
        self._volcano = volcano

    def _grouped(self, groupby):
        """Distance-bin columns of the volcano's DataFrame, grouped."""
        # Reject anything but "date"/"direction" up front, before doing any
        # work, so every stat method gets the same clear error for free.
        if groupby not in GROUPBY_WHITELIST:
            raise ValueError(
                f"'groupby' must be one of {GROUPBY_WHITELIST}, "
                f"found {groupby!r}"
            )
        df = self._volcano.to_dataframe()
        distance_columns = [
            c for c in df.columns if c not in _NON_DISTANCE_COLUMNS
        ]
        # Pre-select the distance columns so callers' pandas method calls
        # (mean, describe, ...) never see date/direction/lat/long.
        return df.groupby(groupby, observed=True)[distance_columns]

    def __getattr__(self, a):
        """x.__getattr__(a) <==> x.a <==> getattr(x, "a")."""
        if a not in self._DF_WHITELIST:
            raise AttributeError(a)

        def method(groupby, **kwargs):
            grouped = self._grouped(groupby)
            return grouped.apply(lambda g: getattr(g, a)(**kwargs))

        return method

    def __dir__(self):
        """x.__dir__() <==> dir(x)."""
        return super().__dir__() + list(self._DF_WHITELIST)
