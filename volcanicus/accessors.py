#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Accessors for :class:`~volcanicus.core.Volcano` instances (e.g. \
``volcano.plots``)."""

# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

from .utils import AccessorABC

# =============================================================================
# CONSTANTS
# =============================================================================

#: Columns in the source CSV that are not distance bins.
#:
#: Duplicated from ``volcanicus.core`` (instead of imported) to avoid a
#: circular import: ``core`` imports :class:`PlotAccessor` from this module.
_NON_DISTANCE_COLUMNS = ("date", "center_lat", "center_long", "direction")

# =============================================================================
# CLASSES
# =============================================================================


class PlotAccessor(AccessorABC):
    """Plotting accessor for a :class:`~volcanicus.core.Volcano` instance \
    (used as ``volcano.plots``).

    Instances are callable: ``volcano.plots(kind, **kwargs)`` dispatches to
    the method named ``kind`` (``"radial_profile"`` by default, see
    :attr:`_default_kind`).

    Parameters
    ----------
    volcano : Volcano
        The :class:`~volcanicus.core.Volcano` instance this accessor belongs
        to.

    """

    _default_kind = "radial_profile"

    def __init__(self, volcano):
        self._volcano = volcano

    def radial_profile(
        self, ax=None, direction_kwds=None, mean_kwds=None, **kwargs
    ):
        """Plot residual values against distance, one line per direction.

        There are two underlying seaborn plots: one line per direction, and
        (optionally) one aggregate line summarizing all directions. Since
        there are only 8 directions, that aggregate uses the median rather
        than the mean, to be less sensitive to any single outlying
        direction. To keep the plot simple and easy to read at a glance, the
        median line never shows error bars — only ``std`` for the
        per-direction lines is configurable. Instead of hard-coding their
        keyword arguments, ``direction_kwds`` and ``mean_kwds`` let the
        caller override or extend what each ``sns.lineplot`` call receives;
        anything not explicitly provided falls back to this method's own
        defaults (``errorbar="sd"``/``err_style="bars"`` for the
        per-direction plot with ``hue="direction"``, and
        ``color="black"``/``estimator="median"``/``label="Median"`` for the
        aggregate plot).

        Uses the raw (non-averaged) measurements so seaborn can compute, for
        every direction, error bars across dates at each distance bin — not
        just for the overall mean.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if not given.
        direction_kwds : dict, optional
            Keyword arguments forwarded to the per-direction ``sns.lineplot``
            call, on top of this method's own defaults.
        mean_kwds : dict, optional
            Keyword arguments forwarded to the aggregate ``sns.lineplot``
            call, on top of this method's own defaults.
        **kwargs
            ``mean`` : bool, default True. Whether to draw the aggregate
            (median) line at all. ``std`` : bool, default False. Whether
            error bars are shown on the per-direction lines (the median line
            never shows error bars); ignored if ``errorbar`` is set
            explicitly via ``direction_kwds``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        """
        mean = kwargs.get("mean", True)
        std = kwargs.get("std", False)

        # Fill in this method's own defaults for whatever the caller didn't
        # already set explicitly via direction_kwds/mean_kwds.
        direction_kwds = {} if direction_kwds is None else direction_kwds
        direction_kwds.setdefault("alpha", .5 if mean else 1)
        direction_kwds.setdefault("hue", "direction")
        direction_kwds.setdefault("errorbar", "sd" if std else None)
        direction_kwds.setdefault("err_style", "bars")

        mean_kwds = {} if mean_kwds is None else mean_kwds
        mean_kwds.setdefault("color", "black")
        mean_kwds.setdefault("linewidth", 2)
        mean_kwds.setdefault("errorbar", None)
        mean_kwds.setdefault("estimator", "median")
        mean_kwds.setdefault("label", "Median")

        # Reshape from one-column-per-distance-bin (wide) to one row per
        # (direction, distance, residual) observation (long), which is what
        # seaborn's lineplot expects for it to compute per-direction error
        # bars across the repeated (raw, non-averaged) observations.
        df = self._volcano.dataframe
        distance_columns = [
            c for c in df.columns if c not in _NON_DISTANCE_COLUMNS
        ]
        long_df = df.melt(
            id_vars="direction",
            value_vars=distance_columns,
            var_name="distance",
            value_name="residual",
        )

        ax = plt.gca() if ax is None else ax

        # Layer 1: one line per direction, drawn from the raw data above.
        sns.lineplot(
            data=long_df, x="distance", y="residual", ax=ax, **direction_kwds
        )

        if mean:
            # Layer 2 (optional): a single aggregate line summarizing all
            # directions. Volcano.radial_profile() already reduces to one
            # row per direction; the median line's errorbar then reflects
            # spread *between directions*, not between raw dates. Median
            # (not mean) because there are only 8 directions, so it's more
            # robust to any single outlying direction.
            by_direction = self._volcano.radial_profile().reset_index()
            mean_long_df = by_direction.melt(
                id_vars="direction", var_name="distance", value_name="residual"
            )
            sns.lineplot(
                data=mean_long_df,
                x="distance",
                y="residual",
                ax=ax,
                **mean_kwds,
            )

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Residual")
        ax.set_title(
            f"{self._volcano.name}: residual vs. distance by direction"
        )
        ax.tick_params(axis="x", rotation=45)
        return ax

    def temporal_profile(
        self, ax=None, direction_kwds=None, mean_kwds=None, **kwargs
    ):
        """Plot residual values against date, one line per direction.

        Temporal mirror of :meth:`radial_profile`: same two-layer plot (one
        line per direction, plus an optional aggregate median line), but
        with ``date`` on the x-axis. Error bars on the per-direction lines
        are computed across distance bins (repeated observations at each
        date), instead of across dates.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if not given.
        direction_kwds : dict, optional
            Keyword arguments forwarded to the per-direction ``sns.lineplot``
            call, on top of this method's own defaults.
        mean_kwds : dict, optional
            Keyword arguments forwarded to the aggregate ``sns.lineplot``
            call, on top of this method's own defaults.
        **kwargs
            ``mean`` : bool, default True. Whether to draw the aggregate
            (median) line at all. ``std`` : bool, default False. Whether
            error bars are shown on the per-direction lines (the median line
            never shows error bars); ignored if ``errorbar`` is set
            explicitly via ``direction_kwds``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        """
        mean = kwargs.get("mean", True)
        std = kwargs.get("std", False)

        # Fill in this method's own defaults for whatever the caller didn't
        # already set explicitly via direction_kwds/mean_kwds.
        direction_kwds = {} if direction_kwds is None else direction_kwds
        direction_kwds.setdefault("alpha", .5 if mean else 1)
        direction_kwds.setdefault("hue", "direction")
        direction_kwds.setdefault("errorbar", "sd" if std else None)
        direction_kwds.setdefault("err_style", "bars")

        mean_kwds = {} if mean_kwds is None else mean_kwds
        mean_kwds.setdefault("color", "black")
        mean_kwds.setdefault("linewidth", 2)
        mean_kwds.setdefault("errorbar", None)
        mean_kwds.setdefault("estimator", "median")
        mean_kwds.setdefault("label", "Median")

        # Reshape from one-column-per-distance-bin (wide) to one row per
        # (date, direction, distance, residual) observation (long); this
        # time keeping date/direction so seaborn can compute, per direction
        # and date, error bars across the repeated distance-bin
        # observations (the temporal mirror of radial_profile's melt).
        df = self._volcano.dataframe
        distance_columns = [
            c for c in df.columns if c not in _NON_DISTANCE_COLUMNS
        ]
        long_df = df.melt(
            id_vars=["date", "direction"],
            value_vars=distance_columns,
            var_name="distance",
            value_name="residual",
        )

        ax = plt.gca() if ax is None else ax

        # Layer 1: one line per direction, drawn from the raw data above.
        sns.lineplot(
            data=long_df, x="date", y="residual", ax=ax, **direction_kwds
        )

        if mean:
            # Layer 2 (optional): a single aggregate line summarizing all
            # directions. Volcano.temporal_profile() already reduces to one
            # row per date; the median line's errorbar then reflects spread
            # *between directions*, not between raw distance bins. Median
            # (not mean) because there are only 8 directions, so it's more
            # robust to any single outlying direction.
            by_date = self._volcano.temporal_profile().reset_index()
            mean_long_df = by_date.melt(
                id_vars="date", var_name="direction", value_name="residual"
            )
            sns.lineplot(
                data=mean_long_df,
                x="date",
                y="residual",
                ax=ax,
                **mean_kwds,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Residual")
        ax.set_title(f"{self._volcano.name}: residual vs. date by direction")
        ax.tick_params(axis="x", rotation=45)
        return ax


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

    #: Columns ``groupby`` is allowed to group the measurements by.
    _GROUPBY_WHITELIST = ("date", "direction")

    _default_kind = "describe"

    def __init__(self, volcano):
        self._volcano = volcano

    def _grouped(self, groupby):
        # Reject anything but "date"/"direction" up front, before doing any
        # work, so every stat method gets the same clear error for free.
        if groupby not in self._GROUPBY_WHITELIST:
            raise ValueError(
                f"'groupby' must be one of {self._GROUPBY_WHITELIST}, "
                f"found {groupby!r}"
            )
        df = self._volcano.dataframe
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

