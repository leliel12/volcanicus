#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Accessors for :class:`~volcanicus.core.Volcano` instances (e.g. \
``volcano.plot``)."""

# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from .constants import (
    DIRECTION_ORDER,
    GROUPBY_WHITELIST,
    NON_DISTANCE_COLUMNS as _NON_DISTANCE_COLUMNS,
)
from .utils import AccessorABC

# =============================================================================
# CLASSES
# =============================================================================


class PlotAccessor(AccessorABC):
    """Plotting accessor for a :class:`~volcanicus.core.Volcano` instance \
    (used as ``volcano.plot``).

    Instances are callable: ``volcano.plot(kind, **kwargs)`` dispatches to
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
        self,
        ax=None,
        direction_kwds=None,
        median_kwds=None,
        directions=True,
        median=True,
        std=False,
    ):
        """Plot residual values against distance, one line per direction.

        There are two underlying seaborn plots: one line per direction, and
        (optionally) one aggregate line summarizing all directions. Since
        there are only 8 directions, that aggregate uses the median rather
        than the mean, to be less sensitive to any single outlying
        direction. To keep the plot simple and easy to read at a glance, the
        median line never shows error bars — only ``std`` for the
        per-direction lines is configurable. Instead of hard-coding their
        keyword arguments, ``direction_kwds`` and ``median_kwds`` let the
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
        median_kwds : dict, optional
            Keyword arguments forwarded to the aggregate ``sns.lineplot``
            call, on top of this method's own defaults.
        directions : bool, default True
            Whether to draw the per-direction lines at all — turn this off
            (with ``median=True``) to get just the aggregate line, e.g. to
            overlay several volcanoes' median curves on the same axes for
            comparison. This is a plotting-only switch: it has no
            equivalent on :meth:`Volcano.radial_profile`.
        median : bool, default True
            Whether to draw the aggregate (median) line at all.
        std : bool, default False
            Whether error bars are shown on the per-direction lines (the
            median line never shows error bars); ignored if ``errorbar`` is
            set explicitly via ``direction_kwds``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        """
        # Fill in this method's own defaults for whatever the caller didn't
        # already set explicitly via direction_kwds/median_kwds.
        direction_kwds = {} if direction_kwds is None else direction_kwds
        direction_kwds.setdefault("alpha", 0.5 if median else 1)
        direction_kwds.setdefault("hue", "direction")
        direction_kwds.setdefault("errorbar", "sd" if std else None)
        direction_kwds.setdefault("err_style", "bars")

        median_kwds = {} if median_kwds is None else median_kwds
        median_kwds.setdefault("color", "black")
        median_kwds.setdefault("linewidth", 2)
        median_kwds.setdefault("errorbar", None)
        median_kwds.setdefault("estimator", "median")
        median_kwds.setdefault("label", "Median")

        ax = plt.gca() if ax is None else ax

        if directions:
            # Layer 1 (optional): one line per direction. Reshape from
            # one-column-per-distance-bin (wide) to one row per (direction,
            # distance, residual) observation (long), which is what
            # seaborn's lineplot expects for it to compute per-direction
            # error bars across the repeated (raw, non-averaged)
            # observations.
            df = self._volcano.to_dataframe()
            distance_columns = [
                c for c in df.columns if c not in _NON_DISTANCE_COLUMNS
            ]
            long_df = df.melt(
                id_vars="direction",
                value_vars=distance_columns,
                var_name="distance",
                value_name="residual",
            )
            sns.lineplot(
                data=long_df,
                x="distance",
                y="residual",
                ax=ax,
                **direction_kwds,
            )

        if median:
            # Layer 2 (optional): a single aggregate line summarizing all
            # directions. Volcano.radial_profile() already reduces to one
            # row per direction; the median line's errorbar then reflects
            # spread *between directions*, not between raw dates. Median
            # (not mean) because there are only 8 directions, so it's more
            # robust to any single outlying direction.
            by_direction = self._volcano.radial_profile().reset_index()
            median_long_df = by_direction.melt(
                id_vars="direction", var_name="distance", value_name="residual"
            )
            sns.lineplot(
                data=median_long_df,
                x="distance",
                y="residual",
                ax=ax,
                **median_kwds,
            )

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Residual")
        ax.set_title(
            f"{self._volcano.name}: residual vs. distance by direction"
        )
        ax.tick_params(axis="x", rotation=45)
        return ax

    def temporal_profile(
        self,
        ax=None,
        direction_kwds=None,
        median_kwds=None,
        directions=True,
        median=True,
        std=False,
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
        median_kwds : dict, optional
            Keyword arguments forwarded to the aggregate ``sns.lineplot``
            call, on top of this method's own defaults.
        directions : bool, default True
            Whether to draw the per-direction lines at all — turn this off
            (with ``median=True``) to get just the aggregate line, e.g. to
            overlay several volcanoes' median curves on the same axes for
            comparison. This is a plotting-only switch: it has no
            equivalent on :meth:`Volcano.temporal_profile`.
        median : bool, default True
            Whether to draw the aggregate (median) line at all.
        std : bool, default False
            Whether error bars are shown on the per-direction lines (the
            median line never shows error bars); ignored if ``errorbar`` is
            set explicitly via ``direction_kwds``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        """
        # Fill in this method's own defaults for whatever the caller didn't
        # already set explicitly via direction_kwds/median_kwds.
        direction_kwds = {} if direction_kwds is None else direction_kwds
        direction_kwds.setdefault("alpha", 0.5 if median else 1)
        direction_kwds.setdefault("hue", "direction")
        direction_kwds.setdefault("errorbar", "sd" if std else None)
        direction_kwds.setdefault("err_style", "bars")

        median_kwds = {} if median_kwds is None else median_kwds
        median_kwds.setdefault("color", "black")
        median_kwds.setdefault("linewidth", 2)
        median_kwds.setdefault("errorbar", None)
        median_kwds.setdefault("estimator", "median")
        median_kwds.setdefault("label", "Median")

        ax = plt.gca() if ax is None else ax

        if directions:
            # Layer 1 (optional): one line per direction. Reshape from
            # one-column-per-distance-bin (wide) to one row per (date,
            # direction, distance, residual) observation (long); this time
            # keeping date/direction so seaborn can compute, per direction
            # and date, error bars across the repeated distance-bin
            # observations (the temporal mirror of radial_profile's melt).
            df = self._volcano.to_dataframe()
            distance_columns = [
                c for c in df.columns if c not in _NON_DISTANCE_COLUMNS
            ]
            long_df = df.melt(
                id_vars=["date", "direction"],
                value_vars=distance_columns,
                var_name="distance",
                value_name="residual",
            )
            sns.lineplot(
                data=long_df, x="date", y="residual", ax=ax, **direction_kwds
            )

        if median:
            # Layer 2 (optional): a single aggregate line summarizing all
            # directions. Volcano.temporal_profile() already reduces to one
            # row per date; the median line's errorbar then reflects spread
            # *between directions*, not between raw distance bins. Median
            # (not mean) because there are only 8 directions, so it's more
            # robust to any single outlying direction.
            by_date = self._volcano.temporal_profile().reset_index()
            median_long_df = by_date.melt(
                id_vars="date", var_name="direction", value_name="residual"
            )
            sns.lineplot(
                data=median_long_df,
                x="date",
                y="residual",
                ax=ax,
                **median_kwds,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Residual")
        ax.set_title(f"{self._volcano.name}: residual vs. date by direction")
        ax.tick_params(axis="x", rotation=45)
        return ax

    def polar(
        self, ax=None, dates=None, cmap="viridis", colorbar=True, **kwargs
    ):
        """Plot every date as a polygon on a compass, colored by date.

        Each date becomes a closed polygon with one vertex per direction:
        the vertex's angle is the direction's compass bearing (north up,
        clockwise) and its radius the residual of that date/direction
        averaged over every distance bin (:meth:`Volcano.temporal_profile`).
        A round polygon means the residual is even around the vent; a
        lopsided one points towards where it's higher.

        Residuals can be negative, so the radial axis starts at the lowest
        value instead of 0; a dashed circle marks the zero residual. A
        direction missing on a date leaves a gap in that date's polygon.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Polar axes to draw on (``projection="polar"``). New ones are
            created if not given.
        dates : list-like, optional
            Dates to plot. Defaults to every date.
        cmap : str or matplotlib.colors.Colormap, default "viridis"
            Colormap mapping each polygon's date to its color.
        colorbar : bool, default True
            Whether to add a colorbar labeling the dates.
        **kwargs
            Forwarded to every polygon's ``ax.plot`` call, on top of this
            method's defaults (``linewidth=1``, ``alpha=0.6``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        Raises
        ------
        ValueError
            If ``ax`` is not a polar axes.

        """
        kwargs.setdefault("linewidth", 1)
        kwargs.setdefault("alpha", 0.6)

        if ax is None:
            ax = plt.subplot(projection="polar")
        elif ax.name != "polar":
            raise ValueError(
                "'ax' must be a polar axes (projection='polar'), "
                f"found {ax.name!r}"
            )

        # One row per date, one column per direction, in compass order (the
        # pivot inside temporal_profile sorts them alphabetically).
        profile = self._volcano.temporal_profile().reindex(
            columns=list(DIRECTION_ORDER)
        )
        if dates is not None:
            profile = profile.loc[pd.to_datetime(dates)]

        # Repeat the first direction at the end so every polygon closes.
        bearings = np.radians(45.0 * np.arange(len(DIRECTION_ORDER) + 1))
        closed = np.column_stack([profile, profile.iloc[:, 0]])

        cmap = plt.get_cmap(cmap)
        when = mdates.date2num(profile.index)
        norm = mcolors.Normalize(when.min(), when.max())
        for date_num, radii in zip(when, closed):
            ax.plot(bearings, radii, color=cmap(norm(date_num)), **kwargs)

        # Zero residual as a reference: the radial axis starts at the
        # lowest residual, so the center isn't 0.
        full_circle = np.linspace(0, 2 * np.pi, 100)
        ax.plot(full_circle, np.zeros_like(full_circle), "k--", linewidth=1)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise, like a compass
        ax.set_xticks(bearings[:-1], DIRECTION_ORDER)
        ax.set_title(f"{self._volcano.name}: residual by direction and date")

        if colorbar:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = ax.figure.colorbar(mappable, ax=ax, pad=0.1)
            # Concise labels adapt to the span: years for decades of
            # data, months/days for a handful of nearby dates.
            locator = mdates.AutoDateLocator()
            cbar.ax.yaxis.set_major_locator(locator)
            cbar.ax.yaxis.set_major_formatter(
                mdates.ConciseDateFormatter(locator)
            )
            cbar.set_label("Date")
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
