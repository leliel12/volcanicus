#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Plotting accessor for :class:`~volcanicus.core.Volcano` instances \
(``volcano.plot``)."""

# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

import pandas as pd

from scipy.stats import gaussian_kde

import seaborn as sns

from .base import AccessorABC
from ..constants import (
    DIRECTION_ORDER,
    METERS_PER_DEGREE,
    NON_DISTANCE_COLUMNS as _NON_DISTANCE_COLUMNS,
)

# =============================================================================
# FUNCTIONS
# =============================================================================


def _polar_axes(ax):
    """Return ``ax`` if it's polar (new polar axes if None), else raise."""
    if ax is None:
        return plt.subplot(projection="polar")
    if ax.name != "polar":
        raise ValueError(
            "'ax' must be a polar axes (projection='polar'), "
            f"found {ax.name!r}"
        )
    return ax


def _draw_grid(ax, row, title, label, colorbar, **kwargs):
    """Draw a scene-shaped row (``(direction, distance)`` index) as an image.

    The row's index must run direction-major in
    :data:`~volcanicus.constants.DIRECTION_ORDER`, as the columns of
    :meth:`~volcanicus.accessors.scenes.ScenesAccessor.grid` do, so it reshapes
    straight into the (direction, distance) grid.
    """
    distances = list(row.index.unique("distance"))
    grid = row.to_numpy(dtype=float).reshape(
        len(DIRECTION_ORDER), len(distances)
    )

    kwargs.setdefault("aspect", "auto")
    ax = plt.gca() if ax is None else ax
    image = ax.imshow(grid, **kwargs)
    ax.set_yticks(range(len(DIRECTION_ORDER)), DIRECTION_ORDER)
    step = max(1, len(distances) // 6)
    ax.set_xticks(range(0, len(distances), step), distances[::step])
    ax.set_xlabel("Distance")
    ax.set_ylabel("Direction")
    ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, label=label)
    return ax


def _as_compass(ax):
    """Lay out polar axes as a compass: north up, clockwise, labeled."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise, like a compass
    bearings = np.radians(45.0 * np.arange(len(DIRECTION_ORDER)))
    ax.set_xticks(bearings, DIRECTION_ORDER)


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

        ax = _polar_axes(ax)

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

        _as_compass(ax)
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

    def scene(self, date, anomaly=False, ax=None, colorbar=True, **kwargs):
        """Plot one scene as an image: directions x distance bins.

        Every row is a direction (in
        :data:`~volcanicus.constants.DIRECTION_ORDER`) and every column a
        distance bin, colored by the residual
        (:meth:`ScenesAccessor.grid`) or, with ``anomaly=True``, by its
        z-score against the volcano's baseline
        (:meth:`AnomaliesAccessor.zscores`). Missing cells stay blank.

        Parameters
        ----------
        date : str or datetime-like
            Date of the scene to plot.
        anomaly : bool, default False
            Whether to plot anomalies instead of raw residuals. They get a
            diverging colormap (``"RdBu_r"``) centered on 0, so red is
            hotter than usual and blue colder. The color range is
            symmetric and clipped at the 95th percentile of ``|z|`` (at
            least 3), so a few extreme cells don't wash out the rest.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if not given.
        colorbar : bool, default True
            Whether to add a colorbar.
        **kwargs
            Forwarded to ``ax.imshow``, on top of this method's defaults
            (``aspect="auto"``; for anomalies also ``cmap="RdBu_r"`` and
            ``vmin``/``vmax`` symmetric around 0).

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        Raises
        ------
        KeyError
            If the volcano has no scene on ``date``.

        """
        date = pd.Timestamp(date)
        scenes = (
            self._volcano.anomalies.zscores()
            if anomaly
            else self._volcano.scenes.grid()
        )
        if date not in scenes.index:
            raise KeyError(f"No scene on {date:%Y-%m-%d}")

        row = scenes.loc[date]
        if anomaly:
            # Symmetric limits keep 0 (the usual value) at the neutral
            # color of the diverging map; the floor of 3 keeps an ordinary
            # scene from looking alarming.
            values = row.to_numpy(dtype=float)
            finite = np.abs(values[np.isfinite(values)])
            limit = max(3.0, np.percentile(finite, 95)) if finite.size else 3.0
            kwargs.setdefault("cmap", "RdBu_r")
            kwargs.setdefault("vmin", -limit)
            kwargs.setdefault("vmax", limit)

        what = "anomaly (z)" if anomaly else "residual"
        return _draw_grid(
            ax,
            row,
            f"{self._volcano.name}: {what} on {date:%Y-%m-%d}",
            what.capitalize(),
            colorbar,
            **kwargs,
        )

    def baseline(
        self, stat="center", min_scenes=10, ax=None, colorbar=True, **kwargs
    ):
        """Plot the volcano's baseline as a directions x distance image.

        Same layout as :meth:`scene`, for one of the two rows of
        :meth:`AnomaliesAccessor.baseline`: every cell's usual value
        (``"center"``, the median over every scene) or its usual spread
        (``"scale"``, ``1.4826 * MAD``). Cells without a baseline (too few
        scenes, or never varying) stay blank.

        Parameters
        ----------
        stat : {"center", "scale"}, default "center"
            Which row of the baseline to plot.
        min_scenes : int, default 10
            Passed on to :meth:`AnomaliesAccessor.baseline`.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if not given.
        colorbar : bool, default True
            Whether to add a colorbar.
        **kwargs
            Forwarded to ``ax.imshow``, on top of ``aspect="auto"``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        Raises
        ------
        ValueError
            If ``stat`` is not ``"center"`` or ``"scale"``.

        """
        labels = {"center": "Center (median)", "scale": "Scale (1.4826 x MAD)"}
        if stat not in labels:
            raise ValueError(
                f"'stat' must be one of {tuple(labels)}, found {stat!r}"
            )
        row = self._volcano.anomalies.baseline(min_scenes).loc[stat]
        return _draw_grid(
            ax,
            row,
            f"{self._volcano.name}: baseline {stat}",
            labels[stat],
            colorbar,
            **kwargs,
        )

    def episodes(
        self,
        threshold=3.0,
        max_gap=90,
        how="median",
        min_scenes=10,
        min_episode_scenes=1,
        ax=None,
    ):
        """Plot the anomaly score over time, with the episodes shaded.

        Draws :meth:`EpisodesAccessor.score`, a dashed line at
        ``threshold`` and a shaded band over every episode
        (:meth:`EpisodesAccessor.find`). A single-scene episode has no
        duration, so its band is widened to 16 days (one satellite
        revisit) to stay visible.

        Parameters
        ----------
        threshold, max_gap, how, min_scenes
            Passed on to :meth:`EpisodesAccessor.find`.
        min_episode_scenes : int, default 1
            Only shade episodes with at least this many scenes. Single-scene
            episodes are mostly chance crossings of the threshold, so ``2``
            is a sensible choice to show only the solid ones.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if not given.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        """
        score = self._volcano.episodes.score(how, min_scenes)
        episodes = self._volcano.episodes.find(
            threshold, max_gap, how, min_scenes
        )
        episodes = episodes[episodes["n_scenes"] >= min_episode_scenes]

        ax = plt.gca() if ax is None else ax
        score.plot(ax=ax, marker=".", linewidth=0.5, color="gray")
        ax.axhline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Threshold",
        )
        # One revisit as a minimum width, so single scenes show up.
        min_width = np.timedelta64(16, "D")
        for i, episode in enumerate(episodes.itertuples()):
            ax.axvspan(
                episode.start,
                max(episode.end, episode.start + min_width),
                color="orange",
                alpha=0.4,
                label="Episode" if i == 0 else None,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Anomaly score")
        ax.set_title(f"{self._volcano.name}: anomaly score and episodes")
        ax.legend()
        return ax

    def azimuth_rose(self, bins=16, weighted=True, ax=None, **kwargs):
        """Plot where the heat points, over every scene, as a compass rose.

        A histogram of the scenes' ``azimuth``
        (:meth:`DescriptorsAccessor.table`) on a compass (north up,
        clockwise). By default every scene is weighted by its
        ``anisotropy``, so scenes where the heat is spread evenly (and the
        azimuth means little) barely count, and the rose shows where the
        heat goes when one side dominates. Scenes without an azimuth are
        left out.

        Parameters
        ----------
        bins : int, default 16
            Number of sectors, centered on the bearings: with 16, every
            sector spans 22.5° and the 8 compass directions fall at the
            center of every other one.
        weighted : bool, default True
            Whether to weight every scene by its ``anisotropy``. If False,
            the rose just counts scenes.
        ax : matplotlib.axes.Axes, optional
            Polar axes to draw on (``projection="polar"``). New ones are
            created if not given.
        **kwargs
            Forwarded to ``ax.bar``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        Raises
        ------
        ValueError
            If ``ax`` is not a polar axes.

        """
        ax = _polar_axes(ax)
        descriptors = self._volcano.descriptors.table().dropna(
            subset=["azimuth"]
        )

        # Shift by half a sector so sectors are centered on the bearings
        # (e.g. N spans -11.25° to 11.25° with 16 bins).
        width = 360.0 / bins
        shifted = (descriptors["azimuth"] + width / 2) % 360
        weights = descriptors["anisotropy"] if weighted else None
        counts, edges = np.histogram(
            shifted, bins=bins, range=(0, 360), weights=weights
        )

        ax.bar(
            np.radians(edges[:-1]), counts, width=np.radians(width), **kwargs
        )
        _as_compass(ax)
        what = "anisotropy-weighted azimuth" if weighted else "azimuth"
        ax.set_title(f"{self._volcano.name}: {what}")
        return ax

    #: Reductions :meth:`rays_rose` accepts for its ``how`` parameter.
    _RAYS_ROSE_HOW = ("median", "mean", "max", "min")

    def rays_rose(self, column="peak", how="median", ax=None, **kwargs):
        """Plot a ray descriptor per direction, over every date, on a compass.

        One bar per direction, as long as ``column`` of
        :meth:`RaysAccessor.descriptors` reduced over every date with
        ``how``: with the defaults, the typical (median) peak of each
        direction, i.e. which side of the vent usually runs hotter. Unlike
        :meth:`azimuth_rose`, which shows where the heat goes when one side
        dominates, this shows the typical state of every direction.

        Parameters
        ----------
        column : str, default "peak"
            Ray descriptor to plot (a column of
            :meth:`RaysAccessor.descriptors`).
        how : str, default "median"
            Reduction over dates: ``"median"``, ``"mean"``, ``"max"`` or
            ``"min"``.
        ax : matplotlib.axes.Axes, optional
            Polar axes to draw on (``projection="polar"``). New ones are
            created if not given.
        **kwargs
            Forwarded to ``ax.bar``, on top of this method's default
            ``width`` (40° per bar).

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        Raises
        ------
        ValueError
            If ``column`` is not a ray descriptor, ``how`` is not a
            supported reduction, or ``ax`` is not a polar axes.

        """
        rays = self._volcano.rays.descriptors()
        if column not in rays.columns:
            raise ValueError(
                f"'column' must be one of {tuple(rays.columns)}, "
                f"found {column!r}"
            )
        if how not in self._RAYS_ROSE_HOW:
            raise ValueError(
                f"'how' must be one of {self._RAYS_ROSE_HOW}, found {how!r}"
            )
        ax = _polar_axes(ax)

        by_direction = rays[column].groupby(level="direction", sort=False)
        values = getattr(by_direction, how)().reindex(DIRECTION_ORDER)

        kwargs.setdefault("width", np.radians(40))
        bearings = np.radians(45.0 * np.arange(len(DIRECTION_ORDER)))
        ax.bar(bearings, values.to_numpy(dtype=float), **kwargs)
        _as_compass(ax)
        ax.set_title(f"{self._volcano.name}: {how} ray {column} by direction")
        return ax

    #: Share of the centers the fill of :meth:`centers_map` covers.
    _CENTERS_MAP_SUPPORT = 0.95

    def centers_map(
        self,
        temperature="peak",
        fill=True,
        centers=True,
        spread=True,
        reference=True,
        cmap="hot",
        vmin=None,
        vmax=None,
        center=None,
        color="k",
        colorbar=True,
        legend=True,
        ax=None,
    ):
        """Map every date's center around their centroid, with its temperature.

        Every date has a center (``center_lat``/``center_long``), the focus
        the measurements are taken around. The map is a compass (north up,
        clockwise) whose origin is the centroid of every center: the angle
        is the bearing from it and the radius the distance, in meters.

        The background is the scenes' ``temperature``
        (:meth:`DescriptorsAccessor.table`) smoothed in space: every
        location gets the mean of the nearby centers, weighted by a Gaussian
        kernel as wide as a KDE of the centers (Scott's rule). It is only
        drawn over the densest region holding 95% of the centers, so it
        doesn't extrapolate where there are none. The fill and the spread
        ellipse are left out when the centers don't span a plane (fewer than
        three dates, or all of them on a line).

        Parameters
        ----------
        temperature : str, default "peak"
            Scene descriptor (a column of
            :meth:`DescriptorsAccessor.table`) that gives every center its
            temperature.
        fill : bool, default True
            Whether to color the background with the smoothed temperature.
        centers : bool, default True
            Whether to draw every center as a dot, colored by its
            temperature on the same scale as the fill: a dot standing out
            from its background is a scene hotter (or colder) than its
            neighbors.
        spread : bool, default True
            Whether to draw the dashed 1-sigma ellipse of the centers.
        reference : bool, default True
            Whether to mark the volcano's GVP location
            (:attr:`Volcano.gvp_lat`, :attr:`Volcano.gvp_lon`) with a star.
        cmap : str or matplotlib.colors.Colormap, default "hot"
            Colormap of the temperature. Values beyond ``vmin``/``vmax`` get
            the end colors.
        vmin, vmax : float, optional
            Temperatures at the ends of the colormap. Default to the 2nd and
            98th percentile of the temperature, so a few extreme scenes don't
            wash out the rest. Fix them (e.g. ``0`` and ``1`` for ``peak``)
            to compare several volcanoes on the same scale.
        center : float, optional
            Temperature at the middle of the colormap, as in
            :func:`seaborn.heatmap`: each half of the colormap stretches to
            cover its side, so the colors are no longer linear across
            ``center``. Must lie strictly between ``vmin`` and ``vmax``.
        color : color, default "k"
            Color of the spread ellipse and the GVP star.
        colorbar : bool, default True
            Whether to add a colorbar of the temperature.
        legend : bool, default True
            Whether to add a legend of the centers, ellipse and star.
        ax : matplotlib.axes.Axes, optional
            Polar axes to draw on (``projection="polar"``). New ones are
            created if not given.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the plot was drawn on.

        Raises
        ------
        ValueError
            If ``ax`` is not a polar axes, or ``center`` is not strictly
            between ``vmin`` and ``vmax``.
        KeyError
            If ``temperature`` is not a scene descriptor.

        """
        ax = _polar_axes(ax)

        # One center per date (every direction of a date shares it), as
        # east/north offsets in meters from their centroid; a degree of
        # longitude shrinks with the cosine of the latitude.
        df = self._volcano.to_dataframe()
        by_date = df.groupby("date")[["center_lat", "center_long"]].first()
        lat0, lon0 = by_date.mean()

        def to_xy(lat, lon):
            north = (lat - lat0) * METERS_PER_DEGREE
            east = (lon - lon0) * METERS_PER_DEGREE * np.cos(np.radians(lat0))
            return np.asarray(east), np.asarray(north)

        def to_polar(east, north):
            # (east, north), not (north, east): bearings start at north
            # and grow towards east.
            return np.arctan2(east, north), np.hypot(east, north)

        points = np.vstack(
            to_xy(by_date["center_lat"], by_date["center_long"])
        )
        temp = (
            self._volcano.descriptors.table()[temperature]
            .reindex(by_date.index)
            .to_numpy(dtype=float)
        )
        low, high = np.nanpercentile(temp, [2, 98])
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
        norm = (
            mcolors.Normalize(vmin, vmax)
            if center is None
            else mcolors.TwoSlopeNorm(center, vmin, vmax)
        )

        # A percentile, not the farthest center, so a few strays don't
        # shrink the rest to the middle.
        rmax = 1.3 * np.percentile(to_polar(*points)[1], 95)

        # The fill and the ellipse need the centers to span both axes;
        # a volcano with one or two dates (or all centers on a line)
        # gets neither.
        two_dimensional = (
            len(temp) > 2 and np.linalg.matrix_rank(np.cov(points)) == 2
        )

        # Whatever is drawn first (fill or dots) feeds the colorbar.
        mappable = None
        if fill and two_dimensional:
            kde = gaussian_kde(points)

            # Polar grid, as east/north locations.
            theta, radius = np.meshgrid(
                np.linspace(0, 2 * np.pi, 181), np.linspace(0, rmax, 120)
            )
            grid = np.vstack(
                [
                    (radius * np.sin(theta)).ravel(),
                    (radius * np.cos(theta)).ravel(),
                ]
            )

            # Kernel-weighted mean of the known temperatures at every grid
            # node (Nadaraya-Watson), with the KDE's kernel.
            known = np.isfinite(temp)
            diff = grid[:, :, None] - points[:, None, known]
            inv = np.linalg.inv(kde.covariance)
            kernel = np.exp(
                -0.5 * np.einsum("imn,ij,jmn->mn", diff, inv, diff)
            )
            field = kernel @ temp[known] / kernel.sum(axis=1)

            # Clipped to the color range (no blank cells beyond it), and
            # blanked outside the densest region holding the support share
            # of the centers: its edge is the density that share of the
            # centers reaches.
            edge = np.quantile(kde(points), 1 - self._CENTERS_MAP_SUPPORT)
            field = np.where(
                kde(grid) >= edge, field.clip(norm.vmin, norm.vmax), np.nan
            )
            mappable = ax.contourf(
                theta,
                radius,
                field.reshape(theta.shape),
                levels=MaxNLocator(10).tick_values(norm.vmin, norm.vmax),
                cmap=cmap,
                norm=norm,
            )

        if centers:
            dots = ax.scatter(
                *to_polar(*points),
                c=temp,
                cmap=cmap,
                norm=norm,
                s=18,
                edgecolors="k",  # visible over a fill of its own color
                linewidths=0.5,
                label=f"Centers (n={len(temp)})",
            )
            mappable = dots if mappable is None else mappable

        if spread and two_dimensional:
            # A unit circle stretched along the principal axes of the
            # centers by the standard deviation along each.
            values, vectors = np.linalg.eigh(np.cov(points))
            t = np.linspace(0, 2 * np.pi, 200)
            circle = np.vstack([np.cos(t), np.sin(t)])
            ellipse = vectors @ (np.sqrt(values)[:, None] * circle)
            ax.plot(
                *to_polar(*ellipse),
                color=color,
                linestyle="--",
                label=r"1$\sigma$ spread",
            )

        if reference:
            gvp = to_xy(self._volcano.gvp_lat, self._volcano.gvp_lon)
            ax.scatter(
                *to_polar(*gvp),
                marker="*",
                s=260,
                c=color,
                edgecolors="white",  # the star often lands on dark fill
                zorder=3,
                label="GVP location",
            )

        if colorbar and mappable is not None:
            # Not the ground's temperature: a dot is its own scene's, the
            # fill the mean of the scenes centered nearby.
            ax.figure.colorbar(
                mappable,
                ax=ax,
                pad=0.1,
                shrink=0.8,
                label=f"{temperature} of the scenes centered here",
            )

        _as_compass(ax)
        if rmax > 0:  # a single center leaves nothing to scale to
            ax.set_ylim(0, rmax)
        ax.set_rlabel_position(22.5)  # between N and NE, off the compass
        ax.set_title(
            f"{self._volcano.name}: centers around their centroid (m)"
        )
        if legend:
            ax.legend(
                loc="lower left", bbox_to_anchor=(-0.25, -0.2), frameon=False
            )
        return ax
