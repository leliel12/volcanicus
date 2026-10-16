#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Volcano class: wraps a DataFrame of volcano deformation/residual \
measurements together with the name of the volcano they belong to.

This is a plain class (no dataclass, no frozen, no caching tricks): internal
attributes are private by convention (single leading underscore) and are only
exposed to the outside through read-only properties.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import methodtools

import numpy as np

import pandas as pd

from .accessors import PlotAccessor, StatsAccessor
from .constants import (
    DIRECTION_ORDER,
    GROUPBY_WHITELIST,
    NON_DISTANCE_COLUMNS,
    REDUCE_DIRECTIONS_WHITELIST,
    SATURATION_LEVEL,
)
from .utils import Bunch

# =============================================================================
# FUNCTIONS
# =============================================================================


def normalize_dataframe(df):
    """Enforce the correct dtypes for a volcano measurement DataFrame.

    The date column is parsed as ``datetime``, the latitude/longitude and all
    distance-bin columns are parsed as numeric, and ``direction`` is parsed as
    a category.

    This is a free function (not a method of :class:`Volcano`) so it can be
    reused on its own, without needing an instance of the class.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to normalize. Modified in place.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame instance that was passed in, modified in place and
        returned so calls can be chained if desired.

    """
    # Parse the date column, if present; anything unparseable becomes NaT
    # instead of raising, since the source CSVs are hand-curated.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Coordinates come in as text in some CSVs; coerce them to numeric.
    for col in ("center_lat", "center_long"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # direction has a small, fixed set of values, so category is cheaper
    # to store and lets pandas/seaborn treat it as such (e.g. for hue).
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype("category")

    # Every remaining column is a distance bin holding a numeric residual.
    distance_columns = [c for c in df.columns if c not in NON_DISTANCE_COLUMNS]
    for col in distance_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rows may have come from a concatenation of multiple CSVs upstream;
    # renumber them so the index is a clean 0..n-1 range.
    df.reset_index(drop=True, inplace=True)
    return df


def _distance_radii(distance_columns):
    """Numeric radius (m) of every distance bin, from names like ``"15m"``."""
    return np.array([float(c.removesuffix("m")) for c in distance_columns])


def _sector_area(radii):
    """Area (m²) of every bin's annular sector: 1/8 of ``2 * pi * r * dr``."""
    return 2 * np.pi * radii * np.gradient(radii) / len(DIRECTION_ORDER)


def _fill_rays(rays, radii):
    """Fill the gaps of every ray (last axis) by linear interpolation.

    Returns two filled copies: ``filled`` also holds the nearest value past
    the first/last known bin (for integrals, where leaving the edges out
    would undercount), while ``interior`` leaves those edges as ``NaN`` (for
    the profile's shape, which shouldn't be extrapolated). Rays with no data
    at all stay ``NaN`` in both.
    """
    filled = rays.copy()
    interior = rays.copy()
    for ray, inner in zip(
        filled.reshape(-1, len(radii)), interior.reshape(-1, len(radii))
    ):
        known = ~np.isnan(ray)
        if known.any() and not known.all():
            ray[:] = np.interp(radii, radii[known], ray[known])
            first, last = np.flatnonzero(known)[[0, -1]]
            span = slice(first, last + 1)
            inner[span] = ray[span]
    return filled, interior


def _peak_and_efold(profiles, radii):
    """Peak and e-folding length of every profile (row) in a 2-D array.

    The e-folding length is the distance from the peak to where the profile
    first drops to ``peak / e``, linearly interpolated between bins; ``NaN``
    when the peak isn't positive or the profile never drops that far.
    """
    peak = np.full(len(profiles), np.nan)
    efold_length = np.full(len(profiles), np.nan)
    for i, profile in enumerate(profiles):
        if np.isnan(profile).all():
            continue
        top = np.nanargmax(profile)
        peak[i] = profile[top]
        if peak[i] <= 0:
            continue
        threshold = peak[i] / np.e
        below = np.flatnonzero(profile[top:] <= threshold)
        if below.size:
            j = top + below[0]
            # j > top always (the peak itself is above threshold), so
            # interpolate the crossing between bins j - 1 and j.
            fraction = (profile[j - 1] - threshold) / (
                profile[j - 1] - profile[j]
            )
            efold_length[i] = (
                radii[j - 1]
                + fraction * (radii[j] - radii[j - 1])
                - radii[top]
            )
    return peak, efold_length


# =============================================================================
# CLASSES
# =============================================================================


class Volcano:
    """Residual measurements for a single volcano.

    Parameters
    ----------
    name : str
        Name of the volcano.
    df : pandas.DataFrame
        Raw measurements, with one row per date/direction and one column per
        distance bin. Normalized in place (see :func:`normalize_dataframe`)
        and stored internally.
    metadata : dict, optional
        Free-form annotations about this instance: the volcano's descriptive
        metadata for the bundled datasets (country, coordinates, GVP
        classification, morphology, ...), plus processing annotations such
        as whether it was imputed. Stored as a
        read-only :class:`~volcanicus.utils.Bunch`; defaults to an empty one.

    Attributes
    ----------
    name : str
        Name of the volcano.
    plot : PlotAccessor
        Plotting accessor, created lazily on first access.
    stats : StatsAccessor
        Statistics accessor, created lazily on first access.
    has_missing : bool
        Whether any distance-bin measurement is missing.
    metadata : Bunch
        Descriptive metadata and processing annotations of this instance.
    m : Bunch
        Shorthand alias of ``metadata``.

    """

    def __init__(self, name, df, metadata=None):
        self._dataframe = normalize_dataframe(df)
        self._name = name
        self._metadata = Bunch("metadata", metadata or {})

    # ACCESSORS (YES, WE USE CACHED PROPERTIES IS THE EASIEST WAY) ============

    @methodtools.lru_cache(maxsize=None)
    @property
    def plot(self):
        """Plotting accessor for this instance.

        Returns
        -------
        PlotAccessor
            Created on first access and reused on subsequent accesses.

        """
        return PlotAccessor(self)

    @methodtools.lru_cache(maxsize=None)
    @property
    def stats(self):
        """Statistics accessor for this instance.

        Returns
        -------
        StatsAccessor
            Created on first access and reused on subsequent accesses.

        """
        return StatsAccessor(self)

    # PROPERTIES ==============================================================

    @property
    def name(self):
        """Name of the volcano.

        Returns
        -------
        str

        """
        return self._name

    @property
    def has_missing(self):
        """Whether any distance-bin measurement is missing (``NaN``).

        Returns
        -------
        bool

        """
        distance_columns = [
            c for c in self._dataframe.columns if c not in NON_DISTANCE_COLUMNS
        ]
        return bool(self._dataframe[distance_columns].isna().any().any())

    @property
    def metadata(self):
        """Free-form annotations about this instance.

        Returns
        -------
        Bunch
            Read-only mapping; values are readable by key or attribute
            (``volcano.metadata["imputed"]`` or ``volcano.metadata.imputed``).

        See Also
        --------
        m : Shorthand alias of this property.

        """
        return self._metadata

    #: Shorthand alias of :attr:`metadata`, for interactive use
    #: (``volcano.m.imputed``).
    m = metadata

    # METHODS =================================================================

    def to_dataframe(self):
        """Copy of the internal, normalized DataFrame.

        Returns
        -------
        pandas.DataFrame
            A copy, so the internal state cannot be mutated from the outside.

        """
        return self._dataframe.copy()

    def to_scenes(self, as_array=False):
        """Reshape the measurements into one scene (date) per row.

        The source DataFrame is "long" on direction (one row per
        date/direction). A scene is everything observed on a single date:
        a grid of every direction x every distance bin. Each scene becomes a
        single row, which is the natural sample unit for a model.

        The grid is always complete: directions follow
        :data:`~volcanicus.constants.DIRECTION_ORDER` and any date/direction
        pair absent from the source data is filled with ``NaN``, so every
        scene has exactly the same shape (and the same columns) no matter
        the volcano.

        Parameters
        ----------
        as_array : bool, default False
            If False, return a DataFrame. If True, return the same values as
            a 3-D ``numpy.ndarray`` of shape
            ``(n_dates, n_directions, n_distances)``.

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            DataFrame indexed by ``date``, with two-level columns
            ``(direction, distance)``; or its values as a cube (see
            ``as_array``). In the cube, axis 0 follows the sorted dates,
            axis 1 follows ``DIRECTION_ORDER`` and axis 2 the distance bins
            in their original (increasing) order.

        Raises
        ------
        ValueError
            If the data can't be laid out as a grid of scenes: a direction
            not in ``DIRECTION_ORDER`` (it would be silently dropped), a
            date that failed to parse (``NaT``) or a repeated
            date/direction pair.

        """
        df = self._dataframe

        # The grid below is reindexed to DIRECTION_ORDER, which would
        # silently turn a misspelled direction into NaN; fail loudly instead.
        unknown = set(df["direction"].astype(str)) - set(DIRECTION_ORDER)
        if unknown:
            raise ValueError(
                f"Unknown direction(s) {sorted(unknown)}; "
                f"expected values in {DIRECTION_ORDER}"
            )
        # normalize_dataframe coerces unparseable dates to NaT, which would
        # otherwise end up as a scene with no date.
        if df["date"].isna().any():
            raise ValueError(
                f"{df['date'].isna().sum()} row(s) have an invalid date (NaT)"
            )
        duplicated = df.duplicated(["date", "direction"])
        if duplicated.any():
            pairs = df.loc[duplicated, ["date", "direction"]]
            raise ValueError(
                "Repeated date/direction pair(s): "
                f"{list(pairs.itertuples(index=False, name=None))}"
            )

        distance_columns = [
            c for c in df.columns if c not in NON_DISTANCE_COLUMNS
        ]
        # (date, direction) -> one row; then direction moves from the rows
        # into the columns, next to the distance bins.
        wide = df.set_index(["date", "direction"])[distance_columns].unstack(
            "direction"
        )

        # unstack leaves (distance, direction) columns in arbitrary order and
        # only for the directions present; force the full, canonical grid so
        # every volcano yields identically shaped scenes.
        full_columns = pd.MultiIndex.from_product(
            [DIRECTION_ORDER, distance_columns],
            names=["direction", "distance"],
        )
        scenes = (
            wide.swaplevel(axis=1).reindex(columns=full_columns).sort_index()
        )

        if as_array:
            return scenes.to_numpy(dtype=float).reshape(
                len(scenes), len(DIRECTION_ORDER), len(distance_columns)
            )
        return scenes

    def reduce_directions(self, how="median"):
        """Collapse the directions of every scene into a single profile.

        Each scene (see :meth:`to_scenes`) holds 8 directions x every
        distance bin; this reduces the 8 directions at each distance with
        ``how``, leaving one radial profile per date.

        ``NaN`` values are skipped while reducing, so a missing direction
        doesn't spoil the others; a distance bin where every direction is
        missing stays ``NaN``.

        Parameters
        ----------
        how : str, default "median"
            Reduction to apply across directions: one of ``"count"``,
            ``"max"``, ``"mean"``, ``"median"``, ``"min"``, ``"std"``,
            ``"sum"`` or ``"var"``.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``date``, with one column per distance bin.

        Raises
        ------
        ValueError
            If ``how`` is not a supported reduction.

        See Also
        --------
        scenes_median : Shorthand for ``reduce_directions("median")``.
        scenes_max : Shorthand for ``reduce_directions("max")``.

        """
        if how not in REDUCE_DIRECTIONS_WHITELIST:
            raise ValueError(
                f"'how' must be one of {REDUCE_DIRECTIONS_WHITELIST}, "
                f"found {how!r}"
            )
        # Transpose so every (direction, distance) pair is a row, then group
        # those rows by distance: each group holds the 8 directions to reduce.
        by_distance = self.to_scenes().T.groupby(level="distance", sort=False)
        reduced = getattr(by_distance, how)().T
        reduced.columns.name = "distance"
        return reduced

    @property
    def scenes_median(self):
        """Typical radial profile of every scene: median across directions.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``date``, with one column per distance bin. Using the
            median (not the mean) keeps a single anomalous direction from
            dragging the profile.

        See Also
        --------
        to_scenes : The full, per-direction scenes this is reduced from.
        scenes_max : Hottest direction of every scene.

        """
        return self.reduce_directions("median")

    @property
    def scenes_max(self):
        """Hottest direction of every scene: maximum across directions.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``date``, with one column per distance bin.

        See Also
        --------
        to_scenes : The full, per-direction scenes this is reduced from.
        scenes_median : Typical (median) profile of every scene.

        """
        return self.reduce_directions("max")

    def scenes_descriptors(self):
        """Summarize every scene with a few interpretable descriptors.

        Each scene (see :meth:`to_scenes`) holds 8 directions x every
        distance bin; this condenses it into five numbers that are
        comparable across dates and volcanoes:

        - ``peak``: maximum of the median (across directions) radial
          profile.
        - ``efold_length``: distance (m) from the peak over which that
          median profile decays to ``peak / e``, the crossing linearly
          interpolated between bins. Measured from the peak (not from the
          vent) so it doesn't depend on which inner bins a scene happens to
          have: for an exponential decay it's the decay length whatever
          the first bin is. ``NaN`` if the peak isn't positive or the
          profile never drops that far within the measured distances.
        - ``integrated``: the residual integrated over the whole disk, each
          bin weighted by the area of its annular sector
          (``2 * pi * r * dr / 8``), in residual x m². A missing direction
          is skipped and the rest are scaled up to the full disk.
        - ``anisotropy``: how concentrated the heat is in one direction, in
          ``[0, 1]``: ``0`` spreads evenly around the vent, ``1`` sits
          entirely in a single direction. It's the length of the vector sum
          of the 8 directions, each as long as its ray's positive ``peak``
          (see :meth:`rays_descriptors`), over the sum of their lengths.
          Using the ray peaks keeps it about the near field: weighting by
          area instead would let the far field, which covers most of the
          disk, decide the direction.
        - ``azimuth``: direction of that vector sum, in degrees clockwise
          from north (``N`` is 0, ``E`` is 90, ``S`` is 180, ``O`` is 270).
          ``NaN`` when no direction dominates (``anisotropy`` is 0).

        ``anisotropy`` and ``azimuth`` need all 8 directions, so they're
        ``NaN`` for a scene with a direction missing entirely (see
        :meth:`impute` to fill those first).

        Gaps along a direction are filled by linear interpolation over
        distance before computing anything; otherwise a bin missing in some
        scenes and not in others would make descriptors incomparable
        between scenes. For the integrals, the edges also take the nearest
        available bin (the innermost one is often missing, and leaving it
        out would undercount). ``peak`` and ``efold_length`` don't
        extrapolate, so a scene missing its innermost bin peaks at the
        first bin it has.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``date``, with the columns ``peak``,
            ``efold_length``, ``integrated``, ``anisotropy`` and
            ``azimuth``.

        See Also
        --------
        rays_descriptors : The same kind of descriptors, per direction.

        """
        scenes = self.to_scenes()
        cube = self.to_scenes(as_array=True)
        n_dates, n_directions, n_distances = cube.shape
        radii = _distance_radii(scenes.columns.unique("distance"))
        filled, interior = _fill_rays(cube, radii)
        missing_direction = np.isnan(cube).all(axis=2)

        # peak & efold_length: shape of the typical (median) profile.
        # A bin with no data in any direction is an expected NaN, not
        # something to warn about.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_profile = np.nanmedian(interior, axis=1)
        peak, efold_length = _peak_and_efold(median_profile, radii)

        # integrated: per-direction integrals; a missing direction is
        # skipped and the mean of the others stands in for it.
        direction_integral = (filled * _sector_area(radii)).sum(axis=2)
        direction_integral[missing_direction] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            integrated = np.nanmean(direction_integral, axis=1) * n_directions

        # anisotropy & azimuth: vector sum of every ray's positive peak,
        # each pointing along its compass bearing.
        ray_peak, _ = _peak_and_efold(interior.reshape(-1, n_distances), radii)
        positive = np.clip(ray_peak.reshape(n_dates, n_directions), 0, None)
        bearings = np.radians(45.0 * np.arange(n_directions))
        east = (positive * np.sin(bearings)).sum(axis=1)
        north = (positive * np.cos(bearings)).sum(axis=1)
        total = positive.sum(axis=1)
        with np.errstate(all="ignore"):
            anisotropy = np.hypot(east, north) / total
        # A perfectly even scene leaves floating-point crumbs in the vector
        # sum; snap those to 0 so they don't fake a bearing.
        anisotropy[np.isclose(anisotropy, 0, atol=1e-9)] = 0
        azimuth = np.degrees(np.arctan2(east, north)) % 360
        undefined = missing_direction.any(axis=1) | ~(total > 0)
        anisotropy[undefined] = np.nan
        azimuth[undefined | (anisotropy == 0)] = np.nan

        return pd.DataFrame(
            {
                "peak": peak,
                "efold_length": efold_length,
                "integrated": integrated,
                "anisotropy": anisotropy,
                "azimuth": azimuth,
            },
            index=scenes.index,
        )

    def rays_descriptors(self):
        """Summarize every ray (one direction of one scene).

        A ray is the radial profile of a single date/direction: one row of
        the source data. Describing rays one by one, instead of whole
        scenes, tells where around the vent things happen, and isolates
        problems that only affect one direction.

        - ``peak``: maximum residual along the ray.
        - ``efold_length``: distance (m) from the peak over which the ray
          decays to ``peak / e``; defined exactly as in
          :meth:`scenes_descriptors`.
        - ``integrated``: residual integrated over the ray's annular
          sector (``2 * pi * r * dr / 8`` per bin), in residual x m². The
          8 rays of a scene add up to its ``integrated``.
        - ``n_saturated``: number of bins at the saturation level
          (:data:`~volcanicus.constants.SATURATION_LEVEL`). Residuals are
          capped there, so a saturated ray's ``peak`` underestimates the
          real anomaly.
        - ``max_abs``: largest absolute residual along the ray, on the raw
          values (see :meth:`flag_rays`).

        Gaps are handled as in :meth:`scenes_descriptors`. Rays missing from
        the data appear with every descriptor ``NaN`` (``n_saturated`` 0),
        so the result always holds 8 rays per scene.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(date, direction)``, directions in
            :data:`~volcanicus.constants.DIRECTION_ORDER`, with the columns
            ``peak``, ``efold_length``, ``integrated``, ``n_saturated`` and
            ``max_abs``.

        """
        scenes = self.to_scenes()
        cube = self.to_scenes(as_array=True)
        n_distances = cube.shape[2]
        radii = _distance_radii(scenes.columns.unique("distance"))
        filled, interior = _fill_rays(cube, radii)

        rays = cube.reshape(-1, n_distances)
        peak, efold_length = _peak_and_efold(
            interior.reshape(-1, n_distances), radii
        )
        integrated = (
            filled.reshape(-1, n_distances) * _sector_area(radii)
        ).sum(axis=1)
        missing = np.isnan(rays).all(axis=1)
        integrated[missing] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_abs = np.nanmax(np.abs(rays), axis=1)

        index = pd.MultiIndex.from_product(
            [scenes.index, DIRECTION_ORDER], names=["date", "direction"]
        )
        return pd.DataFrame(
            {
                "peak": peak,
                "efold_length": efold_length,
                "integrated": integrated,
                "n_saturated": (rays == SATURATION_LEVEL).sum(axis=1),
                "max_abs": max_abs,
            },
            index=index,
        )

    def flag_rays(self, limit=2.0):
        """Flag rays holding a residual too large to be a real measurement.

        Residuals saturate at
        :data:`~volcanicus.constants.SATURATION_LEVEL` (``1.0``), yet a
        handful of rays in the bundled data reach tens or thousands. These
        come in runs along a single direction while the rest of the scene
        looks ordinary, which points to a processing artifact rather than
        to the volcano. The default ``limit`` of twice the saturation level
        keeps the values that only slightly overshoot it.

        Parameters
        ----------
        limit : float, default 2.0
            A ray is flagged when any of its residuals is larger than this
            in absolute value.

        Returns
        -------
        pandas.Series
            Boolean, indexed by ``(date, direction)`` like
            :meth:`rays_descriptors`; ``True`` for a flagged ray.

        See Also
        --------
        mask_rays : Drop the flagged rays' values.

        """
        flags = self.rays_descriptors()["max_abs"] > limit
        flags.name = "flagged"
        return flags

    def mask_rays(self, limit=2.0):
        """Blank out every ray :meth:`flag_rays` flags.

        Every distance bin of a flagged ray becomes ``NaN``, so it's
        treated like any other missing ray afterwards (skipped by the
        reductions, fillable by :meth:`impute`). The whole ray goes, not
        only the offending values: the bins next to an absurd value are
        rarely trustworthy (in Kadovar a single artifact spans 4 bins).

        Parameters
        ----------
        limit : float, default 2.0
            Passed on to :meth:`flag_rays`.

        Returns
        -------
        Volcano
            A new instance with the flagged rays masked. Its metadata
            records the ``mask_limit`` used and how many rays it masked
            (``masked_rays``).

        """
        flags = self.flag_rays(limit)
        flagged = flags[flags].index

        df = self._dataframe.copy()
        distance_columns = [
            c for c in df.columns if c not in NON_DISTANCE_COLUMNS
        ]
        rows = pd.MultiIndex.from_frame(
            df[["date", "direction"]].astype({"direction": str})
        ).isin(flagged)
        df.loc[rows, distance_columns] = np.nan

        metadata = {
            **self._metadata,
            "masked_rays": int(rows.sum()),
            "mask_limit": limit,
        }
        return Volcano(self._name, df, metadata=metadata)

    def scenes_baseline(self, min_scenes=10):
        """Typical value and spread of every cell of the scenes.

        For every ``(direction, distance)`` cell, over every scene of this
        volcano: the median (``center``) and a robust spread (``scale``,
        ``1.4826 * MAD``, which matches the standard deviation for normally
        distributed data). Each cell gets its own baseline because each
        direction can sit on a different background (e.g. a slope facing
        the sun, or the sea).

        This is the volcano's *usual* state, not necessarily a quiet one:
        for a volcano active in most of its scenes, "usual" means active,
        and the anomalies (see :meth:`scenes_anomaly`) measure departures
        from that. Run :meth:`mask_rays` first so corrupted rays don't
        leak into it.

        Parameters
        ----------
        min_scenes : int, default 10
            Cells with fewer non-missing scenes than this get a ``NaN``
            baseline: a median and MAD over a handful of values aren't
            meaningful.

        Returns
        -------
        pandas.DataFrame
            Two rows, ``center`` and ``scale``, with the same two-level
            ``(direction, distance)`` columns as :meth:`to_scenes`. A cell
            whose values never vary (MAD of 0, e.g. always saturated) has a
            ``NaN`` scale.

        """
        scenes = self.to_scenes()
        center = scenes.median()
        scale = 1.4826 * (scenes - center).abs().median()

        too_few = scenes.count() < min_scenes
        center[too_few] = np.nan
        scale[too_few | (scale == 0)] = np.nan
        return pd.DataFrame({"center": center, "scale": scale}).T

    def scenes_anomaly(self, min_scenes=10):
        """Every scene as robust z-scores against the volcano's baseline.

        Each cell becomes ``(value - center) / scale``, with ``center`` and
        ``scale`` from :meth:`scenes_baseline`: how many "typical
        deviations" it sits above (positive) or below (negative) the
        volcano's usual value for that direction and distance. Unlike raw
        residuals, these are comparable across volcanoes with very
        different backgrounds.

        Parameters
        ----------
        min_scenes : int, default 10
            Passed on to :meth:`scenes_baseline`.

        Returns
        -------
        pandas.DataFrame
            Same shape, index and columns as :meth:`to_scenes`. ``NaN``
            where the value is missing or the cell has no baseline.

        """
        baseline = self.scenes_baseline(min_scenes)
        return (self.to_scenes() - baseline.loc["center"]) / baseline.loc[
            "scale"
        ]

    def radial_profile(self):
        """Average the measurements over dates, per direction.

        Every date is averaged out for each direction, producing one row per
        direction with the mean of every distance bin, indexed by
        ``direction``. This is purely a data reduction: things like an
        overall mean across directions or its dispersion are trivial to
        derive from this result, so they're left to the plotting layer
        (:meth:`PlotAccessor.radial_profile`) instead of being
        parameters here.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``direction``, with one column per distance bin.

        """
        distance_columns = [
            c for c in self._dataframe.columns if c not in NON_DISTANCE_COLUMNS
        ]
        grouped = self._dataframe.groupby("direction", observed=True)
        return grouped[distance_columns].mean()

    def temporal_profile(self):
        """Average the measurements over distance bins, per date.

        Temporal mirror of :meth:`radial_profile`: that one averages over
        dates and keeps distance bins as columns, indexed by direction; this
        one averages over distance bins and keeps direction as columns,
        indexed by ``date``.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``date``, with one column per direction.

        """
        distance_columns = [
            c for c in self._dataframe.columns if c not in NON_DISTANCE_COLUMNS
        ]
        # Collapse the distance axis first: one residual value per
        # date/direction row, averaged across every distance bin.
        row_mean = self._dataframe[distance_columns].mean(axis=1)

        # date/direction are already one-per-row, so pair each row's mean
        # with its date and direction...
        long = self._dataframe[["date", "direction"]].assign(residual=row_mean)
        # ...then pivot direction out of the rows and into columns, since
        # (unlike distance bins) it isn't already columnar in the source
        # data.
        return long.pivot_table(
            index="date",
            columns="direction",
            values="residual",
            observed=True,
        )

    #: Columns ``describe`` is allowed to group the measurements by.
    _DESCRIBE_BY_WHITELIST = GROUPBY_WHITELIST

    def describe(self, by="direction"):
        """Summarize missing (``NaN``) distance-bin measurements.

        Parameters
        ----------
        by : str, default "direction"
            Column to group the measurements by before counting: either
            ``"date"`` or ``"direction"``.

        Returns
        -------
        pandas.DataFrame
            One row per distinct value of `by`, plus a trailing
            ``"TOTAL"`` row summarizing everything, with three columns:

            - ``total``: number of measurements possible for that
              row, across every distance bin (``int``).
            - ``n_missing``: number of those that are missing (``int``).
            - ``proportion``: ``n_missing / total``, i.e. the fraction
              missing (``float`` in ``[0, 1]``).

        """
        if by not in self._DESCRIBE_BY_WHITELIST:
            raise ValueError(
                f"'by' must be one of {self._DESCRIBE_BY_WHITELIST}, "
                f"found {by!r}"
            )

        distance_columns = [
            c for c in self._dataframe.columns if c not in NON_DISTANCE_COLUMNS
        ]
        # Count NaNs per row first (one row = one date/direction), then
        # aggregate those per-row counts by the requested grouping key.
        missing_per_row = self._dataframe[distance_columns].isna().sum(axis=1)
        grouped = missing_per_row.groupby(self._dataframe[by], observed=True)
        n_missing = grouped.sum().astype(int)
        total = (grouped.count() * len(distance_columns)).astype(int)

        table = pd.DataFrame(
            {
                "total": total,
                "n_missing": n_missing,
                "proportion": n_missing / total,
            }
        )
        # Appended as its own row (rather than via `.loc["TOTAL"] = [...]`)
        # so the int columns don't get upcast to float to accommodate the
        # mixed-dtype assignment.
        overall = pd.DataFrame(
            {
                "total": [table["total"].sum()],
                "n_missing": [table["n_missing"].sum()],
                "proportion": [
                    table["n_missing"].sum() / table["total"].sum()
                ],
            },
            index=["TOTAL"],
        )
        return pd.concat([table, overall])

    def impute(self, fallback_to_mean=True):
        """Fill missing distance-bin measurements.

        Three-step strategy:

        1. Within each date/direction row, linearly interpolate missing
           distance bins from their neighboring distance bins (interior
           gaps only). The residual decays continuously with distance at a
           fixed point in time, so this is the most defensible fill.
        2. If a row is still entirely missing after step 1 (no distance bin
           left to interpolate from), fill it by averaging the same date's
           two compass-adjacent directions — e.g. a fully-missing ``"N"``
           row is filled from ``"NO"`` and ``"NE"`` on that date.
        3. If ``fallback_to_mean`` is true, anything still missing after
           steps 1-2 (e.g. both compass neighbors were also missing) is
           filled with the global mean of the *original*, unimputed
           measurements. That mean is computed once, up front, from the raw
           data only, so it isn't biased by values steps 1-2 already
           filled in.

        Parameters
        ----------
        fallback_to_mean : bool, default True
            Whether to apply step 3. If False, values steps 1-2 couldn't
            fill are left as ``NaN``.

        Returns
        -------
        Volcano
            A new instance with missing values imputed where possible.

        """
        # .copy() matters here: without it we'd be writing through to
        # self._dataframe and silently mutating this Volcano too.
        df = self._dataframe.copy()
        distance_columns = [
            c for c in df.columns if c not in NON_DISTANCE_COLUMNS
        ]

        # Snapshot the grand mean of the *raw* values before step 1 touches
        # anything, so step 3's fallback isn't computed from data that
        # steps 1-2 already filled in.
        global_mean = df[distance_columns].stack().mean()

        # Step 1: interior-gap interpolation along each row's distance
        # bins. limit_area="inside" is what keeps this from extrapolating
        # past the first/last known value in a row.
        df[distance_columns] = df[distance_columns].interpolate(
            axis=1, limit_area="inside"
        )

        # Step 2: rows step 1 couldn't touch at all (every distance bin
        # NaN) get filled from their compass neighbors on the same date.
        still_missing = df[distance_columns].isna().all(axis=1)
        for idx in df.index[still_missing]:
            date = df.loc[idx, "date"]
            direction = df.loc[idx, "direction"]
            # Look up direction's position in the compass ring so we can
            # grab the direction immediately before and after it (with
            # wraparound, e.g. "N"'s predecessor is "NO").
            pos = DIRECTION_ORDER.index(direction)
            neighbors = (
                DIRECTION_ORDER[pos - 1],
                DIRECTION_ORDER[(pos + 1) % len(DIRECTION_ORDER)],
            )
            same_date = df[df["date"] == date]
            neighbor_rows = same_date[same_date["direction"].isin(neighbors)]
            # neighbor_rows has at most 2 rows (one per neighbor, fewer if
            # one is missing from the data entirely); average whatever is
            # there. If both neighbors are also NaN this is a no-op and
            # the row stays missing, to be picked up by step 3 below.
            df.loc[idx, distance_columns] = neighbor_rows[
                distance_columns
            ].mean()

        # Step 3: anything steps 1-2 still couldn't reach falls back to
        # the pre-computed grand mean, if the caller opted into it.
        if fallback_to_mean:
            df[distance_columns] = df[distance_columns].fillna(global_mean)

        metadata = {
            **self._metadata,
            "imputed": True,
            "impute_fallback_to_mean": fallback_to_mean,
        }
        return Volcano(self._name, df, metadata=metadata)

    # MAGIC ===================================================================

    def _summary_fields(self):
        """Fields shared by :meth:`__repr__` and :meth:`_repr_html_`."""
        distance_columns = [
            c for c in self._dataframe.columns if c not in NON_DISTANCE_COLUMNS
        ]
        return {
            "name": self._name.title(),
            "dates": self._dataframe["date"].nunique(),
            "missing": int(
                self._dataframe[distance_columns].isna().sum().sum()
            ),
            "registers": len(self._dataframe),
        }

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        f = self._summary_fields()
        return (
            f"Volcano(name={f['name']!r}, dates={f['dates']}, "
            f"missing={f['missing']}, registers={f['registers']})"
        )

    def _repr_html_(self):
        """Rich HTML repr, used by Jupyter/IPython instead of __repr__."""
        f = self._summary_fields()
        rows = "".join(
            f"<tr><td>{label}</td><td>{f[key]}</td></tr>"
            for label, key in [
                ("dates", "dates"),
                ("missing", "missing"),
                ("registers", "registers"),
            ]
        )
        return (
            "<table>"
            f"<caption>🌋 <b>{f['name']}</b></caption>"
            f"{rows}"
            "</table>"
        )
