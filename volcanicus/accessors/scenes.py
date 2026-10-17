#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Scene view of :class:`~volcanicus.core.Volcano` instances \
(``volcano.scenes``)."""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd

from .base import AccessorABC
from ..constants import (
    DIRECTION_ORDER,
    NON_DISTANCE_COLUMNS as _NON_DISTANCE_COLUMNS,
    REDUCE_DIRECTIONS_WHITELIST,
)

# =============================================================================
# CLASSES
# =============================================================================


class ScenesAccessor(AccessorABC):
    """Measurements as scenes: one row per date (used as ``volcano.scenes``).

    A scene is everything observed on a single date: a grid of every
    direction x every distance bin. Calling the accessor
    (``volcano.scenes()``) is the same as :meth:`grid`.

    Parameters
    ----------
    volcano : Volcano
        The :class:`~volcanicus.core.Volcano` instance this accessor belongs
        to.

    """

    _default_kind = "grid"

    def __init__(self, volcano):
        self._volcano = volcano

    def grid(self, as_array=False):
        """Reshape the measurements into one scene (date) per row.

        The source DataFrame is "long" on direction (one row per
        date/direction). Each scene becomes a single row, which is the
        natural sample unit for a model.

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
        df = self._volcano.to_dataframe()

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
            c for c in df.columns if c not in _NON_DISTANCE_COLUMNS
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

    def reduce(self, how="median"):
        """Collapse the directions of every scene into a single profile.

        Each scene (see :meth:`grid`) holds 8 directions x every distance
        bin; this reduces the 8 directions at each distance with ``how``,
        leaving one radial profile per date.

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
        median : Shorthand for ``reduce("median")``.
        max : Shorthand for ``reduce("max")``.

        """
        if how not in REDUCE_DIRECTIONS_WHITELIST:
            raise ValueError(
                f"'how' must be one of {REDUCE_DIRECTIONS_WHITELIST}, "
                f"found {how!r}"
            )
        # Transpose so every (direction, distance) pair is a row, then group
        # those rows by distance: each group holds the 8 directions to reduce.
        by_distance = self.grid().T.groupby(level="distance", sort=False)
        reduced = getattr(by_distance, how)().T
        reduced.columns.name = "distance"
        return reduced

    def median(self):
        """Typical radial profile of every scene: median across directions.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``date``, with one column per distance bin. Using the
            median (not the mean) keeps a single anomalous direction from
            dragging the profile.

        See Also
        --------
        grid : The full, per-direction scenes this is reduced from.
        max : Hottest direction of every scene.

        """
        return self.reduce("median")

    def max(self):  # noqa: A003
        """Hottest direction of every scene: maximum across directions.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``date``, with one column per distance bin.

        See Also
        --------
        grid : The full, per-direction scenes this is reduced from.
        median : Typical (median) profile of every scene.

        """
        return self.reduce("max")
