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

from pathlib import Path

import methodtools

import pandas as pd

from .accessors import PlotAccessor, StatsAccessor

# =============================================================================
# CONSTANTS
# =============================================================================

#: Columns in the source CSV that are not distance bins.
NON_DISTANCE_COLUMNS = ("date", "center_lat", "center_long", "direction")

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
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ("center_lat", "center_long"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "direction" in df.columns:
        df["direction"] = df["direction"].astype("category")

    distance_columns = [c for c in df.columns if c not in NON_DISTANCE_COLUMNS]
    for col in distance_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.reset_index(drop=True, inplace=True)
    return df


# =============================================================================
# CLASSES
# =============================================================================


class Volcano:
    """Residual measurements for a single volcano.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Raw measurements, with one row per date/direction and one column per
        distance bin. A normalized copy is stored internally; the original
        DataFrame passed in is not modified.
    name : str
        Name of the volcano.

    Attributes
    ----------
    df : pandas.DataFrame
        Read-only copy of the normalized measurements.
    name : str
        Name of the volcano.
    plots : PlotAccessor
        Plotting accessor, created lazily on first access.
    stats : StatsAccessor
        Statistics accessor, created lazily on first access.
    has_missing : bool
        Whether any distance-bin measurement is missing.

    """

    def __init__(self, df, name):
        self._dataframe = normalize_dataframe(df)
        self._name = name

    # ALTERNATIVE CONSTRUCTORS ================================================

    @classmethod
    def from_csv(cls, path, volcano_name):
        """Build a :class:`Volcano` instance from a CSV file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the CSV file with the measurements.
        volcano_name : str
            Name of the volcano.

        Returns
        -------
        Volcano
            A new instance built from the CSV contents.

        """
        df = pd.read_csv(Path(path))
        return cls(df, volcano_name)

    # ACCESSORS (YES, WE USE CACHED PROPERTIES IS THE EASIEST WAY) ============

    @methodtools.lru_cache(maxsize=None)
    @property
    def plots(self):
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
    def dataframe(self):
        """Copy of the internal, normalized DataFrame.

        Returns
        -------
        pandas.DataFrame
            A copy, so the internal state cannot be mutated from the outside.

        """
        return self._dataframe.copy()

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

    # METHODS =================================================================

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
        row_mean = self._dataframe[distance_columns].mean(axis=1)
        long = self._dataframe[["date", "direction"]].assign(
            residual=row_mean
        )
        return long.pivot_table(
            index="date",
            columns="direction",
            values="residual",
            observed=True,
        )

    def missing_report(self):
        """Summarize missing (``NaN``) distance-bin measurements, by date.

        Returns
        -------
        pandas.Series
            - ``n_missing``: total number of missing measurement values.
            - ``n_dates``: number of distinct dates in the measurements.
            - ``n_dates_with_missing``: dates with at least one missing
              value (in any direction/distance bin that day).
            - ``missing_per_date_min``/``_max``/``_mean``: statistics of
              the number of missing values per date.
            - ``worst_date``: the date with the most missing values.

        """
        distance_columns = [
            c for c in self._dataframe.columns if c not in NON_DISTANCE_COLUMNS
        ]
        missing_per_row = self._dataframe[distance_columns].isna().sum(axis=1)
        missing_per_date = missing_per_row.groupby(
            self._dataframe["date"]
        ).sum()

        return pd.Series(
            {
                "n_missing": int(missing_per_row.sum()),
                "n_dates": self._dataframe["date"].nunique(),
                "n_dates_with_missing": int((missing_per_date > 0).sum()),
                "missing_per_date_min": missing_per_date.min(),
                "missing_per_date_max": missing_per_date.max(),
                "missing_per_date_mean": missing_per_date.mean(),
                "worst_date": missing_per_date.idxmax(),
            }
        )

    # MAGIC ===================================================================

    def __repr__(self):
        distance_columns = [
            c for c in self._dataframe.columns if c not in NON_DISTANCE_COLUMNS
        ]
        n_dates = self._dataframe["date"].nunique()
        n_missing = int(self._dataframe[distance_columns].isna().sum().sum())
        registers = len(self._dataframe)
        return (
            f"Volcano(name={self._name!r}, dates={n_dates}, "
            f"missing={n_missing}, "
            f"registers={registers})"
        )


# =============================================================================
# IO
# =============================================================================


def read_csv(path, volcano_name):
    """Build a :class:`Volcano` instance from a CSV file.

    Thin module-level wrapper around :meth:`Volcano.from_csv`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file with the measurements.
    volcano_name : str
        Name of the volcano.

    Returns
    -------
    Volcano
        A new instance built from the CSV contents.

    """
    return Volcano.from_csv(path, volcano_name)
