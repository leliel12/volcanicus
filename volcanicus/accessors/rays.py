#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Ray view of :class:`~volcanicus.core.Volcano` instances \
(``volcano.rays``)."""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import numpy as np

import pandas as pd

from ._profiles import (
    distance_radii,
    fill_rays,
    peak_and_efold,
    sector_area,
)
from .base import AccessorABC
from ..constants import (
    DIRECTION_ORDER,
    NON_DISTANCE_COLUMNS as _NON_DISTANCE_COLUMNS,
    SATURATION_LEVEL,
)

# =============================================================================
# CLASSES
# =============================================================================


class RaysAccessor(AccessorABC):
    """Measurements as rays: one direction of one scene \
    (used as ``volcano.rays``).

    A ray is the radial profile of a single date/direction: one row of the
    source data. Calling the accessor (``volcano.rays()``) is the same as
    :meth:`descriptors`.

    Parameters
    ----------
    volcano : Volcano
        The :class:`~volcanicus.core.Volcano` instance this accessor belongs
        to.

    """

    _default_kind = "descriptors"

    def __init__(self, volcano):
        self._volcano = volcano

    def descriptors(self):
        """Summarize every ray (one direction of one scene).

        Describing rays one by one, instead of whole scenes, tells where
        around the vent things happen, and isolates problems that only
        affect one direction.

        - ``peak``: maximum residual along the ray.
        - ``efold_length``: distance (m) from the peak over which the ray
          decays to ``peak / e``; defined exactly as in
          :meth:`DescriptorsAccessor.table`.
        - ``integrated``: residual integrated over the ray's annular
          sector (``2 * pi * r * dr / 8`` per bin), in residual x m². The
          8 rays of a scene add up to its ``integrated``.
        - ``n_saturated``: number of bins at the saturation level
          (:data:`~volcanicus.constants.SATURATION_LEVEL`). Residuals are
          capped there, so a saturated ray's ``peak`` underestimates the
          real anomaly.
        - ``max_abs``: largest absolute residual along the ray, on the raw
          values (see :meth:`flag`).

        Gaps are handled as in :meth:`DescriptorsAccessor.table`. Rays
        missing from the data appear with every descriptor ``NaN``
        (``n_saturated`` 0), so the result always holds 8 rays per scene.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(date, direction)``, directions in
            :data:`~volcanicus.constants.DIRECTION_ORDER`, with the columns
            ``peak``, ``efold_length``, ``integrated``, ``n_saturated`` and
            ``max_abs``.

        """
        scenes = self._volcano.scenes.grid()
        cube = self._volcano.scenes.grid(as_array=True)
        n_distances = cube.shape[2]
        radii = distance_radii(scenes.columns.unique("distance"))
        filled, interior = fill_rays(cube, radii)

        rays = cube.reshape(-1, n_distances)
        peak, efold_length = peak_and_efold(
            interior.reshape(-1, n_distances), radii
        )
        integrated = (
            filled.reshape(-1, n_distances) * sector_area(radii)
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

    def flag(self, limit=2.0):
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
            :meth:`descriptors`; ``True`` for a flagged ray.

        See Also
        --------
        mask : Drop the flagged rays' values.

        """
        flags = self.descriptors()["max_abs"] > limit
        flags.name = "flagged"
        return flags

    def mask(self, limit=2.0):
        """Blank out every ray :meth:`flag` flags.

        Every distance bin of a flagged ray becomes ``NaN``, so it's
        treated like any other missing ray afterwards (skipped by the
        reductions, fillable by :meth:`~volcanicus.core.Volcano.impute`).
        The whole ray goes, not only the offending values: the bins next to
        an absurd value are rarely trustworthy (in Kadovar a single
        artifact spans 4 bins).

        Parameters
        ----------
        limit : float, default 2.0
            Passed on to :meth:`flag`.

        Returns
        -------
        Volcano
            A new instance with the flagged rays masked. Its metadata
            records the ``mask_limit`` used and how many rays it masked
            (``masked_rays``).

        """
        volcano = self._volcano
        flags = self.flag(limit)
        flagged = flags[flags].index

        df = volcano.to_dataframe()
        distance_columns = [
            c for c in df.columns if c not in _NON_DISTANCE_COLUMNS
        ]
        rows = pd.MultiIndex.from_frame(
            df[["date", "direction"]].astype({"direction": str})
        ).isin(flagged)
        df.loc[rows, distance_columns] = np.nan

        metadata = {
            **volcano.metadata,
            "masked_rays": int(rows.sum()),
            "mask_limit": limit,
        }
        # type(volcano) instead of importing Volcano, which imports this
        # module.
        return type(volcano)(
            volcano.name,
            df,
            gvp_lat=volcano.gvp_lat,
            gvp_lon=volcano.gvp_lon,
            metadata=metadata,
        )
