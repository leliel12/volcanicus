#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Descriptors view of :class:`~volcanicus.core.Volcano` instances \
(``volcano.descriptors``)."""

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

# =============================================================================
# CLASSES
# =============================================================================


class DescriptorsAccessor(AccessorABC):
    """Every scene condensed into a few interpretable numbers \
    (used as ``volcano.descriptors``).

    Calling the accessor (``volcano.descriptors()``) is the same as
    :meth:`table`.

    Parameters
    ----------
    volcano : Volcano
        The :class:`~volcanicus.core.Volcano` instance this accessor belongs
        to.

    """

    _default_kind = "table"

    def __init__(self, volcano):
        self._volcano = volcano

    def table(self):
        """Summarize every scene with a few interpretable descriptors.

        Each scene (see :meth:`ScenesAccessor.grid`) holds 8 directions x
        every distance bin; this condenses it into five numbers that are
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
          (see :meth:`RaysAccessor.descriptors`), over the sum of their
          lengths. Using the ray peaks keeps it about the near field:
          weighting by area instead would let the far field, which covers
          most of the disk, decide the direction.
        - ``azimuth``: direction of that vector sum, in degrees clockwise
          from north (``N`` is 0, ``E`` is 90, ``S`` is 180, ``O`` is 270).
          ``NaN`` when no direction dominates (``anisotropy`` is 0).

        ``anisotropy`` and ``azimuth`` need all 8 directions, so they're
        ``NaN`` for a scene with a direction missing entirely (see
        :meth:`~volcanicus.core.Volcano.impute` to fill those first).

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
        RaysAccessor.descriptors : The same kind of descriptors, per
            direction.

        """
        scenes = self._volcano.scenes.grid()
        cube = self._volcano.scenes.grid(as_array=True)
        n_dates, n_directions, n_distances = cube.shape
        radii = distance_radii(scenes.columns.unique("distance"))
        filled, interior = fill_rays(cube, radii)
        missing_direction = np.isnan(cube).all(axis=2)

        # peak & efold_length: shape of the typical (median) profile.
        # A bin with no data in any direction is an expected NaN, not
        # something to warn about.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_profile = np.nanmedian(interior, axis=1)
        peak, efold_length = peak_and_efold(median_profile, radii)

        # integrated: per-direction integrals; a missing direction is
        # skipped and the mean of the others stands in for it.
        direction_integral = (filled * sector_area(radii)).sum(axis=2)
        direction_integral[missing_direction] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            integrated = np.nanmean(direction_integral, axis=1) * n_directions

        # anisotropy & azimuth: vector sum of every ray's positive peak,
        # each pointing along its compass bearing.
        ray_peak, _ = peak_and_efold(interior.reshape(-1, n_distances), radii)
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
