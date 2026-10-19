#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Radial profile helpers shared by the scene and ray descriptors."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..constants import DIRECTION_ORDER

# =============================================================================
# FUNCTIONS
# =============================================================================


def distance_radii(distance_columns):
    """Numeric radius (m) of every distance bin, from names like ``"15m"``."""
    return np.array([float(c.removesuffix("m")) for c in distance_columns])


def sector_area(radii):
    """Area (m²) of every bin's annular sector: 1/8 of ``2 * pi * r * dr``."""
    return 2 * np.pi * radii * np.gradient(radii) / len(DIRECTION_ORDER)


def fill_rays(rays, radii):
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


def peak_and_efold(profiles, radii):
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
