#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Anomaly view of :class:`~volcanicus.core.Volcano` instances \
(``volcano.anomalies``)."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

from .base import AccessorABC

# =============================================================================
# CLASSES
# =============================================================================


class AnomaliesAccessor(AccessorABC):
    """Every scene measured against the volcano's usual state \
    (used as ``volcano.anomalies``).

    Calling the accessor (``volcano.anomalies()``) is the same as
    :meth:`zscores`.

    Parameters
    ----------
    volcano : Volcano
        The :class:`~volcanicus.core.Volcano` instance this accessor belongs
        to.

    """

    _default_kind = "zscores"

    def __init__(self, volcano):
        self._volcano = volcano

    def baseline(self, min_scenes=10):
        """Typical value and spread of every cell of the scenes.

        For every ``(direction, distance)`` cell, over every scene of this
        volcano: the median (``center``) and a robust spread (``scale``,
        ``1.4826 * MAD``, which matches the standard deviation for normally
        distributed data). Each cell gets its own baseline because each
        direction can sit on a different background (e.g. a slope facing
        the sun, or the sea).

        This is the volcano's *usual* state, not necessarily a quiet one:
        for a volcano active in most of its scenes, "usual" means active,
        and the anomalies (see :meth:`zscores`) measure departures from
        that. Run :meth:`RaysAccessor.mask` first so corrupted rays don't
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
            ``(direction, distance)`` columns as
            :meth:`ScenesAccessor.grid`. A cell whose values never vary
            (MAD of 0, e.g. always saturated) has a ``NaN`` scale.

        """
        scenes = self._volcano.scenes.grid()
        center = scenes.median()
        scale = 1.4826 * (scenes - center).abs().median()

        too_few = scenes.count() < min_scenes
        center[too_few] = np.nan
        scale[too_few | (scale == 0)] = np.nan
        return pd.DataFrame({"center": center, "scale": scale}).T

    def zscores(self, min_scenes=10):
        """Every scene as robust z-scores against the volcano's baseline.

        Each cell becomes ``(value - center) / scale``, with ``center`` and
        ``scale`` from :meth:`baseline`: how many "typical deviations" it
        sits above (positive) or below (negative) the volcano's usual value
        for that direction and distance. Unlike raw residuals, these are
        comparable across volcanoes with very different backgrounds.

        Parameters
        ----------
        min_scenes : int, default 10
            Passed on to :meth:`baseline`.

        Returns
        -------
        pandas.DataFrame
            Same shape, index and columns as :meth:`ScenesAccessor.grid`.
            ``NaN`` where the value is missing or the cell has no baseline.

        """
        baseline = self.baseline(min_scenes)
        return (
            self._volcano.scenes.grid() - baseline.loc["center"]
        ) / baseline.loc["scale"]
