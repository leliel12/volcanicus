#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Episode view of :class:`~volcanicus.core.Volcano` instances \
(``volcano.episodes``)."""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd

from .base import AccessorABC

# =============================================================================
# CLASSES
# =============================================================================


class EpisodesAccessor(AccessorABC):
    """Periods of unusual activity: runs of anomalous scenes \
    (used as ``volcano.episodes``).

    Calling the accessor (``volcano.episodes()``) is the same as
    :meth:`find`.

    Parameters
    ----------
    volcano : Volcano
        The :class:`~volcanicus.core.Volcano` instance this accessor belongs
        to.

    """

    _default_kind = "find"

    #: Reductions :meth:`score` accepts for its ``how``.
    _SCORE_HOW = ("median", "mean", "max")

    def __init__(self, volcano):
        self._volcano = volcano

    def score(self, how="median", min_scenes=10):
        """How anomalous every scene is, as a single number.

        The scene's anomalies (:meth:`AnomaliesAccessor.zscores`) are
        reduced across the 8 directions with ``how``, leaving one profile of
        z-scores per scene; the score is that profile's maximum over
        distance. With the default ``"median"``, a score of ``z`` means at
        least half the directions sit ``z`` usual deviations above normal
        at some distance: robust to a single odd ray, but blind to heat
        that goes only one way (e.g. a lava flow down one flank); ``"max"``
        catches that, at the cost of more false alarms.

        Parameters
        ----------
        how : str, default "median"
            Reduction across directions: ``"median"``, ``"mean"`` or
            ``"max"``.
        min_scenes : int, default 10
            Passed on to :meth:`AnomaliesAccessor.baseline`.

        Returns
        -------
        pandas.Series
            Indexed by ``date``. ``NaN`` for a scene with no anomaly at
            all (e.g. every cell lacks a baseline).

        Raises
        ------
        ValueError
            If ``how`` is not a supported reduction.

        """
        if how not in self._SCORE_HOW:
            raise ValueError(
                f"'how' must be one of {self._SCORE_HOW}, found {how!r}"
            )
        anomaly = self._volcano.anomalies.zscores(min_scenes)
        by_distance = anomaly.T.groupby(level="distance", sort=False)
        score = getattr(by_distance, how)().T.max(axis=1)
        score.name = "score"
        return score

    def find(self, threshold=3.0, max_gap=90, how="median", min_scenes=10):
        """Find periods of unusual activity: runs of anomalous scenes.

        A scene is anomalous when its :meth:`score` exceeds ``threshold``.
        An episode is a run of consecutive anomalous scenes: it ends at the
        first scene that isn't anomalous, or when two anomalous scenes are
        more than ``max_gap`` days apart with nothing observed in between
        (there's no telling what happened in such a gap). Scenes without a
        score are treated as not observed.

        Parameters
        ----------
        threshold : float, default 3.0
            Score a scene must exceed to count as anomalous.
        max_gap : int, default 90
            Longest gap, in days, allowed between two scenes of the same
            episode. The default leaves room for a few missed passes of a
            satellite revisiting every 16 days.
        how : str, default "median"
            Passed on to :meth:`score`.
        min_scenes : int, default 10
            Passed on to :meth:`AnomaliesAccessor.baseline`.

        Returns
        -------
        pandas.DataFrame
            One row per episode, in chronological order, with the columns
            ``start`` and ``end`` (first and last anomalous scene),
            ``duration_days`` (``0`` for a single scene), ``n_scenes``,
            ``peak_date`` and ``peak_score`` (the highest score).

        """
        score = self.score(how, min_scenes).dropna()
        anomalous = score > threshold

        # A new episode starts at every anomalous scene whose predecessor
        # isn't anomalous or is too far back in time.
        gap_days = score.index.to_series().diff().dt.days
        starts = anomalous & (
            ~anomalous.shift(1, fill_value=False) | (gap_days > max_gap)
        )
        episode_id = starts.cumsum()[anomalous]

        rows = []
        for _, members in score[anomalous].groupby(episode_id):
            rows.append(
                {
                    "start": members.index[0],
                    "end": members.index[-1],
                    "duration_days": (
                        members.index[-1] - members.index[0]
                    ).days,
                    "n_scenes": len(members),
                    "peak_date": members.idxmax(),
                    "peak_score": members.max(),
                }
            )
        columns = [
            "start",
            "end",
            "duration_days",
            "n_scenes",
            "peak_date",
            "peak_score",
        ]
        return pd.DataFrame(rows, columns=columns)
