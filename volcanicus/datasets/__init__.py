#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2026, Rosenfeld, Hernán; Cabral, Juan B.
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The :mod:`volcanicus.datasets` module includes utilities to fetch \
bundled example :class:`~volcanicus.core.Volcano` datasets.

Each dataset lives in its own directory, ``<name>/``, holding the volcano's
descriptive metadata (``<name>.json``) and its measurements, either as
``<name>.csv`` (one row per date/direction, the old format every bundled
volcano except Erta Ale still uses) or ``<name>.parquet`` (one row per pixel
and scene; :func:`fetch` pivots it into the same shape, for whichever
`band` is asked for).

Turning a per-pixel parquet into those wide ring tables is the ``pivot``
stage of ``volcanicus_pipeline_v12.py``, rewritten here to start from an
already processed parquet (as in ``erta_ale.parquet``) and to return
DataFrames instead of writing CSV files. It lives in this module (rather
than importing :class:`~volcanicus.core.Volcano` from a separate one) to
dodge the circular import that would otherwise cause:
:mod:`~volcanicus.core` needs it too, for
:meth:`~volcanicus.core.Volcano.pixels` and
:meth:`~volcanicus.core.Volcano.to_band`, and gets it with a *lazy* import
inside those two methods instead.

Every table has the same layout as the old CSVs: one row per
``(date, direction)`` with the columns ``date, center_lat, center_long,
direction`` followed by one column per 30 m ring (``15m``, ``45m``, ...,
``885m``) holding a statistic of a pixel quantity inside that ring and
sector. :data:`~volcanicus.constants.BANDS` lists the bands a table can be
built for (``"fused"`` by default). Everything below :func:`available` and
:func:`fetch` is internal plumbing (hence the leading underscores): the
only entry points the rest of the package needs are
:func:`_read_binned_pixels` and :func:`_band_table`.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import functools
import json
import os
import pathlib

import numpy as np

import pandas as pd

from ..constants import BANDS
from ..core import Volcano

# =============================================================================
# CONSTANTS
# =============================================================================

_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

#: Width of each distance ring, in meters.
RING_WIDTH_M = 30

#: Outer radius of the profile, in meters.
MAX_RADIUS_M = 900

#: Columns that are not distance rings.
METADATA_COLUMNS = ("date", "center_lat", "center_long", "direction")

#: Sector index (0 = N, clockwise in 45° steps) to compass direction.
SECTOR_TO_DIRECTION = {
    0: "N",
    1: "NE",
    2: "E",
    3: "SE",
    4: "S",
    5: "SO",
    6: "O",
    7: "NO",
}

#: Pixel columns the tables are built from.
REQUIRED_COLUMNS = (
    "stem",
    "date",
    "sector",
    "distance_m",
    "focus_lat",
    "focus_lon",
    "sat",
    "lst_k",
    "bg_tir_k",
    "sat_tir_final",
    "resn_tir",
    "lst_corr_k",
    "sat_swir1",
    "sat_swir2",
    "resn_swir1",
    "resn_swir2",
    "p_tir_w",
    "p_swir_w",
)

#: Fusion parameters. The first three are only defaults: :func:`fetch` and
#: :func:`_band_table` let the caller override them.
#: Temperature (K) at which the TIR of each Landsat sensor (``sat`` column)
#: saturates, as in ``LST_SAT_POR_SENSOR`` of the data collection script. It
#: gives the lower bound of the saturated pixels in the ``"sum"`` fusion and
#: maps to ``lst_norm = 1`` in the ``"blend"`` one.
LST_SATURATION_K = {"L7": 347.5, "L8": 373.0, "L9": 373.0}
#: How TIR and SWIR are combined: ``"sum"`` (TIR + SWIR powers) or the old
#: ``"blend"`` (sigmoid-weighted average, needs ``tir`` and ``swir`` columns).
FUSION_METHOD = "sum"
#: Pixel area (m2), TIR emissivity and Stefan-Boltzmann constant, as in the
#: data collection script (``p_tir_w``).
PIXEL_AREA_M2 = 900.0
EMISSIVITY_TIR = 0.95
SIGMA_SB = 5.670374e-8
#: ``lst_norm`` at which TIR and SWIR weigh the same (only for ``"blend"``).
FUSION_CENTER = 0.82
#: Width of the TIR -> SWIR transition, in ``lst_norm`` units.
FUSION_SCALE = 0.055
#: Only pixels below this ``lst_norm`` calibrate SWIR against TIR.
CALIBRATION_MAX_LST_NORM = 0.90
#: Minimum number of pixels needed to calibrate; otherwise identity.
CALIBRATION_MIN_PIXELS = 50
#: Bounds for the calibration slope and intercept.
CALIBRATION_SLOPE_RANGE = (0.25, 4.0)
CALIBRATION_INTERCEPT_RANGE = (-8.0, 8.0)

# =============================================================================
# PIVOT HELPERS
# =============================================================================


def _ring_labels():
    """Return the ring column names, from ``15m`` to ``885m``."""
    half = RING_WIDTH_M // 2
    return [f"{c}m" for c in range(half, MAX_RADIUS_M, RING_WIDTH_M)]


def _bin_by_distance(pixels):
    """Add the ``ring`` column (ring center in meters) and drop pixels \
    outside the profile radius."""
    in_range = pixels["distance_m"].between(0, MAX_RADIUS_M)
    pixels = pixels[in_range].copy()
    ring_index = np.floor(pixels["distance_m"] / RING_WIDTH_M)
    pixels["ring"] = (ring_index * RING_WIDTH_M + RING_WIDTH_M // 2).astype(
        int
    )
    pixels["direction"] = pixels["sector"].map(SECTOR_TO_DIRECTION)
    return pixels


def _pivot_rings(
    pixels, value, *, quantile=0.5, omni_center=False, decimals=4
):
    """Build a ``(date, direction) x ring`` table with a quantile of \
    ``value``.

    Parameters
    ----------
    pixels : pandas.DataFrame
        Output of :func:`_bin_by_distance`.
    value : str
        Pixel column to aggregate.
    quantile : float, default 0.5
        Quantile taken inside each cell (0.5 is the median).
    omni_center : bool, default False
        The first ring only has a few pixels, so most sectors are empty
        there. When True that ring gets, for every direction, the median
        of all its pixels regardless of sector (the ``centro_omni``
        behavior of v12).
    decimals : int, default 4
        Rounding applied to the result.

    Returns
    -------
    pandas.DataFrame

    """
    by_cell = pixels.groupby(["date", "sector", "ring"])
    cells = by_cell[["focus_lat", "focus_lon"]].mean()
    cells = cells.rename(
        columns={"focus_lat": "center_lat", "focus_lon": "center_long"}
    )
    cells["value"] = by_cell[value].quantile(quantile)
    cells = cells.reset_index()
    cells["direction"] = cells["sector"].map(SECTOR_TO_DIRECTION)
    cells = cells.dropna(subset=["direction"])

    centers = cells.groupby(["date", "direction"])[
        ["center_lat", "center_long"]
    ].mean()

    table = cells.pivot_table(
        index=["date", "direction"], columns="ring", values="value"
    )
    table.columns = [f"{ring}m" for ring in table.columns]
    table = table.join(centers).reset_index()
    table = table.reindex(columns=list(METADATA_COLUMNS) + _ring_labels())

    if omni_center:
        first_ring = RING_WIDTH_M // 2
        center = (
            pixels[pixels["ring"] == first_ring]
            .groupby("date")[value]
            .quantile(quantile)
        )
        table[f"{first_ring}m"] = table["date"].map(center)

    table = table.sort_values(["date", "direction"], ignore_index=True)
    return table.round(decimals)


# =============================================================================
# PIVOT FUSION
# =============================================================================


def _normalized_lst(pixels, lst_saturation_k=LST_SATURATION_K):
    """LST scaled to ``[0, 1]`` between the scene background and the \
    saturation temperature of each sensor."""
    background = pixels["bg_tir_k"]
    saturation = pixels["sat"].map(lst_saturation_k)
    lst_norm = (pixels["lst_k"] - background) / (saturation - background)
    return lst_norm.clip(0, 1)


def _calibrate_swir(tir, swir, lst_norm):
    """Fit ``log1p(tir) = intercept + slope * log1p(swir)``.

    Only pixels where both signals are positive and TIR is far from
    saturation (``lst_norm < CALIBRATION_MAX_LST_NORM``) are used. With
    fewer than :data:`CALIBRATION_MIN_PIXELS` of them the identity
    (slope 1, intercept 0) is returned.

    Returns
    -------
    dict
        ``slope``, ``intercept`` and ``n_pixels`` used in the fit.

    """
    usable = (tir > 0) & (swir > 0) & (lst_norm < CALIBRATION_MAX_LST_NORM)
    x = np.log1p(swir[usable])
    y = np.log1p(tir[usable])
    n_pixels = int(usable.sum())

    slope, intercept = 1.0, 0.0
    if n_pixels >= CALIBRATION_MIN_PIXELS:
        variance = (x * x).mean() - x.mean() ** 2
        if variance > 1e-12:
            slope = ((x * y).mean() - x.mean() * y.mean()) / variance
        intercept = y.mean() - slope * x.mean()

    return {
        "slope": float(np.clip(slope, *CALIBRATION_SLOPE_RANGE)),
        "intercept": float(np.clip(intercept, *CALIBRATION_INTERCEPT_RANGE)),
        "n_pixels": n_pixels,
    }


def _fuse_tir_swir(
    pixels,
    tir="p_tir_w",
    swir="p_swir_w",
    *,
    lst_saturation_k=LST_SATURATION_K,
    fusion_center=FUSION_CENTER,
    fusion_scale=FUSION_SCALE,
    method=FUSION_METHOD,
):
    """Combine the TIR and SWIR signals of every pixel into one signal.

    With ``method="sum"`` (default) the signals are the radiative powers (W)
    and the result is ``TIR + SWIR`` (two-component pixel model, see NOTES),
    each term taken as 0 when missing and pixels with neither NaN. Pixels
    with saturated TIR (``sat_tir_final``) take, instead of the missing TIR
    power, its lower bound ``A * eps * sigma * (T_sat^4 - T_bg^4)`` with
    ``T_sat`` from ``lst_saturation_k``. The rest of this docstring describes
    ``method="blend"``.

    SWIR is first calibrated to the TIR scale (:func:`_calibrate_swir`).
    Its weight follows a sigmoid of :func:`_normalized_lst` centered at
    ``fusion_center``, so it takes over as TIR approaches saturation.
    Pixels with no positive SWIR keep the TIR value, and saturated pixels
    (no TIR residual) take the calibrated SWIR.

    Parameters
    ----------
    pixels : pandas.DataFrame
        Pixels with the ``tir``, ``swir``, ``sat``, ``lst_k`` and
        ``bg_tir_k`` columns (and ``sat_tir_final`` for ``"sum"``).
    tir, swir : str
        Columns holding the TIR and SWIR signals (powers by default).
    lst_saturation_k : dict, default :data:`LST_SATURATION_K`
        Saturation temperature (K) per sensor (``L7``, ``L8``, ``L9``).
    fusion_center : float, default :data:`FUSION_CENTER`
        ``lst_norm`` at which TIR and SWIR weigh the same.
    fusion_scale : float, default :data:`FUSION_SCALE`
        Width of the TIR -> SWIR transition, in ``lst_norm`` units.

    Returns
    -------
    fused : pandas.Series
        Fused signal, NaN where both inputs are missing.
    calibration : dict
        Output of :func:`_calibrate_swir` (None for ``"sum"``).

    """
    tir_values, swir_values = pixels[tir], pixels[swir]

    if method == "sum":
        saturated = pixels["sat_tir_final"].fillna(False).astype(bool)
        swir_saturated = pixels["sat_swir1"].fillna(False).astype(bool)
        t_sat = pixels["sat"].map(lst_saturation_k)
        bound = (
            PIXEL_AREA_M2
            * EMISSIVITY_TIR
            * SIGMA_SB
            * (t_sat**4 - pixels["bg_tir_k"] ** 4)
        ).clip(lower=0)
        tir_part = tir_values.where(~saturated, bound).fillna(0).clip(lower=0)
        fused = tir_part + swir_values.fillna(0).clip(lower=0)
        fused[saturated & swir_saturated] = np.nan
        fused[~saturated & swir_saturated] = np.nan
        fused[tir_values.isna() & ~saturated & ~(swir_values > 0)] = np.nan
        return fused, None
    if method != "blend":
        raise ValueError(f"method must be 'sum' or 'blend', not {method!r}")

    lst_norm = _normalized_lst(pixels, lst_saturation_k)
    calibration = _calibrate_swir(tir_values, swir_values, lst_norm)

    swir_calibrated = np.expm1(
        calibration["intercept"]
        + calibration["slope"] * np.log1p(swir_values.clip(lower=0))
    ).clip(lower=0)
    swir_weight = 1 / (
        1 + np.exp(-(lst_norm.fillna(0) - fusion_center) / fusion_scale)
    )
    tir_part = tir_values.fillna(0).clip(lower=0)

    fused = tir_part.where(
        ~(swir_values > 0),
        (1 - swir_weight) * tir_part + swir_weight * swir_calibrated,
    )
    fused[tir_values.isna() & swir_values.isna()] = np.nan
    return fused, calibration


# =============================================================================
# PIVOT TABLES
# =============================================================================


def _tir_residual_table(pixels):
    """Normalized TIR residual (``resn_tir``), median per ring."""
    return _pivot_rings(pixels, "resn_tir", omni_center=True, decimals=4)


def _lst_table(pixels):
    """Land surface temperature in K (``lst_corr_k``), median per ring."""
    return _pivot_rings(pixels, "lst_corr_k", omni_center=True, decimals=2)


def _swir1_residual_table(pixels):
    """Normalized SWIR1 residual (``resn_swir1``), median per ring."""
    return _pivot_rings(pixels, "resn_swir1", omni_center=True, decimals=6)


def _swir2_residual_table(pixels):
    """Normalized SWIR2 residual (``resn_swir2``), median per ring."""
    return _pivot_rings(pixels, "resn_swir2", omni_center=True, decimals=6)


def _fused_table(pixels, **fusion_params):
    """TIR/SWIR fused power in W (:func:`_fuse_tir_swir`), 90th percentile \
    per ring.

    ``fusion_params`` (``lst_saturation_k``, ``fusion_center``,
    ``fusion_scale``) are passed to :func:`_fuse_tir_swir`. The SWIR
    calibration is kept in ``table.attrs["calibration"]``.

    """
    fused, calibration = _fuse_tir_swir(pixels, **fusion_params)
    table = _pivot_rings(
        pixels.assign(fused=fused),
        "fused",
        quantile=0.9,
        omni_center=True,
        decimals=4,
    )
    table.attrs["calibration"] = calibration
    return table


#: Table-builder for each entry of :data:`~volcanicus.constants.BANDS`,
#: except ``"fused"`` (handled separately by :func:`_band_table` since it
#: alone takes the fusion parameters).
_BAND_TABLE_BUILDERS = {
    "tir": _tir_residual_table,
    "lst": _lst_table,
    "swir1": _swir1_residual_table,
    "swir2": _swir2_residual_table,
}


# =============================================================================
# PIVOT API
# =============================================================================


def _read_binned_pixels(path):
    """Read a per-pixel parquet for one volcano, binned by distance.

    Parameters
    ----------
    path : str or path-like
        Parquet file with one row per pixel and scene, holding the columns
        in :data:`REQUIRED_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        Pixels restricted to the profile radius, with the ``ring`` and
        ``direction`` columns already assigned.

    Raises
    ------
    ValueError
        If the parquet holds pixels for more than one volcano.

    """
    pixels = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    stems = pixels["stem"].unique()
    if len(stems) != 1:
        raise ValueError(
            f"Expected one volcano per parquet, found {sorted(stems)}"
        )
    return _bin_by_distance(pixels)


def _band_table(
    binned_pixels,
    band,
    *,
    lst_saturation_k=LST_SATURATION_K,
    fusion_center=FUSION_CENTER,
    fusion_scale=FUSION_SCALE,
):
    """Build one band's ``(date, direction) x ring`` table.

    Parameters
    ----------
    binned_pixels : pandas.DataFrame
        Output of :func:`_read_binned_pixels`.
    band : str
        One of :data:`~volcanicus.constants.BANDS`.
    lst_saturation_k, fusion_center, fusion_scale
        Only used when ``band == "fused"``; see :func:`_fuse_tir_swir`.

    Returns
    -------
    pandas.DataFrame

    """
    if band not in BANDS:
        raise ValueError(f"'band' must be one of {BANDS}, found {band!r}")
    if band == "fused":
        return _fused_table(
            binned_pixels,
            lst_saturation_k=lst_saturation_k,
            fusion_center=fusion_center,
            fusion_scale=fusion_scale,
        )
    return _BAND_TABLE_BUILDERS[band](binned_pixels)


# =============================================================================
# FUNCTIONS
# =============================================================================


@functools.cache
def available() -> list[str]:
    """List the volcano names bundled with :mod:`volcanicus.datasets`.

    Returns
    -------
    list of str
        Names accepted by :func:`fetch`, sorted alphabetically.

    """
    names = {
        p.parts[-2]
        for p in (*_PATH.glob("*/*.csv"), *_PATH.glob("*/*.parquet"))
        if not p.parts[-2].startswith(".")
    }
    return sorted(names)


def fetch(
    volcano_name,
    band="fused",
    *,
    lst_saturation_k=LST_SATURATION_K,
    fusion_center=FUSION_CENTER,
    fusion_scale=FUSION_SCALE,
):
    """Fetch a bundled example :class:`~volcanicus.core.Volcano` dataset.

    Everything is read from the local package today; the name anticipates
    datasets that, in the future, are fetched remotely instead.

    Parameters
    ----------
    volcano_name : str
        Name of the bundled volcano to fetch (see :func:`available` for the
        valid values). Matched case-sensitively against the dataset
        directory name.
    band : str, default "fused"
        For a volcano backed by a per-pixel parquet (currently only
        ``"erta_ale"``), which of :data:`~volcanicus.constants.BANDS` to
        pivot it into. Ignored for a volcano backed by a plain CSV, which
        has no bands to choose from.
    lst_saturation_k, fusion_center, fusion_scale
        Forwarded to the fusion that builds the ``"fused"`` band (see
        :func:`_fuse_tir_swir`); ignored for every other `band`, and for a
        CSV-backed volcano.

    Returns
    -------
    Volcano
        A new instance built from the bundled measurements, with the
        ``latitude``/``longitude`` of ``<name>.json`` as its GVP reference
        location (:attr:`~volcanicus.core.Volcano.gvp_lat`,
        :attr:`~volcanicus.core.Volcano.gvp_lon`), the rest of the file as
        its :attr:`~volcanicus.core.Volcano.metadata`, and, for a
        parquet-backed volcano, :attr:`~volcanicus.core.Volcano.band` set
        to `band` (``None`` otherwise).

    Raises
    ------
    ValueError
        If ``volcano_name`` is not a bundled dataset, or `band` is not one
        of :data:`~volcanicus.constants.BANDS`.

    """
    if band not in BANDS:
        raise ValueError(f"'band' must be one of {BANDS}, found {band!r}")

    dir_path = _PATH / volcano_name
    
    if not dir_path.exists():
        raise ValueError(f"Unknow volcan  {volcano_name!r}")
    
    parquet_path = dir_path / f"{volcano_name}.parquet"
    if not parquet_path.exists():
        raise ValueError(f"No data for  {volcano_name!r}")
    
    metadata_path = dir_path / f"{volcano_name}.json"
    if not metadata_path.exists():
        raise ValueError(f"No metadata for  {volcano_name!r}")
   


    metadata = json.loads(metadata_path.read_text())
    gvp_lat = metadata.pop("latitude")
    gvp_lon = metadata.pop("longitude")

    fusion_params = {
        "lst_saturation_k": lst_saturation_k,
        "fusion_center": fusion_center,
        "fusion_scale": fusion_scale,
    }
    binned_pixels = _read_binned_pixels(parquet_path)
    table = _band_table(binned_pixels, band, **fusion_params)
    return Volcano(
        name=volcano_name,
        df=table,
        gvp_lat=gvp_lat,
        gvp_lon=gvp_lon,
        metadata=metadata,
        band=band,
        pixel_source=parquet_path,
        fusion_params=fusion_params,
    )


# =============================================================================
# NOTES
# =============================================================================

# Source data
# -----------
# - The input parquet is NOT the ``crudos_limpios.parquet`` written by the
#   ``preparar`` stage of v12. It comes from another version of the
#   pipeline: it has ``res_tir3_k``, ``motivo_tir``, ``pF_tir``, ``bin_i``,
#   ``c_*`` cell stats, etc., and lacks ``p_fused_w``, ``flag_sat``,
#   ``p_tir_w``, ``residuo_plot``, ``res_rad1/2`` and ``th_swir_k``. The v12
#   pivot logic was therefore adapted to these columns.
# - The ``c_*_med`` columns of the parquet are exactly the per-cell
#   (date x sector x ring) medians of ``resn_tir``, ``lst_corr_k``,
#   ``resn_swir1`` and ``resn_swir2``. The tables recompute them from the
#   pixels and match them up to rounding (except the 15 m ring, see below).
# - ``(date, direction)`` pairs where every ring is NaN are dropped by
#   ``pivot_table``, same as v12.
#
# Column choices
# --------------
# - TIR uses ``resn_tir`` (normalized residual), not ``res_tir`` (Kelvin),
#   because it is the closest match to the v12 ``residuo``.
# - LST uses ``lst_corr_k``. It equals ``lst_k`` except on the ~1045
#   saturated TIR pixels, where it is NaN. The v12 emissivity correction is
#   not applied again: the parquet already carries its own correction.
# - The 15 m ring only has ~3 pixels, so most sectors are empty there. TIR
#   and fused fill it with the omnidirectional value per date (v12
#   ``centro_omni``); LST and SWIR do not, also as in v12.
#
# Fusion
# ------
# - Default (``method="sum"``): fused = ``p_tir_w + p_swir_w`` (W), 90th
#   percentile per cell. Two-component pixel model: ``p_tir_w`` is the TIR
#   excess over the predicted background and ``p_swir_w`` the power of the
#   hot component (0 where there is no SWIR detection).
# - Pixels with saturated TIR (``sat_tir_final``) take, instead of the
#   missing TIR power, its lower bound ``A·ε·σ·(T_sat⁴ − T_bg⁴)`` — except
#   when SWIR is also saturated (``sat_swir1``/``sat_swir2``), in which case
#   fused is NaN: the bound assumes ``T = T_sat`` (368 K) but SWIR
#   saturation implies ``T >> 591 K``, so the bound would underestimate by
#   ~7x.
# - The TIR also sees part of the hot component: at most ~16 % of
#   ``p_swir_w`` for a hot component >= 1100 K (own calculation, Planck at
#   10.9 um, background 300 K, emissivity 0.95). It is not subtracted.
# - Saturated-TIR pixels (NaN ``p_tir_w``, ``sat_tir_final``) take the
#   saturation lower bound as TIR term, with the same ``T_sat`` as the data
#   collection script (``LST_SAT_POR_SENSOR``), so they are not
#   underestimated.
# - ``method="blend"`` is the previous scheme (log-linear calibration of
#   SWIR against TIR, sigmoid weight on ``lst_norm``); pass the residual
#   columns ``resn_tir`` and ``resn_swir2`` as ``tir`` and ``swir``.
# - Pixels with neither TIR nor SWIR are NaN here; in v12 they became 0
#   (``GREATEST(NULL, 0)``), which pulled the 90th percentile down.
