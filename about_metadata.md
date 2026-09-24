One row per volcano dataset (187 rows), matching one-to-one with the files in
`volcanicus/datasets/` (CSV) and `volcanicus/datasets_yaml/` (YAML). First
column is `file`.

## What changed from the previous version

This file replaces the previous `metadata.csv` / `about_metadata.md`, which
described a different schema (`gvp_volcano_number`, `region`, `subregion`,
`activity_status` as a free column, etc.) that no longer matches what's
actually in `metadata.csv`. That version is gone; below is the real thing.

Also: **14 volcanoes from the old dataset were dropped** because no updated
source data (pivot tables from the current GEE extraction, script v18) exists
for them: `kick_em_jenny`, `marapi`, `meakandake`, `numazawa`,
`rincon_de_la_vieja`, `tinakula` (had an old metadata row, no new data),
plus `gorely_lago1`, `gorely_lago3`, `krakatau_lago1`, `mugogo`,
`ritter_island_lago1`, `ritter_island_lago2`, `visoke`,
`whakaari_white_island_lago1` (had neither). They can be reintroduced later
if/when new pivot tables for them become available. `dieng` and `kelimutu`,
which used to be single aggregated files, are now split into their real
per-crater/per-lake series (`dieng_sikidang`, `dieng_sileri`,
`kelimutu_lago1/2/3`), matching how the GEE extraction script actually
treats them.

## Coordinates (`latitude`, `longitude`)

These are **not** the manually-defined search-center points hardcoded in the
GEE extraction script (`VOLCANES` array, one `{lat, lon}` per volcano used
only to seed the search radius). They're the **organic focus**: for each
volcano, every scene in the pivot table independently detects its own
hottest/coldest pixel (`foco_lat`, `foco_lon` in the GEE script) within that
search radius, and those detected points repeat across scenes because the
underlying detector is pixel-quantized (~30 m, Landsat resolution). The
`latitude`/`longitude` reported here is the **mode** of those repeated
detections per volcano (ties broken by proximity to the overall mean). This
was chosen over a plain mean/centroid because the mean can land on a
coordinate that was never actually detected in any scene, while the mode is
guaranteed to be a real, repeatedly-observed point.

Cross-checked against the hardcoded search-center points in the GEE script:
187/187 matched within a reasonable radius, average offset ~242 m, median
~236 m, max ~766 m (a handful of large, diffuse volcanic fields). This
confirms the organic focus is a fine correction on top of the original
search seed, not a divergence from it.

`gee_vs_gvp_offset_km` was recomputed from these coordinates against the
Smithsonian GVP 5.4.0 Holocene volcano list (`GVP_2026.xlsx`, matched by
`gvp_number_2026`). Empty = exact match, or the volcano isn't in GVP 5.4.0
(Domuyo and Seulawah Agam).

## Data cleaning: `residuo` outliers

The `residuo` column in the pivot tables (one value per distance bin) should
theoretically stay within `[-1, 1]`, since it's `lst_norm` (already clamped
to `[0, 1]`) minus a per-sector linear-regression prediction. In practice,
0.0076% of cells (264 out of 3,460,570) fall outside that range — up to
±1264 in one extreme case (`kadovar`). This happens when a sector has too
few valid pixels for a given scene (after cloud/snow/shadow masking), making
the regression fit numerically unstable.

**These 264 cells were nulled out** in both the CSV and YAML datasets
shipped here. The original values, with their exact location (file, date,
direction, distance bin), are preserved in `residuo_outliers.csv` at the
repo root, so this cleaning step is fully reversible and auditable — it was
not applied silently.

## Columns

| Column | Group | Source | Description |
|---|---|---|---|
| `file` | Identification | volcanicus | CSV/YAML filename for this series. |
| `volcano_name` | Identification | GVP 4.6.7 (2018) | Volcano name, not updated to GVP 5.4.0 naming. |
| `gvp_number_2026` | Identification | GVP 5.4.0 | Current GVP volcano number (2026 edition). |
| `gvp_number_2018` | Identification | GVP 4.6.7 | 2018 GVP number. Used by PyVOLCANS and the morphology database. Differs from `gvp_number_2026` for 2 volcanoes (Bora-Bericha, Sessagara Hills). |
| `country` | Location | GVP 5.4.0 | Country. |
| `gvp_region_group` | Location | GVP 5.4.0 | Volcanic region group (current GVP taxonomy). |
| `gvp_volcanic_region` | Location | GVP 5.4.0 | Volcanic region (current GVP taxonomy). |
| `latitude` / `longitude` | Location | GEE scripts (organic focus, see above) | Decimal degrees of the mode of detected foci. |
| `elevation_m_2018` | Location | GVP 4.6.7 (2018) | Nominal elevation per the 2018 GVP. |
| `elevation_m_2026` | Location | GVP 5.4.0 (2026) | Nominal elevation per the current GVP. Differs from the 2018 value for 33 volcanoes (not yet decided which to use). |
| `gvp_landform` | GVP | GVP 5.4.0 | Edifice form per GVP (Composite, Shield, Caldera, Cluster, Minor). Cluster/Minor correspond to volcanic fields, cinder cones and fissures here. |
| `dominant_rock_type` | GVP | GVP 5.4.0 | Dominant rock. Manual exception: Tupungatito (potassic andesites). |
| `tectonic_setting` | GVP | GVP 5.4.0 | Tectonic setting. |
| `last_known_eruption` | GVP | GVP 5.4.0 | Last known eruption. |
| `activity_evidence` | GVP | GVP 5.4.0 | Evidence of Holocene activity. |
| `volcano_type_class` | Volcano type | GVP 5.4.0, regrouped | Unified volcano-type naming (Complex, Shield, Stratovolcano, Pyroclastic cone, Fissure vent, Lava dome; Bagana keeps Lava cone). Not the monogenetic/polygenetic split, which isn't a GVP term (see `gvp_landform`). |
| `rock_affinity` | Derived | Own | Regrouping of `dominant_rock_type`. |
| `thermal_regime_class` | Own label | Own | Proposed thermal regime; partly LLM-assisted, pending validation. |
| `crater_lake_mentioned` | Own label | Own | Whether a crater lake is mentioned (Yes/No/Unknown/...). |
| `label_confidence` | Own label | Own | Confidence of the label (high/moderate/.../pending). |
| `thermal_proxy_set` | Own label | Own | Id of the recommended spectral-proxy set; text in `thermal_proxy_sets.csv`. |
| `morph_source` | Morphology | PyVOLCANS | Source of W\* and T: PikeClow1981 = Pike & Clow (1981), USGS OFR 81-1073; Grosse2014 = Grosse et al. (2014), Bull. Volcanol. 76:784. |
| `morph_height_km` | Morphology | PyVOLCANS | Edifice height (H), km. |
| `morph_half_width_km` | Morphology | PyVOLCANS | Basal half-width (W\*), km. |
| `morph_summit_basal_width_ratio` | Morphology | PyVOLCANS | Summit/basal width ratio (T). |
| `morph_crater_diameter_km` | Morphology | PyVOLCANS | Crater diameter (d), km. Many gaps. |
| `morph_crater_depth_km` | Morphology | PyVOLCANS | Crater depth (h), km. Many gaps. |
| `morph_ellipticity_index` | Morphology | PyVOLCANS | Average ellipticity index. |
| `morph_circularity` | Morphology | PyVOLCANS | Circularity (C). Many gaps. |
| `morph_secondary_vents` | Morphology | PyVOLCANS | Number of secondary peaks/vents (sv, GR2014). |
| `morph_note_code` | Morphology | PyVOLCANS | Source observation code; see PyVOLCANS's `README_NUMBERING.md`. |
| `morph_pc1` | Morphology | Derived from PyVOLCANS (PC81/GR2014) | Morphological index 1 (~53% of variance): size and steepness. High = tall, wide, narrow-summit (cone-like); low = short, small, wide-summit. Mean 0, std 1 in the reference base. |
| `morph_pc2` | Morphology | Derived from PyVOLCANS (PC81/GR2014) | Morphological index 2 (~33%): width and flatness. High = wide edifice with a wide summit relative to height (shield/caldera-like). Mean 0, std 1 in the reference base. |
| `gee_vs_gvp_offset_km` | Audit | This process | Distance in km between the GEE point (`latitude`, `longitude`) and the GVP 5.4.0 catalogue coordinate. Empty = exact match, or the volcano isn't in GVP 5.4.0 (Domuyo, Seulawah Agam). |
| `pyvolcans_analogue_vnums` | Analogues | PyVOLCANS 1.3.3 | GVP (2018) number of each volcano in `pyvolcans_analogues`, same order. |
| `activity_status` | Derived | Own | Regrouping of `activity_evidence`. |
| `pyvolcans_analogues` | Analogues | PyVOLCANS 1.3.3 | 5 analogues with country and `total_analogy` (equal weights; ties broken by GVP number). |
| `notes` | Notes | Various | Grouped notes: `[lagos]`, `[morfología]`, `[PyVOLCANS]`. |

## Per-file metadata (YAML)

Each file in `volcanicus/datasets_yaml/` carries its own metadata as the last
entry of the YAML list, under the key `metadata:`, with exactly the row above
that corresponds to that file. This is redundant with the top-level
`metadata.csv` by design, so a single YAML file is self-contained.
