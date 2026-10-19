"""Cluster volcanoes by their scene descriptors — v2.

Diferencias con la v1 (cluster_descriptors.py):

1. integrated -> integral de CAMPO CERCANO (<= 300 m). La de disco completo
   está dominada por el campo lejano, levemente negativo en 148/180 volcanes
   (tutorial 03).
2. + fracción de escenas SATURADAS (residual == 1.0). El cluster "caliente"
   de la v1 era casi exactamente el grupo saturado, donde peak es un piso.
3. + dos features RELATIVAS a partir de episodes.score (tutoriales 05-06).
4. clip + log1p -> arcsinh. El clip apilaba en 0 los 12 volcanes con
   integrated negativo y fabricaba el tercer cluster de k=3.
5. PCA (90 % de varianza) para que el eje "cuán caliente", repetido en
   varias features correlacionadas, cuente una sola vez.
6. k se elige exigiendo silhouette > referencia nula Y estabilidad bootstrap;
   GMM con covarianza diagonal en vez de completa (sobreajustaba).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from volcanicus import datasets
from volcanicus.constants import SATURATION_LEVEL

# Módulo privado de volcanicus: lo usamos para que la integral de campo
# cercano rellene huecos y pese por área EXACTAMENTE igual que
# descriptors.table(). Si la librería suma esa integral, este import se va.
from volcanicus.accessors._profiles import distance_radii, fill_rays, sector_area

warnings.filterwarnings("ignore", category=RuntimeWarning)  # medianas de NaN

KS = range(2, 9)  # mismo rango que v1
COLUMNS = ["peak", "efold_length", "anisotropy"]  # v1 + sin integrated
NEAR = 300.0  # radio (m) del campo cercano, el corte del tutorial 03
rng = np.random.default_rng(0)

# 1. Una fila por volcán. ----------------------------------------------------
rows = {}
for name in datasets.available():
    # Igual que v1: enmascarar rays corruptos antes de nada.
    v = datasets.load(name).rays.mask()
    d = v.descriptors.table()[COLUMNS]

    # NUEVO: integral de campo cercano. Misma cuenta que `integrated` en la
    # librería (huecos interpolados, bin x área de su sector, dirección
    # faltante = promedio de las otras), pero sumando solo bins <= 300 m.
    # En v1, el anillo de 885 m pesaba 59 veces el de 15 m.
    cube = v.scenes.grid(as_array=True)  # (fechas, 8 direcciones, 30 bins)
    radii = distance_radii(v.scenes.grid().columns.unique("distance"))
    filled, _ = fill_rays(cube, radii)
    near = radii <= NEAR
    per_dir = (filled[:, :, near] * sector_area(radii)[near]).sum(axis=2)
    per_dir[np.isnan(cube).all(axis=2)] = np.nan  # dirección sin datos
    d["near_integrated"] = np.nanmean(per_dir, axis=1) * 8

    d = d.dropna()
    if len(d) < 10:  # mismo criterio que v1 (quedan 180 de 186)
        continue

    # Igual que v1: mediana (estado típico) e IQR (variabilidad).
    q1, med, q3 = d.quantile([0.25, 0.5, 0.75]).to_numpy()
    row = dict(zip([f"{c}_med" for c in d], med))
    row.update(zip([f"{c}_iqr" for c in d], q3 - q1))

    # NUEVO: fracción de escenas con algún bin en el tope de 1.0. Le avisa
    # al modelo que ahí `peak` es un piso, no una medida (tutorial 04).
    row["saturated_frac"] = (cube == SATURATION_LEVEL).any(axis=(1, 2)).mean()

    # NUEVO: actividad RELATIVA al estado habitual de cada volcán
    # (z-scores robustos, comparables entre volcanes). No usamos episodes()
    # porque solo 57 volcanes tienen episodios de 2+ escenas: serían casi
    # todo ceros. Usamos la distribución del score, que existe para todos:
    #  - anomaly_rate: fracción de escenas con score > 3 (por azar ~4 %).
    #  - score_q90: tamaño de los apartamientos grandes (p90, no el máximo,
    #    para que una escena rara no decida).
    # Sesgo conocido: un volcán activo siempre (lago de lava) tiene lo
    # activo como "habitual" y acá se ve tranquilo; por eso esto se SUMA a
    # las features absolutas, no las reemplaza.
    score = v.episodes.score().dropna()
    row["anomaly_rate"] = (score > 3).mean()
    row["score_q90"] = score.quantile(0.9)

    rows[name] = row
features = pd.DataFrame(rows).T

# 2. Transformación con signo + escalado robusto. ------------------------------
# v1: clip(lower=0) + log1p -> los negativos colapsaban en 0. arcsinh se
# comporta como log en las colas pero conserva el signo. Dividir por la
# mediana absoluta pone el "codo" en la escala típica de cada feature.
T = np.arcsinh(features / features.abs().median().replace(0, 1))

# Mediana e IQR como el RobustScaler de v1, pero si el IQR es 0 se usa el
# std: saturated_frac vale 0 en la mayoría de los volcanes, su IQR es 0 y
# sin esto quedaría sin escalar y pesaría casi nada en la PCA.
iqr = T.quantile(0.75) - T.quantile(0.25)
T = (T - T.median()) / iqr.where(iqr > 0, T.std())

# 3. PCA para sacar redundancia. -----------------------------------------------
# En v1, peak_med, integrated_med y peak_iqr tenían Spearman 0.6-0.77: el
# eje "cuán caliente" pesaba triple. Las cargas muestran qué mide cada eje.
pca = PCA(n_components=0.90, svd_solver="full").fit(T)
X = pca.transform(T)
print(f"{len(features)} volcanes, {X.shape[1]} componentes PCA")
print(pd.DataFrame(pca.components_.T, index=features.columns).iloc[:, :3].round(2))

# 4. Evaluar cada k. -------------------------------------------------------------
# v1 elegía k por silhouette o BIC a secas. Acá, por cada k:
#  - silhouette de k-means (igual que v1);
#  - null_p95: silhouette con cada columna permutada por separado (mismas
#    distribuciones, sin grupos reales). K-means encuentra "clusters"
#    incluso en ruido; esta es la vara para compararlo;
#  - estabilidad: ARI mediano entre el modelo y reajustes en bootstrap
#    (1 = misma partición, 0 = azar). En v1, k=2 daba 0.68 con p10 de -0.07;
#  - BIC de un GMM DIAGONAL: v1 usaba covarianza completa, ~45 parámetros
#    por componente para 180 volcanes, y sobreajustaba.
km = {k: KMeans(k, n_init=10, random_state=0).fit(X) for k in KS}
gm = {k: GaussianMixture(k, covariance_type="diag", n_init=5, random_state=0).fit(X) for k in KS}
evaluation = {}
for k in KS:
    null = []
    for _ in range(30):
        Xp = np.column_stack([rng.permutation(c) for c in X.T])
        null.append(silhouette_score(Xp, KMeans(k, n_init=10, random_state=0).fit_predict(Xp)))
    boot = []
    for _ in range(50):
        idx = rng.choice(len(X), len(X))
        boot.append(adjusted_rand_score(
            km[k].labels_, KMeans(k, n_init=10, random_state=0).fit(X[idx]).predict(X)
        ))
    evaluation[k] = {
        "silhouette": silhouette_score(X, km[k].labels_),
        "null_p95": np.percentile(null, 95),
        "stability": np.median(boot),
        "gmm_bic": gm[k].bic(X),
        "sizes": np.bincount(km[k].labels_).tolist(),
    }
evaluation = pd.DataFrame(evaluation).T
print(evaluation)

# 5. Elegir k: tiene que superar a la nula Y ser estable (ARI >= 0.75). ------
# Si ninguno pasa, no hay evidencia de grupos discretos: se muestra el k más
# estable solo como descripción, no como "tipos" de volcán.
ok = evaluation[(evaluation.silhouette > evaluation.null_p95) & (evaluation.stability >= 0.75)]
if len(ok):
    k = int(ok.silhouette.astype(float).idxmax())
else:
    k = int(evaluation.stability.astype(float).idxmax())
    print("Ningún k supera la nula con estabilidad: no hay grupos claros.")
features["cluster"] = km[k].labels_
# Igual que v1: el GMM da membresías blandas; confianza baja = volcán "entre"
# grupos, esperable si los datos son un continuo.
features["gmm_confidence"] = gm[k].predict_proba(X).max(axis=1)

# 6. Describir y validar. --------------------------------------------------------
# Medianas en unidades ORIGINALES (no transformadas), para leerlas físicamente.
print(f"\nk={k}")
print(features.groupby("cluster").median().round(3).T)

# Validación externa contra la clasificación térmica de los metadatos
# (parcial: 66 volcanes están "Not classified").
features["thermal_regime_class"] = [
    datasets.load(n).m.thermal_regime_class for n in features.index
]
print(pd.crosstab(features.thermal_regime_class, features.cluster))

features.to_csv("features_clusters_v2.csv")
