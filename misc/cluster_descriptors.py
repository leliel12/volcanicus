"""Cluster volcanoes by their scene descriptors with K-means and a GMM."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from volcanicus import datasets

KS = range(2, 9)
COLUMNS = ["peak", "efold_length", "integrated", "anisotropy"]  # azimuth is circular

# 1. One row per volcano: median and IQR of each descriptor over its scenes.
rows = {}
for name in datasets.available():
    d = datasets.load(name).mask_rays().scenes_descriptors()[COLUMNS].dropna()
    if len(d) >= 10:
        q1, med, q3 = d.quantile([0.25, 0.5, 0.75]).to_numpy()
        rows[name] = np.r_[med, q3 - q1]
features = pd.DataFrame(
    rows, index=[f"{c}_{s}" for s in ("med", "iqr") for c in COLUMNS]
).T

# 2. Log-compress the skewed magnitudes, then scale robustly.
prep = make_pipeline(FunctionTransformer(np.log1p), RobustScaler())
X = prep.fit_transform(features.clip(lower=0))

# 3. Choose k: silhouette for K-means, BIC for the GMM.
km = {k: KMeans(k, n_init=10, random_state=0).fit(X) for k in KS}
gm = {k: GaussianMixture(k, n_init=5, random_state=0).fit(X) for k in KS}
print(pd.DataFrame({
    "kmeans_silhouette": {k: silhouette_score(X, m.labels_) for k, m in km.items()},
    "gmm_bic": {k: m.bic(X) for k, m in gm.items()},
}))

# 4. Labels at the best k of each model (the GMM also gives soft memberships).
k_km = max(KS, key=lambda k: silhouette_score(X, km[k].labels_))
k_gm = min(KS, key=lambda k: gm[k].bic(X))
features["kmeans"] = km[k_km].labels_
features["gmm"] = gm[k_gm].predict(X)
features["gmm_confidence"] = gm[k_gm].predict_proba(X).max(axis=1)

print(f"\nkmeans k={k_km}, gmm k={k_gm}")
print(features.groupby("kmeans")[features.columns[:8]].median())
print(features.groupby("gmm")[features.columns[:8]].median())
