import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# PARÁMETROS
# -------------------------

EPS = 1.5
MIN_SAMPLES = 5

# -------------------------
# CARGAR DATASET
# -------------------------

df = pd.read_csv("dataset_sintetico_FIRE_UdeA.csv")

X = df.drop(columns=["label"])

# -------------------------
# ESCALAR
# -------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# DBSCAN
# -------------------------

db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)

clusters = db.fit_predict(X_scaled)

print("Clusters:",
      len(set(clusters)) - (1 if -1 in clusters else 0))

print("Ruido:", np.sum(clusters == -1))

# -------------------------
# PCA (3 DIMENSIONES)
# -------------------------

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)

print("Varianza explicada por PCA:",
      pca.explained_variance_ratio_)

# -------------------------
# GRÁFICA 3D
# -------------------------

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111, projection='3d')

unique_clusters = set(clusters)

for c in unique_clusters:

    mask = clusters == c

    if c == -1:

        ax.scatter(
            X_pca[mask,0],
            X_pca[mask,1],
            X_pca[mask,2],
            c="gray",
            s=20,
            label="ruido"
        )

    else:

        ax.scatter(
            X_pca[mask,0],
            X_pca[mask,1],
            X_pca[mask,2],
            s=20,
            label=f"cluster {c}"
        )

ax.set_title(f"DBSCAN en PCA 3D (eps={EPS})")

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")

ax.legend()

plt.savefig("dbscan_pca_3d.png", dpi=150)

plt.show()