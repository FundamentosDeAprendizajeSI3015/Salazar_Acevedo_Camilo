import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# -------------------------
# PARÁMETROS
# -------------------------

EPS = 3.0        # radio del vecindario (CAMBIA ESTO)
MIN_SAMPLES = 3

# -------------------------
# FEATURES
# -------------------------

FEATURES = [
    "ingresos_totales",
    "gastos_personal",
    "liquidez",
    "dias_efectivo",
    "cfo",
    "participacion_ley30",
    "participacion_regalias",
    "participacion_servicios",
    "participacion_matriculas",
    "hhi_fuentes",
    "endeudamiento",
    "tendencia_ingresos",
    "gp_ratio",
]

# -------------------------
# CARGAR DATASET
# -------------------------

df = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")

X = df[FEATURES].dropna()

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

print("Clusters encontrados:",
      len(set(clusters)) - (1 if -1 in clusters else 0))

print("Ruido:", np.sum(clusters == -1))

# -------------------------
# PCA
# -------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# -------------------------
# GRÁFICA
# -------------------------

plt.figure(figsize=(7,5))

unique_clusters = set(clusters)

for c in unique_clusters:

    mask = clusters == c

    if c == -1:
        plt.scatter(
            X_pca[mask,0],
            X_pca[mask,1],
            c="gray",
            s=25,
            label="ruido"
        )
    else:
        plt.scatter(
            X_pca[mask,0],
            X_pca[mask,1],
            s=25,
            label=f"cluster {c}"
        )

plt.title(f"DBSCAN realista (eps={EPS})")

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.legend()

plt.savefig("dbscan_dataset_FIRE_realista.png", dpi=150)

plt.show()