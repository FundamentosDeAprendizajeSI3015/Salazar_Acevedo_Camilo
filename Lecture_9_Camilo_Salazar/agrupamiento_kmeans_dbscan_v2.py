"""
Agrupamiento (Clustering) - KMeans y DBSCAN
SI3015 - Fundamentos de Aprendizaje Automático

Dataset: dataset_sintetico_FIRE_UdeA.csv
  - 500 filas, sin valores nulos
  - Features: liquidez, dias_efectivo, cfo, participacion_ley30,
               hhi_fuentes, gastos_personal, tendencia_ingresos
  - Columna 'label' excluida del clustering (es la clase real)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────
# Configuración general
# ─────────────────────────────────────────
random_state = 42
plt.rc('font', family='serif', size=12)

# ─────────────────────────────────────────
# Carga y preparación del dataset
# ─────────────────────────────────────────
df = pd.read_csv("dataset_sintetico_FIRE_UdeA.csv")

features = [
    "liquidez",
    "dias_efectivo",
    "cfo",
    "participacion_ley30",
    "hhi_fuentes",
    "gastos_personal",
    "tendencia_ingresos",
]

data = df[features].values

# Estandarizamos los datos (necesario: las magnitudes varían mucho entre columnas)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Par de features para visualización 2D
x_idx = features.index("liquidez")
y_idx = features.index("dias_efectivo")
x_label = "liquidez"
y_label = "dias_efectivo"

# ─────────────────────────────────────────
# 1. KMeans con K = 2
# ─────────────────────────────────────────
kmeans_k2 = KMeans(n_clusters=2, random_state=random_state, n_init=10)
kmeans_k2.fit(data_scaled)

labels_k2 = kmeans_k2.labels_
inertia_k2 = kmeans_k2.inertia_

print(f"KMeans K=2 — Inercia: {inertia_k2:.4f}")
print(f"Distribución de clusters: {np.unique(labels_k2, return_counts=True)}")

fig, ax = plt.subplots(figsize=(5 * 1.6, 5))
scatter = ax.scatter(data_scaled[:, x_idx], data_scaled[:, y_idx],
                     c=labels_k2, cmap="tab10")
ax.set_title("KMeans — K = 2")
ax.set_xlabel(x_label + " (estandarizado)")
ax.set_ylabel(y_label + " (estandarizado)")
plt.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig("kmeans_k2.png", dpi=120)
plt.show()

# ─────────────────────────────────────────
# 2. Método del Codo
# ─────────────────────────────────────────
K_range = range(1, 11)
inertias = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(data_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(5 * 1.6, 5))
ax.plot(list(K_range), inertias, "bo-")
ax.set_title("Método del Codo")
ax.set_xlabel("Número de Clusters (K)")
ax.set_ylabel("Inercia")
plt.tight_layout()
plt.savefig("metodo_codo.png", dpi=120)
plt.show()

# ─────────────────────────────────────────
# 3. KMeans con el K óptimo (ajusta según el codo)
# ─────────────────────────────────────────
k_optimo = 2  # <-- cambia este valor según la gráfica del codo

kmeans_opt = KMeans(n_clusters=k_optimo, random_state=random_state, n_init=10)
kmeans_opt.fit(data_scaled)

labels_opt = kmeans_opt.labels_
inertia_opt = kmeans_opt.inertia_

print(f"\nKMeans K={k_optimo} (codo) — Inercia: {inertia_opt:.4f}")
print(f"Distribución de clusters: {np.unique(labels_opt, return_counts=True)}")

fig, ax = plt.subplots(figsize=(5 * 1.6, 5))
scatter = ax.scatter(data_scaled[:, x_idx], data_scaled[:, y_idx],
                     c=labels_opt, cmap="tab10")
ax.set_title(f"KMeans — K = {k_optimo} (codo)")
ax.set_xlabel(x_label + " (estandarizado)")
ax.set_ylabel(y_label + " (estandarizado)")
plt.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig(f"kmeans_k{k_optimo}.png", dpi=120)
plt.show()

# ─────────────────────────────────────────
# 4. DBSCAN
# ─────────────────────────────────────────
# Pipeline: StandardScaler → DBSCAN
# eps y min_samples ajustados para 500 muestras y 7 features
clu_dbscan = Pipeline([
    ("scaler", StandardScaler()),
    ("clustering", DBSCAN(eps=1.2, min_samples=5)),
])
clu_dbscan.fit(data)

labels_dbscan = clu_dbscan["clustering"].labels_

unique, counts = np.unique(labels_dbscan, return_counts=True)
n_clusters = len(unique[unique >= 0])
n_ruido = counts[unique == -1][0] if -1 in unique else 0

print(f"\nDBSCAN — Clusters encontrados (sin ruido): {n_clusters}")
print(f"Puntos de ruido (-1): {n_ruido}")
print(f"Etiquetas únicas y conteos: {list(zip(unique, counts))}")

fig, ax = plt.subplots(figsize=(5 * 1.6, 5))
scatter = ax.scatter(data_scaled[:, x_idx], data_scaled[:, y_idx],
                     c=labels_dbscan, cmap="tab10")
ax.set_title("DBSCAN")
ax.set_xlabel(x_label + " (estandarizado)")
ax.set_ylabel(y_label + " (estandarizado)")
plt.colorbar(scatter, ax=ax, label="Cluster (-1 = ruido)")
plt.tight_layout()
plt.savefig("dbscan.png", dpi=120)
plt.show()
