"""
Agrupamiento (Clustering) - KMeans y DBSCAN
SI3015 - Fundamentos de Aprendizaje Automático

Dataset: dataset_sintetico_FIRE_UdeA_realista.csv
Se usan columnas numéricas relevantes para el clustering.
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
df = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")

# Seleccionamos columnas numéricas relevantes para clustering
# (descartamos 'anio', 'unidad' y 'label')
features = [
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

df_clean = df[features].dropna()
data = df_clean.values

# Estandarizamos los datos (importante para DBSCAN y KMeans con distancias)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Para visualización usamos las dos primeras componentes (o las dos más relevantes)
# Aquí usamos liquidez vs gp_ratio como par representativo
x_idx = features.index("liquidez")
y_idx = features.index("gp_ratio")

x_label = "liquidez"
y_label = "gp_ratio"

# ─────────────────────────────────────────
# 1. KMeans con K = 2
# ─────────────────────────────────────────
kmeans_k2 = KMeans(n_clusters=2, random_state=random_state, n_init=10)
kmeans_k2.fit(data_scaled)

labels_k2 = kmeans_k2.labels_
inertia_k2 = kmeans_k2.inertia_

print(f"KMeans K=2 — Inercia: {inertia_k2:.4f}")
print(f"Distribución de clusters: {np.unique(labels_k2, return_counts=True)}")

fig, ax = plt.subplots(figsize=( 5 * 1.6, 5))
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
# 2. Método del Codo para encontrar el mejor K
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
# 3. KMeans con el K seleccionado (codo)
# ─────────────────────────────────────────
# Ajusta este valor según lo que indique la gráfica del codo:
k_optimo = 3

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
# Pipeline con StandardScaler + DBSCAN
# (los datos ya están escalados, pero dejamos el pipeline como en el notebook)
clu_dbscan = Pipeline([
    ("scaler", StandardScaler()),
    ("clustering", DBSCAN(eps=1.5, min_samples=3)),
])
clu_dbscan.fit(data)  # pipeline aplica scaler internamente

labels_dbscan = clu_dbscan["clustering"].labels_

unique, counts = np.unique(labels_dbscan, return_counts=True)
print(f"\nDBSCAN — Clusters encontrados (sin ruido): {len(unique[unique >= 0])}")
print(f"Etiquetas únicas y conteos: {(unique, counts)}")

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
