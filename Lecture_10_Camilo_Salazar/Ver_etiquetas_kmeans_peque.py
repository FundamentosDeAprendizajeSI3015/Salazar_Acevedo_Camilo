import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------
# CARGAR DATASET
# -----------------------------

df = pd.read_csv("dataset_sintetico_FIRE_UdeA.csv")

labels = df["label"]

X = df.drop(columns=["label"])

# -----------------------------
# ESCALAR
# -----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# KMEANS (2 CLUSTERS)
# -----------------------------

kmeans = KMeans(n_clusters=2, random_state=0)

clusters = kmeans.fit_predict(X_scaled)

df_res = pd.DataFrame({
    "cluster": clusters,
    "label": labels
})

# -----------------------------
# COMPOSICIÓN DE CLUSTERS
# -----------------------------

print("\nComposición de clusters:")

tabla = (
    df_res.groupby("cluster")["label"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)

print(tabla)

# -----------------------------
# ASIGNAR CLASE A CADA CLUSTER
# -----------------------------

cluster_pred = {}

for c in tabla.index:

    if tabla.loc[c,1] >= tabla.loc[c,0]:
        cluster_pred[c] = 1
    else:
        cluster_pred[c] = 0

df_res["pred_label"] = df_res["cluster"].map(cluster_pred)

# -----------------------------
# PORCENTAJE CORRECTO
# -----------------------------

print("\nVerificación de etiquetas")

for clase in [0,1]:

    sub = df_res[df_res["label"] == clase]

    correctos = (sub["pred_label"] == sub["label"]).sum()

    print(
        f"label {clase}: {correctos}/{len(sub)} "
        f"({100*correctos/len(sub):.2f}%)"
    )

# -----------------------------
# PCA
# -----------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# GRÁFICA CLUSTERS
# -----------------------------

plt.figure(figsize=(7,5))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="Set1",
    s=20
)

plt.title("Clusters detectados (KMeans)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.savefig("clusters_kmeans_pca.png", dpi=150)

plt.show()

# -----------------------------
# GRÁFICA ETIQUETAS
# -----------------------------

plt.figure(figsize=(7,5))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels,
    cmap="Set1",
    s=20
)

plt.title("Etiquetas originales")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.savefig("labels_originales_pca.png", dpi=150)

plt.show()

# -----------------------------
# POSIBLES ERRORES
# -----------------------------

errores = df_res[df_res["pred_label"] != df_res["label"]]

print("\nPosibles errores de etiqueta:", len(errores))

plt.figure(figsize=(7,5))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c="lightgray",
    s=15,
    alpha=0.5
)

if len(errores) > 0:

    idx = errores.index.values

    plt.scatter(
        X_pca[idx,0],
        X_pca[idx,1],
        c="red",
        s=60
    )

plt.title("Posibles errores de etiquetado")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.savefig("posibles_errores_pca.png", dpi=150)

plt.show()