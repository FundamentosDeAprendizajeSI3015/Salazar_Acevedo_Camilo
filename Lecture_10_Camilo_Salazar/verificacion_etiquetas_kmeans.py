import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------
# CARGAR DATASET
# -----------------------------

df = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")

labels = df["label"]

features = [c for c in df.columns if c not in ["label","anio","unidad"]]

X = df[features].dropna().reset_index(drop=True)
labels = labels.loc[X.index].reset_index(drop=True)
# -----------------------------
# ESCALAR DATOS
# -----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# KMEANS (2 CLUSTERS)
# -----------------------------

kmeans = KMeans(n_clusters=2, random_state=0)

clusters = kmeans.fit_predict(X_scaled)

# -----------------------------
# RESULTADOS
# -----------------------------

df_res = pd.DataFrame({
    "cluster": clusters,
    "label": labels
})

print("\nComposición de clusters")

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
        f"({100*correctos/len(sub):.1f}%)"
    )

# -----------------------------
# REDUCCIÓN DE DIMENSIÓN (PCA)
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
    s=40
)

plt.title("Clusters detectados (KMeans)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.savefig("clusters_kmeans.png", dpi=150)

plt.show()

# -----------------------------
# GRÁFICA ETIQUETAS ORIGINALES
# -----------------------------

plt.figure(figsize=(7,5))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels,
    cmap="Set1",
    s=40
)

plt.title("Etiquetas originales")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.savefig("labels_originales.png", dpi=150)

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
    s=30,
    alpha=0.5
)

if len(errores) > 0:

    plt.scatter(
        X_pca[errores.index,0],
        X_pca[errores.index,1],
        c="red",
        s=80
    )

plt.title("Posibles errores de etiquetado")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.savefig("posibles_errores.png", dpi=150)

plt.show()

# -----------------------------
# IDENTIFICAR FALSOS NEGATIVOS (label 1)
# -----------------------------

# falso negativo: predijo 0 pero el label real es 1
fn_mask = (df_res["pred_label"] == 0) & (df_res["label"] == 1)

falsos_negativos = df_res[fn_mask]

print("\nTotal falsos negativos para label=1:", len(falsos_negativos))

# recuperar la facultad/unidad original
falsos_negativos["unidad"] = df.loc[falsos_negativos.index, "unidad"].values

# contar falsos negativos por facultad
fn_por_facultad = (
    falsos_negativos
    .groupby("unidad")
    .size()
    .sort_values(ascending=False)
)

print("\nFalsos negativos por facultad:")
print(fn_por_facultad)

# facultad con mayor cantidad
if len(fn_por_facultad) > 0:

    peor_facultad = fn_por_facultad.index[0]
    cantidad = fn_por_facultad.iloc[0]

    print("\nFacultad con más falsos negativos en label 1:")
    print(f"{peor_facultad} → {cantidad} casos")

# guardar resultados
fn_por_facultad.to_csv("falsos_negativos_label1_por_facultad.csv")