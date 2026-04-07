"""
═══════════════════════════════════════════════════════════════════════════════
  NETFLIX GENRE CLASSIFICATION — ANÁLISIS COMPLETO
  ─────────────────────────────────────────────────
  1. Preprocesamiento (via pipeline.py)
  2. Clustering No Supervisado: K-Means, Fuzzy C-Means, Subtractive, DBSCAN,
     + familia cluster (Agglomerative, Gaussian Mixture, Mean-Shift)
  3. Reevaluación de etiquetas (~30% ruido)
  4. Modelos Supervisados: Árbol de Decisión, Regresión Logística, Ridge
  5. Comparación: dataset original vs dataset con etiquetas corregidas
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Directorios de salida ─────────────────────────────────────────────────────
OUT = "eda_outputs"
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 0 — Importar pipeline y preparar datos
# ═══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load_and_prepare

CSV_PATH = "netflix_merged_intersection.csv"

print("=" * 65)
print("  CARGANDO Y PREPROCESANDO DATOS (pipeline.py)")
print("=" * 65)

X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, mlb = \
    load_and_prepare(filepath=CSV_PATH)

# Reconstruir el dataset completo para clustering (usamos todo)
import importlib.util
from pipeline import load_and_prepare, group_rare_countries
from sklearn.preprocessing import MultiLabelBinarizer

df_raw = pd.read_csv(CSV_PATH)
df_raw = df_raw[df_raw["director"] != "Not Given"].copy()
df_raw = df_raw[[
    "type_base", "country", "release_year", "rating",
    "duration", "listed_in", "imdb_score", "imdb_votes", "age_certification"
]].copy()
df_raw = df_raw.dropna(subset=["listed_in"])
df_raw["imdb_score"]       = df_raw["imdb_score"].fillna(df_raw["imdb_score"].median())
df_raw["imdb_votes"]       = df_raw["imdb_votes"].fillna(0)
df_raw["country"]          = df_raw["country"].fillna("unknown")
df_raw["age_certification"]= df_raw["age_certification"].fillna("unknown")
df_raw["duration_num"]     = df_raw["duration"].str.extract(r"(\d+)").astype(float)
df_raw.drop(columns=["duration"], inplace=True)
df_raw["country"]          = group_rare_countries(df_raw["country"], min_count=1)

# Preparar etiquetas
df_raw["listed_in_list"] = (
    df_raw["listed_in"].str.lower().str.strip().str.split(",")
    .apply(lambda tags: [t.strip() for t in tags])
)
mlb_full = MultiLabelBinarizer()
Y_full   = mlb_full.fit_transform(df_raw["listed_in_list"])

# Género principal (para evaluación visual de clustering)
df_raw["main_genre"] = df_raw["listed_in_list"].apply(lambda x: x[0] if len(x) > 0 else "unknown")

X_full = df_raw.drop(columns=["listed_in", "listed_in_list", "main_genre"])
GENRES  = mlb_full.classes_
N_GENRES = len(GENRES)
print(f"\n✓ Dataset: {X_full.shape[0]} muestras | {N_GENRES} géneros únicos")
print(f"  Géneros: {list(GENRES[:8])} ...")

# ── Preprocesar TODO el dataset (fit_transform sobre X_full) ─────────────────
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_cols    = ["release_year", "imdb_score", "imdb_votes", "duration_num"]
categorical_cols= ["type_base", "rating", "age_certification", "country"]

preprocessor_full = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])
X_proc = preprocessor_full.fit_transform(X_full)
print(f"  Dimensiones tras OHE: {X_proc.shape}")

# ── PCA para clustering y visualización ──────────────────────────────────────
from sklearn.decomposition import PCA

pca_viz = PCA(n_components=2, random_state=42)
X_2d    = pca_viz.fit_transform(X_proc)

pca_clust = PCA(n_components=20, random_state=42)
X_20d     = pca_clust.fit_transform(X_proc)
print(f"  Varianza explicada 20 componentes: {pca_clust.explained_variance_ratio_.sum()*100:.1f}%")

main_genre_arr = df_raw["main_genre"].values
genre_list_sorted = sorted(df_raw["main_genre"].unique())
genre2idx = {g: i for i, g in enumerate(genre_list_sorted)}
genre_colors = plt.cm.tab20(np.linspace(0, 1, len(genre_list_sorted)))

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score
)

def evaluate_clustering(X, labels, name, true_labels=None):
    """Calcula métricas internas y externas del clustering."""
    unique = np.unique(labels[labels != -1])
    n_clusters = len(unique)
    noise = (labels == -1).sum()

    metrics = {"Algoritmo": name, "n_clusters": n_clusters, "n_ruido": noise}

    if n_clusters >= 2:
        mask = labels != -1
        metrics["Silhouette"]  = round(silhouette_score(X[mask], labels[mask]), 4)
        metrics["DaviesBouldin"]= round(davies_bouldin_score(X[mask], labels[mask]), 4)
        metrics["CalinskiHarabasz"] = round(calinski_harabasz_score(X[mask], labels[mask]), 4)
    else:
        metrics["Silhouette"] = metrics["DaviesBouldin"] = metrics["CalinskiHarabasz"] = np.nan

    if true_labels is not None and n_clusters >= 2:
        true_clean = true_labels[labels != -1] if noise > 0 else true_labels
        lbl_clean  = labels[labels != -1]      if noise > 0 else labels
        metrics["ARI"] = round(adjusted_rand_score(true_clean, lbl_clean), 4)
    else:
        metrics["ARI"] = np.nan

    print(f"  [{name}] clusters={n_clusters} | ruido={noise} | "
          f"Silhouette={metrics['Silhouette']} | DB={metrics['DaviesBouldin']} | ARI={metrics['ARI']}")
    return metrics


def save_cluster_plot(X_2d, labels, title, fname, true_labels=None, subtitle=""):
    fig, axes = plt.subplots(1, 2 if true_labels is not None else 1,
                              figsize=(14 if true_labels is not None else 7, 5))
    if true_labels is None:
        axes = [axes]

    # Clusters asignados
    unique_lbl = np.unique(labels)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(unique_lbl), 1)))
    for i, lbl in enumerate(unique_lbl):
        mask = labels == lbl
        color = "gray" if lbl == -1 else cmap[i % len(cmap)]
        label_str = "Ruido" if lbl == -1 else f"Cluster {lbl}"
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[color], s=8, alpha=0.6, label=label_str)
    axes[0].set_title(f"{title}\n{subtitle}", fontsize=10)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    if len(unique_lbl) <= 15:
        axes[0].legend(markerscale=2, fontsize=7, loc="best")

    # Géneros reales (comparación)
    if true_labels is not None:
        for gi, g in enumerate(genre_list_sorted):
            mask = true_labels == g
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                            c=[genre_colors[gi]], s=8, alpha=0.5, label=g)
        axes[1].set_title("Géneros Reales (referencia)", fontsize=10)
        axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
        if len(genre_list_sorted) <= 20:
            axes[1].legend(markerscale=2, fontsize=6, loc="best",
                           ncol=2, bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → Gráfica guardada: {OUT}/{fname}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 1 — K-MEANS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 1 — K-MEANS CLUSTERING")
print("=" * 65)

from sklearn.cluster import KMeans

# Elbow + Silhouette para seleccionar K
K_range = range(2, 21)
inertias, sil_scores = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_20d)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_20d, lbl))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_range), inertias, "o-", color="#E50914")
ax1.set_xlabel("Número de Clusters (K)"); ax1.set_ylabel("Inercia")
ax1.set_title("K-Means — Método del Codo"); ax1.grid(alpha=0.3)
ax2.plot(list(K_range), sil_scores, "o-", color="#221F1F")
ax2.set_xlabel("Número de Clusters (K)"); ax2.set_ylabel("Silhouette Score")
ax2.set_title("K-Means — Silhouette por K"); ax2.grid(alpha=0.3)
plt.suptitle("Selección de K óptimo", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_kmeans_elbow.png"), dpi=150, bbox_inches="tight")
plt.close()

best_k = list(K_range)[np.argmax(sil_scores)]
print(f"  K óptimo (max Silhouette): {best_k}")

km_best = KMeans(n_clusters=best_k, random_state=42, n_init=20)
km_labels = km_best.fit_predict(X_20d)
km_metrics = evaluate_clustering(X_20d, km_labels, "K-Means", main_genre_arr)
save_cluster_plot(X_2d, km_labels, f"K-Means (K={best_k})",
                  "02_kmeans_clusters.png", main_genre_arr,
                  f"Silhouette={km_metrics['Silhouette']}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 2 — FUZZY C-MEANS (implementación propia)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 2 — FUZZY C-MEANS")
print("=" * 65)


def fuzzy_cmeans(X, c=10, m=2.0, max_iter=150, tol=1e-4, random_state=42):
    """
    Fuzzy C-Means (Bezdek, 1984).
    Parámetros:
      c   — número de clusters
      m   — exponente de difuminación (m > 1; típico = 2)
    Retorna:
      U   — matriz de pertenencia (n_samples × c)
      centers — centroides (c × n_features)
    """
    rng = np.random.default_rng(random_state)
    n   = X.shape[0]
    # Inicialización aleatoria de U (filas suman 1)
    U   = rng.random((n, c))
    U   = U / U.sum(axis=1, keepdims=True)

    for it in range(max_iter):
        U_old = U.copy()
        # Centroides
        um    = U ** m                              # (n, c)
        centers = (um.T @ X) / um.sum(axis=0)[:, None]  # (c, d)
        # Distancias cuadradas (n, c)
        diff  = X[:, None, :] - centers[None, :, :]     # (n, c, d)
        dist2 = (diff ** 2).sum(axis=2)                  # (n, c)
        dist2 = np.maximum(dist2, 1e-10)
        # Actualizar U
        exp   = 2 / (m - 1)
        # ratio[i,j,k] = dist2[i,j] / dist2[i,k]
        ratio = dist2[:, :, None] / dist2[:, None, :]   # (n, c, c)
        U     = 1.0 / ratio.sum(axis=2) ** (1 / (m - 1) * (m - 1))
        # Corrección numérica para la fórmula estándar
        inv   = (dist2[:, :, None] / dist2[:, None, :]) ** (1 / (m - 1))
        U     = 1.0 / inv.sum(axis=2)
        U     = U / U.sum(axis=1, keepdims=True)

        if np.linalg.norm(U - U_old) < tol:
            print(f"    Convergió en iteración {it+1}")
            break

    hard_labels = U.argmax(axis=1)
    return U, centers, hard_labels


C_FCM = best_k
print(f"  Entrenando Fuzzy C-Means con c={C_FCM}, m=2 ...")
U_fcm, centers_fcm, fcm_labels = fuzzy_cmeans(X_20d, c=C_FCM, m=2.0,
                                               max_iter=200, random_state=42)

fcm_metrics = evaluate_clustering(X_20d, fcm_labels, "Fuzzy C-Means", main_genre_arr)
save_cluster_plot(X_2d, fcm_labels, f"Fuzzy C-Means (c={C_FCM})",
                  "03_fcm_clusters.png", main_genre_arr,
                  f"Silhouette={fcm_metrics['Silhouette']}")

# Gráfica de membresía (top 3 clusters por muestra)
max_membership = U_fcm.max(axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(max_membership, bins=40, color="#E50914", alpha=0.8, edgecolor="white")
ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Umbral 0.5")
ax.set_xlabel("Membresía Máxima por Muestra")
ax.set_ylabel("Frecuencia")
ax.set_title("Fuzzy C-Means — Distribución de Membresía Máxima")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_fcm_membership.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 — SUBTRACTIVE CLUSTERING (implementación propia)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 3 — SUBTRACTIVE CLUSTERING")
print("=" * 65)


def subtractive_clustering(X, r_a=0.5, r_b_ratio=1.5, eps_high=0.5,
                            eps_low=0.15, max_centers=30):
    """
    Subtractive Clustering (Chiu, 1994).
    Usa distancias normalizadas. r_a = radio de vecindad principal.
    Retorna:
      centers_idx — índices de los centros encontrados
      labels      — asignación hard de cada punto al centro más cercano
    """
    n, d    = X.shape
    r_b     = r_a * r_b_ratio
    # Normalizar X a [0,1] por columna para que r_a sea comparable
    Xn      = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)

    # Calcular potencial inicial de cada punto
    potential = np.zeros(n)
    for i in range(n):
        diff2  = ((Xn - Xn[i]) ** 2).sum(axis=1)
        potential[i] = np.exp(-4 * diff2 / r_a**2).sum()

    centers_idx = []
    p_first     = None

    for _ in range(max_centers):
        best_i  = np.argmax(potential)
        best_p  = potential[best_i]

        if p_first is None:
            p_first = best_p

        ratio = best_p / p_first

        if ratio > eps_high:
            # Aceptar centro
            centers_idx.append(best_i)
        elif ratio < eps_low:
            break
        else:
            # Prueba montaña: distancia al centro más cercano
            if len(centers_idx) > 0:
                dists = np.array([((Xn[best_i] - Xn[c]) ** 2).sum() ** 0.5
                                  for c in centers_idx])
                d_min = dists.min()
                if ratio + d_min / r_a >= 1.0:
                    centers_idx.append(best_i)
                else:
                    potential[best_i] = 0
                    continue
            else:
                centers_idx.append(best_i)

        # Reducir potencial de los vecinos del nuevo centro
        c_new = centers_idx[-1]
        diff2 = ((Xn - Xn[c_new]) ** 2).sum(axis=1)
        potential -= best_p * np.exp(-4 * diff2 / r_b**2)
        potential = np.maximum(potential, 0)

    if len(centers_idx) == 0:
        centers_idx = [0]

    # Asignación: cada punto al centro más cercano
    centers_X = Xn[centers_idx]
    diff      = Xn[:, None, :] - centers_X[None, :, :]   # (n, c, d)
    dists     = (diff ** 2).sum(axis=2)
    labels    = dists.argmin(axis=1)

    print(f"    Centros encontrados: {len(centers_idx)}")
    return np.array(centers_idx), labels


# Subtractive sobre PCA 10D (menos costoso computacionalmente)
pca_sub = PCA(n_components=10, random_state=42)
X_10d   = pca_sub.fit_transform(X_proc)

sub_centers, sub_labels = subtractive_clustering(X_10d, r_a=0.45, max_centers=25)
sub_metrics = evaluate_clustering(X_10d, sub_labels, "Subtractive", main_genre_arr)
save_cluster_plot(X_2d, sub_labels, f"Subtractive Clustering",
                  "05_subtractive_clusters.png", main_genre_arr,
                  f"Silhouette={sub_metrics['Silhouette']}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 — DBSCAN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 4 — DBSCAN")
print("=" * 65)

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# k-dist para seleccionar eps
nbrs = NearestNeighbors(n_neighbors=5).fit(X_20d)
dists, _ = nbrs.kneighbors(X_20d)
k_dists  = np.sort(dists[:, -1])[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_dists, color="#221F1F", lw=1.5)
ax.set_xlabel("Puntos (ordenados)"); ax.set_ylabel("5-distancia")
ax.set_title("DBSCAN — k-dist para selección de eps")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_dbscan_kdist.png"), dpi=150, bbox_inches="tight")
plt.close()

# Buscar eps óptimo
best_sil_db, best_eps, best_min = -1, 1.5, 5
for eps in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
    for min_s in [3, 5, 8]:
        db = DBSCAN(eps=eps, min_samples=min_s)
        lbl = db.fit_predict(X_20d)
        nc = len(set(lbl) - {-1})
        if nc >= 2:
            mask = lbl != -1
            if mask.sum() >= 50:
                s = silhouette_score(X_20d[mask], lbl[mask])
                if s > best_sil_db:
                    best_sil_db, best_eps, best_min = s, eps, min_s

print(f"  DBSCAN óptimo: eps={best_eps}, min_samples={best_min}")
db_best   = DBSCAN(eps=best_eps, min_samples=best_min)
db_labels = db_best.fit_predict(X_20d)
db_metrics = evaluate_clustering(X_20d, db_labels, "DBSCAN", main_genre_arr)
save_cluster_plot(X_2d, db_labels,
                  f"DBSCAN (eps={best_eps}, min={best_min})",
                  "07_dbscan_clusters.png", main_genre_arr,
                  f"Silhouette={db_metrics['Silhouette']}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 5 — FAMILIA CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 5 — FAMILIA CLUSTER (Agglomerative, GMM, Mean-Shift)")
print("=" * 65)

from sklearn.cluster import AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture

# 5a. Agglomerative Hierarchical
print("  → Agglomerative Clustering ...")
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
agg_labels  = agg.fit_predict(X_20d)
agg_metrics = evaluate_clustering(X_20d, agg_labels, "Agglomerative", main_genre_arr)
save_cluster_plot(X_2d, agg_labels, f"Agglomerative (k={best_k}, Ward)",
                  "08_agglomerative_clusters.png", main_genre_arr,
                  f"Silhouette={agg_metrics['Silhouette']}")

# Dendrogram (submuestra)
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
sample_idx = np.random.choice(len(X_20d), min(150, len(X_20d)), replace=False)
Z = scipy_linkage(X_20d[sample_idx], method="ward")
fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, ax=ax, color_threshold=0.7*max(Z[:,2]),
           labels=None, no_labels=True, leaf_rotation=90)
ax.set_title("Dendrograma Jerárquico (muestra 150 pts)", fontsize=11)
ax.set_ylabel("Distancia Ward")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_dendrogram.png"), dpi=150, bbox_inches="tight")
plt.close()

# 5b. Gaussian Mixture Model
print("  → Gaussian Mixture Model ...")
# Seleccionar n_components por BIC
bic_scores = []
gm_range   = range(2, 16)
for nc in gm_range:
    gm = GaussianMixture(n_components=nc, random_state=42, covariance_type="diag")
    gm.fit(X_20d)
    bic_scores.append(gm.bic(X_20d))

best_gmm_k = list(gm_range)[np.argmin(bic_scores)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(gm_range), bic_scores, "o-", color="#E50914")
ax.axvline(best_gmm_k, color="black", ls="--", lw=1.5, label=f"Óptimo k={best_gmm_k}")
ax.set_xlabel("n_components"); ax.set_ylabel("BIC")
ax.set_title("GMM — Selección de Componentes por BIC")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_gmm_bic.png"), dpi=150, bbox_inches="tight")
plt.close()

gmm_best   = GaussianMixture(n_components=best_gmm_k, random_state=42, covariance_type="diag")
gmm_labels = gmm_best.fit_predict(X_20d)
gmm_metrics= evaluate_clustering(X_20d, gmm_labels, "GMM", main_genre_arr)
save_cluster_plot(X_2d, gmm_labels, f"Gaussian Mixture (k={best_gmm_k})",
                  "11_gmm_clusters.png", main_genre_arr,
                  f"Silhouette={gmm_metrics['Silhouette']}")

# 5c. Mean-Shift
print("  → Mean-Shift (puede tardar) ...")
bw = estimate_bandwidth(X_20d, quantile=0.15, n_samples=500, random_state=42)
ms = MeanShift(bandwidth=bw, bin_seeding=True)
ms_labels  = ms.fit_predict(X_20d)
ms_metrics = evaluate_clustering(X_20d, ms_labels, "Mean-Shift", main_genre_arr)
save_cluster_plot(X_2d, ms_labels, f"Mean-Shift (bw={bw:.2f})",
                  "12_meanshift_clusters.png", main_genre_arr,
                  f"Silhouette={ms_metrics['Silhouette']}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 6 — COMPARATIVA CLUSTERING (tabla + radar)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 6 — COMPARATIVA CLUSTERING")
print("=" * 65)

all_metrics = [km_metrics, fcm_metrics, sub_metrics, db_metrics,
               agg_metrics, gmm_metrics, ms_metrics]
df_metrics  = pd.DataFrame(all_metrics)
df_metrics.to_csv(os.path.join(OUT, "cluster_metrics.csv"), index=False)
print("\n", df_metrics.to_string(index=False))

# Barras comparativas
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_to_plot = ["Silhouette", "DaviesBouldin", "CalinskiHarabasz"]
titles          = ["Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"]
colors          = ["#E50914", "#221F1F", "#B20710"]

for ax, col, title, color in zip(axes, metrics_to_plot, titles, colors):
    vals = df_metrics[col].fillna(0)
    bars = ax.bar(df_metrics["Algoritmo"], vals, color=color, alpha=0.85, edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        if v != 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

plt.suptitle("Comparativa de Algoritmos de Clustering", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "13_cluster_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 7 — REEVALUACIÓN DE ETIQUETAS (~30% ruido)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 7 — REEVALUACIÓN DE ETIQUETAS")
print("=" * 65)

"""
ESTRATEGIA:
  1. Usar el mejor cluster (K-Means o Agglomerative) como "señal no supervisada"
  2. Para cada cluster, identificar el género más frecuente (género dominante)
  3. Si la etiqueta principal de una muestra ≠ género dominante de su cluster
     → marcar como "posiblemente mal etiquetada"
  4. Para esas muestras, proponer el género dominante del cluster como
     etiqueta corregida (solo si la confianza es alta ≥ 0.5 Silhouette-like)
  5. Comparar rendimiento supervisado: etiquetas originales vs corregidas
"""

# Usar el clustering con mejor Silhouette
best_algo = df_metrics.loc[df_metrics["Silhouette"].idxmax(), "Algoritmo"]
print(f"  Mejor clustering: {best_algo}")

algo_map = {
    "K-Means": km_labels, "Fuzzy C-Means": fcm_labels,
    "Subtractive": sub_labels, "DBSCAN": db_labels,
    "Agglomerative": agg_labels, "GMM": gmm_labels, "Mean-Shift": ms_labels
}
best_labels = algo_map[best_algo]

# Género dominante por cluster
df_eval = df_raw.copy()
df_eval["cluster"] = best_labels
df_eval["main_genre_original"] = df_raw["main_genre"]

cluster_dominant = (
    df_eval[df_eval["cluster"] != -1]
    .groupby("cluster")["main_genre_original"]
    .agg(lambda x: x.value_counts().index[0])
    .to_dict()
)

df_eval["cluster_genre"] = df_eval["cluster"].map(cluster_dominant)
df_eval["mislabeled"]    = (
    (df_eval["cluster"] != -1) &
    (df_eval["main_genre_original"] != df_eval["cluster_genre"])
)

n_mislabeled = df_eval["mislabeled"].sum()
pct          = n_mislabeled / len(df_eval) * 100
print(f"  Muestras potencialmente mal etiquetadas: {n_mislabeled} ({pct:.1f}%)")

# Corrección: reemplazar género principal por el dominante del cluster
df_eval["main_genre_corrected"] = df_eval.apply(
    lambda row: row["cluster_genre"]
    if (row["mislabeled"] and row["cluster_genre"] is not None
        and pd.notna(row["cluster_genre"]))
    else row["main_genre_original"],
    axis=1
)

# Guardar reporte
relabeling_report = df_eval[df_eval["mislabeled"]][[
    "main_genre_original", "cluster", "cluster_genre", "main_genre_corrected"
]].copy()
relabeling_report.to_csv(os.path.join(OUT, "relabeling_report.csv"), index=False)
print(f"  Reporte guardado: {OUT}/relabeling_report.csv")

# Visualización: comparación original vs corregido
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, col, title in zip(
    axes,
    ["main_genre_original", "main_genre_corrected"],
    ["Distribución Original", "Distribución Corregida"]
):
    vc = df_eval[col].value_counts().head(20)
    ax.barh(vc.index[::-1], vc.values[::-1], color="#E50914", alpha=0.85, edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Conteo"); ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)

plt.suptitle(f"Reevaluación de Etiquetas — {n_mislabeled} muestras corregidas ({pct:.1f}%)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "14_relabeling_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# Heatmap: confusión original vs corregido (top 15 géneros)
top15 = df_eval["main_genre_original"].value_counts().head(15).index.tolist()
mask15 = df_eval["main_genre_original"].isin(top15) & df_eval["main_genre_corrected"].isin(top15)
confusion_relabel = pd.crosstab(
    df_eval.loc[mask15, "main_genre_original"],
    df_eval.loc[mask15, "main_genre_corrected"],
    rownames=["Original"], colnames=["Corregido"]
)
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(confusion_relabel, annot=True, fmt="d", cmap="Reds",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Conteo"})
ax.set_title("Matriz de Reetiquetado (top 15 géneros)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "15_relabeling_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 8 — MODELOS SUPERVISADOS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 8 — MODELOS SUPERVISADOS")
print("=" * 65)

"""
Entrenamos 3 tipos de modelos en dos escenarios:
  A) Etiquetas ORIGINALES (main_genre)
  B) Etiquetas CORREGIDAS por clustering

Modelos:
  1. Árbol de Decisión
  2. Regresión Logística (multiclase, one-vs-rest)
  3. Regresión Ridge (como clasificador vía umbralización)

Métrica principal: Accuracy + F1 macro (por ser multiclase)
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

# Re-preparar X features (mismas que en el pipeline pero completas)
X_feat = df_eval.drop(columns=[
    "listed_in", "listed_in_list", "main_genre",
    "cluster", "cluster_genre", "mislabeled",
    "main_genre_original", "main_genre_corrected"
], errors="ignore")

numeric_cols2 = ["release_year", "imdb_score", "imdb_votes", "duration_num"]
categorical_cols2 = ["type_base", "rating", "age_certification", "country"]

prep2 = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols2),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols2),
], remainder="drop")

# Encoders para las etiquetas
le_orig = LabelEncoder()
le_corr = LabelEncoder()

y_orig = le_orig.fit_transform(df_eval["main_genre_original"])
y_corr = le_corr.fit_transform(df_eval["main_genre_corrected"])

# Filtrar clases con menos de 2 muestras para poder estratificar
def filter_rare_classes(df_X, y_series, min_count=2):
    counts = y_series.value_counts()
    valid  = counts[counts >= min_count].index
    mask   = y_series.isin(valid)
    return df_X[mask].reset_index(drop=True), y_series[mask].reset_index(drop=True)

df_eval_reset = df_eval.reset_index(drop=True)
X_feat_df = df_eval_reset.drop(columns=[
    "listed_in", "listed_in_list", "main_genre",
    "cluster", "cluster_genre", "mislabeled",
    "main_genre_original", "main_genre_corrected"
], errors="ignore")

y_orig_s = df_eval_reset["main_genre_original"]
y_corr_s = df_eval_reset["main_genre_corrected"]

le_orig = LabelEncoder(); le_corr = LabelEncoder()

X_fo, y_fo_s = filter_rare_classes(X_feat_df, y_orig_s)
X_fc, y_fc_s = filter_rare_classes(X_feat_df, y_corr_s)

y_fo = le_orig.fit_transform(y_fo_s)
y_fc = le_corr.fit_transform(y_fc_s)

X_tr_o, X_te_o, y_tr_o, y_te_o = train_test_split(
    X_fo, y_fo, test_size=0.25, random_state=42, stratify=y_fo)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_fc, y_fc, test_size=0.25, random_state=42, stratify=y_fc)

models = {
    "Árbol de Decisión": DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=5, random_state=42),
    "Regresión Logística": LogisticRegression(
        max_iter=500, C=1.0, solver="lbfgs", random_state=42),
    "Ridge Classifier": RidgeClassifier(alpha=1.0),
}

results = []

for label_type, (X_tr, X_te, y_tr, y_te, le) in {
    "Original": (X_tr_o, X_te_o, y_tr_o, y_te_o, le_orig),
    "Corregido": (X_tr_c, X_te_c, y_tr_c, y_te_c, le_corr),
}.items():
    print(f"\n  ── Etiquetas {label_type} ──")
    for name, clf in models.items():
        pipe = SKPipeline([("prep", prep2), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc  = accuracy_score(y_te, y_pred)
        f1m  = f1_score(y_te, y_pred, average="macro", zero_division=0)
        f1w  = f1_score(y_te, y_pred, average="weighted", zero_division=0)

        results.append({
            "Etiquetas": label_type, "Modelo": name,
            "Accuracy": round(acc, 4),
            "F1 Macro": round(f1m, 4),
            "F1 Weighted": round(f1w, 4),
        })
        print(f"    {name:25s} | Acc={acc:.4f} | F1-macro={f1m:.4f} | F1-w={f1w:.4f}")

        # Matriz de confusión (top 10 géneros)
        class_names = le.classes_
        top10_enc   = [le.transform([g])[0] for g in
                       pd.Series(le.inverse_transform(y_te)).value_counts().head(10).index
                       if g in le.classes_]
        mask_top = np.isin(y_te, top10_enc) & np.isin(y_pred, top10_enc)
        if mask_top.sum() > 0:
            cm = confusion_matrix(y_te[mask_top], y_pred[mask_top], labels=top10_enc)
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Reds", ax=ax,
                        xticklabels=[class_names[i] for i in top10_enc],
                        yticklabels=[class_names[i] for i in top10_enc],
                        linewidths=0.5)
            ax.set_title(f"{name} — Etiquetas {label_type}\nMatriz de Confusión (top 10)",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
            plt.xticks(rotation=40, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            fname = f"cm_{label_type[:4]}_{name[:4].replace(' ', '_')}.png"
            plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
            plt.close()

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUT, "supervised_results.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 9 — COMPARATIVA FINAL (Original vs Corregido)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 9 — COMPARATIVA FINAL")
print("=" * 65)
print("\n", df_results.to_string(index=False))

# Gráfica doble barra: Original vs Corregido por modelo
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics_show = ["Accuracy", "F1 Macro", "F1 Weighted"]
x = np.arange(len(models))
width = 0.35

for ax, metric in zip(axes, metrics_show):
    vals_orig = df_results[df_results["Etiquetas"] == "Original"][metric].values
    vals_corr = df_results[df_results["Etiquetas"] == "Corregido"][metric].values
    b1 = ax.bar(x - width/2, vals_orig, width, label="Original", color="#221F1F", alpha=0.85)
    b2 = ax.bar(x + width/2, vals_corr, width, label="Corregido", color="#E50914", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(list(models.keys()), rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(metric); ax.set_title(metric, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color="#E50914")

plt.suptitle("Comparativa: Dataset Original vs Dataset con Etiquetas Corregidas",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "16_final_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# Tabla HTML de resumen
html_table = df_results.pivot_table(
    index="Modelo", columns="Etiquetas",
    values=["Accuracy", "F1 Macro", "F1 Weighted"]
).round(4).to_html()

with open(os.path.join(OUT, "summary_table.html"), "w") as f:
    f.write(f"<html><head><style>table{{border-collapse:collapse}} "
            f"td,th{{border:1px solid #ccc;padding:6px 12px;}} "
            f"th{{background:#E50914;color:white;}}</style></head><body>"
            f"<h2>Resultados Supervisados — Original vs Corregido</h2>{html_table}</body></html>")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 10 — FEATURE IMPORTANCE (Árbol de Decisión)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SECCIÓN 10 — FEATURE IMPORTANCE")
print("=" * 65)

# Re-entrenar árbol sobre todo el dataset original para importancias
prep_fi  = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols2),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols2),
], remainder="drop")
X_fi = prep_fi.fit_transform(X_feat)
dt_fi = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42)
dt_fi.fit(X_fi, y_orig)

# Nombres de features
cat_names  = prep_fi.named_transformers_["cat"].get_feature_names_out(categorical_cols2)
feat_names = numeric_cols2 + list(cat_names)
importances = dt_fi.feature_importances_
top_idx     = np.argsort(importances)[::-1][:25]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh([feat_names[i] for i in top_idx[::-1]],
        [importances[i] for i in top_idx[::-1]],
        color="#E50914", alpha=0.85, edgecolor="white")
ax.set_xlabel("Importancia (Gini)"); ax.set_title("Top 25 Features — Árbol de Decisión", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "17_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
#  RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ✓ ANÁLISIS COMPLETADO")
print("=" * 65)
print(f"\n  Gráficas generadas en: ./{OUT}/")
files = sorted(os.listdir(OUT))
for f in files:
    print(f"    • {f}")
print()
