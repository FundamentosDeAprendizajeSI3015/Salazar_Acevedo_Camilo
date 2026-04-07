"""
Regenera las gráficas de clusters usando t-SNE para mejor separación visual.
Ejecutar DESPUÉS de analysis.py (reutiliza los mismos datos y labels).
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = "eda_outputs"
os.makedirs(OUT, exist_ok=True)

# ─── Estilo general de gráficas ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#ddd",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#eee",
    "grid.color":       "#333",
    "grid.linewidth":   0.5,
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#555",
})

# ─── 1. Reconstruir datos (igual que en analysis.py) ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load_and_prepare, group_rare_countries
from sklearn.preprocessing import (
    MultiLabelBinarizer, StandardScaler, OneHotEncoder, LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

CSV_PATH = "netflix_merged_intersection.csv"

print("Reconstruyendo dataset...")
df_raw = pd.read_csv(CSV_PATH)
df_raw = df_raw[df_raw["director"] != "Not Given"].copy()
df_raw = df_raw[[
    "type_base", "country", "release_year", "rating",
    "duration", "listed_in", "imdb_score", "imdb_votes", "age_certification"
]].copy()
df_raw = df_raw.dropna(subset=["listed_in"])
df_raw["imdb_score"]        = df_raw["imdb_score"].fillna(df_raw["imdb_score"].median())
df_raw["imdb_votes"]        = df_raw["imdb_votes"].fillna(0)
df_raw["country"]           = df_raw["country"].fillna("unknown")
df_raw["age_certification"] = df_raw["age_certification"].fillna("unknown")
df_raw["duration_num"]      = df_raw["duration"].str.extract(r"(\d+)").astype(float)
df_raw.drop(columns=["duration"], inplace=True)
df_raw["country"]           = group_rare_countries(df_raw["country"], min_count=1)
df_raw["listed_in_list"]    = (
    df_raw["listed_in"].str.lower().str.strip().str.split(",")
    .apply(lambda tags: [t.strip() for t in tags])
)
df_raw["main_genre"] = df_raw["listed_in_list"].apply(
    lambda x: x[0] if len(x) > 0 else "unknown"
)

numeric_cols     = ["release_year", "imdb_score", "imdb_votes", "duration_num"]
categorical_cols = ["type_base", "rating", "age_certification", "country"]

X_full = df_raw.drop(columns=["listed_in", "listed_in_list", "main_genre"])
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])
X_proc = preprocessor.fit_transform(X_full)

# ─── 2. Reducción: PCA → 50 dims → t-SNE → 2 dims ───────────────────────────
print("Calculando PCA(50) → t-SNE(2)... (puede tardar ~1 min)")
# Primero PCA a 50 componentes para acelerar t-SNE y reducir ruido
pca50 = PCA(n_components=50, random_state=42)
X_50d = pca50.fit_transform(X_proc)
print(f"  Varianza explicada PCA-50: {pca50.explained_variance_ratio_.sum()*100:.1f}%")

# t-SNE con parámetros pensados para buena separación
tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate="auto",
    max_iter=1200,
    metric="euclidean",
    init="pca",
    random_state=42,
    n_jobs=-1,
)
X_tsne = tsne.fit_transform(X_50d)
print(f"  t-SNE listo. Rango X: [{X_tsne[:,0].min():.1f}, {X_tsne[:,0].max():.1f}] "
      f"Y: [{X_tsne[:,1].min():.1f}, {X_tsne[:,1].max():.1f}]")

# ─── 3. Paleta de géneros ─────────────────────────────────────────────────────
main_genre_arr    = df_raw["main_genre"].values
genres_sorted     = sorted(df_raw["main_genre"].unique())
N_GENRES          = len(genres_sorted)
# Paleta con suficiente contraste para 39 géneros
tab20  = plt.cm.tab20(np.linspace(0, 1, 20))
tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
all_colors = np.vstack([tab20, tab20b])
genre_palette = {g: all_colors[i % len(all_colors)] for i, g in enumerate(genres_sorted)}

# ─── 4. Helper para guardar plots ─────────────────────────────────────────────
def save_tsne_plot(labels, title, fname, show_genre_ref=True,
                   noise_label=-1, subtitle=""):
    """
    Genera un plot t-SNE de los clusters asignados.
    Si show_genre_ref=True, agrega un segundo panel con los géneros reales.
    """
    ncols = 2 if show_genre_ref else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols + 1, 6.5),
                              facecolor="#0f0f0f")

    unique_lbl = sorted(set(labels))
    n_clusters = len([l for l in unique_lbl if l != noise_label])

    # Paleta de clusters
    cmap_c = plt.cm.gist_ncar(np.linspace(0.05, 0.95, max(n_clusters, 1)))
    cluster_colors = {}
    ci = 0
    for lbl in unique_lbl:
        if lbl == noise_label:
            cluster_colors[lbl] = np.array([0.4, 0.4, 0.4, 0.5])
        else:
            cluster_colors[lbl] = cmap_c[ci % len(cmap_c)]
            ci += 1

    # Panel izquierdo: clusters
    ax = axes[0] if show_genre_ref else axes
    for lbl in unique_lbl:
        mask = labels == lbl
        color = cluster_colors[lbl]
        lbl_str = "Ruido" if lbl == noise_label else f"Cluster {lbl}"
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[color], s=12, alpha=0.75, linewidths=0,
                   label=lbl_str, rasterized=True)

    # Centroides anotados
    for lbl in unique_lbl:
        if lbl == noise_label:
            continue
        mask = labels == lbl
        cx, cy = X_tsne[mask, 0].mean(), X_tsne[mask, 1].mean()
        ax.text(cx, cy, str(lbl), fontsize=9, fontweight="bold", color="white",
                ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold", color="#eee")
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.grid(alpha=0.2)
    if n_clusters <= 20:
        ax.legend(markerscale=2, fontsize=7.5, loc="lower right",
                  framealpha=0.6, ncol=2)

    # Panel derecho: géneros reales
    if show_genre_ref:
        ax2 = axes[1]
        # Ordenar por frecuencia para que los más comunes estén encima
        freq_order = df_raw["main_genre"].value_counts().index.tolist()
        for g in reversed(freq_order):
            mask = main_genre_arr == g
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        c=[genre_palette[g]], s=9, alpha=0.70, linewidths=0,
                        label=g, rasterized=True)
        ax2.set_title("Géneros Reales (referencia)", fontsize=11,
                      fontweight="bold", color="#eee")
        ax2.set_xlabel("t-SNE dim 1", fontsize=9)
        ax2.set_ylabel("t-SNE dim 2", fontsize=9)
        ax2.grid(alpha=0.2)
        # Leyenda fuera del plot (demasiados géneros)
        ax2.legend(markerscale=2.5, fontsize=6.5, ncol=2,
                   bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.7)

    plt.tight_layout()
    path = os.path.join(OUT, fname)
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print(f"  → {path}")


# ─── 5. Reproducir todos los clusterings ─────────────────────────────────────
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

pca20 = PCA(n_components=20, random_state=42)
X_20d = pca20.fit_transform(X_proc)

pca10 = PCA(n_components=10, random_state=42)
X_10d = pca10.fit_transform(X_proc)

print("\nRecalculando clusters...")

# ── K-Means (K óptimo por silhouette) ────────────────────────────────────────
print("  K-Means...")
sil_scores = [silhouette_score(X_20d,
               KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_20d))
               for k in range(2, 16)]
best_k = range(2, 16)[np.argmax(sil_scores)]
km_labels = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit_predict(X_20d)
s = silhouette_score(X_20d, km_labels)
save_tsne_plot(km_labels, f"K-Means  (K={best_k})",
               "tsne_02_kmeans.png", subtitle=f"Silhouette={s:.4f}")

# ── Fuzzy C-Means ─────────────────────────────────────────────────────────────
print("  Fuzzy C-Means...")
def fuzzy_cmeans(X, c=10, m=2.0, max_iter=150, tol=1e-4, random_state=42):
    rng = np.random.default_rng(random_state)
    n   = X.shape[0]
    U   = rng.random((n, c))
    U  /= U.sum(axis=1, keepdims=True)
    for it in range(max_iter):
        U_old = U.copy()
        um    = U ** m
        centers = (um.T @ X) / um.sum(axis=0)[:, None]
        diff2   = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        diff2   = np.maximum(diff2, 1e-10)
        inv     = (diff2[:, :, None] / diff2[:, None, :]) ** (1 / (m - 1))
        U       = 1.0 / inv.sum(axis=2)
        U      /= U.sum(axis=1, keepdims=True)
        if np.linalg.norm(U - U_old) < tol:
            break
    return U, U.argmax(axis=1)

U_fcm, fcm_labels = fuzzy_cmeans(X_20d, c=best_k, m=2.0, max_iter=200)
s = silhouette_score(X_20d, fcm_labels)
save_tsne_plot(fcm_labels, f"Fuzzy C-Means  (c={best_k})",
               "tsne_03_fcm.png", subtitle=f"Silhouette={s:.4f}")

# Gráfica de membresía máxima sobre t-SNE
max_memb = U_fcm.max(axis=1)
fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f0f0f")
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=max_memb,
                cmap="RdYlGn", s=10, alpha=0.8, linewidths=0, vmin=0.3, vmax=1.0)
cb = plt.colorbar(sc, ax=ax)
cb.set_label("Membresía máxima", color="#eee")
cb.ax.yaxis.set_tick_params(color="#aaa"); cb.outline.set_edgecolor("#555")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aaa")
ax.set_title("Fuzzy C-Means — Grado de certeza por muestra", fontweight="bold")
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "tsne_04_fcm_membership.png"), dpi=160,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print(f"  → {OUT}/tsne_04_fcm_membership.png")

# ── Subtractive ───────────────────────────────────────────────────────────────
print("  Subtractive...")
def subtractive_clustering(X, r_a=0.5, r_b_ratio=1.5, eps_high=0.5,
                            eps_low=0.15, max_centers=30):
    n, d = X.shape
    r_b  = r_a * r_b_ratio
    Xn   = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
    potential = np.array([np.exp(-4 * ((Xn - Xn[i])**2).sum(axis=1) / r_a**2).sum()
                          for i in range(n)])
    centers_idx = []; p_first = None
    for _ in range(max_centers):
        best_i = np.argmax(potential); best_p = potential[best_i]
        if p_first is None: p_first = best_p
        ratio = best_p / p_first
        if ratio > eps_high:
            centers_idx.append(best_i)
        elif ratio < eps_low:
            break
        else:
            if len(centers_idx) > 0:
                d_min = min(((Xn[best_i] - Xn[c])**2).sum()**0.5 for c in centers_idx)
                if ratio + d_min / r_a >= 1.0: centers_idx.append(best_i)
                else: potential[best_i] = 0; continue
            else:
                centers_idx.append(best_i)
        c_new = centers_idx[-1]
        diff2 = ((Xn - Xn[c_new])**2).sum(axis=1)
        potential -= best_p * np.exp(-4 * diff2 / r_b**2)
        potential  = np.maximum(potential, 0)
    if not centers_idx: centers_idx = [0]
    centers_X = Xn[np.array(centers_idx)]
    dists = ((Xn[:, None, :] - centers_X[None, :, :])**2).sum(axis=2)
    return np.array(centers_idx), dists.argmin(axis=1)

sub_centers, sub_labels = subtractive_clustering(X_10d, r_a=0.45, max_centers=25)
n_sub = len(sub_centers)
s = silhouette_score(X_10d, sub_labels) if n_sub >= 2 else float("nan")
save_tsne_plot(sub_labels, f"Subtractive Clustering  ({n_sub} centros)",
               "tsne_05_subtractive.png", subtitle=f"Silhouette={s:.4f}")

# ── DBSCAN ────────────────────────────────────────────────────────────────────
print("  DBSCAN...")
best_sil, best_eps, best_min = -1, 1.5, 5
for eps in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
    for ms in [3, 5, 8]:
        lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_20d)
        nc  = len(set(lbl) - {-1})
        if nc >= 2:
            mask = lbl != -1
            if mask.sum() >= 50:
                s = silhouette_score(X_20d[mask], lbl[mask])
                if s > best_sil: best_sil, best_eps, best_min = s, eps, ms
db_labels = DBSCAN(eps=best_eps, min_samples=best_min).fit_predict(X_20d)
n_noise   = (db_labels == -1).sum()
save_tsne_plot(db_labels,
               f"DBSCAN  (eps={best_eps}, min={best_min})",
               "tsne_07_dbscan.png",
               subtitle=f"Silhouette={best_sil:.4f}  |  ruido={n_noise} pts")

# ── Agglomerative ─────────────────────────────────────────────────────────────
print("  Agglomerative...")
agg_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X_20d)
s = silhouette_score(X_20d, agg_labels)
save_tsne_plot(agg_labels, f"Agglomerative  (k={best_k}, Ward)",
               "tsne_08_agglomerative.png", subtitle=f"Silhouette={s:.4f}")

# ── GMM ───────────────────────────────────────────────────────────────────────
print("  GMM...")
bic_scores = [GaussianMixture(n_components=nc, random_state=42,
               covariance_type="diag").fit(X_20d).bic(X_20d)
               for nc in range(2, 16)]
best_gmm_k = range(2, 16)[np.argmin(bic_scores)]
gmm_labels = GaussianMixture(n_components=best_gmm_k, random_state=42,
               covariance_type="diag").fit_predict(X_20d)
s = silhouette_score(X_20d, gmm_labels)
save_tsne_plot(gmm_labels, f"GMM  (k={best_gmm_k})",
               "tsne_11_gmm.png", subtitle=f"Silhouette={s:.4f}")

# ── Mean-Shift ────────────────────────────────────────────────────────────────
print("  Mean-Shift...")
bw = estimate_bandwidth(X_20d, quantile=0.15, n_samples=500, random_state=42)
ms_labels = MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(X_20d)
s = silhouette_score(X_20d, ms_labels)
save_tsne_plot(ms_labels, f"Mean-Shift  (bw={bw:.2f})",
               "tsne_12_meanshift.png", subtitle=f"Silhouette={s:.4f}")

# ─── 6. Panel resumen: todos los algoritmos en una sola figura ────────────────
print("\nGenerando panel resumen...")
algo_labels = {
    "K-Means":         km_labels,
    "Fuzzy C-Means":   fcm_labels,
    "Subtractive":     sub_labels,
    "DBSCAN":          db_labels,
    "Agglomerative":   agg_labels,
    "GMM":             gmm_labels,
    "Mean-Shift":      ms_labels,
}

fig, axes = plt.subplots(2, 4, figsize=(26, 12), facecolor="#0f0f0f")
axes_flat = axes.flatten()

for idx, (name, labels) in enumerate(algo_labels.items()):
    ax = axes_flat[idx]
    unique_lbl = sorted(set(labels))
    n_cl = len([l for l in unique_lbl if l != -1])
    cmap_c = plt.cm.gist_ncar(np.linspace(0.05, 0.95, max(n_cl, 1)))
    ci = 0
    for lbl in unique_lbl:
        mask = labels == lbl
        if lbl == -1:
            color = np.array([0.35, 0.35, 0.35, 0.4])
        else:
            color = cmap_c[ci % len(cmap_c)]; ci += 1
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[color], s=7, alpha=0.75, linewidths=0, rasterized=True)

    # Centroides
    for lbl in unique_lbl:
        if lbl == -1: continue
        mask = labels == lbl
        cx, cy = X_tsne[mask, 0].mean(), X_tsne[mask, 1].mean()
        ax.text(cx, cy, str(lbl), fontsize=7.5, fontweight="bold", color="white",
                ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

    mask_valid = labels != -1
    if mask_valid.sum() >= 2 and n_cl >= 2:
        X_ev = X_20d if name != "Subtractive" else X_10d
        sil = silhouette_score(X_ev[mask_valid], labels[mask_valid])
        subtitle = f"k={n_cl}  Sil={sil:.3f}"
    else:
        subtitle = f"k={n_cl}"
    ax.set_title(f"{name}\n{subtitle}", fontsize=10, fontweight="bold", color="#eee")
    ax.set_xlabel("t-SNE 1", fontsize=8); ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.grid(alpha=0.15); ax.tick_params(labelsize=7)

# Último panel: géneros reales
ax_ref = axes_flat[7]
freq_order = df_raw["main_genre"].value_counts().index.tolist()
for g in reversed(freq_order):
    mask = main_genre_arr == g
    ax_ref.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[genre_palette[g]], s=7, alpha=0.72, linewidths=0,
                   label=g, rasterized=True)
ax_ref.set_title("Géneros Reales\n(referencia)", fontsize=10, fontweight="bold", color="#eee")
ax_ref.set_xlabel("t-SNE 1", fontsize=8); ax_ref.set_ylabel("t-SNE 2", fontsize=8)
ax_ref.grid(alpha=0.15); ax_ref.tick_params(labelsize=7)
ax_ref.legend(markerscale=2, fontsize=5.5, ncol=2,
              bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.6)

plt.suptitle("Comparativa de Algoritmos de Clustering — t-SNE",
             fontsize=14, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "tsne_00_panel_resumen.png"),
            dpi=160, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print(f"  → {OUT}/tsne_00_panel_resumen.png")

# ─── 7. Plot solo de géneros con t-SNE (para presentación) ───────────────────
fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0f0f0f")
freq_order = df_raw["main_genre"].value_counts().head(20).index.tolist()
# Solo top 20 géneros para legibilidad
mask_top20 = np.isin(main_genre_arr, freq_order)
for g in reversed(freq_order):
    mask = main_genre_arr == g
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=[genre_palette[g]], s=14, alpha=0.80, linewidths=0,
               label=g, rasterized=True)
# Anotar centroide de cada género
for g in freq_order:
    mask = main_genre_arr == g
    if mask.sum() < 5: continue
    cx, cy = X_tsne[mask, 0].mean(), X_tsne[mask, 1].mean()
    ax.text(cx, cy, g.replace(" ", "\n"), fontsize=6, color="white", ha="center",
            va="center", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

ax.set_title("Distribución de Géneros — t-SNE (top 20)", fontsize=13,
             fontweight="bold", color="#eee")
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.grid(alpha=0.15)
ax.legend(markerscale=2.5, fontsize=7.5, ncol=2,
          bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "tsne_genres_reference.png"),
            dpi=160, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print(f"  → {OUT}/tsne_genres_reference.png")

print("\n✓ Todas las gráficas t-SNE generadas.")
files_new = [f for f in os.listdir(OUT) if f.startswith("tsne_")]
for f in sorted(files_new):
    print(f"    • {f}")
