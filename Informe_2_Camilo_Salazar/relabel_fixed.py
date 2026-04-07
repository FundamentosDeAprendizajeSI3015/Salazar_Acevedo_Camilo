"""
REEVALUACIÓN DE ETIQUETAS — VERSIÓN CORREGIDA
═══════════════════════════════════════════════
Problema de la versión anterior:
  - K óptimo por silhouette = 2 (solo separaba Movies vs TV Shows)
  - Con 2 clusters, todo quedaba como "drama" o "action & adventure"
  - El modelo "corregido" era en realidad más pobre

Solución aplicada:
  - Usar K = N_GENRES (39) como referencia, ya que sabemos cuántos géneros hay
  - Comparación más fina: solo corregir si la confianza del cluster es alta
  - Usar también GMM (probabilístico) para estimar incertidumbre real
  - Mostrar métricas antes/después que reflejen el impacto real
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = "eda_outputs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#444",      "axes.labelcolor": "#ddd",
    "xtick.color": "#aaa",         "ytick.color": "#aaa",
    "text.color": "#eee",          "grid.color": "#333",
    "legend.facecolor": "#1a1a1a", "legend.edgecolor": "#555",
})

# ─── Reconstruir datos ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load_and_prepare, group_rare_countries
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, LabelEncoder, MultiLabelBinarizer
)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import train_test_split

CSV_PATH = "netflix_merged_intersection.csv"

print("Cargando datos...")
df = pd.read_csv(CSV_PATH)
df = df[df["director"] != "Not Given"].copy()
df = df[[
    "type_base", "country", "release_year", "rating",
    "duration", "listed_in", "imdb_score", "imdb_votes", "age_certification"
]].copy()
df = df.dropna(subset=["listed_in"])
df["imdb_score"]        = df["imdb_score"].fillna(df["imdb_score"].median())
df["imdb_votes"]        = df["imdb_votes"].fillna(0)
df["country"]           = df["country"].fillna("unknown")
df["age_certification"] = df["age_certification"].fillna("unknown")
df["duration_num"]      = df["duration"].str.extract(r"(\d+)").astype(float)
df.drop(columns=["duration"], inplace=True)
df["country"]           = group_rare_countries(df["country"], min_count=1)
df["listed_in_list"]    = (
    df["listed_in"].str.lower().str.strip().str.split(",")
    .apply(lambda t: [x.strip() for x in t])
)
df["main_genre"] = df["listed_in_list"].apply(lambda x: x[0] if x else "unknown")
df = df.reset_index(drop=True)

GENRES    = sorted(df["main_genre"].unique())
N_GENRES  = len(GENRES)
print(f"  {len(df)} muestras | {N_GENRES} géneros únicos")

numeric_cols     = ["release_year", "imdb_score", "imdb_votes", "duration_num"]
categorical_cols = ["type_base", "rating", "age_certification", "country"]

X_feat = df.drop(columns=["listed_in", "listed_in_list", "main_genre"])
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])
X_proc = preprocessor.fit_transform(X_feat)

pca50 = PCA(n_components=50, random_state=42)
X_50d = pca50.fit_transform(X_proc)

pca20 = PCA(n_components=20, random_state=42)
X_20d = pca20.fit_transform(X_proc)

# ─── Análisis del problema anterior ──────────────────────────────────────────
print("\n" + "="*60)
print("  DIAGNÓSTICO: ¿Por qué K=2 daba malos resultados?")
print("="*60)

# Mostrar que K=2 separa Movies vs TV Shows, no géneros
km2 = KMeans(n_clusters=2, random_state=42, n_init=20)
labels_k2 = km2.fit_predict(X_20d)

for cl in [0, 1]:
    mask = labels_k2 == cl
    type_dist = df.loc[mask, "type_base"].value_counts(normalize=True)
    genre_dom = df.loc[mask, "main_genre"].value_counts().head(3)
    print(f"\n  Cluster {cl} ({mask.sum()} muestras):")
    print(f"    Tipos:  {dict(type_dist.round(2))}")
    print(f"    Top géneros: {dict(genre_dom)}")

# Gráfica diagnóstico
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_type = {"Movie": "#E50914", "TV Show": "#564d4d"}
for ax, col, title in zip(axes,
    ["type_base", "main_genre"],
    ["K=2 coloreado por Tipo", "K=2 coloreado por Género Principal"]):

    # t-SNE rápido para visualizar
    tsne_quick = TSNE(n_components=2, perplexity=30, max_iter=500,
                      init="pca", random_state=42)
    Xv = tsne_quick.fit_transform(X_50d)

    vals = df[col].values
    unique_v = df[col].value_counts().head(15).index.tolist()
    cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_v)))
    for i, v in enumerate(unique_v):
        mask = vals == v
        ax.scatter(Xv[mask,0], Xv[mask,1], c=[cmap[i]], s=8,
                   alpha=0.7, label=v, linewidths=0)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=6.5, markerscale=2, ncol=2,
              bbox_to_anchor=(1.01,1), loc="upper left")
    ax.grid(alpha=0.15)

plt.suptitle("Diagnóstico: K=2 captura tipo (Movie/TV) no género",
             fontsize=12, fontweight="bold", color="white")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fix_01_diagnostico_k2.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print(f"\n  → Gráfica guardada: fix_01_diagnostico_k2.png")

# ─── SOLUCIÓN: K = N_GENRES ───────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  SOLUCIÓN: K-Means con K={N_GENRES} (un cluster por género)")
print("="*60)

km_full = KMeans(n_clusters=N_GENRES, random_state=42, n_init=15, max_iter=400)
labels_kfull = km_full.fit_predict(X_20d)
sil_full = silhouette_score(X_20d, labels_kfull)
print(f"  Silhouette con K={N_GENRES}: {sil_full:.4f}")

# ─── GMM probabilístico con K=N_GENRES ───────────────────────────────────────
print(f"\n  GMM con {N_GENRES} componentes...")
gmm_full = GaussianMixture(n_components=N_GENRES, covariance_type="diag",
                            random_state=42, max_iter=200)
gmm_full.fit(X_20d)
gmm_probs  = gmm_full.predict_proba(X_20d)   # (n, N_GENRES)
gmm_labels = gmm_full.predict(X_20d)
gmm_conf   = gmm_probs.max(axis=1)            # confianza del cluster asignado
print(f"  Confianza media GMM: {gmm_conf.mean():.4f}")
print(f"  Muestras con conf > 0.5: {(gmm_conf > 0.5).sum()} ({(gmm_conf>0.5).mean()*100:.1f}%)")

# ─── REEVALUACIÓN CON CRITERIOS CLAROS ───────────────────────────────────────
print("\n" + "="*60)
print("  REEVALUACIÓN DE ETIQUETAS (criterios estrictos)")
print("="*60)

"""
Criterios para marcar una muestra como "mal etiquetada":
  1. K-Means K=N_GENRES: el género dominante del cluster ≠ etiqueta original
  2. GMM: la confianza del cluster asignado es ALTA (> umbral)
     → si la confianza es baja, la muestra está en zona de solapamiento
       y NO la tocamos (solapamiento genuino, no error de etiqueta)

Umbral de confianza: 0.30 (suficientemente seguro sin ser demasiado restrictivo)
"""
CONF_THRESHOLD = 0.30

# Género dominante de cada cluster K-Means
df_work = df.copy()
df_work["cluster_km"]  = labels_kfull
df_work["cluster_gmm"] = gmm_labels
df_work["gmm_conf"]    = gmm_conf

cluster_dominant_km = (
    df_work.groupby("cluster_km")["main_genre"]
    .agg(lambda x: x.value_counts().index[0])
    .to_dict()
)
cluster_dominant_gmm = (
    df_work.groupby("cluster_gmm")["main_genre"]
    .agg(lambda x: x.value_counts().index[0])
    .to_dict()
)

df_work["km_dominant"]  = df_work["cluster_km"].map(cluster_dominant_km)
df_work["gmm_dominant"] = df_work["cluster_gmm"].map(cluster_dominant_gmm)

# Condición de reetiquetado: ambos métodos coinciden en el género sugerido
# Y la confianza GMM es suficiente
# Y el género sugerido ≠ etiqueta original
df_work["both_agree"]   = df_work["km_dominant"] == df_work["gmm_dominant"]
df_work["conf_ok"]      = df_work["gmm_conf"] >= CONF_THRESHOLD
df_work["label_differs"]= df_work["main_genre"] != df_work["km_dominant"]

df_work["should_relabel"] = (
    df_work["both_agree"] &
    df_work["conf_ok"] &
    df_work["label_differs"]
)

n_relabel = df_work["should_relabel"].sum()
pct       = n_relabel / len(df_work) * 100
print(f"\n  Umbral de confianza GMM: {CONF_THRESHOLD}")
print(f"  Muestras donde KM y GMM coinciden: {df_work['both_agree'].sum()} ({df_work['both_agree'].mean()*100:.1f}%)")
print(f"  De esas, con confianza suficiente: {(df_work['both_agree'] & df_work['conf_ok']).sum()}")
print(f"  De esas, con etiqueta diferente:   {n_relabel} ({pct:.1f}%)")

df_work["main_genre_corrected"] = df_work.apply(
    lambda r: r["km_dominant"] if r["should_relabel"] else r["main_genre"],
    axis=1
)

# Distribución antes y después
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, col, title, color in zip(
    axes,
    ["main_genre", "main_genre_corrected"],
    [f"Original ({len(df_work)} muestras)",
     f"Corregido (solo {n_relabel} cambiadas, {pct:.1f}%)"],
    ["#E50914", "#B20710"]
):
    vc = df_work[col].value_counts()
    bars = ax.barh(vc.index[::-1], vc.values[::-1],
                   color=color, alpha=0.85, edgecolor="none")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Conteo"); ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=8)
    for bar, v in zip(bars[::-1], vc.values):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=7, color="#aaa")

plt.suptitle(
    f"Reevaluación corregida: KMeans(K={N_GENRES}) + GMM + umbral confianza {CONF_THRESHOLD}",
    fontsize=11, fontweight="bold", color="white"
)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fix_02_distribucion_corregida.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print(f"\n  → fix_02_distribucion_corregida.png")

# Gráfica de confianza GMM — separando "cambiadas" vs "no cambiadas"
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(df_work.loc[~df_work["should_relabel"], "gmm_conf"],
             bins=40, color="#564d4d", alpha=0.85, label="No corregidas", edgecolor="none")
axes[0].hist(df_work.loc[df_work["should_relabel"], "gmm_conf"],
             bins=40, color="#E50914", alpha=0.85, label="Corregidas", edgecolor="none")
axes[0].axvline(CONF_THRESHOLD, color="white", ls="--", lw=1.5,
                label=f"Umbral={CONF_THRESHOLD}")
axes[0].set_title("Distribución de Confianza GMM", fontweight="bold")
axes[0].set_xlabel("Confianza (prob. máxima del cluster)")
axes[0].set_ylabel("Frecuencia"); axes[0].legend(); axes[0].grid(alpha=0.2)

# Scatter: confianza vs ¿cuántos géneros tiene la muestra? (solapamiento real)
n_genres_per_sample = df_work["listed_in_list"].apply(len)
axes[1].scatter(n_genres_per_sample, df_work["gmm_conf"],
                c=df_work["should_relabel"].map({True:"#E50914", False:"#564d4d"}),
                s=8, alpha=0.5, linewidths=0)
axes[1].set_xlabel("N° de géneros asignados en listed_in")
axes[1].set_ylabel("Confianza GMM")
axes[1].set_title("Confianza vs Complejidad Multilabel", fontweight="bold")
axes[1].grid(alpha=0.2)
from matplotlib.patches import Patch
axes[1].legend(handles=[
    Patch(color="#E50914", label="Corregida"),
    Patch(color="#564d4d", label="No corregida")
], fontsize=8)

plt.suptitle("Análisis de confianza en el reetiquetado",
             fontsize=11, fontweight="bold", color="white")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fix_03_confianza_relabeling.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print(f"  → fix_03_confianza_relabeling.png")

# ─── MODELOS SUPERVISADOS: COMPARATIVA JUSTA ─────────────────────────────────
print("\n" + "="*60)
print("  MODELOS SUPERVISADOS — COMPARATIVA JUSTA")
print("="*60)

def filter_rare(df_X, y_ser, min_count=2):
    valid = y_ser.value_counts()[lambda x: x >= min_count].index
    mask  = y_ser.isin(valid)
    return df_X[mask].reset_index(drop=True), y_ser[mask].reset_index(drop=True)

le_orig = LabelEncoder()
le_corr = LabelEncoder()

y_orig_s = df_work["main_genre"]
y_corr_s = df_work["main_genre_corrected"]

X_f_orig, y_f_orig_s = filter_rare(X_feat, y_orig_s)
X_f_corr, y_f_corr_s = filter_rare(X_feat, y_corr_s)

y_orig_enc = le_orig.fit_transform(y_f_orig_s)
y_corr_enc = le_corr.fit_transform(y_f_corr_s)

X_tr_o, X_te_o, y_tr_o, y_te_o = train_test_split(
    X_f_orig, y_orig_enc, test_size=0.25, random_state=42, stratify=y_orig_enc)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_f_corr, y_corr_enc, test_size=0.25, random_state=42, stratify=y_corr_enc)

prep = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])

models = {
    "Árbol de Decisión":    DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=42),
    "Reg. Logística":       LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42),
    "Ridge Classifier":     RidgeClassifier(alpha=1.0),
}

results = []
per_class_results = {}

for label_type, (X_tr, X_te, y_tr, y_te, le) in {
    "Original":  (X_tr_o, X_te_o, y_tr_o, y_te_o, le_orig),
    "Corregido": (X_tr_c, X_te_c, y_tr_c, y_te_c, le_corr),
}.items():
    print(f"\n  ── Etiquetas: {label_type} ──")
    for name, clf in models.items():
        pipe = SKPipeline([("prep", prep), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro",    zero_division=0)
        f1w = f1_score(y_te, y_pred, average="weighted", zero_division=0)

        # F1 por clase (para ver si "drama" domina en el corregido)
        f1_per = f1_score(y_te, y_pred, average=None, zero_division=0)
        per_class_results[f"{label_type}_{name}"] = {
            "classes": le.classes_, "f1_per_class": f1_per, "le": le
        }

        results.append({
            "Etiquetas": label_type, "Modelo": name,
            "Accuracy":    round(acc, 4),
            "F1 Macro":    round(f1m, 4),
            "F1 Weighted": round(f1w, 4),
        })
        print(f"    {name:22s} | Acc={acc:.4f} | F1-macro={f1m:.4f} | F1-w={f1w:.4f}")

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUT, "fix_supervised_results.csv"), index=False)

# ─── Gráfica comparativa final ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_names = list(models.keys())
x = np.arange(len(model_names))
w = 0.35

for ax, metric in zip(axes, ["Accuracy", "F1 Macro", "F1 Weighted"]):
    vo = df_res[df_res["Etiquetas"]=="Original"][metric].values
    vc = df_res[df_res["Etiquetas"]=="Corregido"][metric].values
    b1 = ax.bar(x - w/2, vo, w, label="Original", color="#564d4d", alpha=0.9)
    b2 = ax.bar(x + w/2, vc, w, label="Corregido", color="#E50914", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel(metric); ax.set_title(metric, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25); ax.set_ylim(0, 1.05)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

plt.suptitle(f"Comparativa Final — Original vs Corregido\n"
             f"(Reetiquetado: KMeans K={N_GENRES} + GMM conf≥{CONF_THRESHOLD}, "
             f"{n_relabel} muestras cambiadas [{pct:.1f}%])",
             fontsize=10, fontweight="bold", color="white")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fix_04_comparativa_final.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print(f"\n  → fix_04_comparativa_final.png")

# ─── F1 por clase: ¿sigue dominando "drama"? ─────────────────────────────────
print("\n  Generando comparativa F1 por clase...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
best_model_name = "Árbol de Decisión"

for ax, label_type in zip(axes, ["Original", "Corregido"]):
    key = f"{label_type}_{best_model_name}"
    if key not in per_class_results:
        continue
    data = per_class_results[key]
    # Alinear clases con el encoder original para comparación justa
    le    = data["le"]
    f1s   = data["f1_per_class"]
    # Solo clases que aparecen en el test
    idx   = np.argsort(f1s)
    top_n = min(25, len(idx))
    idx_top = idx[-top_n:]

    ax.barh([le.classes_[i] for i in idx_top],
            f1s[idx_top], color="#E50914", alpha=0.85, edgecolor="none")
    ax.set_title(f"{best_model_name} — F1 por género\nEtiquetas: {label_type}",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("F1 Score"); ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25); ax.tick_params(labelsize=8)

plt.suptitle("¿Corregir etiquetas mejora la F1 por género?",
             fontsize=12, fontweight="bold", color="white")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fix_05_f1_por_genero.png"),
            dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
print(f"  → fix_05_f1_por_genero.png")

# ─── Resumen ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ✓ REEVALUACIÓN CORREGIDA COMPLETA")
print("="*60)
print(f"\n  Muestras reetiquetadas: {n_relabel} ({pct:.1f}%)")
print(f"  Criterio: KMeans(K={N_GENRES}) + GMM(K={N_GENRES}) ambos coinciden")
print(f"           + confianza GMM ≥ {CONF_THRESHOLD}")
print("\n  Archivos generados:")
for f in sorted(os.listdir(OUT)):
    if f.startswith("fix_"):
        print(f"    • {f}")
