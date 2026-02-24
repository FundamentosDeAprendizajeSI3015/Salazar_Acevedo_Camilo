"""
Lecture_6_Camilo_Salazar.py
─────────
1. Carga datos de pipeline.py
2. Detecta 3 clusters con KMeans (equivalente visual al UMAP):
     Cluster 0 → "Clásicos/Premium"      (~205 filas, películas antiguas, IMDB alto)
     Cluster 1 → "Entretenimiento Mod."  (~812 filas, recientes, IMDB bajo)
     Cluster 2 → "Contenido Diverso"     (~1190 filas, mix moderno)
3. Aplica undersampling para balancear las 3 clases
4. Entrena Random Forest y Gradient Boosting
5. Genera métricas y gráficas para Train y Validación
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os, time, warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from pipeline import load_and_prepare

# ════════════════════════════════════════════════════════════
#  0. Cargar datos desde pipeline
# ════════════════════════════════════════════════════════════
print("=" * 64)
print("  CARGANDO DATOS DESDE PIPELINE")
print("=" * 64)

(X_train, X_val, X_test,
 y_train_ml, y_val_ml, y_test_ml,
 preprocessor, mlb) = load_and_prepare()

os.makedirs("outputs", exist_ok=True)

NUMERIC_COLS     = ["release_year", "imdb_score", "imdb_votes", "duration_num"]
CATEGORICAL_COLS = ["type_base", "rating", "age_certification", "country"]

CLUSTER_NAMES = {
    "best_fit": None,   # se asigna dinámicamente tras analizar clusters
}

# ════════════════════════════════════════════════════════════
#  1. Identificar clusters con KMeans sobre TODO el dataset
#     (reproduciendo la lógica del UMAP)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  IDENTIFICACIÓN DE CLUSTERS (KMeans k=3)")
print("=" * 64)

X_all = pd.concat([X_train, X_val, X_test]).reset_index(drop=True)
y_all = np.vstack([y_train_ml, y_val_ml, y_test_ml])

# Preprocesar para KMeans
prep_km = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
])
X_proc_all = prep_km.fit_transform(X_all)

km = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters_all = km.fit_predict(X_proc_all)

# Caracterizar cada cluster para asignarle un nombre legible
cluster_info = {}
for c in range(3):
    mask = clusters_all == c
    cluster_info[c] = {
        "n":           mask.sum(),
        "year_mean":   X_all[mask]["release_year"].mean(),
        "score_mean":  X_all[mask]["imdb_score"].mean(),
        "dur_mean":    X_all[mask]["duration_num"].mean(),
    }

# Regla de asignación basada en el análisis exploratorio:
#   - Año más antiguo + duración más larga → "Clásicos/Premium"
#   - IMDB más bajo + más reciente         → "Entretenimiento Popular"
#   - Resto                                → "Contenido Diverso"
sorted_by_year  = sorted(cluster_info, key=lambda c: cluster_info[c]["year_mean"])
sorted_by_score = sorted(cluster_info, key=lambda c: cluster_info[c]["score_mean"])

oldest_cluster   = sorted_by_year[0]           # año más antiguo
lowest_score_cl  = sorted_by_score[0]          # IMDB más bajo
diverse_cluster  = [c for c in range(3)
                    if c != oldest_cluster and c != lowest_score_cl][0]

label_map = {
    oldest_cluster:  "Clásicos/Premium",
    lowest_score_cl: "Entretenimiento Popular",
    diverse_cluster: "Contenido Diverso",
}
# Si oldest y lowest coinciden, el tercero que queda se reparte
if oldest_cluster == lowest_score_cl:
    remaining = [c for c in range(3) if c != oldest_cluster]
    label_map[remaining[0]] = "Entretenimiento Popular"
    label_map[remaining[1]] = "Contenido Diverso"

print("\nCaracterísticas por cluster:")
print(f"  {'Cluster':<6} {'Etiqueta':<26} {'N':>6} {'Año':>8} {'IMDB':>7} {'Dur':>7}")
print(f"  {'-'*62}")
for c in range(3):
    info = cluster_info[c]
    print(f"  {c:<6} {label_map[c]:<26} {info['n']:>6} "
          f"{info['year_mean']:>8.1f} {info['score_mean']:>7.2f} {info['dur_mean']:>7.1f}")

CLASS_NAMES = [label_map[c] for c in range(3)]

# ════════════════════════════════════════════════════════════
#  2. Asignar etiquetas de cluster a train y val
#     (KMeans predice el cluster según el centroide más cercano)
# ════════════════════════════════════════════════════════════
# Usamos el mismo KMeans global (km) para predecir sobre train y val
# El preprocesador ya está ajustado (prep_km), lo reutilizamos
X_train_proc = prep_km.transform(X_train)
X_val_proc   = prep_km.transform(X_val)

raw_train_clusters = km.predict(X_train_proc)
raw_val_clusters   = km.predict(X_val_proc)

y_train_cl = np.array([label_map[c] for c in raw_train_clusters])
y_val_cl   = np.array([label_map[c] for c in raw_val_clusters])

print(f"\nDistribución train (antes de undersampling):")
for name in CLASS_NAMES:
    print(f"  {name}: {(y_train_cl == name).sum()}")

# ════════════════════════════════════════════════════════════
#  3. Undersampling — igualar al tamaño de la clase minoritaria
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  UNDERSAMPLING")
print("=" * 64)

min_size = min((y_train_cl == name).sum() for name in CLASS_NAMES)
print(f"\n  Clase minoritaria: {min_size} muestras → todas las clases quedan en ~{min_size}")

X_parts, y_parts = [], []
for name in CLASS_NAMES:
    mask = y_train_cl == name
    X_c  = X_train[mask].reset_index(drop=True)
    y_c  = y_train_cl[mask]
    X_rs, y_rs = resample(X_c, y_c, n_samples=min_size,
                          replace=False, random_state=42)
    X_parts.append(X_rs)
    y_parts.append(y_rs)

X_train_bal = pd.concat(X_parts).reset_index(drop=True)
y_train_bal = np.concatenate(y_parts)

print(f"\nDistribución train (después de undersampling):")
for name in CLASS_NAMES:
    print(f"  {name}: {(y_train_bal == name).sum()}")
print(f"  Total: {len(y_train_bal)}")

# ════════════════════════════════════════════════════════════
#  4. Definir y entrenar pipelines
# ════════════════════════════════════════════════════════════
def make_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])

rf_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier",   RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
    )),
])

gb_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("classifier",   GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )),
])

print("\n" + "=" * 64)
print("  ENTRENAMIENTO")
print("=" * 64)

print("\n▶ Random Forest...")
t0 = time.time()
rf_pipeline.fit(X_train_bal, y_train_bal)
print(f"  ✓ {time.time()-t0:.1f}s")

print("▶ Gradient Boosting...")
t0 = time.time()
gb_pipeline.fit(X_train_bal, y_train_bal)
print(f"  ✓ {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════
#  5. Predicciones — train y validación
# ════════════════════════════════════════════════════════════
rf_pred_train = rf_pipeline.predict(X_train_bal)
rf_pred_val   = rf_pipeline.predict(X_val)
gb_pred_train = gb_pipeline.predict(X_train_bal)
gb_pred_val   = gb_pipeline.predict(X_val)

# ════════════════════════════════════════════════════════════
#  6. Métricas en consola
# ════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, model_name, split):
    acc = accuracy_score(y_true, y_pred)
    p_m = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    r_m = recall_score(   y_true, y_pred, average="macro",    zero_division=0)
    f_m = f1_score(       y_true, y_pred, average="macro",    zero_division=0)
    p_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    r_w = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
    f_w = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'═'*62}")
    print(f"  {model_name}  │  {split}")
    print(f"{'═'*62}")
    print(f"  {'Accuracy':<14}: {acc:.4f}")
    print(f"\n  {'Métrica':<14} {'Macro':>10} {'Weighted':>12}")
    print(f"  {'-'*40}")
    print(f"  {'Precision':<14} {p_m:>10.4f} {p_w:>12.4f}")
    print(f"  {'Recall':<14} {r_m:>10.4f} {r_w:>12.4f}")
    print(f"  {'F1':<14} {f_m:>10.4f} {f_w:>12.4f}")

    return dict(acc=acc, p_m=p_m, r_m=r_m, f_m=f_m, p_w=p_w, r_w=r_w, f_w=f_w)


print("\n" + "=" * 64)
print("  MÉTRICAS")
print("=" * 64)

rf_tr = compute_metrics(y_train_bal, rf_pred_train, "Random Forest",     "Train")
rf_vl = compute_metrics(y_val_cl,    rf_pred_val,   "Random Forest",     "Validación")
gb_tr = compute_metrics(y_train_bal, gb_pred_train, "Gradient Boosting", "Train")
gb_vl = compute_metrics(y_val_cl,    gb_pred_val,   "Gradient Boosting", "Validación")

print("\n\n── Classification Report: Random Forest — Validación ───────")
print(classification_report(y_val_cl, rf_pred_val,
                             target_names=CLASS_NAMES, zero_division=0))
print("── Classification Report: Gradient Boosting — Validación ──")
print(classification_report(y_val_cl, gb_pred_val,
                             target_names=CLASS_NAMES, zero_division=0))

# ════════════════════════════════════════════════════════════
#  7. Gráficas
# ════════════════════════════════════════════════════════════
DARK   = "#141414"
PANEL  = "#1e1e1e"
WHITE  = "#FFFFFF"
GRAY   = "#aaaaaa"
GRID   = "#2e2e2e"
RED    = "#E50914"
BLUE   = "#0080FF"
RED_L  = "#FF6B6B"
BLUE_L = "#66B2FF"

SHORT = ["Clásicos", "Entret.\nPopular", "Contenido\nDiverso"]
cmap_cm = LinearSegmentedColormap.from_list("nf", ["#1e1e1e", RED])

print("\n\nGenerando gráficas...")

# ── 7a. Matrices de confusión 2×2 (modelo × split) ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor=DARK)
fig.suptitle("Matrices de Confusión — Clusters UMAP",
             fontsize=17, fontweight="bold", color=WHITE, y=1.01)

combos = [
    (y_train_bal, rf_pred_train, "Random Forest — Train"),
    (y_val_cl,    rf_pred_val,   "Random Forest — Validación"),
    (y_train_bal, gb_pred_train, "Gradient Boosting — Train"),
    (y_val_cl,    gb_pred_val,   "Gradient Boosting — Validación"),
]

for ax, (yt, yp, title) in zip(axes.flat, combos):
    ax.set_facecolor(PANEL)
    cm = confusion_matrix(yt, yp, labels=CLASS_NAMES)
    ax.imshow(cm, cmap=cmap_cm, interpolation="nearest")
    ax.set_title(title, color=WHITE, fontsize=12, pad=10)
    ax.set_xlabel("Predicho", color=GRAY, fontsize=10)
    ax.set_ylabel("Real",     color=GRAY, fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(SHORT, color=WHITE, fontsize=8, rotation=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(SHORT, color=WHITE, fontsize=8)
    ax.tick_params(colors=WHITE, length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color=DARK if cm[i, j] > thresh else WHITE)

plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("  → outputs/confusion_matrices.png")

# ── 7b. Train vs Validación por modelo ───────────────────────────────────────
metric_keys   = ["acc", "p_m", "r_m", "f_m"]
metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"]
x     = np.arange(len(metric_keys))
width = 0.32

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=DARK)
fig.suptitle("Train vs Validación por Modelo — Clusters UMAP",
             fontsize=14, fontweight="bold", color=WHITE, y=1.02)

for ax, (name, m_tr, m_vl, col_tr, col_vl) in zip(axes, [
    ("Random Forest",     rf_tr, rf_vl, RED,  RED_L),
    ("Gradient Boosting", gb_tr, gb_vl, BLUE, BLUE_L),
]):
    ax.set_facecolor(PANEL)
    v_tr = [m_tr[k] for k in metric_keys]
    v_vl = [m_vl[k] for k in metric_keys]

    b_tr = ax.bar(x - width/2, v_tr, width, label="Train",
                  color=col_tr, alpha=0.92, edgecolor=DARK, linewidth=0.8)
    b_vl = ax.bar(x + width/2, v_vl, width, label="Validación",
                  color=col_vl, alpha=0.92, edgecolor=DARK, linewidth=0.8)

    for bar in list(b_tr) + list(b_vl):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=WHITE, fontweight="bold")

    ax.set_title(name, color=WHITE, fontsize=12, pad=8)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, color=WHITE, fontsize=9)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Score", color=GRAY, fontsize=9)
    ax.tick_params(colors=WHITE, length=0)
    ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#2a2a2a", labelcolor=WHITE, fontsize=9)

plt.tight_layout()
plt.savefig("outputs/train_vs_val.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("  → outputs/train_vs_val.png")

# ── 7c. RF vs GB todas las métricas en validación ────────────────────────────
all_keys   = ["acc", "p_m", "r_m", "f_m", "p_w", "r_w", "f_w"]
all_labels = ["Accuracy", "Precision\nMacro", "Recall\nMacro", "F1\nMacro",
              "Precision\nWeighted", "Recall\nWeighted", "F1\nWeighted"]
x     = np.arange(len(all_keys))
width = 0.32

fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=DARK)
ax.set_facecolor(PANEL)
rf_vals = [rf_vl[k] for k in all_keys]
gb_vals = [gb_vl[k] for k in all_keys]

b_rf = ax.bar(x - width/2, rf_vals, width, label="Random Forest",
              color=RED,  alpha=0.92, edgecolor=DARK, linewidth=0.8)
b_gb = ax.bar(x + width/2, gb_vals, width, label="Gradient Boosting",
              color=BLUE, alpha=0.92, edgecolor=DARK, linewidth=0.8)

for bar in list(b_rf) + list(b_gb):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.006,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=8, color=WHITE, fontweight="bold")

ax.set_title("RF vs GB — Validación (todas las métricas)",
             color=WHITE, fontsize=13, pad=10)
ax.set_xticks(x); ax.set_xticklabels(all_labels, color=WHITE, fontsize=9)
ax.set_ylim(0, 1.18); ax.set_ylabel("Score", color=GRAY, fontsize=9)
ax.tick_params(colors=WHITE, length=0)
ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.6)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")
ax.legend(facecolor="#2a2a2a", labelcolor=WHITE, fontsize=10)

plt.tight_layout()
plt.savefig("outputs/rf_vs_gb_val.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("  → outputs/rf_vs_gb_val.png")

# ── 7d. Distribución de clases antes y después de undersampling ──────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK)
fig.suptitle("Distribución de Clases — Antes vs Después del Undersampling",
             fontsize=13, fontweight="bold", color=WHITE, y=1.02)

colors_dist = [RED, BLUE, "#00C853"]

for ax, (title, y_data) in zip(axes, [
    ("Antes (desbalanceado)", y_train_cl),
    ("Después (balanceado)",  y_train_bal),
]):
    ax.set_facecolor(PANEL)
    counts = [(name, (y_data == name).sum()) for name in CLASS_NAMES]
    bars = ax.bar([c[0] for c in counts], [c[1] for c in counts],
                  color=colors_dist, alpha=0.9, edgecolor=DARK, linewidth=0.8)
    for bar, (_, cnt) in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(cnt), ha="center", va="bottom",
                fontsize=11, color=WHITE, fontweight="bold")
    ax.set_title(title, color=WHITE, fontsize=11, pad=8)
    ax.set_xticklabels([c[0] for c in counts], color=WHITE, fontsize=8, rotation=12)
    ax.set_ylabel("Nº de muestras", color=GRAY, fontsize=9)
    ax.tick_params(colors=WHITE, length=0)
    ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("  → outputs/class_distribution.png")

# ════════════════════════════════════════════════════════════
#  8. Resumen final
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 64)
print("  RESUMEN FINAL")
print("=" * 64)
print(f"\n  {'Modelo':<22} {'Split':<14} {'Acc':>7} {'F1-macro':>10} {'F1-w':>8}")
print(f"  {'-'*65}")
for name, m, split in [
    ("Random Forest",     rf_tr, "Train"),
    ("Random Forest",     rf_vl, "Validación"),
    ("Gradient Boosting", gb_tr, "Train"),
    ("Gradient Boosting", gb_vl, "Validación"),
]:
    print(f"  {name:<22} {split:<14} {m['acc']:>7.4f} {m['f_m']:>10.4f} {m['f_w']:>8.4f}")

print("\n✓ Gráficas en outputs/")