"""
pipeline_FIRE_UdeA.py
─────────────────────────────────────────────────────────────────────────────
Pipeline de Machine Learning para detección de riesgo financiero (FIRE UdeA).

Modelos : XGBoost  y  LightGBM  (Gradient Boosting)
Objetivo: label  →  0 = Sin riesgo financiero | 1 = Con riesgo financiero

División de datos: aleatoria estratificada 60 / 20 / 20
  Train : 60 %  (48 registros)
  Valid : 20 %  (16 registros)
  Test  : 20 %  (16 registros)

Métricas reportadas (mismas que el baseline original):
  split, n, prevalencia, roc_auc, pr_auc, brier, log_loss,
  precision, recall, f1, tn, fp, fn, tp

Baseline a superar (modelo original):
  Train  →  ROC AUC: 1.000  |  Log Loss: 0.409  |  F1: 0.600
  Valid  →  ROC AUC: 0.933  |  Log Loss: 0.239  |  F1: 0.909
  Test   →  ROC AUC: 0.417  |  Log Loss: 4.877  |  F1: 0.857

Uso:
  Coloca este script en la misma carpeta que el dataset y ejecuta:
  python pipeline_FIRE_UdeA.py
─────────────────────────────────────────────────────────────────────────────
"""

# ── Librerías ──────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════

DATA_PATH    = "dataset_sintetico_FIRE_UdeA_realista.csv"
METRICS_PATH = "reporte_metricas_FIRE_UdeA_realista.csv"
OUTPUT_DIR   = "output_udea"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED         = 42

# Baseline original (para comparación)
BASELINE = {
    "train": {"roc_auc":1.000, "pr_auc":1.000, "brier":0.155, "log_loss":0.409,
              "precision":0.429, "recall":1.000, "f1":0.600},
    "valid": {"roc_auc":0.933, "pr_auc":0.933, "brier":0.083, "log_loss":0.239,
              "precision":0.833, "recall":1.000, "f1":0.909},
    "test":  {"roc_auc":0.417, "pr_auc":0.725, "brier":0.257, "log_loss":4.877,
              "precision":0.750, "recall":1.000, "f1":0.857},
}

# ══════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  PIPELINE FIRE-UdeA — XGBoost & LightGBM")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print(f"\n[1] Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"    Unidades: {df['unidad'].nunique()} unidades academicas")
print(f"    Anos    : {sorted(df['anio'].unique())}")
print(f"    Label   : {dict(df['label'].value_counts().sort_index())}  "
      f"(0=sin riesgo, 1=en riesgo)")

target_col      = "label"
numeric_cols    = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\n    Variables numericas   : {len(numeric_cols)}")
print(f"    Variables categoricas : {categorical_cols}  -> One-Hot Encoding")

# ══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════

print("\n[2] Preprocesamiento...")

# Pipeline numerico: imputacion mediana + RobustScaler (resistente a outliers)
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  RobustScaler()),
])

# Pipeline categorico: imputacion + One-Hot Encoding
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot",  OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
])

# Combinar ambos
preprocessor = ColumnTransformer([
    ("num", numeric_transformer,    numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# ── Division aleatoria estratificada 60/20/20 ─────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
)

print(f"    Train : {len(y_train)} muestras | label=1: {y_train.sum()} ({y_train.mean():.1%})")
print(f"    Valid : {len(y_val)}  muestras | label=1: {y_val.sum()} ({y_val.mean():.1%})")
print(f"    Test  : {len(y_test)}  muestras | label=1: {y_test.sum()} ({y_test.mean():.1%})")

# Aplicar preprocesamiento
X_train_p = preprocessor.fit_transform(X_train)
X_val_p   = preprocessor.transform(X_val)
X_test_p  = preprocessor.transform(X_test)

n_features = X_train_p.shape[1]
print(f"\n    Features totales tras preproceso: {n_features} "
      f"({len(numeric_cols)} numericas + ~{n_features - len(numeric_cols)} categoricas)")

# ══════════════════════════════════════════════════════════════════════════
# 3. FUNCIÓN DE EVALUACIÓN (mismas metricas que el baseline)
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_tr, y_tr, X_va, y_va, X_te, y_te, name):
    """Calcula las mismas 13 metricas que el reporte original."""
    results = []
    for split_name, Xs, ys in [("train", X_tr, y_tr),
                                ("valid", X_va, y_va),
                                ("test",  X_te, y_te)]:
        prob = model.predict_proba(Xs)[:, 1]
        pred = model.predict(Xs)
        try:
            tn, fp, fn, tp = confusion_matrix(ys, pred).ravel()
        except ValueError:
            tn = fp = fn = tp = 0

        results.append({
            "model"      : name,
            "split"      : split_name,
            "n"          : len(ys),
            "prevalencia": round(float(ys.mean()), 3),
            "roc_auc"    : round(roc_auc_score(ys, prob), 4),
            "pr_auc"     : round(average_precision_score(ys, prob), 4),
            "brier"      : round(brier_score_loss(ys, prob), 4),
            "log_loss"   : round(log_loss(ys, prob), 4),
            "accuracy"   : round(accuracy_score(ys, pred), 4),
            "precision"  : round(precision_score(ys, pred, zero_division=0), 4),
            "recall"     : round(recall_score(ys, pred, zero_division=0), 4),
            "f1"         : round(f1_score(ys, pred, zero_division=0), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════════════
# 4. MODELO 1 — XGBoost
# ══════════════════════════════════════════════════════════════════════════

print("\n[3] Entrenando XGBoost...")

xgb = XGBClassifier(
    n_estimators         = 100,
    max_depth            = 5,
    learning_rate        = 0.1,
    random_state         = SEED,
    eval_metric          = "logloss",
    early_stopping_rounds= 10,
)
xgb.fit(X_train_p, y_train,
        eval_set=[(X_val_p, y_val)],
        verbose=False)

xgb_results = evaluate_model(xgb, X_train_p, y_train,
                              X_val_p, y_val, X_test_p, y_test, "XGBoost")
print(f"    Arboles usados (early stopping): {xgb.best_iteration + 1}")

# ══════════════════════════════════════════════════════════════════════════
# 5. MODELO 2 — LightGBM
# ══════════════════════════════════════════════════════════════════════════

print("\n[4] Entrenando LightGBM...")

lgbm = LGBMClassifier(
    n_estimators  = 100,
    max_depth     = 5,
    learning_rate = 0.1,
    random_state  = SEED,
    verbose       = -1,
)
lgbm.fit(X_train_p, y_train,
         eval_set=[(X_val_p, y_val)])

lgbm_results = evaluate_model(lgbm, X_train_p, y_train,
                               X_val_p, y_val, X_test_p, y_test, "LightGBM")
print(f"    Arboles entrenados: {lgbm.n_estimators}")

# ══════════════════════════════════════════════════════════════════════════
# 6. COMPARACIÓN CON BASELINE
# ══════════════════════════════════════════════════════════════════════════

all_results = pd.concat([xgb_results, lgbm_results], ignore_index=True)
all_results.to_csv(f"{OUTPUT_DIR}/comparacion_modelos.csv", index=False)

METRICS = ["roc_auc", "pr_auc", "brier", "log_loss", "precision", "recall", "f1"]

print("\n[5] Resultados vs Baseline")
print()

for split in ["train", "valid", "test"]:
    base = BASELINE[split]
    xgb_row  = xgb_results[xgb_results["split"]  == split].iloc[0]
    lgbm_row = lgbm_results[lgbm_results["split"] == split].iloc[0]

    print(f"  [{split.upper()}]  n={xgb_row['n']}  prevalencia={xgb_row['prevalencia']:.1%}")
    print(f"  {'Metrica':<12} {'Baseline':>10} {'XGBoost':>10} {'LightGBM':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for m in METRICS:
        b  = base[m]
        xv = xgb_row[m]
        lv = lgbm_row[m]
        # Para brier y log_loss: menor es mejor
        better = lambda new, old: "mejor" if (new < old if m in ["brier","log_loss"] else new > old) else ("igual" if abs(new-old) < 0.001 else "peor")
        print(f"  {m:<12} {b:>10.4f} {xv:>10.4f} {lv:>10.4f}   {better(xv,b)} / {better(lv,b)}")
    print()

# ══════════════════════════════════════════════════════════════════════════
# 7. VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════

print("[6] Generando visualizaciones...")
sns.set_theme(style="whitegrid")
COLORS = {"Baseline": "#B0BEC5", "XGBoost": "#4C9BE8", "LightGBM": "#E8614C"}
SPLITS = ["train", "valid", "test"]

# ── Fig 1: Comparación de métricas ────────────────────────────────────────
METRICS_PLOT = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
fig, axes = plt.subplots(1, len(METRICS_PLOT), figsize=(22, 5))
fig.suptitle("Comparacion de Metricas: Baseline vs XGBoost vs LightGBM",
             fontsize=13, fontweight="bold")

for ax, metric in zip(axes, METRICS_PLOT):
    base_vals = [BASELINE[s][metric] for s in SPLITS]
    xgb_vals  = xgb_results.set_index("split")[metric].reindex(SPLITS).values
    lgbm_vals = lgbm_results.set_index("split")[metric].reindex(SPLITS).values
    x = np.arange(len(SPLITS))
    w = 0.25
    ax.bar(x - w,   base_vals, width=w, label="Baseline", color=COLORS["Baseline"], edgecolor="white")
    ax.bar(x,       xgb_vals,  width=w, label="XGBoost",  color=COLORS["XGBoost"],  edgecolor="white")
    ax.bar(x + w,   lgbm_vals, width=w, label="LightGBM", color=COLORS["LightGBM"], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(SPLITS)
    ax.set_ylim(0, 1.2)
    ax.set_title(metric.upper().replace("_","-"), fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    if metric == METRICS_PLOT[0]:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_comparacion_metricas.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ── Fig 2: Curvas ROC y PR en test ────────────────────────────────────────
xgb_prob  = xgb.predict_proba(X_test_p)[:, 1]
lgbm_prob = lgbm.predict_proba(X_test_p)[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Curvas ROC y Precision-Recall — Test Set", fontsize=13, fontweight="bold")

for prob, name, color in [(xgb_prob, "XGBoost", COLORS["XGBoost"]),
                           (lgbm_prob, "LightGBM", COLORS["LightGBM"])]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax1.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2, color=color)

ax1.plot([0,1],[0,1], "k--", linewidth=0.8, label="Baseline (AUC=0.417)")
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("Curva ROC", fontweight="bold"); ax1.legend()
ax1.spines[["top","right"]].set_visible(False)

for prob, name, color in [(xgb_prob, "XGBoost", COLORS["XGBoost"]),
                           (lgbm_prob, "LightGBM", COLORS["LightGBM"])]:
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax2.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=2, color=color)

ax2.axhline(y_test.mean(), color="k", linestyle="--", linewidth=0.8,
            label=f"Baseline (AP={y_test.mean():.3f})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Curva Precision-Recall", fontweight="bold"); ax2.legend()
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_roc_pr.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ── Fig 3: Matrices de confusión ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Matrices de Confusion — Test Set", fontsize=13, fontweight="bold")

for ax, model, prob, name, color in [
    (axes[0], xgb,  xgb_prob,  "XGBoost",  COLORS["XGBoost"]),
    (axes[1], lgbm, lgbm_prob, "LightGBM", COLORS["LightGBM"]),
]:
    cm = confusion_matrix(y_test, model.predict(X_test_p))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Sin riesgo","Con riesgo"],
                yticklabels=["Sin riesgo","Con riesgo"],
                annot_kws={"size": 14})
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_confusion.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ── Fig 4: Feature importance ─────────────────────────────────────────────
# Reconstruir nombres de features tras One-Hot
cat_encoder   = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_features  = cat_encoder.get_feature_names_out(categorical_cols).tolist()
feature_names = numeric_cols + cat_features

xgb_imp  = pd.DataFrame({"feature": feature_names, "importance": xgb.feature_importances_}).sort_values("importance", ascending=False)
lgbm_imp = pd.DataFrame({"feature": feature_names, "importance": lgbm.feature_importances_}).sort_values("importance", ascending=False)

TOP_N = min(15, len(xgb_imp[xgb_imp["importance"] > 0]))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Top variables importantes", fontsize=14, fontweight="bold")

for ax, imp_df, name, color in [
    (axes[0], xgb_imp,  "XGBoost",  "#4E79A7"),
    (axes[1], lgbm_imp, "LightGBM", "#E15759"),
]:
    data = imp_df.head(TOP_N).sort_values("importance")  # ascendente para barh
    ax.barh(data["feature"], data["importance"], color=color, edgecolor="white", alpha=0.9)
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_feature_importance.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print("    4 figuras generadas: fig1 a fig4")

# ── Fig 5: Árboles de decisión con sklearn plot_tree ───────────────────────
#
# Se entrena un DecisionTreeClassifier por cada profundidad para visualizar
# la jerarquía de variables con nombres reales, usando exactamente el mismo
# método que Modelo__1_.py (sklearn.tree.plot_tree).
#
# Cómo leer el diagrama:
#   Raíz          → variable más informativa (primera división)
#   Nodo interior → "variable <= umbral"  |  rama izq = Verdadero, rama der = Falso
#   Hoja          → clase predicha (la mayoritaria en ese nodo)
#   Gini          → impureza del nodo: 0 = puro (todos de una clase), 0.5 = máximo desorden
#   Samples       → cuántas muestras de entrenamiento llegan a ese nodo
#   Value         → [cantidad sin riesgo, cantidad con riesgo] en ese nodo
#   Color azul    → mayoría sin riesgo (0)  |  Color naranja → mayoría con riesgo (1)
#   Intensidad    → mayor intensidad = nodo más puro
#

print("\n[7] Generando visualizaciones de arboles...")

from sklearn.tree import DecisionTreeClassifier, plot_tree

configs = [
    (2, "Arbol #1 — Profundidad 2 (vision general, variables raiz)"),
    (3, "Arbol #2 — Profundidad 3 (segunda capa de decisiones)"),
    (4, "Arbol #3 — Profundidad 4 (reglas completas)"),
]

for depth, subtitle in configs:
    dt = DecisionTreeClassifier(
        max_depth=depth,
        criterion="gini",
        class_weight="balanced",
        random_state=42,
    )
    dt.fit(X_train_p, y_train)
    root_feat = feature_names[dt.tree_.feature[0]]

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=["Sin riesgo", "Con riesgo"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        impurity=True,
        proportion=False,
    )
    ax.set_title(
        f"{subtitle}\n"
        f"Raiz: \'{root_feat}\'  |  "
        f"Azul=sin riesgo  Naranja=con riesgo  |  "
        f"Gini: impureza (0=puro)  Samples: muestras  Value: [sin riesgo, con riesgo]",
        fontsize=10, fontweight="bold", pad=15
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig5_arbol_d{depth}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Guardado: fig5_arbol_d{depth}.png  (raiz: {root_feat})")

print("    Arboles generados: 3 imagenes (profundidad 2, 3 y 4)")

# ══════════════════════════════════════════════════════════════════════════
# 8. RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════

xgb_te  = xgb_results[xgb_results["split"]  == "test"].iloc[0]
lgbm_te = lgbm_results[lgbm_results["split"] == "test"].iloc[0]

best_name = "XGBoost" if xgb_te["roc_auc"] >= lgbm_te["roc_auc"] else "LightGBM"
best_te   = xgb_te if best_name == "XGBoost" else lgbm_te

print(f"""
{'='*65}
  RESUMEN FINAL — TEST SET
{'='*65}
  {'Metrica':<12} {'Baseline':>10} {'XGBoost':>10} {'LightGBM':>10}
  {'-'*12} {'-'*10} {'-'*10} {'-'*10}
  {'ROC AUC':<12} {0.417:>10.3f} {xgb_te['roc_auc']:>10.3f} {lgbm_te['roc_auc']:>10.3f}
  {'Log Loss':<12} {4.877:>10.3f} {xgb_te['log_loss']:>10.3f} {lgbm_te['log_loss']:>10.3f}
  {'Precision':<12} {0.750:>10.3f} {xgb_te['precision']:>10.3f} {lgbm_te['precision']:>10.3f}
  {'Recall':<12} {1.000:>10.3f} {xgb_te['recall']:>10.3f} {lgbm_te['recall']:>10.3f}
  {'F1':<12} {0.857:>10.3f} {xgb_te['f1']:>10.3f} {lgbm_te['f1']:>10.3f}

  Mejor modelo en test: {best_name}
  ROC AUC: 0.417 (baseline) -> {best_te['roc_auc']:.3f}
  Log Loss: 4.877 (baseline) -> {best_te['log_loss']:.3f}

  Archivos guardados en output_udea/:
    comparacion_modelos.csv
    fig1_comparacion_metricas.png
    fig2_roc_pr.png
    fig3_confusion.png
    fig4_feature_importance.png
    fig5_arbol_d2.png           (arbol profundidad 2 — vision general)
    fig5_arbol_d3.png           (arbol profundidad 3 — segunda capa)
    fig5_arbol_d4.png           (arbol profundidad 4 — reglas completas)
{'='*65}
""")