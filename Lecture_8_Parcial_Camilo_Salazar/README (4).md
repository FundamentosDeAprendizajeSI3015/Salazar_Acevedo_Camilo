# Lecture 8 — Parcial Camilo Salazar

Pipeline completo de Machine Learning para **detección de riesgo financiero** en unidades académicas de la UdeA (proyecto FIRE), comparando modelos de Gradient Boosting (**XGBoost** y **LightGBM**) contra un baseline previo, con visualizaciones detalladas y árboles de decisión interpretables.

---

## Estructura

```
Lecture_8_Parcial_Camilo_Salazar/
│
├── pipeline_FIRE_UdeA.py                        # Script principal
├── dataset_sintetico_FIRE_UdeA.csv              # Dataset base
├── dataset_sintetico_FIRE_UdeA_realista.csv     # Dataset extendido (usado por el pipeline)
│
└── output_udea/                                 # Resultados generados automáticamente
    ├── comparacion_modelos.csv
    ├── fig1_comparacion_metricas.png
    ├── fig2_roc_pr.png
    ├── fig3_confusion.png
    ├── fig4_feature_importance.png
    ├── fig5_arbol_d2.png
    ├── fig5_arbol_d3.png
    └── fig5_arbol_d4.png
```

---

## Archivo principal

### `pipeline_FIRE_UdeA.py`

**Objetivo:** clasificar unidades académicas como `0 = Sin riesgo financiero` o `1 = Con riesgo financiero`.

**Flujo:**

1. **Carga de datos** — `dataset_sintetico_FIRE_UdeA_realista.csv`
2. **Preprocesamiento**
   - Variables numéricas: imputación por mediana + `RobustScaler`
   - Variables categóricas: imputación + `OneHotEncoder`
   - División estratificada **60 / 20 / 20** (Train / Valid / Test)
3. **Entrenamiento de modelos**
   - `XGBoostClassifier` con early stopping en validación
   - `LGBMClassifier`
4. **Evaluación** con las mismas 13 métricas del baseline original: ROC AUC, PR AUC, Brier, Log Loss, Accuracy, Precision, Recall, F1, TN, FP, FN, TP
5. **Comparación contra baseline** (impresa en consola por split)
6. **Visualizaciones** generadas en `output_udea/`
7. **Árboles de decisión** con `sklearn.tree.plot_tree` a profundidades 2, 3 y 4

---

## Baseline a superar

| Split | ROC AUC | Log Loss | F1    |
|-------|---------|----------|-------|
| Train | 1.000   | 0.409    | 0.600 |
| Valid | 0.933   | 0.239    | 0.909 |
| Test  | 0.417   | 4.877    | 0.857 |

---

## Salidas (`output_udea/`)

| Archivo | Descripción |
|---|---|
| `comparacion_modelos.csv` | Tabla completa de métricas por modelo y split |
| `fig1_comparacion_metricas.png` | Barras comparativas de ROC AUC, PR AUC, F1, Precision, Recall |
| `fig2_roc_pr.png` | Curvas ROC y Precision-Recall en el test set |
| `fig3_confusion.png` | Matrices de confusión de XGBoost y LightGBM |
| `fig4_feature_importance.png` | Top variables importantes por modelo |
| `fig5_arbol_d2.png` | Árbol de decisión — profundidad 2 (visión general) |
| `fig5_arbol_d3.png` | Árbol de decisión — profundidad 3 (segunda capa) |
| `fig5_arbol_d4.png` | Árbol de decisión — profundidad 4 (reglas completas) |

---

## Cómo ejecutar

```bash
# Instalar dependencias si es necesario
pip install xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn

# Ejecutar desde la carpeta del proyecto
python pipeline_FIRE_UdeA.py
```

Los resultados se guardan automáticamente en la carpeta `output_udea/`.

---

## Datasets

| Archivo | Descripción |
|---|---|
| `dataset_sintetico_FIRE_UdeA.csv` | Dataset base del proyecto FIRE |
| `dataset_sintetico_FIRE_UdeA_realista.csv` | Versión extendida con columnas `label`, `anio`, `unidad` y features financieras adicionales — **usado por el pipeline** |

---

## Dependencias

```
pandas · numpy · matplotlib · seaborn · scikit-learn · xgboost · lightgbm
```
