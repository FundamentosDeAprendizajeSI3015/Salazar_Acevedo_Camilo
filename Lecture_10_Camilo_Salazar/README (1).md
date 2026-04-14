# Lecture 10 — Camilo Salazar

Exploración de algoritmos de clustering no supervisado (**KMeans** y **DBSCAN**) aplicados a datasets sintéticos del proyecto FIRE UdeA, con visualización mediante reducción de dimensionalidad PCA.

---

## Archivos

### `kmeans_fire.py`
Clustering con **KMeans (k=2)** sobre `dataset_sintetico_FIRE_UdeA.csv`.

**Flujo:**
1. Carga y escalado estándar de features
2. Ajuste de KMeans con 2 clusters
3. Asignación automática de cada cluster a una clase (0 o 1) según composición mayoritaria
4. Reporte de porcentaje de acierto por clase
5. Visualización PCA 2D de clusters, etiquetas originales y posibles errores de etiquetado

**Salidas:** `clusters_kmeans_pca.png`, `labels_originales_pca.png`, `posibles_errores_pca.png`

---

### `dbscan_3d.py`
Clustering con **DBSCAN** sobre `dataset_sintetico_FIRE_UdeA.csv` con visualización en **3 dimensiones**.

**Parámetros clave:** `eps=1.5`, `min_samples=5`

**Flujo:**
1. Escalado de features
2. Ajuste de DBSCAN e impresión de número de clusters y puntos de ruido
3. Reducción a 3 componentes con PCA
4. Gráfica 3D interactiva con clusters coloreados y ruido en gris

**Salida:** `dbscan_pca_3d.png`

---

### `dbscan_realista.py`
Clustering con **DBSCAN** sobre `dataset_sintetico_FIRE_UdeA_realista.csv`, usando un subconjunto específico de 13 features financieras.

**Parámetros clave:** `eps=3.0`, `min_samples=3`

**Features usadas:** ingresos totales, gastos de personal, liquidez, días de efectivo, CFO, participaciones por fuente de ingreso, HHI de fuentes, endeudamiento, tendencia de ingresos, ratio GP.

**Flujo:**
1. Selección y escalado de features financieras
2. Ajuste de DBSCAN e impresión de clusters y ruido
3. Reducción a 2 componentes con PCA y visualización

**Salida:** `dbscan_dataset_FIRE_realista.png`

---

## Dataset

| Archivo | Descripción |
|---|---|
| `dataset_sintetico_FIRE_UdeA.csv` | Dataset base con features financieras y columna `label` |
| `dataset_sintetico_FIRE_UdeA_realista.csv` | Versión extendida con columnas `label`, `anio`, `unidad` y features adicionales |

---

## Dependencias

```
pandas · numpy · matplotlib · scikit-learn
```
