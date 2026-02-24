# Lecture 6 Netflix Genre Clustering — ML Pipeline

Proyecto de clasificación sobre el dataset de Netflix (`netflix_merged_intersection.csv`), que combina datos de Netflix con scores de IMDB. El objetivo es predecir a qué grupo temático pertenece un título usando **Random Forest** y **Gradient Boosting**.

---

## Estructura del proyecto

```                
├── models.py                    # Clustering, balanceo, entrenamiento y métricas
├── netflix_merged_intersection.csv
└── outputs/
    ├── confusion_matrices.png
    ├── train_vs_val.png
    ├── rf_vs_gb_val.png
    └── class_distribution.png
```

---

## Pipeline (`pipeline.py`)

### Limpieza y transformación

- Se eliminan las filas donde `director == "Not Given"` (~853 filas).
- Se seleccionan las columnas relevantes: `type_base`, `country`, `release_year`, `rating`, `duration`, `listed_in`, `imdb_score`, `imdb_votes`, `age_certification`.
- Manejo de valores nulos:
  - `imdb_score` → rellena con la mediana.
  - `imdb_votes` → rellena con `0`.
  - `country` → rellena con `"unknown"`.
  - `age_certification` → rellena con `"unknown"`.
  - Filas sin `listed_in` → se eliminan.
- La columna `duration` se convierte a número extrayendo los dígitos (`duration_num`) y se elimina la original.
- Los países que aparecen **una sola vez** se agrupan en `"others"` para reducir cardinalidad.

### Filtro de géneros raros

La columna `listed_in` es multilabel (una fila puede tener varios géneros). Se parsean, se normalizan a minúsculas y se eliminan los géneros con menos de **30 apariciones** globales, reduciendo de 41 géneros a **18 géneros válidos**.  
Las filas que quedan sin ningún género válido tras el filtro también se eliminan (~16 filas).

### Features y preprocesamiento

| Tipo | Columnas | Transformación |
|---|---|---|
| Numéricas | `release_year`, `imdb_score`, `imdb_votes`, `duration_num` | `StandardScaler` |
| Categóricas | `type_base`, `rating`, `age_certification`, `country` | `OneHotEncoder` |

### Split del dataset

División en tres conjuntos con `train_test_split` encadenado:

```
60% Train  →  1324 filas
20% Val    →   441 filas
20% Test   →   442 filas
```

La función `load_and_prepare()` retorna `X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, mlb` y puede importarse directamente desde `models.py`.

---

## Modelos (`models.py`)

### Identificación de clusters (variable objetivo)

En lugar de usar etiquetas de género directamente, se aplica **KMeans (k=3)** sobre los features procesados para identificar agrupaciones naturales en los datos, equivalentes a los 3 clusters visibles en la proyección UMAP del dataset.

Los clusters se caracterizan automáticamente por sus promedios de `release_year`, `imdb_score` y `duration_num`, y se etiquetan con nombres interpretables:

| Cluster | Etiqueta | N | Año medio | IMDB medio | Duración media |
|---|---|---|---|---|---|
| 0 | **Clásicos/Premium** | 205 | 1997.7 | 6.89 | 132.4 min |
| 1 | **Entretenimiento Popular** | 812 | 2017.3 | 5.17 | 102.0 min |
| 2 | **Contenido Diverso** | 1190 | 2017.3 | 6.94 | 92.3 min |

> El cluster "Clásicos/Premium" corresponde al grupo aislado a la izquierda del UMAP: películas antiguas, largas y bien valoradas que el espacio de features separa claramente del resto.

### Undersampling

Las tres clases están desbalanceadas (132 / 493 / 699 en train). Para evitar que los modelos favorezcan la clase mayoritaria se aplica **undersampling aleatorio** sin reemplazo, reduciendo todas las clases al tamaño de la minoritaria:

```
Antes  →  132 / 493 / 699
Después → 132 / 132 / 132  (396 filas totales de entrenamiento)
```

### Entrenamiento

Se definen dos pipelines independientes, cada uno con su propio `ColumnTransformer` para evitar data leakage:

**Random Forest** (bagging de árboles de decisión):
```python
RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
```

**Gradient Boosting** (boosting secuencial de árboles):
```python
GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
```

### Métricas (Validación)

| Modelo | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| Random Forest | 0.9229 | 0.8767 | 0.9278 |
| Gradient Boosting | **0.9252** | **0.8877** | **0.9283** |

Ambos modelos reportan accuracy perfecta en train (1.0), lo que indica **overfitting**, aunque generalizan bien a validación (~0.92). Gradient Boosting supera ligeramente a Random Forest en todas las métricas de validación.

El punto débil de ambos modelos es la **precisión en "Clásicos/Premium"** (~0.59–0.64), ya que esa clase tiene pocas muestras de evaluación (32 en validación). El recall es perfecto (1.0) porque sus features son muy distintivos.

---

## Gráficas generadas

- **`confusion_matrices.png`** — cuatro matrices de confusión en grilla 2×2 (modelo × split: train y validación).
- **`train_vs_val.png`** — barras comparando Accuracy, Precision, Recall y F1 (macro) entre train y validación, una gráfica por modelo.
- **`rf_vs_gb_val.png`** — comparativa de todas las métricas (macro y weighted) entre Random Forest y Gradient Boosting sobre el conjunto de validación.
- **`class_distribution.png`** — distribución de clases antes y después del undersampling.

---

## Dependencias

```
pandas
numpy
scikit-learn
matplotlib
```

## Uso

```bash
# Solo el pipeline
python pipeline.py

# Entrenamiento completo + métricas + gráficas
python models.py
```
