# Lecture 5 — Camilo Salazar

Implementación y comparación de modelos de **regresión lineal regularizada** (Ridge y Lasso) y **regresión logística** sobre el dataset del Titanic, con búsqueda de hiperparámetros mediante `RandomizedSearchCV` y pipelines de preprocesamiento.

---

## Archivos

### `Lecture_5_Camilo_Salazar_RLineal.py`
Predicción del precio del pasaje (`Fare`) a partir de la edad (`Age`) usando regresión polinomial regularizada.

**Flujo:**

1. Carga del dataset y selección de columnas `Age` y `Fare`
2. División entrenamiento/prueba (80/20)
3. Visualización del scatter plot de train vs test
4. Definición de pipelines con `PolynomialFeatures` + `StandardScaler` + modelo (Ridge / Lasso)
5. Búsqueda aleatoria de hiperparámetros (`degree` ∈ {1,2,3,4}, `alpha` ∈ distribución recíproca) con 5-fold CV
6. Evaluación con **R²** y **MAE** sobre el conjunto de prueba
7. Visualización de las curvas de predicción de Ridge y Lasso sobre los datos de entrenamiento

**Métricas reportadas:** R², MAE

---

### `Lecture_5_Camilo_Salazar_RLogistica.py`
Clasificación de supervivencia (`Survived`) a partir de `Age` y `Fare` usando regresión logística polinomial.

**Flujo:**

1. Carga del dataset y selección de columnas `Age`, `Fare` y `Survived`
2. División entrenamiento/prueba (80/20)
3. Pipeline con `PolynomialFeatures` + `StandardScaler` + `LogisticRegression`
4. Búsqueda aleatoria de hiperparámetros (`degree` ∈ {1,2,3}, `C` ∈ distribución recíproca) con 5-fold CV
5. Evaluación con **Accuracy** y **F1-score**
6. Visualización de la frontera de decisión sobre el espacio Age–Fare
7. Matriz de confusión con `ConfusionMatrixDisplay`

**Métricas reportadas:** Accuracy, F1-score

---

## Dataset

| Archivo | Descripción |
|---|---|
| `Titanic-Dataset.csv` | Dataset clásico del Titanic con variables demográficas, socioeconómicas y de supervivencia |

---

## Dependencias

```
pandas · numpy · scipy · matplotlib · scikit-learn
```
