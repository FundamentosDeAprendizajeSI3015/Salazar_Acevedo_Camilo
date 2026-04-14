# Lecture 4 — Camilo Salazar

Análisis exploratorio de datos (EDA) y preprocesamiento sobre el dataset del **Titanic**, cubriendo estadística descriptiva, detección de outliers, codificación de variables categóricas, correlación y escalado de features.

---

## Archivos

### `taller4.py`
Script principal de análisis sobre `Titanic-Dataset.csv`.

**Flujo:**

1. **Estadística descriptiva**
   - Media, mediana y moda de `Survived`, `Age` y `Fare`
   - Varianza, desviación estándar y rango
   - Cuartiles (Q1, Q2, Q3) y percentiles (10, 25, 50, 75, 90)

2. **Detección y eliminación de outliers**
   - Método IQR sobre `Age` y `Fare`
   - Reporte de cantidad de outliers por columna antes de filtrar

3. **Visualizaciones**
   - Histogramas de `Age` y `Fare`
   - Scatter plot de Edad vs Precio del Pasaje coloreado por supervivencia

4. **Codificación de variables categóricas**
   - **One-Hot Encoding** sobre `Sex`
   - **Label Encoding** sobre `Embarked`
   - **Binary Encoding** manual sobre `Embarked`

5. **Matriz de correlación**
   - Calculada sobre `Survived`, `Age`, `Fare`, `Sex_male` y `Embarked_label`
   - Visualizada con `imshow`

6. **Escalado y transformaciones**
   - `StandardScaler` sobre `Age` y `Fare`
   - Transformación logarítmica (`log1p`) sobre `Fare`, con histogramas comparativos

---

## Dataset

| Archivo | Descripción |
|---|---|
| `Titanic-Dataset.csv` | Dataset clásico del Titanic con variables demográficas, socioeconómicas y de supervivencia |

---

## Conclusión

El análisis evidencia que la supervivencia en el Titanic estuvo fuertemente influenciada por el sexo y el nivel socioeconómico. La edad y el puerto de embarque tuvieron un impacto menor. Las transformaciones aplicadas mejoran la interpretabilidad de los datos y los preparan para etapas posteriores de modelado predictivo.

---

## Dependencias

```
pandas · numpy · matplotlib · scikit-learn
```
