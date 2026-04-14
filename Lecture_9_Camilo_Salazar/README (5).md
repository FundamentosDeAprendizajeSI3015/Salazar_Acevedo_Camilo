# Lecture 9 — Camilo Salazar

Implementación y comparación de algoritmos de clustering no supervisado (**KMeans** y **DBSCAN**) sobre datasets del proyecto FIRE UdeA, incluyendo selección del número óptimo de clusters mediante el **Método del Codo**.

---

## Archivos

### `agrupamiento_kmeans_dbscan.py`
Clustering sobre `dataset_sintetico_FIRE_UdeA.csv` (500 filas, sin valores nulos).

**Features usadas:** `liquidez`, `dias_efectivo`, `cfo`, `participacion_ley30`, `hhi_fuentes`, `gastos_personal`, `tendencia_ingresos`

**Visualización 2D:** `liquidez` vs `dias_efectivo`

**Flujo:**
1. Carga, selección de features y escalado con `StandardScaler`
2. KMeans con K=2 → gráfica y reporte de inercia
3. Método del Codo (K de 1 a 10) para seleccionar K óptimo
4. KMeans con K óptimo (ajustable según la gráfica del codo, default `k=2`)
5. DBSCAN con pipeline `StandardScaler → DBSCAN(eps=1.2, min_samples=5)` → reporte de clusters y puntos de ruido

**Salidas:** `kmeans_k2.png`, `metodo_codo.png`, `kmeans_k{k_optimo}.png`, `dbscan.png`

---

### `agrupamiento_kmeans_dbscan_v2.py`
Clustering sobre `dataset_sintetico_FIRE_UdeA_realista.csv`, con un conjunto más amplio de features financieras.

**Features usadas:** `ingresos_totales`, `gastos_personal`, `liquidez`, `dias_efectivo`, `cfo`, `participacion_ley30`, `participacion_regalias`, `participacion_servicios`, `participacion_matriculas`, `hhi_fuentes`, `endeudamiento`, `tendencia_ingresos`, `gp_ratio`

**Visualización 2D:** `liquidez` vs `gp_ratio`

**Flujo:**
1. Carga, selección de features (descartando `anio`, `unidad`, `label`) y escalado
2. KMeans con K=2 → gráfica y reporte de inercia
3. Método del Codo (K de 1 a 10)
4. KMeans con K óptimo (ajustable, default `k=3`)
5. DBSCAN con pipeline `StandardScaler → DBSCAN(eps=1.5, min_samples=3)` → reporte de clusters y ruido

**Salidas:** `kmeans_k2.png`, `metodo_codo.png`, `kmeans_k{k_optimo}.png`, `dbscan.png`

---

## Diferencias entre versiones

| | `v1` | `v2` |
|---|---|---|
| Dataset | `FIRE_UdeA.csv` | `FIRE_UdeA_realista.csv` |
| Nº de features | 7 | 13 |
| Visualización | liquidez vs dias_efectivo | liquidez vs gp_ratio |
| DBSCAN eps | 1.2 | 1.5 |
| DBSCAN min_samples | 5 | 3 |
| K óptimo (default) | 2 | 3 |

---

## Datasets

| Archivo | Descripción |
|---|---|
| `dataset_sintetico_FIRE_UdeA.csv` | Dataset base, 500 filas, 7 features financieras y columna `label` |
| `dataset_sintetico_FIRE_UdeA_realista.csv` | Versión extendida con 13 features, columnas `anio` y `unidad` |

---

## Dependencias

```
pandas · numpy · matplotlib · scikit-learn
```
