# Informe 2 — Análisis y Reclasificación de Géneros con Aprendizaje No Supervisado

Este proyecto corresponde al **Informe 2** de un análisis de aprendizaje automático aplicado a un dataset de **películas y series de Netflix**.  
El objetivo principal es **evaluar y corregir posibles errores en las etiquetas de género** utilizando **métodos de aprendizaje no supervisado**, y posteriormente **entrenar modelos supervisados** para comparar el rendimiento antes y después de la corrección.

El proyecto parte de la hipótesis de que **una proporción significativa de las etiquetas del dataset puede estar incorrecta** (hasta aproximadamente un 30%), por lo que se utilizan técnicas de clustering para detectar estructuras naturales en los datos y reevaluar dichas etiquetas.

---

# Objetivos del proyecto

- Analizar la estructura del dataset mediante **métodos de clustering**.
- Detectar **posibles inconsistencias en las etiquetas de género**.
- Realizar una **reclasificación de muestras** basada en los clusters encontrados.
- Comparar el rendimiento de **modelos supervisados entrenados con etiquetas originales vs etiquetas corregidas**.
- Visualizar la distribución de los datos mediante **reducción de dimensionalidad con t-SNE**.

---

# Dataset

El dataset contiene información sobre **películas y series de Netflix**, incluyendo múltiples características asociadas al contenido.

Debido a que una producción puede pertenecer a **varios géneros simultáneamente**, para este análisis se seleccionó **un único género principal por muestra**, con el fin de simplificar el problema de clasificación.

El dataset se encuentra incluido en el repositorio.

---

# Metodología

El flujo general del proyecto es el siguiente:

1. **Preprocesamiento de datos**
2. **Aplicación de métodos de clustering**
3. **Análisis de la distribución de clusters**
4. **Reevaluación de etiquetas**
5. **Entrenamiento de modelos supervisados**
6. **Comparación de resultados**
7. **Visualización de clusters con t-SNE**

---


---

# Descripción de archivos

## `pipeline.py`

Este archivo contiene el **pipeline de preprocesamiento de datos**.

Las tareas principales incluyen:

- Limpieza de datos
- Transformación de variables categóricas
- Vectorización de características
- Preparación del dataset para los algoritmos de machine learning

Este pipeline ya estaba implementado previamente y es utilizado por los demás scripts del proyecto.

---

## `analisis.py`

Archivo donde se realizó el **análisis inicial con métodos de clustering**.

Incluye la aplicación de distintos algoritmos de aprendizaje no supervisado, entre ellos:

- **K-Means**
- **Fuzzy C-Means**
- **Subtractive Clustering**
- **DBSCAN**
- Métodos de la familia **Cluster**

Durante esta fase se utilizó el **método del codo (Elbow Method)** para estimar el número óptimo de clusters.

Sin embargo, este método sugirió **K = 2**, lo que produjo una **agrupación excesivamente simplificada**, colapsando gran parte del dataset en un solo género dominante.

Este comportamiento generó una distribución de etiquetas poco realista.

---

## `relabel_fixed.py`

Este archivo contiene la **versión corregida del proceso de reclasificación**.

Debido a los problemas detectados en el clustering inicial, se realizaron las siguientes modificaciones:

- Se **forzó un número mayor de clusters** para capturar mejor la diversidad del dataset.
- Se utilizó una combinación de:
  
  - **K-Means**
  - **Gaussian Mixture Models (GMM)**

- Se aplicó un **umbral de confianza** para evitar cambios arbitrarios en las etiquetas.

Esto permitió:

- Evitar el colapso de clases
- Detectar posibles errores en las etiquetas originales
- Generar una redistribución más realista de los géneros

Posteriormente, en este mismo archivo se realiza nuevamente el **entrenamiento de modelos supervisados**, utilizando:

- Árboles de decisión
- Regresión logística
- Regresión lineal

Finalmente se comparan los resultados obtenidos con:

- Dataset original
- Dataset corregido

---

## `visualize_clusters.py`

Este script se utiliza para **visualizar la estructura de los datos mediante t-SNE**.

t-SNE es un método de **reducción de dimensionalidad no lineal** que permite proyectar datos de alta dimensionalidad a **dos dimensiones**, facilitando la visualización de clusters.

Es importante notar que:

> t-SNE se utiliza únicamente para visualización y no afecta el entrenamiento de los algoritmos de clustering.

Esto permite observar de forma más clara la distribución de los datos y la separación entre grupos.

---

# Resultados generales

El análisis inicial reveló que:

- El método del codo sugería **K = 2 clusters**, lo que provocaba una agrupación excesiva en un solo género (principalmente **drama**).
- Esto generaba **métricas artificialmente altas** en los modelos supervisados, debido a la pérdida de diversidad en las clases.

Tras aplicar la corrección en el proceso de clustering:

- Se logró una **distribución de géneros más equilibrada**.
- Se detectó que aproximadamente **un 30% de las etiquetas podrían estar incorrectamente asignadas**, lo cual coincide con la hipótesis planteada inicialmente.
- Los modelos supervisados entrenados con el dataset corregido presentan **resultados más realistas y representativos del problema**.

---

# Tecnologías utilizadas

- Python
- NumPy
- Pandas
- Scikit-learn
- Scikit-fuzzy
- Matplotlib
- Seaborn
- t-SNE

---

# Autor:
Camilo Salazar Acevedo
