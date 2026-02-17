
#  Informe 1 – Exploratory Data Analysis & Preprocessing

##  Objetivo

Realizar un análisis exploratorio completo sobre un dataset combinado de contenido de Netflix e información adicional de IMDb, aplicando técnicas de:

- Limpieza de datos  
- Manejo de valores faltantes  
- Detección de outliers  
- Análisis estadístico descriptivo  
- Visualización  
- Transformación de variables  
- Codificación de variables categóricas  
- Preparación para modelado multietiqueta  

---

##  Dataset

Se trabajó con un dataset resultante de la intersección entre:

- Información base de títulos de Netflix  
- Información externa con métricas de IMDb  

Variables utilizadas:

- `type_base`  
- `country`  
- `release_year`  
- `rating`  
- `duration`  
- `listed_in` (variable objetivo multietiqueta)  
- `imdb_score`  
- `imdb_votes`  
- `age_certification`  

---

##  Preprocesamiento Realizado

###  Manejo de valores faltantes

- Eliminación de filas sin target  
- Imputación por mediana en variables numéricas  
- Imputación categórica con etiqueta `"unknown"`  

###  Feature Engineering

- Extracción numérica de duración (`duration_num`)  
- Transformación de la variable objetivo en formato multietiqueta  
- Codificación mediante `MultiLabelBinarizer`  

###  Separación de Variables

- Definición de `X` (features)  
- Definición de `y` (target multietiqueta)  

---

##  Análisis Exploratorio (EDA)

Se realizaron:

- Medidas de tendencia central (media, mediana)  
- Medidas de dispersión (varianza, desviación estándar)  
- Cuartiles y rangos  
- Detección de outliers mediante método IQR  
- Histogramas con KDE  
- Gráficos de dispersión entre variables relevantes  
- Matriz de correlación  

Todos los gráficos se generan automáticamente y se almacenan en:
`/eda_outputs`
