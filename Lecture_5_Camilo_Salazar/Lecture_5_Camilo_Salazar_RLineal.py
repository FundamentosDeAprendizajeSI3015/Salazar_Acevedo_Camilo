import pandas as pd
import numpy as np

from scipy.stats import reciprocal

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt

# ---------------------------
# 1. Cargar dataset
# ---------------------------
df = pd.read_csv("Titanic-Dataset.csv")

# Nos quedamos solo con Age y Fare
df = df[['Age', 'Fare']].dropna()

X = df[['Age']]
y = df['Fare']

# ---------------------------
# 2. División entrenamiento / prueba
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 3. Graficar train y test
# ---------------------------
plt.figure()
plt.scatter(X_train, y_train, label="Train")
plt.scatter(X_test, y_test, label="Test")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend()
plt.show()

# ---------------------------
# 4. Definir pipelines
# ---------------------------
ridge_pipeline = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

lasso_pipeline = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=10000))
])

# ---------------------------
# 5. Definir distribuciones de parámetros
# ---------------------------
param_dist = {
    "poly__degree": [1, 2, 3, 4],
    "model__alpha": reciprocal(0.001, 100)
}

# ---------------------------
# 6. Búsqueda aleatoria con CV
# ---------------------------
ridge_search = RandomizedSearchCV(
    ridge_pipeline,
    param_dist,
    n_iter=30,
    cv=5,
    random_state=42
)

lasso_search = RandomizedSearchCV(
    lasso_pipeline,
    param_dist,
    n_iter=30,
    cv=5,
    random_state=42
)

# ---------------------------
# 7. Entrenar modelos
# ---------------------------
ridge_search.fit(X_train, y_train)
lasso_search.fit(X_train, y_train)

# ---------------------------
# 8. Mejores parámetros
# ---------------------------
print("Mejores parámetros Ridge:", ridge_search.best_params_)
print("Mejores parámetros Lasso:", lasso_search.best_params_)

# ---------------------------
# 9. Evaluación
# ---------------------------
ridge_pred = ridge_search.predict(X_test)
lasso_pred = lasso_search.predict(X_test)

ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

ridge_mae = mean_absolute_error(y_test, ridge_pred)
lasso_mae = mean_absolute_error(y_test, lasso_pred)

print("Ridge R2:", ridge_r2)
print("Ridge MAE:", ridge_mae)

print("Lasso R2:", lasso_r2)
print("Lasso MAE:", lasso_mae)

# ---------------------------
# 10. Graficar predicción
# ---------------------------
X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1,1)

ridge_plot = ridge_search.predict(X_plot)
lasso_plot = lasso_search.predict(X_plot)

plt.figure()
plt.scatter(X_train, y_train)
plt.plot(X_plot, ridge_plot)
plt.plot(X_plot, lasso_plot)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(["Datos entrenamiento", "Ridge", "Lasso"])
plt.show()