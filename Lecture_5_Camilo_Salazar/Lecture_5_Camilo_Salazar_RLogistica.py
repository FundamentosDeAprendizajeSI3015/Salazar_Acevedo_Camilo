import pandas as pd
import numpy as np

from scipy.stats import reciprocal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix, accuracy_score

# ---------------------------
# 1. Cargar dataset
# ---------------------------
df = pd.read_csv("Titanic-Dataset.csv")

df = df[['Age', 'Fare', 'Survived']].dropna()

X = df[['Age', 'Fare']]
y = df['Survived']

# ---------------------------
# 2. División entrenamiento / prueba
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 3. Definir pipeline
# ---------------------------
pipeline = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=10000))
])

# ---------------------------
# 4. Distribución de parámetros
# ---------------------------
param_dist = {
    "poly__degree": [1, 2, 3],
    "model__C": reciprocal(0.001, 100),
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"]
}

# ---------------------------
# 5. Búsqueda aleatoria + CV
# ---------------------------
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=30,
    cv=5,
    random_state=42
)

# ---------------------------
# 6. Entrenamiento
# ---------------------------
search.fit(X_train, y_train)

# ---------------------------
# 7. Mejores parámetros
# ---------------------------
print("Mejores parámetros:", search.best_params_)

# ---------------------------
# 8. Evaluación
# ---------------------------
y_pred = search.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1-score:", f1)

# ---------------------------
# 9. Graficar frontera de decisión
# ---------------------------
X_set, y_set = X_train.values, y_train.values

X1, X2 = np.meshgrid(
    np.linspace(X_set[:,0].min(), X_set[:,0].max(), 200),
    np.linspace(X_set[:,1].min(), X_set[:,1].max(), 200)
)

grid = np.c_[X1.ravel(), X2.ravel()]
Z = search.predict(grid)
Z = Z.reshape(X1.shape)

plt.figure()
plt.contourf(X1, X2, Z, alpha=0.3)
plt.scatter(X_set[:,0], X_set[:,1], c=y_set)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Frontera de decisión - Logistic Regression")
plt.show()

# ---------------------------
# 10. Matriz de confusión
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()