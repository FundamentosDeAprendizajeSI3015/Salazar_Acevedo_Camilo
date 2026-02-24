import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# ── Transformer personalizado para agrupar países poco frecuentes ──────────────
class CountryGrouper(BaseEstimator, TransformerMixin):
    """Agrupa en 'others' los países que aparecen una sola vez en el conjunto."""

    def fit(self, X, y=None):
        # X es un array 2D (salida de ColumnTransformer), tomamos la columna country
        # Pero lo usaremos directamente sobre la Series antes del pipeline
        return self

    def transform(self, X, y=None):
        return X


def group_rare_countries(series: pd.Series, min_count: int = 1) -> pd.Series:
    """
    Reemplaza por 'others' los valores cuya frecuencia sea <= min_count.
    Por defecto agrupa los que aparecen exactamente 1 vez.
    """
    counts = series.value_counts()
    rare = counts[counts <= min_count].index
    return series.replace(rare, "others")


# ── Función principal que carga, limpia y divide el dataset ───────────────────
def load_and_prepare(
    filepath: str = "netflix_merged_intersection.csv",
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
):
    """
    Retorna:
        X_train, X_val, X_test  → DataFrames de características
        y_train, y_val, y_test  → arrays MultiLabel binarizados
        preprocessor            → ColumnTransformer (sin ajustar)
        mlb                     → MultiLabelBinarizer ajustado
    """

    # ── 1. Carga ───────────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)

    # ── 2. Limpieza ────────────────────────────────────────────────────────────
    # Eliminar filas donde director es "Not Given"
    df = df[df["director"] != "Not Given"].copy()

    # Seleccionar columnas relevantes
    df = df[
        [
            "type_base",
            "country",
            "release_year",
            "rating",
            "duration",
            "listed_in",
            "imdb_score",
            "imdb_votes",
            "age_certification",
        ]
    ].copy()

    # Manejo de NaNs
    df = df.dropna(subset=["listed_in"])
    df["imdb_score"] = df["imdb_score"].fillna(df["imdb_score"].median())
    df["imdb_votes"] = df["imdb_votes"].fillna(0)
    df["country"] = df["country"].fillna("unknown")
    df["age_certification"] = df["age_certification"].fillna("unknown")

    # Convertir duración a número
    df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df.drop(columns=["duration"], inplace=True)

    # ── 3. Agrupar países con frecuencia <= 1 en "others" ─────────────────────
    df["country"] = group_rare_countries(df["country"], min_count=1)

    # ── 4. Target (multilabel) ────────────────────────────────────────────────
    df["listed_in"] = (
        df["listed_in"]
        .str.lower()
        .str.strip()
        .str.split(",")
        .apply(lambda tags: [t.strip() for t in tags])
    )

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["listed_in"])

    # ── 5. Features ───────────────────────────────────────────────────────────
    X = df.drop(columns=["listed_in"])

    numeric_cols = ["release_year", "imdb_score", "imdb_votes", "duration_num"]
    categorical_cols = ["type_base", "rating", "age_certification", "country"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    # ── 6. Split 60 / 20 / 20 ────────────────────────────────────────────────
    # Primero separamos el 60% de entrenamiento del 40% restante
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,          # 40% temporal (val + test)
        random_state=random_state,
    )

    # Del 40% restante, 50% es validación y 50% es test → cada uno = 20% del total
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=random_state,
    )

    print(f"Train:      {X_train.shape[0]} filas ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validación: {X_val.shape[0]} filas ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test:       {X_test.shape[0]} filas ({X_test.shape[0]/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, mlb


# ── Ejecución directa ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, mlb = load_and_prepare()
    print("\nClases detectadas:", mlb.classes_[:5], "...")

'''
#diagramas
os.makedirs("eda_outputs", exist_ok=True)

stats = pd.DataFrame()



stats["mean"] = df[numeric_cols].mean()
stats["median"] = df[numeric_cols].median()
stats["std"] = df[numeric_cols].std()
stats["var"] = df[numeric_cols].var()
stats["min"] = df[numeric_cols].min()
stats["25%"] = df[numeric_cols].quantile(0.25)
stats["50%"] = df[numeric_cols].quantile(0.50)
stats["75%"] = df[numeric_cols].quantile(0.75)
stats["max"] = df[numeric_cols].max()

stats.to_csv("stadistical_summary.csv")

outliers_summary = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    outliers_summary[col] = len(outliers)

pd.DataFrame.from_dict(outliers_summary, orient="index", columns=["n_outliers"])\
    .to_csv("outliers_summary.csv")

#histogramas

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Hisotgrama de {col}")
    plt.savefig(f"eda_outputs/hist_{col}.png")
    plt.close()

#gráficos de dispersión

pairs = [
    ("imdb_score", "imdb_votes"),
    ("release_year", "imdb_score"),
    ("duration_num", "imdb_score"),
]

for x, y in pairs:
    plt.figure()
    sns.scatterplot(x=df[x], y=df[y])
    plt.title(f"{x} vs {y}")
    plt.savefig(f"eda_outputs/scatter_{x}_vs{y}.png")

    plt.close()

plt.figure()
corr_matrix = df[numeric_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("matriz de correlación")
plt.savefig("eda_outputs/correlation_matrix.png")
plt.close()




from sklearn.decomposition import PCA

# =============================
# Aplicar preprocesamiento
# =============================

X_processed = preprocessor.fit_transform(X)

# =============================
# PCA a 2 componentes
# =============================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# =============================
# Crear DataFrame para graficar
# =============================

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

# Guardar varianza explicada
explained_variance = pca.explained_variance_ratio_
pd.DataFrame(
    explained_variance,
    index=["PC1", "PC2"],
    columns=["Explained Variance Ratio"]
).to_csv("eda_outputs/pca_explained_variance.csv")

# =============================
# Gráfica PCA
# =============================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=pca_df["PC1"],
    y=pca_df["PC2"],
    hue=df["type_base"]  # colorear por tipo
)

plt.title("PCA - Proyección en 2 Componentes")
plt.xlabel(f"PC1 ({explained_variance[0]*100:.2f}% varianza)")
plt.ylabel(f"PC2 ({explained_variance[1]*100:.2f}% varianza)")

plt.legend()
plt.savefig("eda_outputs/pca_projection.png")
plt.close()


# Crear columna con género principal
df["main_genre"] = df["listed_in"].apply(lambda x: x[0] if len(x) > 0 else "unknown")

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=pca_df["PC1"],
    y=pca_df["PC2"],
    hue=df["main_genre"],
    palette="tab20",
    legend=False
)

plt.title("PCA - Coloreado por Género Principal")
plt.savefig("eda_outputs/pca_by_main_genre.png")
plt.close()


from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=1000
)

X_tsne = tsne.fit_transform(X_processed)

tsne_df = pd.DataFrame(
    X_tsne,
    columns=["Dim1", "Dim2"]
)

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=tsne_df["Dim1"],
    y=tsne_df["Dim2"],
    hue=df["main_genre"],
    palette="tab20",
    legend=False,
    alpha=0.7
)

plt.title("t-SNE Projection")
plt.savefig("eda_outputs/tsne_projection.png")
plt.close()


import umap

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

X_umap = umap_model.fit_transform(X_processed)

umap_df = pd.DataFrame(
    X_umap,
    columns=["Dim1", "Dim2"]
)

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=umap_df["Dim1"],
    y=umap_df["Dim2"],
    hue=df["main_genre"],
    palette="tab20",
    legend=False,
    alpha=0.7
)

plt.title("UMAP Projection")
plt.savefig("eda_outputs/umap_projection.png")
plt.close()

'''

