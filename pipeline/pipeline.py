import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("netflix_merged_intersection.csv")

'''

df.drop(columns=["type_external"])

df.to_csv("netflix_merged_intersection.csv")

'''
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
        "age_certification"
    ]
].copy()

#Manejo de NaNs

df = df.dropna(subset=["listed_in"])

df["imdb_score"] = df["imdb_score"].fillna(df["imdb_score"].median())
df["imdb_votes"] = df["imdb_votes"].fillna(0)
df["country"] = df["country"].fillna("unkown")
df["age_certification"] = df["age_certification"].fillna("unknown")

df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)
df.drop(columns=["duration"], inplace=True)

#Target
df["listed_in"] = (
    df["listed_in"]
    .str.lower()
    .str.strip()
    .str.split(",")
)

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["listed_in"])

#def X
X = df.drop(columns=["listed_in"])

#def categoricas y numéricas

numeric_cols = [
    "release_year",
    "imdb_score",
    "imdb_votes",
    "duration_num"
]

categorical_cols = [
    "type_base",
    "rating",
    "age_certification",
    "country"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)


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



