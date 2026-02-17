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

