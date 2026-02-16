import pandas as pd
import numpy as np
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

df["imdb_score"] = df["imdb_score"].fillna(df["imdb_score"].median)
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
    "type",
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

