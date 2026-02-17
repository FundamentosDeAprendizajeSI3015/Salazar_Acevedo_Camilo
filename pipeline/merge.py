import pandas as pd

# =========================
# 1. Cargar datasets
# =========================

df_base = pd.read_csv("netflix1.csv")
df_tv_movies = pd.read_csv("Netflix TV Shows and Movies.csv")
'''
# =========================
# 2. Normalizar títulos
# =========================

def normalize_title(title):
    if pd.isna(title):
        return ""
    return title.strip().lower()

df_base["title_norm"] = df_base["title"].apply(normalize_title)
df_tv_movies["title_norm"] = df_tv_movies["title"].apply(normalize_title)

# =========================
# 3. Asegurar tipo numérico en release_year
# =========================

df_base["release_year"] = pd.to_numeric(df_base["release_year"], errors="coerce")
df_tv_movies["release_year"] = pd.to_numeric(df_tv_movies["release_year"], errors="coerce")

# =========================
# 4. Merge (INTERSECCIÓN)
# =========================

df_merged = pd.merge(
    df_base,
    df_tv_movies,
    on=["title_norm", "release_year"],
    how="inner",
    suffixes=("_base", "_external")
)

# =========================
# 5. Eliminar columna auxiliar
# =========================

df_merged.drop(columns=["title_norm"], inplace=True)

# =========================
# 6. Guardar nuevo CSV
# =========================

df_merged.to_csv("netflix_merged_intersection.csv", index=False)

print("Merge completado.")
print("Filas resultantes:", df_merged.shape[0])
print("Columnas resultantes:", df_merged.shape[1])

'''
df_merged = pd.read_csv("netflix_merged_intersection.csv")

df_merged.drop(columns=["type_external"], inplace=True)

df_merged.to_csv("netflix_merged_intersection.csv")