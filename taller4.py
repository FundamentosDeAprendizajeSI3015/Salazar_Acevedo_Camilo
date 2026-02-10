import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

route = "Titanic-Dataset.csv"

df = pd.read_csv(route)
print(df)

#Media supervivientes
media_super = df['Survived'].mean()
print(f"Media de Supervivientes: {media_super:.2f}")

#Media edad
media_edad = df['Age'].mean()
print(f"Media de Edad: {media_edad:.0f}")

#Media precio
media_precio = df['Fare'].mean()
print(f"Media de Precio: {media_precio:.2f}")

# Mediana de supervivientes, edad, y precio
mediana_todas_col = df[['Survived', 'Age', 'Fare']].median()
print("\nMediana de todas las columnas numéricas:")
print(mediana_todas_col)

# Moda de supervivientes, edad, precio y sexo
moda_todas = df[['Survived' , 'Age', 'Fare', 'Sex']].mode()
print("\n Moda de las columnas anteriores, y a sexo")
print(moda_todas)

# Varianza
varianza = df[['Survived', 'Age', 'Fare']].var()
print("\nVarianza:")
print(varianza)

# Desviación estándar
desv_estandar = df[['Survived', 'Age', 'Fare']].std()
print("\nDesviación estándar:")
print(desv_estandar)

# Rango
rango = df[['Survived', 'Age', 'Fare']].max() - df[['Survived', 'Age', 'Fare']].min()
print("\nRango:")
print(rango)

# Cuartiles
cuartiles = df[['Age', 'Fare']].quantile([0.25, 0.50, 0.75])
print("\nCuartiles (Q1, Q2, Q3):")
print(cuartiles)

# Percentiles (ejemplo: 10, 25, 50, 75, 90)
percentiles = df[['Age', 'Fare']].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
print("\nPercentiles:")
print(percentiles)

data_cols = ['Age', 'Fare']

# Detectar y eliminar outliers usando el rango intercuartílico (IQR)
for col in data_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"\nOutliers detectados en {col}: {len(outliers)}")
    if len(outliers) > 0:
        print(outliers[[col]].head())
 
# Eliminar outliers de todas las columnas numéricas principales
for col in data_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper) | df[col].isnull()]
 
print("\nDataFrame después de eliminar outliers (primeras filas):")
print(df.head())

# Histograma de edad
plt.figure()
plt.hist(df['Age'].dropna(), bins=20)
plt.title('Histograma de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

# Histograma de pasaje
plt.figure()
plt.hist(df['Fare'].dropna(), bins=20)
plt.title('Histograma de Precio del Pasaje')
plt.xlabel('Fare')
plt.ylabel('Frecuencia')
plt.show()

# gráfico de dispersión Age vs Fare
plt.figure()

# No sobrevivientes
plt.scatter(
    df[df['Survived'] == 0]['Age'],
    df[df['Survived'] == 0]['Fare'],
    label='No sobrevivió'
)

# Sobrevivientes
plt.scatter(
    df[df['Survived'] == 1]['Age'],
    df[df['Survived'] == 1]['Fare'],
    label='Sobrevivió'
)

plt.xlabel('Edad')
plt.ylabel('Precio del Pasaje')
plt.title('Edad vs Precio del Pasaje según Supervivencia')
plt.legend()
plt.show()

# one hot encoding sobre género
df_onehot = pd.get_dummies(df, columns=['Sex'], drop_first=True)

print("\nDataFrame con One Hot Encoding (Sex):")
print(df_onehot[['Sex_male']].head())


df['Embarked'] = df['Embarked'].fillna('Missing')

# LABEL ENCODING
embarked_map = {cat: idx for idx, cat in enumerate(df['Embarked'].unique())}

df['Embarked_label'] = df['Embarked'].map(embarked_map)

print("\nMapeo Embarked → Entero:")
print(embarked_map)
print(df[['Embarked', 'Embarked_label']].head())

num_bits = int(np.ceil(np.log2(df['Embarked_label'].nunique())))

def to_binary_array(x, bits):
    return list(map(int, format(x, f'0{bits}b')))

binary_cols = df['Embarked_label'].apply(lambda x: to_binary_array(x, num_bits))

binary_df = pd.DataFrame(
    binary_cols.tolist(),
    columns=[f'Embarked_bin_{i}' for i in range(num_bits)],
    index=df.index
)

# Unir al DataFrame original
df = pd.concat([df, binary_df], axis=1)

print("\nBinary Encoding de Embarked (manual):")
print(df[[f'Embarked_bin_{i}' for i in range(num_bits)]].head())

corr_cols = [
    'Survived', 'Age', 'Fare', 
    'Sex_male', 
    'Embarked_label'
]

df_corr = df_onehot.copy()

df_corr['Embarked'] = df_corr['Embarked'].fillna('Missing')

embarked_map = {cat: idx for idx, cat in enumerate(df_corr['Embarked'].unique())}
df_corr['Embarked_label'] = df_corr['Embarked'].map(embarked_map)

correlation_matrix = df_corr[corr_cols].corr()

print("\nMatriz de correlación:")
print(correlation_matrix)

plt.figure()
plt.imshow(correlation_matrix)
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Matriz de Correlación")
plt.show()


# Standard Scaler
scaler = StandardScaler()

df[['Age_scaled', 'Fare_scaled']] = scaler.fit_transform(
    df[['Age', 'Fare']]
)

print("\nColumnas escaladas con StandardScaler:")
print(df[['Age_scaled', 'Fare_scaled']].head())

# transformación logaritmica
df['Fare_log'] = np.log1p(df['Fare'])

print("\nTransformación logarítmica de Fare:")
print(df[['Fare', 'Fare_log']].head())

plt.figure()
plt.hist(df['Fare'], bins=20)
plt.title("Fare original")
plt.show()

plt.figure()
plt.hist(df['Fare_log'], bins=20)
plt.title("Fare con transformación logarítmica")
plt.show()


print("En conjunto, el análisis evidencia que la supervivencia en el Titanic estuvo fuertemente influenciada por factores sociales como el sexo y el nivel socioeconómico, mientras que variables como la edad y el puerto de embarque tuvieron un impacto menor. Las transformaciones aplicadas mejoraron la interpretabilidad de los datos y los dejaron preparados para etapas posteriores de modelado predictivo.")